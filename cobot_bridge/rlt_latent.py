"""RLT-Latent route: latent residual injected into the SmolVLA flow denoise.

New test route, distinct from the deployed action-residual route.  A
lightweight actor emits a small latent residual ``r_lat`` which is injected
into the flow-matching *conditioning* of the frozen VLA:

    c'_t = c_t + W r_lat        (at every denoise step t)

where ``c_t`` is the action-query (suffix) embedding the VLM conditions on.
The VLA runs the full 10-step denoise (optionally with guided RTC); the
decoded 30-step chunk is sliced to its first 10 steps (C=10 window) to
produce the executed action ``a_exec``.

Guarantees:
  * Stage 1 (frozen VLA + RLT encoder) and the reference model are reused
    unchanged; the SmolVLA denoise only gained an additive ``suffix_offset``
    hook (``None`` = exact base path).
  * The latent actor head is zero-init (residual mode), so at init
    ``r_lat = 0`` and the decoded chunk is bit-identical to the base VLA:
    a safe passthrough.
  * Because the correction is applied *inside* the denoiser, the corrected
    chunk stays a coherent flow-matching sample and guided RTC can blend its
    prefix against the previous chunk's leftover (inter-chunk fluctuation).

The twin-Q critic stays over the EXECUTED action ``a_exec`` (unchanged iv50
semantics); actor loss terms are computed in decode space through the frozen
VLA (prefix KV under no_grad, gradients only through the 10 suffix steps).
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .convert_lerobot_joint_to_eef import _joint_to_eef
from .piper_eef_metrics import PiperForwardKinematics
from .rlt_policy_server import _decode_jpeg, _normalizer, _policy_key
from .run_lerobot_policy import _load_policy
from .rlt_stage2_data import ACTION_CHUNK_FRAME_STRIDE, contiguous_action_chunk

from lerobot.policies.smolvla.relative_eef import (
    chunk_to_relative,
    relative_chunk_to_absolute,
)

LOG = logging.getLogger("cobot_bridge.rlt_latent")

#: Semantic tag recorded in latent Stage-2 checkpoints.
LATENT_ACTOR_OUTPUT_SEMANTICS = "latent_residual_v1"

_CAMERAS = ("front", "left", "right")


# ─────────────────────────────────────────────────────────────────────────────
# Latent residual projector W: r_lat (B, C, action_dim) -> offset (B, C, H)
# ─────────────────────────────────────────────────────────────────────────────
class LatentProjector(nn.Module):
    """Projects a per-step latent residual to a suffix-embedding offset.

    ``W`` maps each step's ``action_dim``-dim residual to the VLM expert
    hidden size ``H`` (one shared weight across the C steps, matching
    ``c'_t = c_t + W r_t``).  No bias: a zero residual must map to a zero
    shift (passthrough guarantee).
    """

    def __init__(self, action_dim: int, expert_hidden: int) -> None:
        super().__init__()
        if action_dim <= 0 or expert_hidden <= 0:
            raise ValueError("action_dim and expert_hidden must be positive")
        self.action_dim = int(action_dim)
        self.expert_hidden = int(expert_hidden)
        self.weight = nn.Parameter(torch.empty(action_dim, expert_hidden))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, r_lat: Tensor) -> Tensor:
        """r_lat (B, C, D) -> offset (B, C, H)."""
        if r_lat.shape[-1] != self.action_dim:
            raise ValueError(
                f"r_lat last dim {r_lat.shape[-1]} != action_dim {self.action_dim}"
            )
        return torch.einsum("bcd,de->bce", r_lat, self.weight)


def build_suffix_offset(
    projector: LatentProjector, r_lat: Tensor, chunk_size: int
) -> Tensor:
    """Map r_lat (B, C, D) to a zero-padded suffix offset (B, chunk_size, H).

    Only the first ``C = r_lat.shape[1]`` action-query tokens receive a
    nonzero offset; the remaining tokens keep the base conditioning so the
    actor only owns its own C-step window.
    """
    offset = projector(r_lat)  # (B, C, H)
    n_steps = offset.shape[1]
    if n_steps > chunk_size:
        raise ValueError(f"r_lat has {n_steps} steps but chunk_size={chunk_size}")
    if n_steps < chunk_size:
        pad = offset.new_zeros(
            (offset.shape[0], chunk_size - n_steps, offset.shape[-1])
        )
        offset = torch.cat([offset, pad], dim=1)
    return offset


def seeded_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> Tensor:
    """Deterministic Gaussian noise for reproducible flow denoise."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime: frozen VLA + RLT encoder + (trainable) latent actor + projector
# ─────────────────────────────────────────────────────────────────────────────
class LatentRuntime:
    """GPU-resident decoder for the RLT-Latent route.

    Loads the frozen SmolVLA (Stage-1 ``smolvla_path``) and the RLT encoder
    (Stage-1 checkpoint).  The latent actor + projector are created at init
    (zero-init passthrough); :meth:`load_stage2` replaces them with trained
    weights.
    """

    def __init__(
        self,
        *,
        stage1_checkpoint: Path | str,
        device: str = "cuda:0",
        cameras: tuple[str, ...] = ("front", "right"),
    ) -> None:
        self.device = torch.device(device)
        if any(c not in _CAMERAS for c in cameras):
            raise ValueError(f"cameras must be a subset of {_CAMERAS}, got {cameras}")
        self.cameras = tuple(cameras)

        stage1_path = Path(stage1_checkpoint).expanduser().resolve()
        if not stage1_path.is_file():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_path}")
        self.stage1_path = stage1_path
        checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
        if not (
            checkpoint.get("stage") == "rlt_stage1_joint"
            and checkpoint.get("rlt_architecture") == "rlinf_prefix_v1"
        ):
            raise RuntimeError("unsupported RLT Stage 1 architecture")

        policy_args = argparse.Namespace(
            policy_path=str(Path(checkpoint["smolvla_path"]).expanduser().resolve()),
            device=device,
            vlm_model_path=None,
            n_action_steps=None,
        )
        self.policy, self.preprocessor, self.postprocessor = _load_policy(policy_args)
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import (
            SmolVLARLTConfig,
        )
        from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
            RLActorMLP,
            RLTokenEncoder,
        )

        config_payload = dict(checkpoint["rlt_config"])
        config_payload.pop("rlt_max_reconstruction_tokens", None)
        config_payload.setdefault("rlt_prefix_seq_len", 1024)
        config_payload["rlt_architecture"] = "rlinf_prefix_v1"
        config_payload["action_stride"] = ACTION_CHUNK_FRAME_STRIDE
        config_payload["mode"] = "online_rl"
        self._config_payload = dict(config_payload)
        self.config = SmolVLARLTConfig(**config_payload)

        self.encoder = RLTokenEncoder(self.config).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        self.encoder.eval().requires_grad_(False)

        self.expert_hidden = int(self.policy.model.vlm_with_expert.expert_hidden_size)
        self.chunk_size = int(self.policy.config.chunk_size)
        self.max_action_dim = int(self.policy.config.max_action_dim)
        self.num_denoise_steps = int(self.policy.config.num_steps)
        self.n_action_steps_rl = int(self.config.n_action_steps_rl)
        self.action_dim = int(self.config.action_dim)

        # Latent actor (zero-init residual head => r_lat=0 => passthrough).
        latent_cfg_payload = dict(self._config_payload)
        latent_cfg_payload["actor_output_mode"] = "residual"
        self.actor = RLActorMLP(SmolVLARLTConfig(**latent_cfg_payload)).to(
            self.device
        )
        self.actor.eval().requires_grad_(False)
        self.projector = LatentProjector(self.action_dim, self.expert_hidden).to(
            self.device
        )
        self.projector.requires_grad_(False)

        self.normalizer = _normalizer(self.preprocessor)
        action_stats = self.normalizer._tensor_stats.get("action", {})
        state_stats = self.normalizer._tensor_stats.get("state", {})
        self.action_mean = (
            action_stats["mean"].to(self.device, dtype=torch.float32)
            if "mean" in action_stats
            else None
        )
        self.action_std = (
            action_stats["std"].to(self.device, dtype=torch.float32)
            if "std" in action_stats
            else None
        )
        self.state_mean = (
            state_stats["mean"].to(self.device, dtype=torch.float32)
            if "mean" in state_stats
            else None
        )
        self.state_std = (
            state_stats["std"].to(self.device, dtype=torch.float32)
            if "std" in state_stats
            else None
        )

        self.kinematics = PiperForwardKinematics()
        self.actor_loaded = False
        self.rtc_enabled = False
        LOG.info(
            "LatentRuntime ready device=%s cameras=%s expert_hidden=%d chunk=%d "
            "c_window=%d denoise_steps=%d action_dim=%d use_cache=%s",
            self.device,
            self.cameras,
            self.expert_hidden,
            self.chunk_size,
            self.n_action_steps_rl,
            self.num_denoise_steps,
            self.action_dim,
            bool(self.policy.config.use_cache),
        )

    # -- checkpoint loading -------------------------------------------------
    def load_stage2(self, stage2_path: Path | str) -> dict[str, Any]:
        """Load a latent Stage-2 checkpoint (actor + projector)."""
        stage2_path = Path(stage2_path).expanduser().resolve()
        if not stage2_path.is_file():
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_path}")
        ckpt = torch.load(stage2_path, map_location="cpu", weights_only=False)
        semantics = ckpt.get("actor_output_semantics")
        if semantics is not None and semantics != LATENT_ACTOR_OUTPUT_SEMANTICS:
            raise RuntimeError(
                f"Stage 2 checkpoint is not the latent route: "
                f"actor_output_semantics={semantics!r}"
            )
        actor_sd = ckpt.get("actor_state_dict")
        projector_sd = ckpt.get("projector_state_dict")
        if actor_sd is None or projector_sd is None:
            raise RuntimeError("Stage 2 checkpoint missing actor/projector state dict")
        if (
            int(ckpt.get("expert_hidden_size", self.expert_hidden))
            != self.expert_hidden
        ):
            raise RuntimeError(
                "checkpoint expert_hidden_size does not match the loaded VLA"
            )
        policy_sd = ckpt.get("policy_state_dict")
        if policy_sd:
            missing, unexpected = self.policy.load_state_dict(policy_sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"policy_state_dict mismatch: {missing[:3]} {unexpected[:3]}")
            LOG.info("loaded fine-tuned reference VLA weights from stage2 checkpoint")
        self.actor.load_state_dict(actor_sd, strict=True)
        if isinstance(projector_sd, dict) and "weight" in projector_sd:
            if projector_sd["weight"].shape != self.projector.weight.shape:
                raise RuntimeError("projector weight shape mismatch")
            self.projector.load_state_dict(projector_sd)
        else:
            raise RuntimeError("projector_state_dict malformed")
        self.actor.eval().requires_grad_(False)
        self.projector.requires_grad_(False)
        self.actor_loaded = True
        self._stage2_meta = {
            key: ckpt[key]
            for key in (
                "total_updates",
                "stage",
                "q_weight",
                "bc_weight",
                "critic_target_mode",
                "ready_for_online",
            )
            if key in ckpt
        }
        LOG.info(
            "loaded latent Stage 2 actor + projector from %s (total_updates=%s)",
            stage2_path.name,
            ckpt.get("total_updates"),
        )
        return self._stage2_meta

    # -- RTC ----------------------------------------------------------------
    def enable_rtc(
        self,
        *,
        execution_horizon: int | None = None,
        max_guidance_weight: float | None = None,
    ) -> None:
        """Turn on guided RTC on the frozen VLA (inference-only)."""
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        base = self.policy.config.rtc_config
        cfg = RTCConfig(
            enabled=True,
            prefix_attention_schedule=(
                base.prefix_attention_schedule
                if base is not None
                else RTCConfig().prefix_attention_schedule
            ),
            max_guidance_weight=(
                max_guidance_weight
                if max_guidance_weight is not None
                else (base.max_guidance_weight if base is not None else 10.0)
            ),
            execution_horizon=(
                execution_horizon
                if execution_horizon is not None
                else (base.execution_horizon if base is not None else 10)
            ),
        )
        self.policy.config.rtc_config = cfg
        # SmolVLAPolicy.init_rtc_processor handles the post-model-creation case
        # (it re-attaches the processor to model.rtc_processor).
        self.policy.init_rtc_processor()
        self.rtc_enabled = True
        LOG.info(
            "RTC enabled execution_horizon=%d max_guidance_weight=%s",
            cfg.execution_horizon,
            cfg.max_guidance_weight,
        )

    # -- observation prep ---------------------------------------------------
    def _prepare_batch(
        self, state: np.ndarray, images: dict[str, np.ndarray], task: str
    ) -> dict[str, Tensor]:
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.action_dim,):
            raise ValueError(f"state must be ({self.action_dim},), got {state.shape}")
        if set(images) != set(self.cameras):
            raise ValueError(f"images keys {set(images)} != cameras {set(self.cameras)}")
        observation: dict[str, np.ndarray] = {"observation.state": state}
        for name, image in images.items():
            observation[_policy_key(name)] = np.asarray(image, dtype=np.uint8)
        batch = prepare_observation_for_inference(
            observation, self.device, task, "cobot_magic_piper"
        )
        return self.preprocessor(batch)

    def eef_state(self, puppet_joint: np.ndarray) -> np.ndarray:
        """Piper joint (7,) -> physical gripper-center EEF state (7,)."""
        joint = np.asarray(puppet_joint, dtype=np.float32).reshape(1, 7)
        return np.asarray(
            _joint_to_eef(joint, np.ones((1,), dtype=bool), self.kinematics),
            dtype=np.float32,
        )[0]

    @staticmethod
    def load_observation(
        raw_h5: str | Path, frame_index: int, cameras: tuple[str, ...]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
        """Decode (images, raw puppet joint, task) at ``frame_index``."""
        import h5py

        with h5py.File(raw_h5, "r") as handle:
            images = {
                name: _decode_jpeg(
                    bytes(np.asarray(handle[f"images/{name}/jpeg"][frame_index]))
                )
                for name in cameras
            }
            state = np.asarray(
                handle["frames/puppet_position"][frame_index], dtype=np.float32
            )
            task = str(handle.attrs["task"])
        return images, state, task

    def load_observation_eef(
        self, raw_h5: str | Path, frame_index: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
        """(images, physical EEF state, task) for the model, matching the
        reconstruction pipeline that produced the replay bundles."""
        images, joint, task = self.load_observation(raw_h5, frame_index, self.cameras)
        return images, self.eef_state(joint), task

    def relative_chunk_to_absolute(
        self, eef_anchor: np.ndarray, relative_chunk: np.ndarray
    ) -> np.ndarray:
        return relative_chunk_to_absolute(
            np.asarray(eef_anchor, dtype=np.float64),
            np.asarray(relative_chunk, dtype=np.float64),
        ).astype(np.float32)

    # -- decode -------------------------------------------------------------
    def _offset(self, r_lat: Tensor | None) -> Tensor | None:
        if r_lat is None:
            return None
        return build_suffix_offset(
            self.projector, r_lat.to(self.device, dtype=torch.float32), self.chunk_size
        )

    def decode(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        *,
        r_lat: Tensor | None = None,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: float | None = None,
        execution_horizon: int | None = None,
        seed: int = 0,
        keep_graph: bool = False,
    ) -> dict[str, Any]:
        """Run one frozen-VLA decode with an optional latent offset.

        Returns normalized-space tensors under the ``_`` keys (kept on device,
        graph-connected when ``keep_graph=True``) plus plain lists.
        """
        from lerobot.utils.constants import OBS_STATE

        batch = self._prepare_batch(state, images, task)
        noise = seeded_noise(
            (1, self.chunk_size, self.max_action_dim), seed, self.device
        )
        offset = self._offset(r_lat)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            and bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        started = time.perf_counter()
        if keep_graph:
            actions_full, z_rl = self._decode_grad(
                batch,
                noise,
                offset,
                amp,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )
        else:
            # RTC's denoise_step runs autograd.grad(x1_t, x_t), which inference_mode()
            # forbids; drop inference mode only when a leftover chunk is supplied.
            infer_ctx = (
                nullcontext() if prev_chunk_left_over is not None
                else torch.inference_mode()
            )
            with infer_ctx, amp:
                actions_full, prefix, prefix_mask = (
                    self.policy.predict_action_chunk_with_rlt_prefix(
                        batch,
                        noise=noise,
                        suffix_offset=offset,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        execution_horizon=execution_horizon,
                    )
                )
                z_rl = self.encoder(prefix.float(), prefix_mask)
        normalized_state = batch[OBS_STATE]
        if normalized_state.ndim > 2:
            normalized_state = normalized_state[:, -1, :]
        normalized_state = normalized_state[:, : self.action_dim]

        reference_full = actions_full if keep_graph else actions_full.detach()
        reference = contiguous_action_chunk(
            reference_full,
            action_steps=self.n_action_steps_rl,
            action_dim=self.action_dim,
        )
        return {
            "dimensions": self.action_dim,
            "z_rl": z_rl.squeeze(0).detach().cpu().float().tolist(),
            "state": normalized_state.squeeze(0).detach().cpu().float().tolist(),
            "reference_actions_full": reference_full.squeeze(0)
            .detach()
            .cpu()
            .float()
            .tolist(),
            "reference_actions": reference.squeeze(0).detach().cpu().float().tolist(),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "_actions_full": actions_full,
            "_reference": reference,
            "_z_rl": z_rl,
            "_normalized_state": normalized_state,
        }

    def _decode_grad(
        self,
        batch,
        noise: Tensor,
        offset: Tensor | None,
        amp,
        *,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: float | None = None,
        execution_horizon: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Grad-enabled decode: no-grad prefix KV + grad denoise steps.

        The VLM prefix (KV cache) is computed under ``no_grad``: the VLA is
        frozen and the KV is a constant w.r.t. the actor.  Gradients flow
        only through the suffix denoise steps (30 tokens x 10 steps), reaching
        the latent offset (and thus the actor + projector).
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        model = self.policy.model
        if not bool(model.config.use_cache):
            raise RuntimeError("KV-cache decode requires use_cache=True in the VLA")
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang = batch[OBS_LANGUAGE_TOKENS]
        lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        bsize = state.shape[0]

        with amp:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                    images, img_masks, lang, lang_mask, state=state
                )
                att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                (prefix_out, _suffix_out), kv = model.vlm_with_expert.forward(
                    attention_mask=att_2d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                    fill_kv_cache=True,
                )
                z_rl = self.encoder(prefix_out.float(), prefix_pad_masks)

            dt = -1.0 / self.num_denoise_steps
            x_t = noise
            for step in range(self.num_denoise_steps):
                t_val = 1.0 + step * dt
                time_tensor = torch.tensor(
                    t_val, dtype=torch.float32, device=self.device
                ).expand(bsize)

                def _denoise(input_x_t, current_timestep=time_tensor):
                    return model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=kv,
                        x_t=input_x_t,
                        timestep=current_timestep,
                        suffix_offset=offset,
                    )

                if self._rtc_enabled():
                    v_t = model.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=t_val,
                        original_denoise_step_partial=_denoise,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = _denoise(x_t)
                x_t = x_t + dt * v_t
        return x_t, z_rl

    def _rtc_enabled(self) -> bool:
        cfg = getattr(self.policy.config, "rtc_config", None)
        return cfg is not None and bool(cfg.enabled)

    # -- physical-space helpers ---------------------------------------------
    def unnormalize_chunk(self, normalized: Tensor) -> np.ndarray:
        """normalized (.., action_dim) -> physical (.., action_dim)."""
        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0).unsqueeze(0)
        elif normalized.ndim == 2:
            normalized = normalized.unsqueeze(0)
        if normalized.shape[-1] != self.action_dim:
            raise ValueError("normalized chunk must end with action_dim")
        rows = [
            self.postprocessor(normalized[:, index, :])
            for index in range(normalized.shape[1])
        ]
        return (
            torch.stack(rows, dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    def state_to_physical(self, normalized: Tensor) -> np.ndarray:
        """normalized state (.., 7) -> physical state via the normalizer stats."""
        if self.state_mean is None or self.state_std is None:
            raise RuntimeError("no state stats available for physical mapping")
        mean = self.state_mean.to(normalized.device)
        std = self.state_std.to(normalized.device)
        physical = normalized.float() * std + mean
        return physical.detach().cpu().numpy()
