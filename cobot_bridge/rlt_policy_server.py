"""Persistent SmolVLA + RL-token rollout server for Cobot Magic.

The frozen SmolVLA and Stage 1 encoder stay resident on the rollout GPU.
``latest.pt`` is inspected and atomically adopted only by ``BEGIN_EPISODE``;
an actor is never changed in the middle of an episode.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import zmq
from torch import Tensor, nn

from .rlt_policy_client import (
    RLTPolicyServerError,
    decode_rlt_json,
    encode_rlt_json,
    make_rlt_response,
    validate_rlt_request,
)
from .rlt_stage2_data import (
    ACTION_CHUNK_FRAME_STRIDE,
    ACTION_CHUNK_TIMEBASE,
    ACTOR_OUTPUT_SEMANTICS,
    DEFAULT_ACTOR_EXPLORATION_STD,
    STEAM_OFFLINE_ACTOR_STAGE,
    SUPPORTED_ACTOR_CHECKPOINT_STAGES,
    SUPPORTED_ACTOR_OUTPUT_SEMANTICS,
    calibrated_actor_exploration_std,
    apply_eef_relative_exploration_profile,
    compose_actor_action,
    contiguous_action_chunk,
)
from .run_lerobot_policy import _load_policy
from .policy_server import _processor_uses_relative_eef_actions, _restore_relative_eef_chunk


LOG = logging.getLogger("cobot_bridge.rlt_policy_server")
CAMERAS = ("front", "left", "right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5561")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=("front", "right"))
    parser.add_argument(
        "--actor-exploration-std-per-dim",
        type=float,
        nargs=7,
        default=None,
        metavar="STD",
        help=(
            "Raw-space exploration noise override per action dimension "
            "(x y z rx ry rz gripper). When set it replaces both the scalar "
            "--actor-exploration-std calibration and the automatic EEF-relative "
            "profile shaping."
        ),
    )
    return parser


def _decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RLTPolicyServerError("invalid RLT policy JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _policy_key(name: str) -> str:
    return f"observation.images.camera_{name[0] if name != 'front' else 'f'}"


def _normalizer(preprocessor):
    from lerobot.processor import NormalizerProcessorStep

    matches = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one action normalizer, found {len(matches)}")
    return matches[0]


def normalize_action_chunk(normalizer, raw_actions: np.ndarray, device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
    return normalizer._normalize_action(values, inverse=False)


def unnormalize_action_chunk(postprocessor, normalized: torch.Tensor) -> np.ndarray:
    rows = [postprocessor(normalized[:, index, :]) for index in range(normalized.shape[1])]
    result = torch.stack(rows, dim=1).squeeze(0).detach().cpu().float().numpy()
    if result.ndim != 2 or result.shape[1] != 7 or not np.isfinite(result).all():
        raise RuntimeError(f"invalid unnormalized action chunk: {result.shape}")
    return result.astype(np.float32, copy=False)


class LegacyActionConditionedRLTokenEncoder(nn.Module):
    """Exact encoder architecture used by the immutable format-v2 Stage 1 run."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        expert_hidden_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_hidden_dim, hidden_dim)
        self.expert_proj = nn.Linear(expert_hidden_dim, hidden_dim)
        self.rl_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vlm_embeddings: Tensor, expert_embeddings: Tensor) -> Tensor:
        batch_size = vlm_embeddings.shape[0]
        sequence = torch.cat(
            [
                self.vlm_proj(vlm_embeddings),
                self.expert_proj(expert_embeddings),
                self.rl_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        return self.norm(self.encoder(sequence)[:, -1, :])


class RLTPolicyRuntime:
    """GPU-resident inference state with episode-boundary actor publication."""

    def __init__(
        self,
        *,
        stage1_checkpoint: Path,
        stage2_checkpoint: Path,
        device: str,
        cameras: tuple[str, ...],
        action_steps: int | None = None,
    ) -> None:
        self.stage1_path = stage1_checkpoint.expanduser().resolve()
        self.stage2_path = stage2_checkpoint.expanduser().resolve()
        checkpoint = torch.load(self.stage1_path, map_location="cpu", weights_only=False)
        prefix_stage1 = bool(
            checkpoint.get("stage") == "rlt_stage1_joint"
            and checkpoint.get("rlt_architecture") == "rlinf_prefix_v1"
        )
        legacy_stage1 = bool(
            checkpoint.get("stage") == "rlt_stage1"
            and int(checkpoint.get("format_version", 0)) == 2
            and "expert_proj.weight" in checkpoint.get("encoder_state_dict", {})
        )
        if not prefix_stage1 and not legacy_stage1:
            raise RuntimeError("unsupported RLT Stage 1 architecture")
        self.rlt_architecture = (
            "rlinf_prefix_v1" if prefix_stage1 else "legacy_action_conditioned_v2"
        )
        policy_args = argparse.Namespace(
            policy_path=str(Path(checkpoint["smolvla_path"]).expanduser().resolve()),
            device=device,
            vlm_model_path=None,
            n_action_steps=None,
        )
        self.policy, self.preprocessor, self.postprocessor = _load_policy(policy_args)
        # The saved processor is the action-space contract.  Relative-EFF
        # checkpoints emit chunk poses in the conditioning EEF frame; the
        # rollout client must receive absolute model-frame targets instead.
        self.uses_relative_eef_actions = _processor_uses_relative_eef_actions(
            self.preprocessor
        )
        from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
        from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP, RLTokenEncoder

        config_payload = dict(checkpoint["rlt_config"])
        expert_hidden_dim = int(config_payload.pop("expert_hidden_dim", 0))
        config_payload.pop("rlt_max_reconstruction_tokens", None)
        config_payload.setdefault("rlt_prefix_seq_len", 1024)
        config_payload["rlt_architecture"] = "rlinf_prefix_v1"
        # A chunk is a sequence of consecutive 30 Hz actions.  The historical
        # action_stride=2 setting incorrectly selected [a0, a2, ...] and then
        # executed that sequence at 30 Hz, compressing its time axis by 2x.
        config_payload["action_stride"] = ACTION_CHUNK_FRAME_STRIDE
        config_payload["mode"] = "online_rl"
        self.config = SmolVLARLTConfig(**config_payload)
        # Stage 1 owns the frozen reference VLA and can retain its native
        # chunk horizon (30).  Stage 2 is allowed to own a shorter contiguous
        # execution/action horizon (for example 10) without retraining the
        # encoder.  ``action_steps`` is used by offline replay reconstruction;
        # online inference adopts the horizon declared by its Stage-2 actor.
        self.reference_action_steps = int(self.config.n_action_steps_rl)
        if action_steps is not None and not 0 < int(action_steps) <= self.reference_action_steps:
            raise ValueError(
                "action_steps must be in (0, Stage 1 reference action horizon]"
            )
        self._requested_action_steps = (
            int(action_steps) if action_steps is not None else self.reference_action_steps
        )
        self.execution_action_steps = self._requested_action_steps
        self.device = torch.device(device)
        if legacy_stage1:
            if expert_hidden_dim <= 0:
                raise RuntimeError("legacy Stage 1 is missing expert_hidden_dim")
            self.encoder = LegacyActionConditionedRLTokenEncoder(
                vlm_hidden_dim=int(self.config.vlm_hidden_dim),
                expert_hidden_dim=expert_hidden_dim,
                hidden_dim=int(self.config.rlt_hidden_dim),
                num_heads=int(self.config.rlt_num_heads),
                num_layers=int(self.config.rlt_encoder_layers),
                dropout=float(self.config.rlt_dropout),
            ).to(self.device)
        else:
            self.encoder = RLTokenEncoder(self.config).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        self.encoder.eval().requires_grad_(False)
        self.actor = RLActorMLP(self.config).to(self.device)
        self.actor.eval().requires_grad_(False)
        self.actor_config = self.config
        self.actor_output_semantics = ACTOR_OUTPUT_SEMANTICS
        self.actor_checkpoint_stage: str | None = None
        self.actor_read_only = False
        self.actor_rejection_reason: str | None = None
        self.actor_gate_passed: bool | None = None
        self.normalizer = _normalizer(self.preprocessor)
        action_stats = self.normalizer._tensor_stats.get("action", {})
        action_physical_std = action_stats.get("std")
        if action_physical_std is None:
            raise RuntimeError("RLT policy normalizer has no physical action std")
        self.action_physical_std = action_physical_std[: self.config.action_dim].to(
            device=self.device, dtype=torch.float32
        )
        self.cameras = cameras
        self.base_policy_path = str(Path(checkpoint["smolvla_path"]).expanduser().resolve())
        self.active_episode = False
        self.actor_allowed = False
        self.actor_ready = False
        self.actor_generation = 0
        self.actor_total_updates = 0
        self.actor_exploration_std = DEFAULT_ACTOR_EXPLORATION_STD
        self.actor_action_scale = torch.ones(
            (1, 1, self.config.action_dim), dtype=torch.float32, device=self.device
        )
        self._loaded_stage2_signature: tuple[int, int, int, int, int] | None = None
        # Twin-Q critic for online Q-guided candidate selection.  Loaded from
        # the Stage 2 checkpoint alongside the actor; None = selection
        # disabled (the actor uses a single exploration draw, legacy path).
        self.critic = None
        # Number of candidates scored per decision when Q selection is on.
        # 0 = legacy single exploration draw; 1 = clean actor output only;
        # >= 2 = K candidates (clean + K-1 noise draws), execute Q-argmax.
        self.q_select_k = 0

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "z_dim": int(self.config.rlt_hidden_dim),
            "state_dim": int(self.config.state_dim),
            "action_steps": int(self.execution_action_steps),
            "action_dim": int(self.config.action_dim),
        }

    def _stage2_signature(self) -> tuple[int, int, int, int, int] | None:
        try:
            stat = self.stage2_path.stat()
        except FileNotFoundError:
            return None
        # The trainer publishes with os.replace(), so inode/ctime distinguish
        # two same-size checkpoints even on coarse or extremely fast filesystems.
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size

    def _lineage_generation(self, total_updates: int) -> int:
        """Count published actor checkpoints in the Stage 2 lineage that are
        strictly older than ``total_updates`` (by the checkpoint filename's
        total_updates, e.g. ``checkpoint_00011064.pt`` -> 11064).

        Derived from disk, so the generation is stable across server
        restarts (the previous in-memory counter reset to 0 on every restart)
        and reflects the true rollout->training publication count.  Example:
        lineage 11000/11032/11064/11096/11128 -> loading 11128 yields
        generation 4.  A rollback to an older checkpoint naturally yields
        that checkpoint's own generation.
        """
        pattern = re.compile(r"^checkpoint_(\d+)\.pt$")
        count = 0
        for path in self.stage2_path.parent.glob("checkpoint_*.pt"):
            match = pattern.match(path.name)
            if match and int(match.group(1)) < total_updates:
                count += 1
        return count

    def _reload_actor_at_boundary(self) -> bool:
        signature = self._stage2_signature()
        if signature is None:
            self.actor_ready = False
            self.actor_total_updates = 0
            self.actor_checkpoint_stage = None
            self.actor_read_only = False
            self.actor_rejection_reason = None
            self.critic = None
            self._loaded_stage2_signature = None
            self.execution_action_steps = self._requested_action_steps
            return False
        if signature == self._loaded_stage2_signature:
            return False
        stage2 = torch.load(self.stage2_path, map_location="cpu", weights_only=False)
        checkpoint_stage = stage2.get("stage")
        if checkpoint_stage not in SUPPORTED_ACTOR_CHECKPOINT_STAGES:
            raise RuntimeError(
                "expected a supported Stage 2 actor checkpoint, "
                f"got {checkpoint_stage!r}"
            )
        output_semantics = stage2.get("actor_output_semantics")
        if output_semantics not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS:
            raise RuntimeError(
                "Stage 2 checkpoint uses incompatible actor output semantics"
            )
        if stage2.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE:
            raise RuntimeError("Stage 2 checkpoint uses an incompatible action timebase")
        actual_stage1 = str(Path(stage2["stage1_checkpoint"]).expanduser().resolve())
        if actual_stage1 != str(self.stage1_path):
            raise RuntimeError("Stage 2 checkpoint belongs to another Stage 1 checkpoint")
        # One-step held-out action MSE is a diagnostic, not a real-robot
        # admission gate: it cannot represent the control effect of consuming
        # a native 30-step VLA chunk at a 10-step receding horizon. A valid
        # checkpoint is therefore always eligible for Actor rollout.
        self.actor_gate_passed = True
        actor_config_payload = stage2.get("rlt_config")
        if not isinstance(actor_config_payload, dict):
            raise RuntimeError("Stage 2 checkpoint has no actor architecture config")
        from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
        from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP

        actor_config_payload = dict(actor_config_payload)
        actor_config_payload["mode"] = "inference"
        actor_config = SmolVLARLTConfig(**actor_config_payload)
        for field in ("rlt_hidden_dim", "state_dim", "action_dim"):
            if getattr(actor_config, field) != getattr(self.config, field):
                raise RuntimeError(f"Stage 2 actor {field} differs from Stage 1 runtime")
        actor_steps = int(actor_config.n_action_steps_rl)
        if not 0 < actor_steps <= self.reference_action_steps:
            raise RuntimeError(
                "Stage 2 actor action horizon must be in "
                f"(0, {self.reference_action_steps}], got {actor_steps}"
            )
        loaded_actor = RLActorMLP(actor_config).to(self.device)
        loaded_actor.load_state_dict(stage2["actor_state_dict"], strict=True)
        loaded_actor.eval().requires_grad_(False)
        loaded_total_updates = int(stage2.get("total_updates", stage2.get("step", 0)))
        action_scale = torch.as_tensor(
            stage2.get("actor_action_scale", []), dtype=torch.float32, device=self.device
        )
        if (
            action_scale.shape != (self.config.action_dim,)
            or not torch.isfinite(action_scale).all()
            or not bool(torch.all(action_scale > 0))
        ):
            raise RuntimeError("Stage 2 checkpoint has invalid actor action scale")
        self.actor = loaded_actor
        self.actor_config = actor_config
        self.execution_action_steps = actor_steps
        self.actor_output_semantics = output_semantics
        self.actor_checkpoint_stage = str(checkpoint_stage)
        self.actor_read_only = checkpoint_stage == STEAM_OFFLINE_ACTOR_STAGE
        self.actor_rejection_reason = None
        self.actor_action_scale = action_scale.reshape(1, 1, -1)
        self.actor_total_updates = loaded_total_updates
        self.actor_ready = True
        self._loaded_stage2_signature = signature
        self.actor_generation = self._lineage_generation(loaded_total_updates)
        # Load the twin-Q critic published with the actor checkpoint.  It is
        # the same network the Stage 2 trainer regressed to RECAP returns, so
        # online Q-guided selection reuses the calibrated critic.  Missing
        # critic_state_dict (legacy checkpoints) -> selection falls back to
        # the single-draw exploration path with a warning.
        self.critic = None
        if stage2.get("critic_state_dict"):
            from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLCriticMLP

            critic = RLCriticMLP(actor_config).to(self.device)
            critic.load_state_dict(stage2["critic_state_dict"], strict=True)
            critic.eval().requires_grad_(False)
            self.critic = critic
        return True

    def status(self) -> dict[str, Any]:
        current_signature = self._stage2_signature()
        exploration_std_per_dim = self._exploration_std_per_dim()
        return {
            "ready": True,
            "device": str(self.device),
            "cameras": list(self.cameras),
            "stage1_checkpoint": str(self.stage1_path),
            "stage2_checkpoint": str(self.stage2_path),
            "base_policy_path": self.base_policy_path,
            "rlt_architecture": self.rlt_architecture,
            "dimensions": self.dimensions,
            "active_episode": self.active_episode,
            "actor_ready": self.actor_ready,
            "actor_allowed": self.actor_allowed,
            "actor_enabled": bool(self.actor_ready and self.actor_allowed),
            "actor_generation": self.actor_generation,
            "actor_total_updates": self.actor_total_updates,
            "actor_checkpoint_stage": getattr(self, "actor_checkpoint_stage", None),
            "actor_read_only": bool(getattr(self, "actor_read_only", False)),
            "actor_rejection_reason": getattr(self, "actor_rejection_reason", None),
            "actor_gate_passed": getattr(self, "actor_gate_passed", None),
            "actor_output_semantics": self.actor_output_semantics,
            "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
            "reference_action_steps": int(self.reference_action_steps),
            "reference_action_steps": int(self.reference_action_steps),
            "action_space": (
                "eef_target" if bool(getattr(self, "uses_relative_eef_actions", False)) else "joint"
            ),
            "actor_exploration_std": self.actor_exploration_std,
            "q_select_k": self.q_select_k,
            "q_select_critic_ready": self.critic is not None,
            "actor_exploration_std_per_dim": exploration_std_per_dim.cpu().tolist(),
            "actor_action_scale": self.actor_action_scale.flatten().cpu().tolist(),
            "actor_update_pending": bool(
                current_signature is not None and current_signature != self._loaded_stage2_signature
            ),
            "server_time_ns": time.time_ns(),
        }

    def _exploration_std_per_dim(self) -> torch.Tensor:
        override = getattr(self, "actor_exploration_std_per_dim_override", None)
        if override is not None:
            return override.to(device=self.device, dtype=torch.float32)
        action_physical_std = getattr(self, "action_physical_std", None)
        if (
            self.actor_output_semantics == ACTOR_OUTPUT_SEMANTICS
            and action_physical_std is not None
        ):
            per_dim = calibrated_actor_exploration_std(
                self.actor_exploration_std,
                self.actor_action_scale,
                action_physical_std,
                residual_scale=self.actor_config.actor_residual_scale,
            )
            if bool(getattr(self, "uses_relative_eef_actions", False)):
                per_dim = apply_eef_relative_exploration_profile(per_dim)
            return per_dim
        return torch.full(
            (self.config.action_dim,),
            float(self.actor_exploration_std),
            dtype=torch.float32,
            device=self.device,
        )

    def begin_episode(
        self,
        *,
        workflow_mode: int,
        allow_actor: bool,
        actor_exploration_std: float,
        q_select_k: int = 0,
    ) -> dict[str, Any]:
        if self.active_episode:
            raise RLTPolicyServerError("an RLT episode is already active")
        if workflow_mode not in (1, 2):
            raise RLTPolicyServerError(f"workflow_mode must be 1 or 2, got {workflow_mode}")
        exploration_std = float(actor_exploration_std)
        if not np.isfinite(exploration_std) or exploration_std < 0 or exploration_std > 0.1:
            raise RLTPolicyServerError("actor_exploration_std must be in [0, 0.1]")
        q_select_k = int(q_select_k)
        if q_select_k < 0 or q_select_k > 64:
            raise RLTPolicyServerError("q_select_k must be in [0, 64]")
        reloaded = self._reload_actor_at_boundary()
        self.policy.reset()
        self.actor_allowed = bool(allow_actor)
        # Reference-only is a baseline, not a short-horizon actor ablation.
        # A loaded Stage-2 actor may use a shorter action head (e.g. 10
        # steps), but that must never silently shorten the native Stage-1 VLA
        # chunk when its residual is disabled.  Otherwise "Reference Only"
        # would execute ref[:10] and replan instead of reproducing the
        # reference policy's full 30-step action horizon.
        self.execution_action_steps = (
            int(self.actor_config.n_action_steps_rl)
            if self.actor_allowed and self.actor_ready
            else self.reference_action_steps
        )
        # Steam offline actors are evaluation-only artifacts.  They must be
        # deterministic and must never accidentally enter the online trainer.
        if bool(getattr(self, "actor_read_only", False)):
            exploration_std = 0.0
            q_select_k = 0
        self.actor_exploration_std = exploration_std
        self.q_select_k = q_select_k
        self.active_episode = True
        return {**self.status(), "workflow_mode": workflow_mode, "actor_reloaded": reloaded}

    def end_episode(self) -> dict[str, Any]:
        was_active = self.active_episode
        self.active_episode = False
        self.actor_allowed = False
        return {**self.status(), "ended": was_active}

    def normalize_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        if not self.active_episode:
            raise RLTPolicyServerError("BEGIN_EPISODE is required before NORMALIZE_ACTION")
        raw = np.asarray(raw_actions, dtype=np.float32)
        expected = (self.execution_action_steps, self.config.action_dim)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise RLTPolicyServerError(f"raw_actions must have shape {expected}, got {raw.shape}")
        normalized = normalize_action_chunk(self.normalizer, raw, self.device)
        return normalized.detach().cpu().float().numpy().astype(np.float32, copy=False)

    def predict(
        self, *, state: np.ndarray, images: dict[str, np.ndarray], task: str, robot_type: str
    ) -> dict[str, Any]:
        if not self.active_episode:
            raise RLTPolicyServerError("BEGIN_EPISODE is required before PREDICT")
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.policies.smolvla.modeling_smolvla import pad_vector
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        observation: dict[str, np.ndarray] = {"observation.state": state}
        for name, image in images.items():
            observation[_policy_key(name)] = image
        batch = prepare_observation_for_inference(observation, self.device, task, robot_type)
        batch = self.preprocessor(batch)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        started = time.perf_counter()
        with torch.inference_mode(), amp:
            reference_full, prefix_embedding, prefix_mask = (
                self.policy.predict_action_chunk_with_rlt_prefix(batch)
            )
            if self.rlt_architecture == "legacy_action_conditioned_v2":
                images_prepared, image_masks = self.policy.prepare_images(batch)
                state_padded = self.policy.prepare_state(batch)
                dummy_actions = pad_vector(reference_full, self.config.max_action_dim)
                vlm_embedding, expert_embedding = (
                    self.policy.model.extract_legacy_rlt_embeddings(
                        images_prepared,
                        image_masks,
                        batch[OBS_LANGUAGE_TOKENS],
                        batch[OBS_LANGUAGE_ATTENTION_MASK],
                        state_padded,
                        dummy_actions,
                    )
                )
                z_rl = self.encoder(vlm_embedding.float(), expert_embedding.float())
            else:
                z_rl = self.encoder(prefix_embedding.float(), prefix_mask)
            # Keep the native Stage-1 VLA chunk available independently of
            # the (possibly shorter) Stage-2 actor horizon.  The actor and
            # critic always condition on the contiguous first actor horizon;
            # before the phase gate latches, the physical executor instead
            # needs this complete reference plan to reproduce BC behavior.
            reference_full_chunk = contiguous_action_chunk(
                reference_full,
                action_steps=self.reference_action_steps,
                action_dim=self.config.action_dim,
            )
            reference = contiguous_action_chunk(
                reference_full_chunk,
                action_steps=self.execution_action_steps,
                action_dim=self.config.action_dim,
            )
            normalized_state = batch[OBS_STATE]
            if normalized_state.ndim > 2:
                normalized_state = normalized_state[:, -1, :]
            normalized_state = normalized_state[:, : self.config.state_dim]
            actor_enabled = bool(self.actor_ready and self.actor_allowed)
            q_select_info: dict[str, Any] | None = None
            if actor_enabled:
                policy_action_raw, _ = self.actor(z_rl, normalized_state, reference)
                if self.q_select_k > 1 and self.critic is not None:
                    # Q-guided candidate selection: K candidates where index 0
                    # is the clean actor output and 1..K-1 are exploration
                    # draws (raw + randn x per-dim sigma, pre-tanh).  All are
                    # composed to absolute actions, scored by the twin-Q min,
                    # and the argmax is executed.  Bounded by construction:
                    # the worst pick is a band-width deviation from the actor.
                    raw_base = policy_action_raw.squeeze(0)  # (steps, dim)
                    sigma = self._exploration_std_per_dim().reshape(1, -1)
                    raw_candidates = [raw_base]
                    for _ in range(self.q_select_k - 1):
                        raw_candidates.append(raw_base + torch.randn_like(raw_base) * sigma)
                    raw_stack = torch.stack(raw_candidates, dim=0)  # (K, steps, dim)
                    candidate_actions = compose_actor_action(
                        raw_stack,
                        reference.expand(raw_stack.shape[0], -1, -1),
                        self.actor_action_scale,
                        output_semantics=self.actor_output_semantics,
                        residual_scale=self.actor_config.actor_residual_scale,
                    )
                    q1, q2 = self.critic(
                        z_rl.expand(raw_stack.shape[0], -1),
                        normalized_state.expand(raw_stack.shape[0], -1),
                        candidate_actions,
                    )
                    q_values = torch.minimum(q1, q2).squeeze(-1)
                    chosen = int(q_values.argmax().item())
                    policy_action_raw = raw_stack[chosen].unsqueeze(0)
                    policy_actions = candidate_actions[chosen : chosen + 1]
                    q_select_info = {
                        "k": int(self.q_select_k),
                        "chosen": chosen,
                        "q": [round(float(v), 6) for v in q_values.detach().cpu().tolist()],
                        "q_spread": round(float((q_values.max() - q_values.min()).item()), 6),
                        "fallback": False,
                    }
                else:
                    if self.q_select_k > 1:
                        logging.warning(
                            "Q selection requested (q_select_k=%d) but the twin-Q "
                            "critic is not loaded; falling back to a single "
                            "exploration draw.", self.q_select_k,
                        )
                    if self.q_select_k == 0 and self.actor_exploration_std > 0:
                        policy_action_raw = policy_action_raw + (
                            torch.randn_like(policy_action_raw)
                            * self._exploration_std_per_dim().reshape(1, 1, -1)
                        )
                    policy_actions = compose_actor_action(
                        policy_action_raw,
                        reference,
                        self.actor_action_scale,
                        output_semantics=self.actor_output_semantics,
                        residual_scale=self.actor_config.actor_residual_scale,
                    )
                    if self.q_select_k == 1:
                        q_select_info = {"k": 1, "chosen": 0, "clean_only": True}
            else:
                # Read-only/reference use must never expose proposals from an
                # unavailable or deliberately disabled actor.
                policy_actions = reference
            executed = policy_actions if actor_enabled else reference
            raw_reference_chunk = unnormalize_action_chunk(self.postprocessor, reference)
            raw_reference_full_chunk = unnormalize_action_chunk(
                self.postprocessor, reference_full_chunk
            )
            raw_policy_chunk = unnormalize_action_chunk(self.postprocessor, policy_actions)
            raw_reference_chunk = _restore_relative_eef_chunk(
                self.preprocessor, state, raw_reference_chunk
            )
            raw_reference_full_chunk = _restore_relative_eef_chunk(
                self.preprocessor, state, raw_reference_full_chunk
            )
            raw_policy_chunk = _restore_relative_eef_chunk(
                self.preprocessor, state, raw_policy_chunk
            )
            if not (
                np.isfinite(raw_reference_chunk).all()
                and np.isfinite(raw_reference_full_chunk).all()
                and np.isfinite(raw_policy_chunk).all()
            ):
                raise RLTPolicyServerError("RLT EEF restoration produced non-finite targets")
            raw_action_chunk = raw_policy_chunk if actor_enabled else raw_reference_chunk
        tensors = (z_rl, normalized_state, reference, policy_actions, executed)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RLTPolicyServerError("RLT policy produced non-finite tensors")
        return {
            "dimensions": self.dimensions,
            "reference_action_steps": int(self.reference_action_steps),
            "z_rl": z_rl.squeeze(0).detach().cpu().float().tolist(),
            "state": normalized_state.squeeze(0).detach().cpu().float().tolist(),
            "reference_actions": reference.squeeze(0).detach().cpu().float().tolist(),
            "policy_actions": policy_actions.squeeze(0).detach().cpu().float().tolist(),
            "executed_actions": executed.squeeze(0).detach().cpu().float().tolist(),
            "raw_reference_chunk": raw_reference_chunk.tolist(),
            "raw_reference_full_chunk": raw_reference_full_chunk.tolist(),
            "raw_policy_chunk": raw_policy_chunk.tolist(),
            "raw_action_chunk": raw_action_chunk.tolist(),
            "action_space": (
                "eef_target" if bool(getattr(self, "uses_relative_eef_actions", False)) else "joint"
            ),
            "action_source": "actor" if actor_enabled else "reference",
            "q_select": q_select_info,
            "actor_generation": self.actor_generation,
            "actor_total_updates": self.actor_total_updates,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=args.device,
        cameras=tuple(args.cameras),
    )
    if args.actor_exploration_std_per_dim is not None:
        runtime.actor_exploration_std_per_dim_override = torch.tensor(
            args.actor_exploration_std_per_dim, dtype=torch.float32
        )
        LOG.info(
            "actor exploration per-dim override active: %s",
            args.actor_exploration_std_per_dim,
        )
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    LOG.info(
        "RLT policy server ready bind=%s device=%s stage1=%s stage2=%s cameras=%s",
        args.bind,
        args.device,
        runtime.stage1_path,
        runtime.stage2_path,
        runtime.cameras,
    )
    shutdown_requested = False
    try:
        while not shutdown_requested:
            frames = socket.recv_multipart()
            request: dict[str, Any] = {}
            try:
                if not frames:
                    raise RLTPolicyServerError("empty RLT policy request")
                request = validate_rlt_request(decode_rlt_json(frames[0]))
                command = request["command"]
                if command == "PING":
                    result = runtime.status()
                elif command == "BEGIN_EPISODE":
                    result = runtime.begin_episode(
                        workflow_mode=int(request.get("workflow_mode", 0)),
                        allow_actor=bool(request.get("allow_actor", False)),
                        actor_exploration_std=float(
                            request.get(
                                "actor_exploration_std", DEFAULT_ACTOR_EXPLORATION_STD
                            )
                        ),
                        q_select_k=int(request.get("q_select_k", 0)),
                    )
                elif command == "END_EPISODE":
                    result = runtime.end_episode()
                elif command == "NORMALIZE_ACTION":
                    normalized = runtime.normalize_actions(
                        np.asarray(request.get("raw_actions", []), dtype=np.float32)
                    )
                    result = {"normalized_actions": normalized.tolist()}
                elif command == "SHUTDOWN":
                    if runtime.active_episode:
                        raise RLTPolicyServerError("cannot stop RLT server during an active episode")
                    shutdown_requested = True
                    result = {"shutting_down": True}
                else:
                    cameras = tuple(str(name) for name in request.get("cameras", ()))
                    if cameras != runtime.cameras:
                        raise RLTPolicyServerError(
                            f"camera mismatch: request={cameras} configured={runtime.cameras}"
                        )
                    if len(frames) != len(cameras) + 1:
                        raise RLTPolicyServerError(
                            f"expected {len(cameras)} JPEG frames, received {len(frames) - 1}"
                        )
                    state = np.asarray(request.get("state", []), dtype=np.float32)
                    if state.shape != (7,) or not np.isfinite(state).all():
                        raise RLTPolicyServerError("state must contain seven finite values")
                    images = {
                        name: _decode_jpeg(data)
                        for name, data in zip(cameras, frames[1:], strict=True)
                    }
                    result = runtime.predict(
                        state=state,
                        images=images,
                        task=str(request.get("task", "")),
                        robot_type=str(request.get("robot_type", "cobot_magic_piper")),
                    )
                response = make_rlt_response(request, ok=True, result=result)
            except Exception as exc:
                LOG.exception("RLT policy request failed")
                response = make_rlt_response(request, ok=False, error=str(exc))
            socket.send_multipart([encode_rlt_json(response)])
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
