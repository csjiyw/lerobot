"""SmolVLA with RL Token (RLT) — core model implementation.

Implements the RLT paper (Physical Intelligence) for SmolVLA:
  Stage 1: Train encoder-decoder to extract z_rl from frozen VLA embeddings
  Stage 2: Train lightweight actor-critic using z_rl for online RL

Architecture:
  Frozen SmolVLA → [RLT Encoder] → z_rl → [Actor MLP] → refined actions
                                        → [Critic MLP] → Q-values
"""

import copy
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.utils.constants import ACTION


# ─────────────────────────────────────────────────────────────────────────────
# RLT Encoder: VLA embeddings → z_rl
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal_positions(length: int, dim: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding


class _GeGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, inputs: Tensor) -> Tensor:
        values, gates = self.proj(inputs).chunk(2, dim=-1)
        return values * F.gelu(gates)


class _RLTSelfAttentionLayer(nn.Module):
    """RLinf-compatible pre-norm self-attention block."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        mlp_dim = dim * 4
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.Dropout(dropout),
            _GeGLU(mlp_dim),
            nn.Linear(mlp_dim, dim),
        )

    def forward(
        self, inputs: Tensor, mask: Tensor | None = None, *, causal_mask: Tensor | None = None
    ) -> Tensor:
        inputs = inputs.to(device=self.attn_norm.weight.device, dtype=self.attn_norm.weight.dtype)
        key_padding_mask = None if mask is None else ~mask.to(device=inputs.device, dtype=torch.bool)
        residual = inputs
        normalized = self.attn_norm(inputs)
        attended = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        inputs = residual + attended
        return inputs + self.mlp(self.mlp_norm(inputs))


class RLTokenEncoder(nn.Module):
    """Compress action-independent VLA prefix states into one flat ``z_rl``."""

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        self.max_prefix_tokens = int(config.rlt_prefix_seq_len)
        self.input_proj = nn.Linear(config.vlm_hidden_dim, config.rlt_hidden_dim)
        self.rl_token = nn.Parameter(_sinusoidal_positions(1, config.rlt_hidden_dim))
        self.prefix_position = nn.Parameter(
            _sinusoidal_positions(self.max_prefix_tokens, config.rlt_hidden_dim)
        )
        self.rl_position = nn.Parameter(_sinusoidal_positions(1, config.rlt_hidden_dim))
        self.layers = nn.ModuleList(
            [
                _RLTSelfAttentionLayer(
                    config.rlt_hidden_dim, config.rlt_num_heads, config.rlt_dropout
                )
                for _ in range(config.rlt_encoder_layers)
            ]
        )

    def forward(self, prefix_embeddings: Tensor, prefix_mask: Tensor | None = None) -> Tensor:
        prefix_embeddings = prefix_embeddings.to(dtype=self.input_proj.weight.dtype)
        prefix = self.input_proj(prefix_embeddings)
        batch_size, length = prefix.shape[:2]
        if length > self.max_prefix_tokens:
            raise ValueError(
                f"prefix length {length} exceeds rlt_prefix_seq_len={self.max_prefix_tokens}"
            )
        prefix = prefix + self.prefix_position[:length].to(prefix)
        token = (self.rl_token + self.rl_position).to(prefix).unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat([prefix, token], dim=1)
        if prefix_mask is not None:
            token_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=sequence.device)
            prefix_mask = torch.cat(
                [prefix_mask.to(device=sequence.device, dtype=torch.bool), token_mask], dim=1
            )
        for layer in self.layers:
            sequence = layer(sequence, prefix_mask)
        return sequence[:, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# RLT Decoder: z_rl → reconstructed VLA embeddings
# ─────────────────────────────────────────────────────────────────────────────

class RLTokenDecoder(nn.Module):
    """Autoregressively reconstruct only the frozen VLA prefix states."""

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        self.max_prefix_tokens = int(config.rlt_prefix_seq_len)
        self.teacher_proj = nn.Linear(config.vlm_hidden_dim, config.rlt_hidden_dim)
        self.position = nn.Parameter(
            _sinusoidal_positions(self.max_prefix_tokens, config.rlt_hidden_dim)
        )
        self.layers = nn.ModuleList(
            [
                _RLTSelfAttentionLayer(
                    config.rlt_hidden_dim, config.rlt_num_heads, config.rlt_dropout
                )
                for _ in range(config.rlt_decoder_layers)
            ]
        )
        self.output_proj = nn.Linear(config.rlt_hidden_dim, config.vlm_hidden_dim)

    def forward(
        self, z_rl: Tensor, prefix_targets: Tensor, prefix_mask: Tensor | None = None
    ) -> Tensor:
        length = prefix_targets.shape[1]
        if length > self.max_prefix_tokens:
            raise ValueError(
                f"prefix length {length} exceeds rlt_prefix_seq_len={self.max_prefix_tokens}"
            )
        frozen = prefix_targets.detach().to(device=z_rl.device, dtype=z_rl.dtype)
        shifted = frozen[:, :-1]
        if prefix_mask is not None:
            shifted = shifted.masked_fill(
                ~prefix_mask[:, :-1].to(device=shifted.device, dtype=torch.bool).unsqueeze(-1), 0
            )
        decoder_input = torch.cat([z_rl.unsqueeze(1), self.teacher_proj(shifted)], dim=1)
        decoder_input = decoder_input + self.position[:length].to(decoder_input)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=z_rl.device), diagonal=1
        )
        decoder_mask = None
        if prefix_mask is not None:
            start = torch.ones(prefix_mask.shape[0], 1, dtype=torch.bool, device=z_rl.device)
            decoder_mask = torch.cat(
                [start, prefix_mask[:, :-1].to(device=z_rl.device, dtype=torch.bool)], dim=1
            )
        for layer in self.layers:
            decoder_input = layer(decoder_input, decoder_mask, causal_mask=causal_mask)
        return self.output_proj(decoder_input)


# ─────────────────────────────────────────────────────────────────────────────
# Actor MLP: z_rl + state + ref_actions → refined actions
# ─────────────────────────────────────────────────────────────────────────────

class _ModalityProjector(nn.Module):
    """Project one modality to a shared width before multimodal fusion."""

    def __init__(self, input_dim: int, output_dim: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.Tanh() if activation == "tanh" else nn.SiLU()
        gain = nn.init.calculate_gain("tanh") if activation == "tanh" else 1.0
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.linear(inputs)))


class RLActorMLP(nn.Module):
    """Lightweight actor that refines VLA reference actions using z_rl.

    Input: z_rl(256) + state(7) + flattened ref_actions(10*7=70) = 333
    Output: action mean (10*7=70) + learnable log_std
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        self.config = config
        action_flat = config.n_action_steps_rl * config.action_dim  # 10 * 7 = 70
        projection_dim = int(config.modality_projection_dim)
        if projection_dim:
            self.z_projector = _ModalityProjector(
                config.rlt_hidden_dim, projection_dim, config.modality_projection_activation
            )
            self.state_projector = _ModalityProjector(
                config.state_dim, projection_dim, config.modality_projection_activation
            )
            self.ref_projector = _ModalityProjector(
                action_flat, projection_dim, config.modality_projection_activation
            )
            input_dim = 3 * projection_dim
        else:
            self.z_projector = nn.Identity()
            self.state_projector = nn.Identity()
            self.ref_projector = nn.Identity()
            input_dim = config.rlt_hidden_dim + config.state_dim + action_flat

        layers = []
        prev_dim = input_dim
        for h in config.actor_hidden_dims:
            activation = nn.Tanh() if config.actor_activation == "tanh" else nn.SiLU()
            layers.extend([nn.Linear(prev_dim, h), activation])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_flat))
        self.mlp = nn.Sequential(*layers)
        if config.actor_output_mode == "residual":
            # A freshly initialized residual actor reproduces the base VLA
            # exactly once its zero residual is composed with the reference.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        # Stage 2 applies RLinf's fixed Gaussian exploration around this mean
        # outside the network. Keep the second return value for API
        # compatibility without registering an unused trainable parameter.
        self.register_buffer("log_std", torch.zeros(action_flat), persistent=False)

        self.action_flat = action_flat
        self.ref_dropout = config.ref_dropout

    def forward(
        self,
        z_rl: Tensor,           # (B, rlt_hidden_dim)
        state: Tensor,           # (B, state_dim)
        ref_actions: Tensor,     # (B, n_action_steps_rl, action_dim)
        training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Returns (action_mean, log_std) for the RL action chunk.

        action_mean: (B, n_action_steps_rl, action_dim)
        log_std: (action_flat,)
        """
        B = z_rl.shape[0]
        ref_flat = ref_actions.reshape(B, -1)  # (B, 70)

        # Reference action dropout during training
        if training and self.ref_dropout > 0:
            mask = (torch.rand(B, 1, device=ref_flat.device) > self.ref_dropout).float()
            ref_flat = ref_flat * mask

        z_features = self.z_projector(z_rl)
        state_features = self.state_projector(state)
        ref_features = self.ref_projector(ref_flat)

        if self.config.actor_input_order == "ref_z_state":
            # RLinf RLTMLPPolicy order: reference chunk, RL token, proprioception.
            x = torch.cat([ref_features, z_features, state_features], dim=-1)
        else:
            x = torch.cat([z_features, state_features, ref_features], dim=-1)
        action_mean = self.mlp(x)  # (B, 70)

        # Reshape to (B, n_action_steps_rl, action_dim)
        action_mean = action_mean.reshape(B, self.config.n_action_steps_rl, self.config.action_dim)

        return action_mean, self.log_std


# ─────────────────────────────────────────────────────────────────────────────
# Twin Critic MLP: z_rl + state + actions → Q-values
# ─────────────────────────────────────────────────────────────────────────────

class _ProjectedQNetwork(nn.Module):
    """One Q tower with independent z/state/action projection parameters."""

    def __init__(self, config: SmolVLARLTConfig, action_flat: int):
        super().__init__()
        projection_dim = int(config.modality_projection_dim)
        self.z_projector = _ModalityProjector(
            config.rlt_hidden_dim, projection_dim, config.modality_projection_activation
        )
        self.state_projector = _ModalityProjector(
            config.state_dim, projection_dim, config.modality_projection_activation
        )
        self.action_projector = _ModalityProjector(
            action_flat, projection_dim, config.modality_projection_activation
        )
        layers: list[nn.Module] = []
        prev_dim = 3 * projection_dim
        for hidden_dim in config.critic_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.critic_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh() if config.critic_activation == "tanh" else nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_rl: Tensor, state: Tensor, actions_flat: Tensor) -> Tensor:
        features = torch.cat(
            [
                self.z_projector(z_rl),
                self.state_projector(state),
                self.action_projector(actions_flat),
            ],
            dim=-1,
        )
        return self.mlp(features)


class RLCriticMLP(nn.Module):
    """Twin Q-network for TD3-style critic.

    Input: z_rl(256) + state(7) + flattened actions(70) = 333
    Output: Two scalar Q-values (min used for target)
    """

    def __init__(self, config: SmolVLARLTConfig):
        super().__init__()
        self.config = config
        action_flat = config.n_action_steps_rl * config.action_dim
        if config.modality_projection_dim:
            self.q1 = _ProjectedQNetwork(config, action_flat)
            self.q2 = _ProjectedQNetwork(config, action_flat)
            if config.critic_activation == "tanh":
                self._initialize_rlinf_q(self.q1.mlp)
                self._initialize_rlinf_q(self.q2.mlp)
        else:
            input_dim = config.rlt_hidden_dim + config.state_dim + action_flat
            self.q1 = self._build_mlp(input_dim, config.critic_hidden_dims)
            self.q2 = self._build_mlp(input_dim, config.critic_hidden_dims)
            if config.critic_activation == "tanh":
                self._initialize_rlinf_q(self.q1)
                self._initialize_rlinf_q(self.q2)

    def _build_mlp(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if self.config.critic_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(
                nn.Tanh() if self.config.critic_activation == "tanh" else nn.SiLU()
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _initialize_rlinf_q(network: nn.Sequential) -> None:
        linears = [module for module in network if isinstance(module, nn.Linear)]
        gain = nn.init.calculate_gain("tanh")
        for layer in linears[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linears[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(linears[-1].bias)

    def forward(
        self,
        z_rl: Tensor,       # (B, rlt_hidden_dim)
        state: Tensor,       # (B, state_dim)
        actions: Tensor,     # (B, n_action_steps_rl, action_dim)
    ) -> tuple[Tensor, Tensor]:
        """Returns (Q1, Q2) each of shape (B, 1)."""
        B = z_rl.shape[0]
        actions_flat = actions.reshape(B, -1)
        if self.config.modality_projection_dim:
            return self.q1(z_rl, state, actions_flat), self.q2(z_rl, state, actions_flat)
        x = torch.cat([z_rl, state, actions_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self,
        z_rl: Tensor,
        state: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Returns min(Q1, Q2) of shape (B, 1)."""
        q1, q2 = self.forward(z_rl, state, actions)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# SmolVLARLTPolicy: wraps everything together
# ─────────────────────────────────────────────────────────────────────────────

class SmolVLARLTPolicy(PreTrainedPolicy):
    """Full RLT policy wrapping frozen SmolVLA + trainable RLT components.

    Supports 3 modes:
      - "rlt_training": Stage 1 — train encoder/decoder on demo data
      - "online_rl": Stage 2 — train actor/critic with online RL
      - "inference": Deploy on robot
    """

    config_class = SmolVLARLTConfig
    name = "smolvla_rlt"

    def __init__(self, config: SmolVLARLTConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # ── RLT encoder + decoder ───────────────────────────────────────
        self.rlt_encoder = RLTokenEncoder(config)
        self.rlt_decoder = RLTokenDecoder(config)

        # ── Actor + Critic ──────────────────────────────────────────────
        self.actor = RLActorMLP(config)
        self.critic = RLCriticMLP(config)
        self.critic_target = RLCriticMLP(config)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Frozen SmolVLA (loaded lazily) ──────────────────────────────
        self._vla_loaded = False
        self._frozen_vla = None

        # ── Action queue for deployment ─────────────────────────────────
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps_rl)}

    def load_frozen_vla(self, vla_model):
        """Attach a frozen SmolVLA model (VLAFlowMatching instance).

        Call this after constructing the policy, passing in the VLAFlowMatching
        model loaded from a pretrained SmolVLA checkpoint.
        """
        self._frozen_vla = vla_model
        self._frozen_vla.eval()
        for p in self._frozen_vla.parameters():
            p.requires_grad = False
        self._vla_loaded = True

    def _ensure_vla(self):
        if not self._vla_loaded:
            raise RuntimeError(
                "Frozen SmolVLA not loaded. Call policy.load_frozen_vla(vla_model) first."
            )

    # ── Embedding extraction from frozen VLA ────────────────────────────

    @torch.no_grad()
    def extract_vla_embeddings(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return action-independent prefix states and their validity mask."""
        self._ensure_vla()
        return self._frozen_vla.extract_rlt_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )

    # ── Stage 1: RLT Training (encoder-decoder) ────────────────────────

    def forward_rlt_training(
        self,
        prefix_embeddings: Tensor,
        prefix_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Stage 1 forward: encode → z_rl → decode → reconstruction loss."""
        target = prefix_embeddings.detach().to(dtype=self.rlt_encoder.input_proj.weight.dtype)
        z_rl = self.rlt_encoder(target, prefix_mask)
        reconstruction = self.rlt_decoder(z_rl, target, prefix_mask)
        squared_error = (reconstruction - target).square()
        if prefix_mask is None:
            reconstruction_loss = squared_error.mean()
        else:
            weights = prefix_mask.to(squared_error).unsqueeze(-1)
            reconstruction_loss = (squared_error * weights).sum() / (
                weights.sum().clamp_min(1) * squared_error.shape[-1]
            )

        return {
            "loss": reconstruction_loss,
            "rlt_loss": reconstruction_loss,
            "z_rl": z_rl,
        }

    # ── Stage 2: Online RL (actor-critic) ───────────────────────────────

    def forward_critic_loss(
        self,
        z_rl: Tensor,           # (B, 256)
        state: Tensor,           # (B, state_dim)
        actions: Tensor,         # (B, n_action_steps_rl, action_dim)
        rewards: Tensor,         # (B, 1)
        next_z_rl: Tensor,       # (B, 256)
        next_state: Tensor,      # (B, state_dim)
        next_ref_actions: Tensor,  # (B, n_action_steps_rl, action_dim)
        dones: Tensor,           # (B, 1)
    ) -> dict[str, Tensor]:
        """Compute TD3-style twin critic loss."""
        with torch.no_grad():
            # Target actions from actor (no exploration noise for simplicity)
            next_action_mean, _ = self.actor(next_z_rl, next_state, next_ref_actions)
            # Target Q from target critic
            target_q = self.critic_target.q_min(next_z_rl, next_state, next_action_mean)
            td_target = rewards + self.config.discount * (1.0 - dones) * target_q

        q1, q2 = self.critic(z_rl, state, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        return {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
            "td_target_mean": td_target.mean(),
        }

    def forward_actor_loss(
        self,
        z_rl: Tensor,           # (B, 256)
        state: Tensor,           # (B, state_dim)
        ref_actions: Tensor,     # (B, n_action_steps_rl, action_dim)
    ) -> dict[str, Tensor]:
        """Compute actor loss: maximize Q + BC regularization."""
        action_mean, _ = self.actor(z_rl, state, ref_actions, training=True)

        # Q-value maximization
        q_value = self.critic.q_min(z_rl.detach(), state, action_mean)
        q_loss = -q_value.mean()

        # BC regularization: stay close to reference actions
        bc_loss = F.mse_loss(action_mean, ref_actions)

        actor_loss = q_loss + self.config.bc_weight * bc_loss

        return {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
        }

    def update_target_critic(self):
        """Soft update target critic via EMA."""
        tau = self.config.target_tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # ── Inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a complete RLT action chunk for the current observation."""
        self.eval()
        return self._get_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for robot execution.

        Manages an action queue: only calls the full pipeline when the
        queue is empty, otherwise pops the next pre-computed action.
        """
        self.eval()

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            # Queue shape: (n_action_steps_rl, batch_size, action_dim)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Full inference pipeline: VLA → z_rl → actor → actions.

        Returns: (B, n_action_steps_rl, action_dim)
        """
        self._ensure_vla()

        # Import here to avoid circular dependency
        from lerobot.policies.smolvla.modeling_smolvla import pad_vector
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        # Prepare inputs (same as SmolVLA)
        images, img_masks = self._prepare_images(batch)
        state_raw = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state_padded = pad_vector(state_raw, self.config.max_state_dim)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Get VLA reference actions (full 50-step chunk)
        ref_actions_full = self._frozen_vla.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded,
        )
        # Unpad to actual action dim
        ref_actions_full = ref_actions_full[:, :, :self.config.action_dim]

        # Keep consecutive policy ticks. A replay-window start stride may
        # generate [a0:C], [a2:2+C], ... as separate transitions; selecting
        # [a0, a2, ...] inside one chunk would compress its control time axis.
        ref_actions_sub = ref_actions_full[:, :self.config.n_action_steps_rl, :]

        # Extract action-independent prefix embeddings.
        prefix_emb, prefix_mask = self.extract_vla_embeddings(
            images, img_masks, lang_tokens, lang_masks, state_padded
        )

        # Encode to z_rl
        z_rl = self.rlt_encoder(prefix_emb, prefix_mask)

        # Actor: refine reference actions
        state_rl = state_raw[:, :self.config.state_dim]  # Original unpadded state
        action_mean, _ = self.actor(z_rl, state_rl, ref_actions_sub)

        return action_mean

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Prepare images using SmolVLA's preprocessing pattern."""
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        images = []
        img_masks = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    # ── Standard forward (dispatches based on mode) ─────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass — dispatches based on config.mode.

        For rlt_training mode, batch contains prefix embeddings and mask.

        Returns (loss, loss_dict).
        """
        if self.config.mode == "rlt_training":
            result = self.forward_rlt_training(
                batch["prefix_embeddings"],
                batch.get("prefix_mask"),
            )
            return result["loss"], {k: v.item() if isinstance(v, Tensor) and v.ndim == 0 else v for k, v in result.items() if k != "z_rl"}
        else:
            raise ValueError(
                f"forward() not supported for mode='{self.config.mode}'. "
                "Use forward_critic_loss/forward_actor_loss for online_rl."
            )

    def get_optim_params(self) -> dict:
        """Return parameters to optimize based on current mode."""
        if self.config.mode == "rlt_training":
            return list(self.rlt_encoder.parameters()) + list(self.rlt_decoder.parameters())
        elif self.config.mode == "online_rl":
            return list(self.actor.parameters()) + list(self.critic.parameters())
        else:
            return []
