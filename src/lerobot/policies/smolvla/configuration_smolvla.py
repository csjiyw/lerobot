# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Express every EEF target in a chunk relative to the single current
    # observation state, matching GR00T relative_chunking semantics.
    # The normalization of state/action is controlled by the standard
    # `normalization_mapping` (IDENTITY / MIN_MAX / QUANTILES / MEAN_STD); the
    # checkpoint's saved processor persists the mapping together with the exact
    # statistics training used, and is loaded back at inference and on resume.
    use_relative_eef_actions: bool = False
    relative_eef_action_stats_path: str | None = None

    # When True (and use_relative_eef_actions), the chunk-relative EEF action
    # transform is computed per-sample inside the dataset worker (parallel
    # across DataLoader workers) instead of in the policy preprocessor (main
    # process, per batch). The preprocessor's relative-EEF step is skipped in
    # that case. Action normalization statistics are unchanged either way.
    eef_relative_in_dataset: bool = False

    # When True (and use_relative_eef_actions), the chunk-relative EEF action
    # transform is computed by a vectorized torch step in the preprocessor,
    # placed after the device move so it runs on GPU in float32 (batched,
    # no per-sample python loop). Mutually exclusive with eef_relative_in_dataset.
    # The preprocessor's CPU relative-EEF step is skipped in that case; action
    # normalization statistics are unchanged either way.
    eef_relative_in_gpu: bool = False
    use_sdpa: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    # Hide the complete normalized state vector for a training sample with
    # this probability. Inference never applies this dropout.
    state_dropout: float = 0.0
    # Fixed indices in normalized observation.state that are hidden from the
    # policy in both training and inference. Action labels and relative-EEF
    # anchoring continue to use the complete, unmasked state.
    state_mask_indices: tuple[int, ...] = ()
    # v4 metal-task additions (2026-10-14): anti-state-shortcut + half-unfreeze vision.
    state_noise_std: float = 0.0
    camera_dropout: float = 0.0
    unfreeze_vision_last_blocks: int = 0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if not 0.0 <= self.state_dropout <= 1.0:
            raise ValueError(f"`state_dropout` must be in [0, 1], got {self.state_dropout}.")
        if len(set(self.state_mask_indices)) != len(self.state_mask_indices):
            raise ValueError("`state_mask_indices` must not contain duplicates.")
        invalid_state_indices = [
            index
            for index in self.state_mask_indices
            if not isinstance(index, int) or index < 0 or index >= self.max_state_dim
        ]
        if invalid_state_indices:
            raise ValueError(
                "`state_mask_indices` must be integers in "
                f"[0, {self.max_state_dim}); got {invalid_state_indices}."
            )
        if self.state_noise_std < 0:
            raise ValueError(f"`state_noise_std` must be >= 0, got {self.state_noise_std}.")
        if not 0.0 <= self.camera_dropout <= 1.0:
            raise ValueError(f"`camera_dropout` must be in [0, 1], got {self.camera_dropout}.")
        if self.unfreeze_vision_last_blocks and not self.freeze_vision_encoder:
            raise ValueError("`unfreeze_vision_last_blocks` requires freeze_vision_encoder=True.")
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
