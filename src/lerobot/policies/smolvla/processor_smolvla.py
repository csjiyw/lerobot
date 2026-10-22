#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.smolvla.relative_eef import chunk_to_relative


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    if config.eef_relative_in_dataset and config.eef_relative_in_gpu:
        raise ValueError("eef_relative_in_dataset and eef_relative_in_gpu are mutually exclusive")

    normalization_stats = dataset_stats
    if config.use_relative_eef_actions:
        if not config.relative_eef_action_stats_path:
            raise ValueError("relative EEF actions require relative_eef_action_stats_path")
        stats_path = Path(config.relative_eef_action_stats_path).expanduser()
        relative_stats = json.loads(stats_path.read_text())
        normalization_stats = dict(dataset_stats or {})
        # Action statistics must describe the relative chunk, not the absolute
        # EEF targets stored in the dataset. State and the remaining features
        # follow the standard normalization_mapping + dataset stats machinery.
        normalization_stats["action"] = relative_stats["action"]

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # Skipped when the dataset worker already produced relative actions
        # (eef_relative_in_dataset) or the vectorized GPU step below performs
        # the transform (eef_relative_in_gpu): the normalizer still applies the
        # relative-space statistics in all cases.
        *([SmolVLARelativeEEFChunkProcessor()] if config.use_relative_eef_actions and not config.eef_relative_in_dataset and not config.eef_relative_in_gpu else []),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        # Vectorized GPU relative-EEF transform: runs here (after the device
        # move) so the whole (B, H, 7) batch is transformed in float32 on the
        # GPU with a handful of kernels instead of a per-sample python/numpy
        # loop on the main thread.
        *([SmolVLARelativeEEFChunkGPUProcessor()] if config.use_relative_eef_actions and config.eef_relative_in_gpu else []),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=normalization_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=normalization_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="smolvla_relative_eef_chunk")
class SmolVLARelativeEEFChunkProcessor(ProcessorStep):
    """Convert an absolute EEF action chunk relative to its current EEF state."""

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION)
        if action is None or observation is None or "observation.state" not in observation:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"relative EEF action must be a tensor, got {type(action)}")
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"relative EEF state must be a tensor, got {type(state)}")
        unbatched = action.ndim == 2
        actions = action.unsqueeze(0) if unbatched else action
        states = state.unsqueeze(0) if state.ndim == 1 else state
        if actions.ndim != 3 or actions.shape[-1] != 7 or states.shape[-1] != 7:
            raise ValueError(f"expected EEF state/action (...,7), got {states.shape}/{actions.shape}")
        converted = np.stack(
            [chunk_to_relative(s, a) for s, a in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy())]
        )
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.as_tensor(
            converted, dtype=action.dtype, device=action.device
        )[0] if unbatched else torch.as_tensor(converted, dtype=action.dtype, device=action.device)
        return new_transition

    def transform_features(self, features):
        return features


def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    """v: (..., 3) -> (..., 3, 3) skew-symmetric matrices."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def _rotvec_to_matrix_batch(rv: torch.Tensor) -> torch.Tensor:
    """Vectorized Rodrigues: rv (..., 3) -> rotation matrices (..., 3, 3)."""
    theta = rv.norm(dim=-1, keepdim=True)  # (..., 1)
    k = _skew_batch(rv)  # (..., 3, 3)
    k2 = k @ k
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    small = theta < 1e-8
    # Coefficients of the closed form R = I + c1 * K + c2 * K^2 (K = skew(rv)):
    #   c1 = sin(theta)/theta -> 1,  c2 = (1 - cos(theta))/theta^2 -> 1/2.
    c1 = torch.where(small, torch.ones_like(theta), sin_t / theta.clamp(min=1e-12))
    c2 = torch.where(small, torch.full_like(theta, 0.5), (1.0 - cos_t) / theta.clamp(min=1e-12) ** 2)
    eye = torch.eye(3, device=rv.device, dtype=rv.dtype).expand_as(k)
    return eye + c1.unsqueeze(-1) * k + c2.unsqueeze(-1) * k2


def _matrix_to_rotvec_batch(rot: torch.Tensor) -> torch.Tensor:
    """Vectorized inverse Rodrigues: rot (..., 3, 3) -> rotvecs (..., 3)."""
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos)  # (...)
    # Off-diagonal cross vector (2 * sin(angle) * axis).
    r = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )  # (..., 3)
    sin_t = torch.sin(angle)
    general = r * (angle / (2.0 * sin_t.clamp(min=1e-12))).unsqueeze(-1)
    small_angle = r * 0.5
    # Degenerate pi branch: axis is the largest-eigenvector of (R + I) / 2.
    eye = torch.eye(3, device=rot.device, dtype=rot.dtype).expand_as(rot)
    eig_vals, eig_vecs = torch.linalg.eigh((rot + eye) * 0.5)
    axis = eig_vecs[..., :, -1]
    pi_rotvec = (axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)) * angle.unsqueeze(-1)
    small_mask = (angle < 1e-8).unsqueeze(-1)
    pi_mask = (angle > (math.pi - 1e-6)).unsqueeze(-1)
    return torch.where(small_mask, small_angle, torch.where(pi_mask, pi_rotvec, general))


def _chunk_to_relative_gpu(state: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Batched GPU equivalent of relative_eef.chunk_to_relative.

    state: (B, 7) [pos, rotvec, gripper]; targets: (B, H, 7). Returns (B, H, 7).
    """
    r_state = _rotvec_to_matrix_batch(state[:, 3:6])  # (B, 3, 3)
    r_target = _rotvec_to_matrix_batch(targets[..., 3:6])  # (B, H, 3, 3)
    r_state_t = r_state.transpose(-1, -2)  # (B, 3, 3)
    r_rel = torch.matmul(r_state_t.unsqueeze(1), r_target)  # (B, H, 3, 3)
    # numpy: R_state.T @ (t - s); with row vectors that is (t - s) @ R_state.
    t_rel = (targets[..., :3] - state[:, :3].unsqueeze(1)) @ r_state  # (B, H, 3)
    rot_rel = _matrix_to_rotvec_batch(r_rel)  # (B, H, 3)
    return torch.cat([t_rel, rot_rel, targets[..., 6:7]], dim=-1)


@dataclass
@ProcessorStepRegistry.register(name="smolvla_relative_eef_chunk_gpu")
class SmolVLARelativeEEFChunkGPUProcessor(ProcessorStep):
    """Vectorized GPU relative-EEF transform (batched, float32, no python loop).

    Must run on the device the batch already lives on (place after the
    preprocessor's DeviceProcessorStep).
    """

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION)
        if action is None or observation is None or "observation.state" not in observation:
            return transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"relative EEF action must be a tensor, got {type(action)}")
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"relative EEF state must be a tensor, got {type(state)}")
        if state.ndim == 3 and state.shape[1] == 1:
            # Dataset produces (B, 1, 7) (singleton time dim from delta
            # timestamps); the EEF transform uses the single conditioning state.
            state = state[:, 0]
        unbatched = action.ndim == 2
        actions = action.unsqueeze(0) if unbatched else action
        states = state.unsqueeze(0) if state.ndim == 1 else state
        if actions.ndim != 3 or actions.shape[-1] != 7 or states.shape[-1] != 7:
            raise ValueError(f"expected EEF state/action (...,7), got {states.shape}/{actions.shape}")
        converted = _chunk_to_relative_gpu(
            states.to(action.dtype, non_blocking=True), actions.to(action.dtype, non_blocking=True)
        )
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = converted[0] if unbatched else converted
        return new_transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
