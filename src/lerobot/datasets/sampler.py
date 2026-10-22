#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterator

import numpy as np
import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class ConsecutiveFrameBatchSampler:
    """Shuffle episode-local blocks, keeping every eligible frame once per epoch.

    A batch contains several independently shuffled blocks, not one whole
    trajectory. Short episode tails are retained rather than oversampled.
    """

    def __init__(self, starts, ends, batch_size, frames_per_block, *, drop_n_last_frames=0):
        if batch_size < 1 or not 1 <= frames_per_block <= batch_size:
            raise ValueError("frames_per_block must be in [1, batch_size]")
        if drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be non-negative")
        self.batch_size = batch_size
        self.blocks = []
        for start, end in zip(starts, ends, strict=True):
            start, end = int(start), int(end) - drop_n_last_frames
            self.blocks.extend(list(range(i, min(i + frames_per_block, end)))
                               for i in range(start, end, frames_per_block))
        self.num_frames = sum(map(len, self.blocks))

    def __iter__(self):
        batch = []
        for block_index in torch.randperm(len(self.blocks)).tolist():
            batch.extend(self.blocks[block_index])
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        return (self.num_frames + self.batch_size - 1) // self.batch_size


class GroupWeightedBatchSampler:
    """Environment-balanced batch sampling with replacement (v4, 2026-10-14).

    Blocks (``frames_per_block`` consecutive frames, clipped at episode end) are
    drawn i.i.d.: a group is picked with probability ``weights[g]``, then a block
    uniformly from that group. Expected frames of g per batch = weights[g]*batch_size
    regardless of group frame counts, i.e. replay per frame =
    weights[g]*steps*batch/frames_g. No physical episode duplication.
    """

    def __init__(
        self,
        starts,
        ends,
        groups,
        weights: dict,
        batch_size: int,
        frames_per_block: int = 1,
        *,
        drop_n_last_frames: int = 0,
        seed: int = 0,
    ):
        import numpy as np

        if batch_size < 1 or not 1 <= frames_per_block <= batch_size:
            raise ValueError("frames_per_block must be in [1, batch_size]")
        rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.frames_per_block = frames_per_block
        pools: dict[str, list[list[int]]] = {}
        counts: dict[str, int] = {}
        for start, end, g in zip(starts, ends, groups, strict=True):
            start, end = int(start), int(end) - drop_n_last_frames
            for i in range(start, end, frames_per_block):
                pools.setdefault(g, []).append(list(range(i, min(i + frames_per_block, end))))
                counts[g] = counts.get(g, 0) + min(i + frames_per_block, end) - i
        missing = set(pools) - set(weights)
        if missing:
            raise ValueError(f"no sampling weight for groups {sorted(missing)}")
        self.groups = sorted(pools)
        self.pools = pools
        w = np.array([float(weights[g]) for g in self.groups], dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("weights must be positive")
        self.probs = w / w.sum()
        self.group_frames = {g: counts[g] for g in self.groups}
        self.total_frames = sum(counts.values())
        self.seed = seed
        self._epoch = 0
        self._rng = rng

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self._epoch)
        self._epoch += 1
        target = len(self)
        for _ in range(target):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                g = self.groups[int(rng.choice(len(self.groups), p=self.probs))]
                pool = self.pools[g]
                batch.extend(pool[int(rng.integers(len(pool)))])
            yield batch[: self.batch_size]

    def __len__(self):
        return (self.total_frames + self.batch_size - 1) // self.batch_size
