"""End-effector workspace gate for critical-phase RLT takeover.

The gate is deliberately independent of ROS and never publishes commands.  It
maps measured Piper joint positions to the gripper-center pose and latches once
the tool enters the insertion workspace for a configurable number of samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .piper_eef_metrics import DEFAULT_PIPER_URDF, PiperForwardKinematics


@dataclass(frozen=True)
class EEFWorkspace:
    """Axis-aligned insertion workspace in the right Piper base frame."""

    minimum: tuple[float, float, float] = (0.31, 0.30, 0.060)
    maximum: tuple[float, float, float] = (0.39, 0.35, 0.130)

    def __post_init__(self) -> None:
        lower = np.asarray(self.minimum, dtype=np.float64)
        upper = np.asarray(self.maximum, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("EEF workspace bounds must each contain x, y, z")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("EEF workspace bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("each EEF workspace minimum must be below its maximum")

    def contains(self, position: Sequence[float]) -> bool:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("EEF position must be a finite xyz vector")
        return bool(
            np.all(value >= np.asarray(self.minimum))
            and np.all(value <= np.asarray(self.maximum))
        )


class PiperEEFInsertionGate:
    """Latch RLT takeover after consecutive measured EEF workspace hits."""

    def __init__(
        self,
        *,
        workspace: EEFWorkspace = EEFWorkspace(),
        confirmation_samples: int = 2,
        urdf_path=DEFAULT_PIPER_URDF,
    ) -> None:
        if int(confirmation_samples) <= 0:
            raise ValueError("confirmation_samples must be positive")
        self.workspace = workspace
        self.confirmation_samples = int(confirmation_samples)
        self.kinematics = PiperForwardKinematics(
            urdf_path=urdf_path, frames=("gripper_center",)
        )
        self.consecutive_hits = 0
        self.latched = False
        self.last_position: np.ndarray | None = None

    def update(self, robot_state: Sequence[float]) -> bool:
        state = np.asarray(robot_state, dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Piper state must be a finite 7D joint/gripper vector")
        position = self.kinematics.forward(state[None, :6])["gripper_center"][0][0]
        self.last_position = position.copy()
        if self.latched:
            return True
        if self.workspace.contains(position):
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0
        self.latched = self.consecutive_hits >= self.confirmation_samples
        return self.latched


class PiperGraspedEEFInsertionGate(PiperEEFInsertionGate):
    """Task-phase gate: grasp first, then enter the calibrated fixture box.

    Piper's recorded gripper coordinate is near zero when open and rises to
    roughly .055-.08 when closed. A single closed test is insufficient because
    the arm may be reset with a stale gripper reading, so the gate requires an
    observed open->closed cycle before accepting consecutive EEF workspace
    hits.
    """

    def __init__(
        self,
        *,
        workspace: EEFWorkspace,
        confirmation_samples: int = 2,
        open_threshold: float = 0.02,
        closed_threshold: float = 0.04,
        urdf_path=DEFAULT_PIPER_URDF,
    ) -> None:
        if not 0.0 <= float(open_threshold) < float(closed_threshold):
            raise ValueError("require 0 <= open_threshold < closed_threshold")
        super().__init__(
            workspace=workspace,
            confirmation_samples=confirmation_samples,
            urdf_path=urdf_path,
        )
        self.open_threshold = float(open_threshold)
        self.closed_threshold = float(closed_threshold)
        self.seen_open = False
        self.grasp_latched = False

    def update(self, robot_state: Sequence[float]) -> bool:
        state = np.asarray(robot_state, dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Piper state must be a finite 7D joint/gripper vector")
        if state[6] <= self.open_threshold:
            self.seen_open = True
        if self.seen_open and state[6] >= self.closed_threshold:
            self.grasp_latched = True
        if not self.grasp_latched:
            self.consecutive_hits = 0
            position = self.kinematics.forward(state[None, :6])["gripper_center"][0][0]
            self.last_position = position.copy()
            return False
        return super().update(state)
