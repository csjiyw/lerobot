"""Robot-like PC interface over :mod:`cobot_bridge.client`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .client import BridgeError, CobotBridgeClient
from .eef_trajectory_limiter import CartesianTrajectoryLimiter
from .piper_eef_transform import (
    LEGACY_MODEL_EEF_FRAME,
    PiperEEFFrameTransform,
    SDK_EEF_FRAME,
    SUPPORTED_EEF_FRAMES,
    resolve_piper_urdf,
)
from .piper_pink_ik import (
    PiperIKSettings,
    PiperPinkIK,
    hardware_to_pink_joints,
    pink_to_hardware_joints,
)


SAFE = 0
POLICY_PUPPET = 1
RLT_SYNC = 2
TEACH = 3
RESYNC_REQUIRED = 4
ALIGN_SYNC = 5


class ControlNotReady(BridgeError):
    """An action was requested while TEACH/SAFE/resync owns control."""


@dataclass
class RemoteObservation:
    raw: dict[str, Any]
    image_bytes: dict[str, bytes]
    side: str

    @staticmethod
    def _position(section: dict[str, Any], role: str, side: str) -> np.ndarray:
        sample = section.get(role, {}).get(side)
        if not isinstance(sample, dict):
            raise ControlNotReady(f"missing_{role}_{side}_state")
        values = np.asarray(sample.get("position", []), dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ControlNotReady(f"invalid_{role}_{side}_state")
        return values

    @staticmethod
    def _eef_position(section: dict[str, Any], role: str, side: str) -> np.ndarray:
        sample = section.get(role, {}).get(side)
        if not isinstance(sample, dict):
            raise ControlNotReady(f"missing_{role}_{side}_eef_state")
        values = np.asarray(sample.get("position", []), dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ControlNotReady(f"invalid_{role}_{side}_eef_state")
        return values

    @property
    def master_state(self) -> np.ndarray:
        return self._position(self.raw, "master", self.side)

    @property
    def puppet_state(self) -> np.ndarray:
        return self._position(self.raw, "puppet", self.side)

    @property
    def master_eef_state(self) -> np.ndarray:
        return self._eef_position(self.raw, "master_eef", self.side)

    @property
    def puppet_eef_state(self) -> np.ndarray:
        return self._eef_position(self.raw, "puppet_eef", self.side)

    def eef_position(self, role: str, side: str) -> np.ndarray:
        if role not in ("master", "puppet"):
            raise ValueError(f"invalid role {role!r}")
        if side not in ("left", "right"):
            raise ValueError(f"invalid side {side!r}")
        return self._eef_position(
            self.raw,
            "master_eef" if role == "master" else "puppet_eef",
            side,
        )

    def position(self, role: str, side: str) -> np.ndarray:
        if role not in ("master", "puppet"):
            raise ValueError(f"invalid role {role!r}")
        if side not in ("left", "right"):
            raise ValueError(f"invalid side {side!r}")
        return self._position(self.raw, role, side)

    @property
    def control(self) -> dict[str, Any]:
        value = self.raw.get("control", {})
        return value if isinstance(value, dict) else {}

    @property
    def epoch(self) -> int:
        return int(self.control.get("epoch", -1))

    @property
    def effective_mode(self) -> int:
        return int(self.control.get("effective_mode_id", SAFE))

    @property
    def teach_active(self) -> bool:
        return bool(self.control.get("teach_active", False))

    @property
    def ready(self) -> bool:
        return bool(self.raw.get("ready", False))

    def executed_puppet_action(self) -> np.ndarray | None:
        sample = self.raw.get("commands", {}).get("puppet", {}).get(self.side)
        if not isinstance(sample, dict) or not sample.get("fresh", False):
            return None
        values = np.asarray(sample.get("position", []), dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            return None
        return values

    def decode_image(self, name: str) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to decode camera JPEGs") from exc
        data = self.image_bytes.get(name)
        if data is None:
            raise ControlNotReady(f"missing_{name}_image")
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ControlNotReady(f"invalid_{name}_jpeg")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CobotMagicRemoteRobot:
    """Thin stateful wrapper used by collection, IL and RLT runners."""

    def __init__(
        self,
        client: CobotBridgeClient,
        *,
        side: str = "left",
        images: Iterable[str] = ("front", "left", "right"),
        model_eef_frame: str = LEGACY_MODEL_EEF_FRAME,
        eef_urdf_path: str | None = None,
        eef_ik_backend: str = "sdk",
        eef_cartesian_smooth: bool = False,
        pink_urdf_path: str | None = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"invalid side {side!r}")
        if model_eef_frame not in SUPPORTED_EEF_FRAMES:
            raise ValueError(
                f"invalid model_eef_frame {model_eef_frame!r}; "
                f"expected one of {SUPPORTED_EEF_FRAMES}"
            )
        if eef_ik_backend not in ("sdk", "pink"):
            raise ValueError("eef_ik_backend must be 'sdk' or 'pink'")
        self.client = client
        self.side = side
        self.images = tuple(images)
        self.model_eef_frame = str(model_eef_frame)
        self.eef_urdf_path = eef_urdf_path
        self.eef_ik_backend = eef_ik_backend
        self.eef_cartesian_smooth = bool(eef_cartesian_smooth)
        self.pink_urdf_path = pink_urdf_path
        self._eef_transform: PiperEEFFrameTransform | None = None
        self._pink_ik: PiperPinkIK | None = None
        self._pink_control_epoch: int | None = None
        self._eef_limiter: CartesianTrajectoryLimiter | None = None
        self._eef_limiter_epoch: int | None = None
        self.sequence = 0
        self.last_observation: RemoteObservation | None = None

    def _get_eef_transform(self) -> PiperEEFFrameTransform:
        if self._eef_transform is None:
            self._eef_transform = PiperEEFFrameTransform(self.eef_urdf_path)
        return self._eef_transform

    def _get_pink_ik(self) -> PiperPinkIK:
        if self._pink_ik is None:
            # Pink runs on the PC/rollout process and emits only joint targets.
            # It never publishes ROS/CAN output itself.
            self._pink_ik = PiperPinkIK(
                resolve_piper_urdf(self.pink_urdf_path or self.eef_urdf_path),
                # Recorded demonstrations reach about 0.53 rad/s.  The old
                # 0.45-rad/s runtime cap accumulated severe Cartesian lag
                # over a complete 30 Hz episode; 0.60 rad/s tracks the same
                # trajectory offline without solver failures.
                settings=PiperIKSettings(
                    control_hz=200.0,
                    max_joint_velocity=0.60,
                ),
            )
        return self._pink_ik

    @staticmethod
    def _rotvec_target_to_se3(ik: PiperPinkIK, target: np.ndarray) -> Any:
        rotation = ik.pin.exp3(np.asarray(target[3:6], dtype=float))
        return ik.pin.SE3(rotation, np.asarray(target[:3], dtype=float))

    def _pink_joint_chunk_from_target(
        self,
        sdk_target: np.ndarray,
        *,
        observation: RemoteObservation,
        ticks: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Turn one link6 EEF target into a bounded 200 Hz joint chunk."""

        if ticks < 1:
            raise ValueError(f"Pink IK ticks must be positive, got {ticks}")
        ik = self._get_pink_ik()
        hardware_joints = np.asarray(observation.puppet_state[:6], dtype=float)
        pink_joints = ik.coerce_measured_joints(
            hardware_to_pink_joints(hardware_joints)
        )
        target = np.asarray(sdk_target, dtype=float).reshape(-1)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"SDK EEF target must be finite shape-(7,), got {target.shape}")
        # Initialize the differential controller only when ownership changes.
        # Resetting it for every 30 Hz policy waypoint used to discard the
        # 200 Hz velocity history and move the posture target to the measured
        # pose on every call.  Near the Piper zero-pose singularity that turns
        # a Cartesian trajectory into a sequence of tiny local corrections.
        initialize = self._pink_control_epoch != observation.epoch
        if initialize:
            ik.reset(pink_joints)
            ik.set_posture_reference(pink_joints)
            self._pink_control_epoch = observation.epoch
        if self.eef_cartesian_smooth:
            if self._eef_limiter is None:
                self._eef_limiter = CartesianTrajectoryLimiter()
            if self._eef_limiter_epoch != observation.epoch:
                self._eef_limiter.reset(observation.eef_position("puppet", self.side))
                self._eef_limiter_epoch = observation.epoch
            eef_targets = self._eef_limiter.advance(target, ticks)
        else:
            eef_targets = [target] * ticks
        chunk: list[np.ndarray] = []
        last_diagnostics: dict[str, Any] | None = None
        for eef_target in eef_targets:
            # Keep the generated 200 Hz trajectory continuous for the complete
            # control epoch. Re-anchoring to lagging hardware feedback at each
            # 30 Hz waypoint truncates the planned motion: every target then
            # starts from the old measured pose and can advance by only a few
            # IK ticks. Runtime feedback remains an external tracking/safety
            # signal and a new epoch performs the next explicit reset.
            step = ik.step(self._rotvec_target_to_se3(ik, eef_target))
            hardware_target = pink_to_hardware_joints(step.joints)
            action = np.concatenate((hardware_target, np.asarray([eef_target[6]], dtype=float)))
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("Pink IK produced a non-finite hardware joint target")
            chunk.append(action.astype(np.float32, copy=False))
            last_diagnostics = step.diagnostics.as_dict()
        assert last_diagnostics is not None
        return np.stack(chunk), last_diagnostics

    def model_eef_state(
        self,
        *,
        role: str = "puppet",
        side: str | None = None,
        observation: RemoteObservation | None = None,
    ) -> np.ndarray:
        """Return the measured EEF in the frame expected by the policy.

        Bridge feedback is always the SDK ``link6`` pose.  Existing converted
        EEF datasets/checkpoints use ``gripper_center`` by default, so this is
        the single observation-side frame boundary for all policy runners.
        """

        selected_side = self.side if side is None else side
        current = observation or self.last_observation or self.observe()
        raw = current.eef_position(role, selected_side)
        if self.model_eef_frame == SDK_EEF_FRAME:
            return raw.astype(np.float32, copy=True)
        return self._get_eef_transform().transform(
            raw,
            source_frame=SDK_EEF_FRAME,
            target_frame=self.model_eef_frame,
        )

    def observe(self) -> RemoteObservation:
        # Both side states are small and are needed to provide a stationary
        # hold target for the non-policy side during all-side Mode 2 sync.
        raw, images = self.client.get_observation(sides=("left", "right"), images=self.images)
        observation = RemoteObservation(raw=raw, image_bytes=images, side=self.side)
        self.last_observation = observation
        return observation

    def request_mode(self, mode: int) -> dict[str, Any]:
        return self.client.set_mode(mode)

    def set_output(self, enabled: bool) -> dict[str, Any]:
        return self.client.set_output(enabled)

    def resync_if_required(self) -> RemoteObservation:
        observation = self.observe()
        if observation.effective_mode == RESYNC_REQUIRED:
            self.client.resync()
            observation = self.observe()
        return observation

    def send_action(
        self, action: Sequence[float], *, observation: RemoteObservation | None = None
    ) -> dict[str, Any]:
        return self.send_action_for_side(self.side, action, observation=observation)

    def send_action_for_side(
        self,
        side: str,
        action: Sequence[float],
        *,
        observation: RemoteObservation | None = None,
    ) -> dict[str, Any]:
        if side not in ("left", "right"):
            raise ValueError(f"invalid side {side!r}")
        current = observation or self.last_observation or self.observe()
        if current.teach_active or current.effective_mode == TEACH:
            raise ControlNotReady("teach_active")
        if current.effective_mode == RESYNC_REQUIRED:
            raise ControlNotReady("resync_required")
        if current.effective_mode not in (POLICY_PUPPET, RLT_SYNC):
            raise ControlNotReady("policy_not_owner")
        if current.epoch < 0:
            raise ControlNotReady("missing_epoch")
        self.sequence += 1
        return self.client.set_action(
            side,
            [float(value) for value in action],
            epoch=current.epoch,
            sequence=self.sequence,
        )

    def send_eef_target(
        self,
        target: Sequence[float],
        *,
        observation: RemoteObservation | None = None,
    ) -> dict[str, Any]:
        """Send an absolute EEF target through the protected ROS PosCmd path."""

        current = observation or self.last_observation or self.observe()
        if current.teach_active or current.effective_mode == TEACH:
            raise ControlNotReady("teach_active")
        if current.effective_mode == RESYNC_REQUIRED:
            raise ControlNotReady("resync_required")
        if current.effective_mode not in (POLICY_PUPPET, RLT_SYNC):
            raise ControlNotReady("policy_not_owner")
        if current.epoch < 0:
            raise ControlNotReady("missing_epoch")
        self.sequence += 1
        return self.client.set_action(
            self.side,
            [float(value) for value in target],
            epoch=current.epoch,
            sequence=self.sequence,
            action_space="eef_target",
        )

    def send_model_eef_target(
        self,
        target: Sequence[float],
        *,
        observation: RemoteObservation | None = None,
        pink_ticks: int | None = None,
    ) -> dict[str, Any]:
        """Convert a policy-frame target back to the SDK ``link6`` frame.

        ``send_eef_target`` intentionally remains the raw SDK-frame method so
        callers that already have a link6 target cannot be silently changed.
        Policy runners must use this explicit model-boundary method.
        """

        values = np.asarray(target, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"model EEF target must be finite shape-(7,), got {values.shape}")
        if self.model_eef_frame == SDK_EEF_FRAME:
            sdk_target = values
        else:
            sdk_target = self._get_eef_transform().transform(
                values,
                source_frame=self.model_eef_frame,
                target_frame=SDK_EEF_FRAME,
            )
        if self.eef_ik_backend == "pink":
            current = observation or self.last_observation or self.observe()
            # A policy tick normally represents one 30 Hz interval.  The
            # runner supplies the cumulative 6/7-tick value so 200 Hz timing
            # stays aligned over long episodes.
            requested_ticks = 1 if pink_ticks is None else int(pink_ticks)
            joint_chunk, diagnostics = self._pink_joint_chunk_from_target(
                sdk_target,
                observation=current,
                ticks=requested_ticks,
            )
            result = self.send_action_chunk(
                joint_chunk,
                waypoint_hz=200.0,
                observation=current,
            )
            result = dict(result)
            result.update(
                {
                    "action_space": "joint",
                    "eef_ik_backend": "pink",
                    "pink_ticks": requested_ticks,
                    "pink_diagnostics": diagnostics,
                }
            )
            return result
        return self.send_eef_target(sdk_target.tolist(), observation=observation)

    def model_eef_target_to_joint_target(
        self,
        target: Sequence[float],
        *,
        observation: RemoteObservation | None = None,
        pink_ticks: int = 1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Plan one model-frame EEF target to a joint target without sending it.

        This is the EEF-to-joint boundary for rollout-side command shaping.
        Pink owns only the differential IK continuity; the caller remains the
        sole owner of the 30 Hz joint velocity/acceleration limiter and bridge
        transmission.
        """

        if self.eef_ik_backend != "pink":
            raise RuntimeError(
                "model_eef_target_to_joint_target requires eef_ik_backend='pink'"
            )
        values = np.asarray(target, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"model EEF target must be finite shape-(7,), got {values.shape}")
        if self.model_eef_frame == SDK_EEF_FRAME:
            sdk_target = values
        else:
            sdk_target = self._get_eef_transform().transform(
                values,
                source_frame=self.model_eef_frame,
                target_frame=SDK_EEF_FRAME,
            )
        current = observation or self.last_observation or self.observe()
        joint_chunk, diagnostics = self._pink_joint_chunk_from_target(
            sdk_target, observation=current, ticks=int(pink_ticks)
        )
        return joint_chunk[-1].astype(np.float32, copy=False), diagnostics

    def send_action_chunk(
        self,
        actions: Sequence[Sequence[float]],
        *,
        waypoint_hz: float,
        observation: RemoteObservation | None = None,
    ) -> dict[str, Any]:
        current = observation or self.last_observation or self.observe()
        if current.teach_active or current.effective_mode == TEACH:
            raise ControlNotReady("teach_active")
        if current.effective_mode == RESYNC_REQUIRED:
            raise ControlNotReady("resync_required")
        if current.effective_mode not in (POLICY_PUPPET, RLT_SYNC):
            raise ControlNotReady("policy_not_owner")
        if current.epoch < 0:
            raise ControlNotReady("missing_epoch")
        # The bridge is persistent across short-lived rollout clients and keeps
        # the last chunk sequence per side.  Use a process-independent monotonic
        # wire value so a new episode cannot restart at 1 and be rejected stale.
        self.sequence = max(self.sequence + 1, time.time_ns())
        wire_actions = [[float(value) for value in action] for action in actions]
        return self.client.set_action_chunk(
            self.side,
            wire_actions,
            epoch=current.epoch,
            sequence=self.sequence,
            waypoint_hz=float(waypoint_hz),
        )

    def send_alignment_target(
        self, action: Sequence[float], *, observation: RemoteObservation | None = None
    ) -> dict[str, Any]:
        return self.send_alignment_target_for_side(self.side, action, observation=observation)

    def send_alignment_target_for_side(
        self,
        side: str,
        action: Sequence[float],
        *,
        observation: RemoteObservation | None = None,
    ) -> dict[str, Any]:
        if side not in ("left", "right"):
            raise ValueError(f"invalid side {side!r}")
        current = observation or self.last_observation or self.observe()
        if current.teach_active:
            raise ControlNotReady("teach_active")
        if current.effective_mode != ALIGN_SYNC:
            raise ControlNotReady("alignment_not_owner")
        if current.epoch < 0:
            raise ControlNotReady("missing_epoch")
        self.sequence += 1
        return self.client.set_action(
            side,
            [float(value) for value in action],
            epoch=current.epoch,
            sequence=self.sequence,
        )
