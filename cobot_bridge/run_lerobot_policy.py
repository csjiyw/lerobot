"""Run a trained LeRobot policy against the remote Cobot bridge.

The default is inference-only: actions are computed and logged but not sent.
``--send-actions`` is an explicit opt-in and the AGX arbiter still enforces its
own output, teach, epoch and process-level dry-run gates.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .client import CobotBridgeClient
from .move_init_pose import INIT_POSE
from .policy_client import PersistentPolicyClient
from .remote_robot import (
    ALIGN_SYNC,
    POLICY_PUPPET,
    RESYNC_REQUIRED,
    RLT_SYNC,
    SAFE,
    TEACH,
    CobotMagicRemoteRobot,
)
from .piper_eef_transform import (
    LEGACY_MODEL_EEF_FRAME,
    SUPPORTED_EEF_FRAMES,
    apply_eef_pose_delta,
)
from .eef_trajectory_limiter import blend_eef_chunk_edge


LOG = logging.getLogger("cobot_bridge.run_lerobot_policy")
SIDES = ("left", "right")
POLICY_DECISION_TRACE_FORMAT = "cobot_magic_policy_decisions_v1"


def resolve_action_steps(requested: int | None, available: int) -> int:
    """Resolve the execution horizon without exceeding the model response."""

    if available < 1:
        raise RuntimeError(f"policy server returned an empty action chunk: {available}")
    if requested is None:
        return available
    if requested < 1:
        raise RuntimeError(f"n_action_steps must be positive, got {requested}")
    return min(requested, available)


class CumulativeControlTickScheduler:
    """Distribute a policy-rate interval over integer 200 Hz ticks.

    At 30 FPS this emits the repeating sequence 7, 7, 6.  Accumulating the
    fractional timing avoids the long-run drift caused by always using 6 or 7
    ticks for every model action.
    """

    def __init__(self, *, policy_hz: float, control_hz: float = 200.0) -> None:
        if not np.isfinite(policy_hz) or policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        self.policy_hz = float(policy_hz)
        self.control_hz = float(control_hz)
        self.index = 0

    def next(self) -> int:
        self.index += 1
        current = int(round(self.index * self.control_hz / self.policy_hz))
        previous = int(round((self.index - 1) * self.control_hz / self.policy_hz))
        return max(1, current - previous)


def eef_target_from_delta(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply one converted dataset delta to the measured EEF state."""

    current = np.asarray(state, dtype=np.float32).reshape(-1)
    action = np.asarray(delta, dtype=np.float32).reshape(-1)
    if current.shape != (7,) or action.shape != (7,):
        raise ValueError(f"EEF state/action must both have shape (7,), got {current.shape}/{action.shape}")
    if not np.isfinite(current).all() or not np.isfinite(action).all():
        raise ValueError("EEF state/action contains non-finite values")
    return apply_eef_pose_delta(current, action)


def _append_policy_decision_trace(
    path: Path | None,
    *,
    observation: Any,
    model_state: np.ndarray,
    model_native_action_chunk: np.ndarray,
    model_absolute_action_chunk: np.ndarray,
    rollout_action_chunk: np.ndarray,
    model_output_semantics: str,
    action_space: str,
    consumed_actions: int,
    latency_ms: Any,
    rtc_prefix_steps: Any,
) -> None:
    """Append one inexpensive, exact inference decision for debug capture.

    This runs once per model chunk, never in the 200 Hz IK loop.  The passive
    debug worker later merges this sidecar into its finalized raw HDF5 record,
    keyed by the bridge sequence captured with the model observation.
    """

    if path is None:
        return
    state = np.asarray(model_state, dtype=np.float32)
    native = np.asarray(model_native_action_chunk, dtype=np.float32)
    absolute = np.asarray(model_absolute_action_chunk, dtype=np.float32)
    rollout = np.asarray(rollout_action_chunk, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"trace model state must be finite shape (7,), got {state.shape}")
    if (
        native.ndim != 2
        or native.shape[1:] != (7,)
        or absolute.shape != native.shape
        or rollout.shape != native.shape
        or not np.isfinite(native).all()
        or not np.isfinite(absolute).all()
        or not np.isfinite(rollout).all()
    ):
        raise ValueError("policy decision trace chunks must be finite matching (steps, 7) arrays")
    if model_output_semantics not in ("absolute_joint", "absolute_eef", "relative_eef"):
        raise ValueError(f"unsupported trace model output semantics: {model_output_semantics!r}")
    raw = observation.raw
    payload = {
        "format": POLICY_DECISION_TRACE_FORMAT,
        "time_ns": time.time_ns(),
        "bridge_time_ns": int(raw.get("bridge_time_ns", 0)),
        "bridge_sequence": int(raw.get("sequence", -1)),
        "model_state": state.tolist(),
        "model_native_action_chunk": native.tolist(),
        "model_absolute_action_chunk": absolute.tolist(),
        "rollout_action_chunk": rollout.tolist(),
        "model_output_semantics": model_output_semantics,
        "action_space": str(action_space),
        "consumed_actions": int(consumed_actions),
        "latency_ms": None if latency_ms is None else float(latency_ms),
        "rtc_prefix_steps": None if rtc_prefix_steps is None else int(rtc_prefix_steps),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--policy-path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--action-space",
        choices=("joint", "eef_delta", "eef_target"),
        default="joint",
        help=(
            "policy state/action contract; eef_delta uses one-step SDK PosCmd, "
            "eef_target uses a complete absolute EEF action chunk"
        ),
    )
    parser.add_argument(
        "--eef-chunk-edge-max-steps", type=int, default=12,
        help="V5 maximum dynamic 30 Hz SE(3) handoff frames at a policy chunk edge",
    )
    parser.add_argument(
        "--model-eef-frame",
        choices=SUPPORTED_EEF_FRAMES,
        default=LEGACY_MODEL_EEF_FRAME,
        help=(
            "absolute EEF frame used by the checkpoint; bridge feedback and "
            "SDK targets are converted through link6"
        ),
    )
    parser.add_argument(
        "--eef-ik-backend",
        choices=("pink", "sdk"),
        default="pink",
        help="EEF target backend: Pink IK emits bounded JointState chunks; sdk uses Piper PosCmd",
    )
    parser.add_argument(
        "--pink-urdf",
        default=None,
        help="optional official Piper URDF path for the Pink IK backend",
    )
    parser.add_argument(
        "--sync-sides",
        nargs="+",
        choices=SIDES,
        default=SIDES,
        help="robot-side participants that must receive a Mode 2 alignment target",
    )
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument(
        "--eef-cartesian-smooth", action="store_true",
        help="V5: smooth absolute EEF targets in SE(3) before 200 Hz Pink IK",
    )
    parser.add_argument("--policy-server-endpoint")
    parser.add_argument("--policy-server-timeout", type=float, default=20.0)
    parser.add_argument(
        "--policy-decision-trace",
        type=Path,
        help=(
            "debug-only JSONL sidecar: persist native model chunks and restored "
            "EEF targets once per inference decision"
        ),
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="actions consumed from each predicted chunk before observing and predicting again",
    )
    parser.add_argument(
        "--guided-rtc-terminal",
        action="store_true",
        help=(
            "synchronous Guided RTC: execute every action in each native model chunk; "
            "on the next prediction, guide its first action toward the previous chunk terminal target"
        ),
    )
    parser.add_argument(
        "--cameras", nargs="+", choices=("front", "left", "right"), default=("front", "right")
    )
    parser.add_argument(
        "--startup-init-hold",
        action="store_true",
        help=(
            "optional: before rollout, verify the arm is near init and hold/open "
            "the gripper; omitted rollouts start at the current measured pose"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--vlm-model-path")
    return parser


def _target_reached(state: np.ndarray) -> bool:
    error = np.abs(state - INIT_POSE)
    return float(np.max(error[:6])) <= 0.04 and float(error[6]) <= 0.012


def _hold_startup_init(
    robot: CobotMagicRemoteRobot,
    *,
    sync_sides: tuple[str, ...] = SIDES,
    timeout_s: float = 35.0,
) -> dict[str, np.ndarray]:
    observation = robot.observe()
    arm_error = float(np.max(np.abs(observation.puppet_state[:6] - INIT_POSE[:6])))
    if arm_error > 0.05:
        raise RuntimeError(
            f"startup init hold refused: arm is {arm_error:.3f} rad from init; "
            "run move_init_pose first"
        )
    initial_mode = observation.effective_mode
    auxiliary_holds: dict[str, np.ndarray] = {}
    if initial_mode == ALIGN_SYNC:
        result = robot.send_alignment_target(INIT_POSE.tolist(), observation=observation)
        if not result.get("accepted", False):
            raise RuntimeError(f"startup init hold rejected: {result}")
        for side in sync_sides:
            if side == robot.side:
                continue
            # Keep the unused side's master stationary. Its puppet independently
            # aligns to this same target, satisfying the all-side arbiter gate.
            target = observation.position("master", side).astype(np.float32, copy=True)
            result = robot.send_alignment_target_for_side(
                side, target.tolist(), observation=observation
            )
            if not result.get("accepted", False):
                raise RuntimeError(f"{side} alignment target rejected: {result}")
            auxiliary_holds[side] = target
    elif initial_mode in (POLICY_PUPPET, RLT_SYNC):
        result = robot.send_action(INIT_POSE.tolist(), observation=observation)
        # The mode response and the observation cache are delivered by
        # separate ROS/ZMQ paths. Immediately after SET_MODE the first cached
        # observation can therefore still carry the previous epoch. A stale
        # epoch is a safe rejection, so refresh once and retry only while the
        # requested policy mode still owns actions.
        if not result.get("accepted", False) and result.get("reason") == "stale_epoch":
            observation = robot.observe()
            if observation.effective_mode != initial_mode:
                raise RuntimeError(
                    "startup init hold lost mode while refreshing stale epoch: "
                    f"effective_mode={observation.effective_mode}"
                )
            result = robot.send_action(INIT_POSE.tolist(), observation=observation)
    else:
        raise RuntimeError(
            f"startup init hold refused: effective mode {initial_mode} does not own actions"
        )
    if not result.get("accepted", False):
        raise RuntimeError(f"startup init hold rejected: {result}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = robot.observe()
        if observation.teach_active or observation.effective_mode == TEACH:
            raise RuntimeError("startup init hold interrupted by Teach")
        if observation.effective_mode in (SAFE, RESYNC_REQUIRED):
            raise RuntimeError(
                "startup init hold lost control ownership: "
                f"effective_mode={observation.effective_mode}"
            )
        if initial_mode == ALIGN_SYNC:
            if observation.effective_mode != RLT_SYNC:
                time.sleep(0.05)
                continue
            # Satisfy the robot-side post-alignment watchdog immediately. This
            # keeps both arms at the alignment target while the policy resets
            # and computes its first fresh action.
            for side, target in auxiliary_holds.items():
                hold_result = robot.send_action_for_side(
                    side, target.tolist(), observation=observation
                )
                if not hold_result.get("accepted", False):
                    raise RuntimeError(f"post-alignment {side} hold rejected: {hold_result}")
            hold_result = robot.send_action(INIT_POSE.tolist(), observation=observation)
            if not hold_result.get("accepted", False):
                raise RuntimeError(f"post-alignment init hold rejected: {hold_result}")
            both_arms_reached = _target_reached(observation.master_state) and _target_reached(
                observation.puppet_state
            )
            if not both_arms_reached:
                raise RuntimeError(
                    "alignment entered RLT_SYNC before both arms reached the init target"
                )
            LOG.info(
                "Mode 2 alignment reached init: master=%s puppet=%s",
                observation.master_state.tolist(),
                observation.puppet_state.tolist(),
            )
            return auxiliary_holds
        if observation.effective_mode != initial_mode:
            raise RuntimeError(
                "startup init hold changed control mode unexpectedly: "
                f"{initial_mode} -> {observation.effective_mode}"
            )
        if _target_reached(observation.puppet_state):
            LOG.info("startup init hold reached: %s", observation.puppet_state.tolist())
            return auxiliary_holds
        time.sleep(0.05)
    if initial_mode == ALIGN_SYNC:
        raise RuntimeError(
            "Mode 2 alignment timed out: "
            f"effective_mode={observation.effective_mode}, "
            f"master_error={np.round(observation.master_state - INIT_POSE, 4).tolist()}, "
            f"puppet_error={np.round(observation.puppet_state - INIT_POSE, 4).tolist()}"
        )
    raise RuntimeError(
        "startup init hold timed out: "
        + str(np.round(observation.puppet_state - INIT_POSE, 4).tolist())
    )


def _load_policy(args: argparse.Namespace):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(args.policy_path)
    if args.device:
        config.device = args.device
    vlm_model_path = getattr(args, "vlm_model_path", None)
    if vlm_model_path and hasattr(config, "vlm_model_name"):
        config.vlm_model_name = vlm_model_path
        # The policy checkpoint contains the complete VLM state dict. Construct
        # from the local config and load the checkpoint once, avoiding a second
        # 1.9 GB base-weight load on an 8 GB robot GPU.
        config.load_vlm_weights = False
    requested_action_steps = getattr(args, "n_action_steps", None)
    if requested_action_steps is not None:
        if not 1 <= requested_action_steps <= config.chunk_size:
            raise ValueError(
                "n_action_steps must be between 1 and the checkpoint chunk_size "
                f"({config.chunk_size}), got {requested_action_steps}"
            )
        config.n_action_steps = requested_action_steps
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        args.policy_path,
        config=config,
    )
    policy.to(config.device)
    device_override = {"device": str(config.device)}
    preprocessor_overrides = {"device_processor": device_override}
    if vlm_model_path:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": vlm_model_path,
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides={"device_processor": device_override},
    )
    if bool(getattr(config, "use_relative_eef_actions", False)):
        # Relative-EEF checkpoints load their saved processor, which carries
        # the normalization_mapping and the exact statistics training used.
        # Log the resulting contract so rollout runs are auditable.
        for step in preprocessor.steps:
            step_stats = getattr(step, "stats", None)
            norm_map = getattr(step, "norm_map", None)
            if not step_stats or not norm_map:
                continue
            state_mode = norm_map.get("STATE")
            state_stats_present = "observation.state" in step_stats
            if state_stats_present:
                LOG.warning("state normalization contract: %s (stats from saved processor)", state_mode)
            elif state_mode and state_mode != "IDENTITY":
                LOG.warning(
                    "state normalization contract: EFFECTIVELY RAW - saved processor "
                    "declares STATE=%s but contains no observation.state statistics",
                    state_mode,
                )
            else:
                LOG.warning("state normalization contract: IDENTITY (raw state)")
    policy.eval()
    policy.reset()
    return policy, preprocessor, postprocessor


def _policy_observation(
    observation,
    cameras: tuple[str, ...],
    *,
    action_space: str = "joint",
    eef_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if action_space in ("eef_delta", "eef_target"):
        state = (
            np.asarray(eef_state, dtype=np.float32)
            if eef_state is not None
            else observation.puppet_eef_state
        )
    else:
        state = observation.puppet_state
    result: dict[str, np.ndarray] = {"observation.state": state}
    for name in cameras:
        key = f"observation.images.camera_{name[0] if name != 'front' else 'f'}"
        result[key] = observation.decode_image(name)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.action_space in ("eef_delta", "eef_target")
        and args.mode == 2
        and args.eef_ik_backend != "pink"
    ):
        parser.error(
            "Mode 2 EEF actions require Pink IK: convert the policy EEF target to a "
            "joint chunk on the PC before the robot arbiter synchronizes master and puppet"
        )
    if args.action_space == "eef_delta":
        if args.n_action_steps not in (None, 1):
            parser.error("eef_delta requires --n-action-steps 1 because each delta is state-relative")
        args.n_action_steps = 1
    if args.guided_rtc_terminal:
        if not args.policy_server_endpoint:
            parser.error("--guided-rtc-terminal requires --policy-server-endpoint")
        if args.action_space == "eef_delta":
            parser.error("--guided-rtc-terminal requires absolute joint or eef_target actions")
        if args.n_action_steps is not None:
            parser.error(
                "--guided-rtc-terminal always executes the complete native action chunk; "
                "leave --n-action-steps unset"
            )
    if not args.policy_server_endpoint and not args.policy_path:
        parser.error("--policy-path is required when --policy-server-endpoint is not used")
    if args.policy_decision_trace is not None and not args.policy_server_endpoint:
        parser.error("--policy-decision-trace requires --policy-server-endpoint")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy_client: PersistentPolicyClient | None = None
    if args.policy_server_endpoint:
        policy = preprocessor = postprocessor = device = None
        policy_client = PersistentPolicyClient(
            args.policy_server_endpoint, timeout_s=args.policy_server_timeout
        )
        policy_server_info = policy_client.ping()
        if args.guided_rtc_terminal and policy_server_info.get("inference_chunking") != "guided_rtc":
            raise RuntimeError(
                "--guided-rtc-terminal requires a model server started with "
                "--inference-chunking guided_rtc"
            )
        if args.guided_rtc_terminal:
            LOG.info("guided RTC initial reset=%s", policy_client.reset())
        LOG.info("policy_server=%s", policy_server_info)
    else:
        policy, preprocessor, postprocessor = _load_policy(args)
        from lerobot.utils.control_utils import predict_action

        device = torch.device(policy.config.device)
    cameras = tuple(args.cameras)
    period = 1.0 / max(1.0, args.fps)
    pink_tick_scheduler = (
        CumulativeControlTickScheduler(policy_hz=args.fps)
        if args.action_space in ("eef_delta", "eef_target")
        and args.eef_ik_backend == "pink"
        else None
    )
    sent = 0
    computed = 0
    remote_action_queue: deque[np.ndarray] = deque()
    previous_eef_chunk_end: np.ndarray | None = None

    def reset_policy() -> None:
        nonlocal previous_eef_chunk_end
        if policy_client is not None:
            remote_action_queue.clear()
            previous_eef_chunk_end = None
            if args.guided_rtc_terminal:
                LOG.info("guided RTC reset=%s", policy_client.reset())
        else:
            policy.reset()

    def compute_action(observation) -> np.ndarray:
        nonlocal previous_eef_chunk_end
        if policy_client is not None:
            if not remote_action_queue:
                policy_state = (
                    robot.model_eef_state(observation=observation)
                    if args.action_space in ("eef_delta", "eef_target")
                    else observation.puppet_state
                )
                prediction = policy_client.predict_chunk_details(
                    state=policy_state,
                    images=observation.image_bytes,
                    cameras=cameras,
                    task=args.task,
                    action_space=args.action_space,
                )
                model_native_chunk = prediction["model_native_action_chunk"]
                model_absolute_chunk = prediction["action_chunk"]
                action_chunk = model_absolute_chunk.copy()
                if args.eef_cartesian_smooth and args.action_space == "eef_target":
                    action_chunk, edge_steps = blend_eef_chunk_edge(
                        previous_eef_chunk_end,
                        action_chunk,
                        policy_hz=args.fps,
                        max_steps=args.eef_chunk_edge_max_steps,
                    )
                    if edge_steps:
                        LOG.info("EEF V5 chunk edge handoff frames=%d", edge_steps)
                    previous_eef_chunk_end = action_chunk[-1].copy()
                actions_to_consume = 1 if args.action_space == "eef_delta" else resolve_action_steps(
                    args.n_action_steps, action_chunk.shape[0]
                )
                if args.guided_rtc_terminal and actions_to_consume != action_chunk.shape[0]:
                    raise RuntimeError(
                        "Guided RTC terminal continuity must consume the complete returned action chunk"
                    )
                if args.n_action_steps is not None and args.n_action_steps > action_chunk.shape[0]:
                    LOG.warning(
                        "n_action_steps=%d exceeds returned chunk size=%d; consuming all %d actions",
                        args.n_action_steps,
                        action_chunk.shape[0],
                        actions_to_consume,
                    )
                _append_policy_decision_trace(
                    args.policy_decision_trace,
                    observation=observation,
                    model_state=policy_state,
                    model_native_action_chunk=model_native_chunk,
                    model_absolute_action_chunk=model_absolute_chunk,
                    rollout_action_chunk=action_chunk,
                    model_output_semantics=prediction["model_output_semantics"],
                    action_space=args.action_space,
                    consumed_actions=actions_to_consume,
                    latency_ms=prediction["latency_ms"],
                    rtc_prefix_steps=prediction["rtc_prefix_steps"],
                )
                remote_action_queue.extend(action_chunk[:actions_to_consume])
                LOG.info(
                    "received action chunk size=%d consuming=%d%s",
                    action_chunk.shape[0],
                    actions_to_consume,
                    (
                        " guided_rtc_terminal=first"
                        if args.guided_rtc_terminal and computed == 0
                        else " guided_rtc_terminal=1"
                        if args.guided_rtc_terminal
                        else ""
                    ),
                )
            return remote_action_queue.popleft()
        action = predict_action(
            observation=_policy_observation(
                observation,
                cameras,
                action_space=args.action_space,
                eef_state=(
                    robot.model_eef_state(observation=observation)
                    if args.action_space in ("eef_delta", "eef_target")
                    else None
                ),
            ),
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(policy.config, "use_amp", False)),
            task=args.task,
            robot_type="cobot_magic_piper",
        )
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    try:
        with CobotBridgeClient(args.endpoint, timeout_s=5.0) as client:
            robot = CobotMagicRemoteRobot(
                client,
                side=args.side,
                images=cameras,
                model_eef_frame=args.model_eef_frame,
                eef_ik_backend=args.eef_ik_backend,
                pink_urdf_path=args.pink_urdf,
                eef_cartesian_smooth=args.eef_cartesian_smooth,
            )
            control_started = False
            teach_seen = False
            try:
                if args.send_actions:
                    LOG.info("mode_result=%s", robot.request_mode(args.mode))
                    LOG.info("output_result=%s", robot.set_output(True))
                    control_started = True
                    auxiliary_holds: dict[str, np.ndarray] = {}
                    if args.startup_init_hold:
                        auxiliary_holds = _hold_startup_init(
                            robot, sync_sides=tuple(args.sync_sides)
                        )
                        reset_policy()
                deadline = time.monotonic() + args.duration
                while time.monotonic() < deadline:
                    started = time.perf_counter()
                    observation = robot.observe()
                    if observation.teach_active:
                        if not teach_seen:
                            LOG.warning("Teach intervention detected; policy queue cleared")
                            reset_policy()
                        teach_seen = True
                        time.sleep(max(0.0, period - (time.perf_counter() - started)))
                        continue
                    if teach_seen:
                        LOG.warning("Teach released; terminating policy episode without resync")
                        break
                    if observation.effective_mode == RESYNC_REQUIRED:
                        reset_policy()
                        LOG.warning("RESYNC_REQUIRED; terminating policy episode without resync")
                        break
                    if not observation.ready:
                        LOG.warning("observation not ready: %s", observation.raw.get("missing_or_stale"))
                        time.sleep(max(0.0, period - (time.perf_counter() - started)))
                        continue

                    values = compute_action(observation)
                    if values.shape != (7,) or not np.isfinite(values).all():
                        raise RuntimeError(f"policy returned invalid action {values}")
                    computed += 1
                    if args.send_actions:
                        for side, target in auxiliary_holds.items():
                            hold_result = robot.send_action_for_side(
                                side, target, observation=observation
                            )
                            if not hold_result.get("accepted", False):
                                LOG.warning("%s hold rejected: %s", side, hold_result)
                        if args.action_space == "eef_delta":
                            model_state = robot.model_eef_state(observation=observation)
                            target = eef_target_from_delta(
                                model_state, values
                            )
                            result = robot.send_model_eef_target(
                                target,
                                observation=observation,
                                pink_ticks=(
                                    pink_tick_scheduler.next()
                                    if pink_tick_scheduler is not None
                                    else None
                                ),
                            )
                        elif args.action_space == "eef_target":
                            # Absolute actions are in the checkpoint's model
                            # frame. Convert once to the SDK link6 frame; do
                            # not add the measured pose a second time.
                            result = robot.send_model_eef_target(
                                values,
                                observation=observation,
                                pink_ticks=(
                                    pink_tick_scheduler.next()
                                    if pink_tick_scheduler is not None
                                    else None
                                ),
                            )
                        else:
                            result = robot.send_action(values, observation=observation)
                        if not result.get("accepted", False):
                            LOG.warning("action rejected: %s", result)
                        else:
                            sent += 1
                    elif computed == 1 or computed % 20 == 0:
                        LOG.info("computed dry action %d: %s", computed, values.tolist())
                    time.sleep(max(0.0, period - (time.perf_counter() - started)))
            finally:
                if control_started:
                    LOG.info("disable_output=%s", robot.set_output(False))
                    LOG.info("safe_mode=%s", robot.request_mode(0))
    finally:
        if policy_client is not None:
            policy_client.close()
    LOG.info("runner complete: computed=%d sent=%d", computed, sent)


if __name__ == "__main__":
    main()
