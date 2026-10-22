"""Cobot Magic RLT Stage 2 online episode runner.

The runner pairs a 30 Hz raw recorder with per-inference RLT decision logs.
Mode 1 is autonomous and never records Teach as intervention. Mode 2 records
Teach-controlled master targets as human action chunks. Real motion requires
both ``--send-actions`` and ``--confirm-motion``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .client import CobotBridgeClient
from .rlt_eef_gate import EEFWorkspace, PiperGraspedEEFInsertionGate
from .record_raw import read_result_outcome
from .remote_robot import RESYNC_REQUIRED, TEACH, CobotMagicRemoteRobot, ControlNotReady
from .rlt_policy_client import PersistentRLTPolicyClient
from .rlt_policy_server import normalize_action_chunk, unnormalize_action_chunk
from .rlt_stage2_data import (
    DEFAULT_ACTOR_EXPLORATION_STD,
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
    repair_mode2_human_action_chunks,
)
from .run_lerobot_policy import _hold_startup_init
from .run_lerobot_policy_edge_v4 import adaptive_edge_blend_steps, edge_blend_action
from .synchronized_chunk_replanner import (
    DEFAULT_POLICY_ACCELERATION_LIMITS,
    DEFAULT_POLICY_VELOCITY_LIMITS,
    PolicyRateCommandLimiter,
)


LOG = logging.getLogger("cobot_bridge.rlt_online")
STOP_REQUESTED = False

_CONTROL_TRANSITION_REASONS = {
    "teach_active",
    "resync_required",
    "owner_safe",
    "output_disabled",
    "stale_epoch",
}

RLT_SMOOTH_PROFILES = ("edge_v4", "raw", "smooth_v3", "smooth_v2")
RLT_EEF_PINK_TICKS = 7  # 200 Hz bridge / 30 Hz policy clock, rounded upward.
# The fixture insertion workspace. The grasp gate uses Piper's physical
# convention (near-zero=open, .055-.08=closed), then requires two EEF hits in
# this region before Actor takeover.
DEFAULT_ACTOR_EEF_MIN = (0.34, 0.05, 0.08)
DEFAULT_ACTOR_EEF_MAX = (0.42, 0.11, 0.18)


def _stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _requested_outcome(result_file: Path) -> str | None:
    """Return an atomically submitted terminal label, if one exists."""

    return read_result_outcome(result_file) if result_file.exists() else None


def _is_mode2_intervention_boundary(
    workflow_mode: int,
    observation,
    *,
    rejection_reason: str | None = None,
) -> bool:
    """Recognize Teach taking ownership while a policy command is in flight.

    Inference and bridge requests are asynchronous with the physical Teach
    button.  A Mode 2 observation can therefore be policy-owned at inference
    start and Teach-owned by the time its action is submitted.  That is a
    normal intervention boundary, not a failed episode.
    """

    if workflow_mode != 2:
        return False
    return bool(
        rejection_reason in {"teach_active", "resync_required"}
        or observation.teach_active
        or observation.effective_mode in (TEACH, RESYNC_REQUIRED)
    )


def _raise_for_rejected_action(label: str, result: dict) -> None:
    """Turn ownership transitions into a typed error for the rollout loop."""

    reason = str(result.get("reason", "action_rejected"))
    if reason in _CONTROL_TRANSITION_REASONS:
        raise ControlNotReady(reason)
    raise RuntimeError(f"{label} rejected: {result}")


class DecisionTakeoverGate:
    """Latch actor takeover after this many reference decisions have executed."""

    def __init__(self, decisions: int) -> None:
        self._needed = max(1, int(decisions))
        self._seen = 0
        self.latched = False
        # Mirrors PiperEEFInsertionGate: the per-decision log line reads this
        # attribute for the eef_xyz field.
        self.last_position: np.ndarray | None = None
        from .piper_eef_metrics import DEFAULT_PIPER_URDF, PiperForwardKinematics

        self._kinematics = PiperForwardKinematics(
            urdf_path=DEFAULT_PIPER_URDF, frames=("gripper_center",)
        )

    def update(self, position: np.ndarray | None) -> bool:
        state = np.asarray(position, dtype=np.float64)
        if state.shape == (7,) and np.isfinite(state).all():
            self.last_position = self._kinematics.forward(
                state[None, :6]
            )["gripper_center"][0][0].copy()
        if self.latched:
            return True
        if self._seen >= self._needed:
            self.latched = True
            return True
        self._seen += 1
        return False


def _actor_control_allowed(
    gate_open: bool,
    actor_decisions_executed: int,
    actor_max_decisions: int,
) -> bool:
    """Apply an optional finite Actor exposure budget after the phase gate."""

    return bool(
        gate_open
        and (
            actor_max_decisions <= 0
            or actor_decisions_executed < actor_max_decisions
        )
    )


def resample_action_chunk(actions: np.ndarray, speed_scale: float) -> np.ndarray:
    """Retimestamp a chunk while preserving its path and both endpoints."""

    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or len(chunk) < 2 or not np.isfinite(chunk).all():
        raise ValueError("actions must be a finite (N, D) array with N >= 2")
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("speed_scale must be in (0, 1]")
    output_steps = max(2, int(round(len(chunk) / float(speed_scale))))
    if output_steps == len(chunk):
        return chunk.copy()
    source_phase = np.linspace(0.0, 1.0, len(chunk), dtype=np.float64)
    output_phase = np.linspace(0.0, 1.0, output_steps, dtype=np.float64)
    output = np.stack(
        [np.interp(output_phase, source_phase, chunk[:, dim]) for dim in range(chunk.shape[1])],
        axis=1,
    )
    return output.astype(np.float32)


def resample_chunk_to_output_hz(
    actions: np.ndarray,
    *,
    source_hz: float,
    output_hz: float,
) -> np.ndarray:
    """Expand a policy-time chunk onto the robot command-time grid.

    The policy remains a 30 Hz producer. This helper only performs the final
    uniform-grid resampling; physical velocity and acceleration constraints
    are applied by ``SynchronizedChunkReplanner.plan_trajectory`` before the
    result is sent to the robot.
    """

    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or len(chunk) < 2 or not np.isfinite(chunk).all():
        raise ValueError("actions must be a finite (N, D) array with N >= 2")
    if not np.isfinite(source_hz) or source_hz <= 0:
        raise ValueError("source_hz must be positive")
    if not np.isfinite(output_hz) or output_hz <= 0:
        raise ValueError("output_hz must be positive")

    duration = (len(chunk) - 1) / float(source_hz)
    output_steps = max(1, int(np.ceil(duration * float(output_hz) - 1e-12)))
    # Keep a uniform output grid. If the rates are not integer multiples, the
    # final sample is held at the last policy waypoint for the residual tick.
    output_times = np.arange(output_steps + 1, dtype=np.float64) / float(output_hz)
    source_times = np.arange(len(chunk), dtype=np.float64) / float(source_hz)
    output = np.stack(
        [
            np.interp(output_times, source_times, chunk[:, dim])
            for dim in range(chunk.shape[1])
        ],
        axis=1,
    )
    return output.astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--policy-server-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--policy-server-timeout", type=float, default=30.0)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--workflow-mode", type=int, choices=(1, 2), required=True)
    parser.add_argument("--rollout-phase", choices=("actor", "reference"), required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("front", "left", "right"),
        default=("front", "right"),
    )
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--raw-fps", type=int, default=30)
    parser.add_argument("--chunk-hz", type=float, default=30.0)
    parser.add_argument("--rollout-speed", type=float, default=1.0)
    parser.add_argument(
        "--smooth-profile",
        choices=RLT_SMOOTH_PROFILES,
        default="edge_v4",
        help=(
            "RLT physical executor: edge_v4 keeps full chunks with adaptive "
            "edge blending and joint velocity/acceleration limits; raw sends "
            "each 30 Hz target unchanged (the robot bridge still interpolates "
            "the pair at 200 Hz); smooth_v3 and smooth_v2 are legacy options"
        ),
    )
    parser.add_argument(
        "--replan-actions", action=argparse.BooleanOptionalAction, default=False,
        help="deprecated compatibility option; RLT uses --smooth-profile",
    )
    parser.add_argument(
        "--replan-max-time-scale",
        type=float,
        default=60.0,
        help="deprecated compatibility option; ignored by the selected executor",
    )
    parser.add_argument(
        "--planner-output-hz",
        type=float,
        default=200.0,
        help="deprecated compatibility option; the bridge always interpolates at 200 Hz",
    )
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    parser.add_argument(
        "--startup-init-hold",
        action="store_true",
        help=(
            "optional init-pose hold; omitted rollouts start at the current "
            "measured pose"
        ),
    )
    parser.add_argument("--enable-actor", action="store_true")
    parser.add_argument(
        "--actor-eef-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--actor-eef-min", nargs=3, type=float, default=DEFAULT_ACTOR_EEF_MIN,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--actor-eef-max", nargs=3, type=float, default=DEFAULT_ACTOR_EEF_MAX,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--actor-eef-confirmations", type=int, default=2)
    parser.add_argument(
        "--actor-takeover-decisions",
        type=int,
        default=0,
        help=(
            "Optional fixed decision-count takeover. 0 (default) uses the "
            "calibrated grasped-fixture EEF phase gate."
        ),
    )
    parser.add_argument(
        "--actor-max-decisions",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help=(
            "Maximum number of Actor chunks physically executed after takeover. "
            "1-3 enables bounded online calibration; 0 keeps unlimited takeover."
        ),
    )
    parser.add_argument(
        "--actor-exploration-std", type=float, default=DEFAULT_ACTOR_EXPLORATION_STD
    )
    parser.add_argument(
        "--q-select",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Q-guided candidate selection: 0 = legacy single exploration draw; "
            "1 = clean actor output only; >=2 = K candidates (1 clean + K-1 "
            "noise draws), execute twin-Q argmax"
        ),
    )
    parser.add_argument("--stage2-output-dir", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--archive-repo-id")
    parser.add_argument("--trainer-device", default="cuda:1")
    parser.add_argument("--pipeline-status-file", type=Path)
    parser.add_argument("--pipeline-log", type=Path)
    parser.add_argument("--recap-checkpoint", type=Path)
    parser.add_argument("--recap-device", default="cuda:1")
    parser.add_argument("--actor-validation-cache", type=Path)
    parser.add_argument("--il-anchor-weight", type=float, default=30.0)
    parser.add_argument("--q-weight", type=float, default=0.01)
    parser.add_argument("--bc-weight", type=float, default=5.0)
    parser.add_argument(
        "--intervention-bc-weight",
        type=float,
        default=1.0,
        help="Extra BC multiplier on is_intervention=1 transitions (human corrections).",
    )
    parser.add_argument(
        "--actor-output-semantics",
        choices=("residual", "direct"),
        default="residual",
        help="Actor output semantics: residual (delta + reference) or direct "
        "(independent full action). Must match the selected actor lineage.",
    )
    parser.add_argument(
        "--ref-anchor-weight",
        type=float,
        default=0.5,
        help="Weight of the reference-anchor (actor -> VLA reference) loss term.",
    )
    parser.add_argument(
        "--ref-anchor-mask",
        choices=("none", "outside_box"),
        default="none",
        help="Apply the reference anchor only on transitions whose physical EEF "
        "state is outside the gate box (requires --action-stats).",
    )
    parser.add_argument(
        "--action-stats",
        type=Path,
        default=None,
        help="Dataset meta/stats.json with 7-dim 'action' mean/std (for the box mask).",
    )
    return parser


def _start_raw_recorder(args) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "cobot_bridge.record_raw",
        "--endpoint",
        args.endpoint,
        "--output",
        str(args.raw_output),
        "--stop-file",
        str(args.result_file),
        "--ready-file",
        str(args.recorder_ready_file),
        "--task",
        args.task,
        "--side",
        args.side,
        "--cameras",
        *args.cameras,
        "--fps",
        str(args.raw_fps),
        "--max-seconds",
        str(args.max_seconds),
        "--policy-id",
        str(args.stage1_checkpoint),
        "--collection-type",
        "RL",
    ]
    return subprocess.Popen(command, start_new_session=False)


def _wait_for_raw_recorder(process: subprocess.Popen, ready_file: Path, *, timeout_s: float = 15.0) -> dict:
    """Wait until the recorder has persisted its first valid observation."""

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if ready_file.is_file():
            try:
                payload = json.loads(ready_file.read_text())
            except json.JSONDecodeError:
                time.sleep(0.01)
                continue
            if int(payload.get("raw_frame_index", -1)) < 0:
                raise RuntimeError(f"invalid raw recorder ready marker: {payload}")
            return payload
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"raw recorder exited before ready with return code {returncode}")
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for raw recorder marker {ready_file}")


def _wait_recorder(process: subprocess.Popen, timeout: float = 20.0) -> None:
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        returncode = process.wait(timeout=5.0)
    if returncode != 0:
        raise RuntimeError(f"raw recorder failed with return code {returncode}")


def _launch_post_episode(args) -> int | None:
    required = (
        args.stage2_output_dir,
        args.archive_root,
        args.archive_repo_id,
        args.pipeline_status_file,
        args.pipeline_log,
        args.recap_checkpoint,
    )
    if all(value is None for value in required):
        return None
    if any(value is None for value in required):
        raise ValueError("all post-episode pipeline paths must be provided together")
    # The RECAP-era auto-training pipeline is built around actor-BC lineage
    # (it requires the IL validation cache). Actors trained with the
    # actor-BC-free remote-aligned recipe carry no cache by design; for them
    # automatic per-episode updates are disabled and retraining happens in
    # manual batches. Skip cleanly instead of raising so the episode still
    # finalizes with exit code 0.
    if args.actor_validation_cache is None:
        LOG.info(
            "post-episode auto-training skipped: actor has no actor-BC "
            "validation cache (manual batch retraining mode)"
        )
        return None
    command = [
        sys.executable,
        "-m",
        "cobot_bridge.rlt_stage2_pipeline",
        "--stage1-checkpoint",
        str(args.stage1_checkpoint),
        "--replay-root",
        str(args.replay_output.parent),
        "--stage2-output-dir",
        str(args.stage2_output_dir),
        "--trainer-device",
        args.trainer_device,
        "--actor-exploration-std",
        str(args.actor_exploration_std),
        "--rollout-phase",
        args.rollout_phase,
        "--raw",
        str(args.raw_output),
        "--decision-output",
        str(args.decision_output),
        "--replay-output",
        str(args.replay_output),
        "--task",
        args.task,
        "--archive-root",
        str(args.archive_root),
        "--archive-repo-id",
        args.archive_repo_id,
        "--fps",
        str(args.raw_fps),
        "--status-file",
        str(args.pipeline_status_file),
        "--recap-checkpoint",
        str(args.recap_checkpoint),
        "--recap-device",
        args.recap_device,
        "--actor-validation-cache",
        str(args.actor_validation_cache),
        "--il-anchor-weight",
        str(args.il_anchor_weight),
        "--q-weight",
        str(args.q_weight),
        "--bc-weight",
        str(args.bc_weight),
        "--intervention-bc-weight",
        str(args.intervention_bc_weight),
        "--actor-output-semantics",
        args.actor_output_semantics,
        "--ref-anchor-weight",
        str(args.ref_anchor_weight),
        "--ref-anchor-mask",
        args.ref_anchor_mask,
    ]
    if args.action_stats is not None:
        command.extend(["--action-stats", str(args.action_stats)])
    args.pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    with args.pipeline_log.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    LOG.info("detached post-episode pipeline pid=%d status=%s", process.pid, args.pipeline_status_file)
    return process.pid


ABORTED_BIN = "_discarded"
ABORTED_SUFFIX = ".aborted"


def _quarantine_aborted_episode(args) -> None:
    """Move every artifact of an aborted episode into ``<batch>/_discarded/<stem>/``.

    The raw recorder and the phase sidecar are written while the rollout runs, so
    an episode discarded before its first action chunk still leaves files behind.
    Unlinking them threw away data that is usually fully recoverable, and left a
    lone ``.phases.jsonl`` in the batch directory.  Every moved file gains an
    ``.aborted`` suffix, which is what keeps the bin invisible to the
    ``*.raw.h5`` / ``rglob("*.h5")`` scans that would otherwise offer a discarded
    episode for rebuilding.
    """

    stem = args.raw_output.name.removesuffix(".raw.h5")
    bin_dir = args.raw_output.parent / ABORTED_BIN / (stem or args.raw_output.stem)
    artifacts = [
        args.raw_output,
        args.raw_output.with_name(args.raw_output.name + ".partial"),
        args.recorder_ready_file,
        Path(f"{args.raw_output}.phases.jsonl"),
        args.decision_output,
        args.decision_output.with_name(args.decision_output.name + ".q_select.json"),
        args.replay_output,
        args.replay_output.with_suffix(args.replay_output.suffix + ".json"),
        args.result_file,
    ]
    moved: dict[str, int] = {}
    quarantine = True
    try:
        bin_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        LOG.exception("cannot create %s; deleting aborted episode files instead", bin_dir)
        quarantine = False
    for source in artifacts:
        if not source.is_file():
            continue
        if quarantine:
            target = bin_dir / (source.name + ABORTED_SUFFIX)
            try:
                os.replace(source, target)
                moved[target.name] = target.stat().st_size
                continue
            except OSError:
                LOG.exception("failed to quarantine %s; deleting", source)
        source.unlink(missing_ok=True)
    if moved:
        (bin_dir / "abort_manifest.json").write_text(
            json.dumps(
                {"episode": stem, "aborted_at_ns": time.time_ns(), "files": moved},
                indent=1,
                sort_keys=True,
            )
        )
    LOG.info(
        "episode aborted; %d file(s) %s (%.1f MiB, restore by dropping the %s suffix)",
        len(moved),
        f"moved to {bin_dir}" if moved else "deleted",
        sum(moved.values()) / 1e6,
        ABORTED_SUFFIX,
    )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.send_actions and not args.confirm_motion:
        raise RuntimeError("real motion requires --confirm-motion")
    if args.rollout_phase == "actor" and not args.enable_actor:
        raise RuntimeError("actor rollout requires --enable-actor")
    if args.rollout_phase == "reference":
        if args.enable_actor:
            raise RuntimeError("reference-only rollout must not enable the Actor")
        if args.workflow_mode != 1:
            raise RuntimeError("reference-only rollout requires workflow Mode 1")
        if args.actor_exploration_std != 0:
            raise RuntimeError("reference-only rollout requires zero exploration")
    if args.raw_fps <= 0 or args.chunk_hz <= 0 or args.max_seconds <= 0:
        raise ValueError("rates and max-seconds must be positive")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    args.recorder_ready_file = args.raw_output.with_name(args.raw_output.name + ".ready")
    for path in (
        args.raw_output,
        args.decision_output,
        args.replay_output,
        args.result_file,
        args.recorder_ready_file,
    ):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    decisions: RLTDecisionWriter | None = None
    recorder: subprocess.Popen | None = None
    control_started = False
    server_episode_started = False
    auxiliary_holds: dict[str, np.ndarray] = {}
    outcome = "aborted"
    action_space = "joint"
    q_select_records: list[dict] = []
    cameras = tuple(args.cameras)
    try:
        with (
            PersistentRLTPolicyClient(
                args.policy_server_endpoint, timeout_s=args.policy_server_timeout
            ) as rlt_client,
            CobotBridgeClient(args.endpoint, timeout_s=5.0) as client,
        ):
            server_status = rlt_client.ping()
            expected_stage1 = str(args.stage1_checkpoint.expanduser().resolve())
            if str(server_status.get("stage1_checkpoint")) != expected_stage1:
                raise RuntimeError(
                    "RLT server Stage 1 mismatch: "
                    f"loaded={server_status.get('stage1_checkpoint')} requested={expected_stage1}"
                )
            # RLT relative-EEF checkpoints always execute through local Pink
            # IK.  The resulting joint target is subsequently shaped by Edge
            # V4 at 30 Hz; the bridge merely interpolates that legal pair at
            # 200 Hz.
            robot = CobotMagicRemoteRobot(
                client,
                side=args.side,
                images=cameras,
                eef_ik_backend="pink",
                eef_cartesian_smooth=False,
            )
            if args.send_actions:
                mode_result = robot.request_mode(args.workflow_mode)
                output_result = robot.set_output(True)
                if not mode_result.get("accepted", False) or not output_result.get("accepted", False):
                    raise RuntimeError(f"control start rejected: mode={mode_result} output={output_result}")
                control_started = True
                # New AGX bridges answer SET_MODE(2) with the intermediate
                # ALIGN_SYNC(5) state ("waiting_sync_first_target") instead of
                # going straight to RLT_SYNC.  The state expects one first
                # sync/alignment target per side before it hands control over;
                # the send guards below legitimately reject every other mode,
                # so without this handshake the first action chunk dies with
                # policy_not_owner.  Mirror move_init_pose: wait for
                # ALIGN_SYNC+output, send the current puppet poses as
                # zero-motion alignment targets, then wait for the real mode.
                align_obs = None
                align_ctrl: dict = {}
                for _ in range(40):
                    time.sleep(0.05)
                    align_obs, _ = client.get_observation(
                        sides=("left", "right"), images=()
                    )
                    align_ctrl = align_obs.get("control", {})
                    if align_ctrl.get("teach_active", False):
                        raise RuntimeError(
                            "startup alignment interrupted by Teach"
                        )
                    if (
                        align_ctrl.get("effective_mode_id") == 5
                        and align_ctrl.get("output_enabled", False)
                    ):
                        break
                else:
                    align_obs = None
                if align_obs is not None:
                    align_epoch = int(align_ctrl.get("epoch", -1))
                    if align_epoch < 0:
                        raise RuntimeError(
                            f"startup alignment epoch missing: {align_ctrl}"
                        )
                    LOG.info(
                        "bridge entered ALIGN_SYNC (epoch %d); sending "
                        "zero-motion first sync targets from current poses",
                        align_epoch,
                    )
                    for side in ("left", "right"):
                        puppet_pos = np.asarray(
                            align_obs["puppet"][side]["position"],
                            dtype=np.float64,
                        )
                        align_result = client.set_action(
                            side,
                            puppet_pos.tolist(),
                            epoch=align_epoch,
                            sequence=1,
                            alignment_speed=0.3,
                        )
                        if not align_result.get("accepted", False):
                            raise RuntimeError(
                                f"{side} startup alignment target rejected: "
                                f"{align_result}"
                            )
                    alignment_deadline = time.monotonic() + 10.0
                    while time.monotonic() < alignment_deadline:
                        time.sleep(0.05)
                        align_obs, _ = client.get_observation(
                            sides=("left", "right"), images=()
                        )
                        align_ctrl = align_obs.get("control", {})
                        if align_ctrl.get("teach_active", False):
                            raise RuntimeError(
                                "startup alignment interrupted by Teach"
                            )
                        align_mode = int(
                            align_ctrl.get("effective_mode_id", -1)
                        )
                        if align_mode == int(args.workflow_mode):
                            LOG.info(
                                "startup alignment complete: mode=%d (%.2fs)",
                                align_mode,
                                10.0
                                - (alignment_deadline - time.monotonic()),
                            )
                            break
                        if align_mode not in (5,):
                            raise RuntimeError(
                                "startup alignment entered unexpected mode: "
                                f"{align_ctrl}"
                            )
                    else:
                        raise RuntimeError(
                            "startup alignment timed out before RLT_SYNC: "
                            f"{align_ctrl}"
                        )
                    # The AGX control node seeds RLT_SYNC with an internal
                    # "align_hold" joint action that fail-safes back to SAFE
                    # (post_align_action_timeout = 1.0 s) unless a fresh
                    # policy-source joint action replaces it.  EEF chunks do
                    # not refresh that cache, so send one zero-motion joint
                    # hold per side from the just-aligned poses.
                    fuse_obs = robot.observe()
                    for fuse_side in ("left", "right"):
                        fuse_target = (
                            fuse_obs.puppet_state
                            if fuse_side == robot.side
                            else fuse_obs.position("puppet", fuse_side)
                        )
                        try:
                            fuse_result = robot.send_action_for_side(
                                fuse_side,
                                np.asarray(fuse_target, dtype=float).tolist(),
                                observation=fuse_obs,
                            )
                        except ControlNotReady as exc:
                            LOG.warning(
                                "post-alignment hold on %s failed: %s",
                                fuse_side,
                                exc,
                            )
                        else:
                            LOG.info(
                                "post-alignment zero-motion hold sent on %s "
                                "(accepted=%s) to clear the AGX align_hold "
                                "failsafe fuse",
                                fuse_side,
                                fuse_result.get("accepted", False),
                            )
                if args.startup_init_hold:
                    auxiliary_holds = _hold_startup_init(robot)
            episode_status = rlt_client.begin_episode(
                workflow_mode=args.workflow_mode,
                allow_actor=args.enable_actor,
                actor_exploration_std=args.actor_exploration_std,
                q_select_k=args.q_select,
            )
            server_episode_started = True
            dimension_values = dict(episode_status.get("dimensions") or {})
            dimensions = RLTDimensions(
                int(dimension_values.get("z_dim", 0)),
                int(dimension_values.get("state_dim", 0)),
                int(dimension_values.get("action_steps", 0)),
                int(dimension_values.get("action_dim", 0)),
            )
            action_space = str(episode_status.get("action_space", "joint"))
            if action_space not in ("eef_target", "joint"):
                raise RuntimeError(
                    f"unsupported RLT action space; server reported action_space={action_space!r}"
                )
            if args.smooth_profile not in ("edge_v4", "raw"):
                raise ValueError(
                    "RLT online rollout supports edge_v4 or raw only; "
                    "relative-EEF Stage 1 adds Pink IK before robot bridge interpolation"
                )
            if not 0.0 < args.rollout_speed <= 1.0:
                raise ValueError("--rollout-speed must be in (0, 1]")
            if not 0.0 < args.planner_output_hz <= 200.0:
                raise ValueError("--planner-output-hz must be in (0, 200]")
            sent_action_steps = max(
                2, int(round(dimensions.action_steps / float(args.rollout_speed)))
            )
            actor_ready = bool(episode_status.get("actor_enabled", False))
            actor_decisions_executed = 0
            # ``raw`` deliberately bypasses PC-side edge blending and v/a
            # limiting. It is useful to separate policy/chunk behavior from
            # the Edge V4 shaper; the robot bridge continues to interpolate
            # every submitted pair at 200 Hz.
            v4_limiter = None
            if args.smooth_profile == "edge_v4":
                v4_limiter = PolicyRateCommandLimiter(
                    action_hz=args.chunk_hz,
                    velocity_limits=DEFAULT_POLICY_VELOCITY_LIMITS,
                    acceleration_limits=DEFAULT_POLICY_ACCELERATION_LIMITS,
                )
                v4_limiter.reset(robot.observe().puppet_state)
            v4_last_command = robot.observe().puppet_state.astype(np.float64, copy=True)
            actor_gate = None
            if args.rollout_phase == "actor":
                if args.actor_takeover_decisions > 0:
                    actor_gate = DecisionTakeoverGate(args.actor_takeover_decisions)
                    LOG.info(
                        "actor takeover gate: %d complete reference decisions "
                        "before Actor exposure",
                        args.actor_takeover_decisions,
                    )
                elif args.actor_eef_gate:
                    actor_gate = PiperGraspedEEFInsertionGate(
                        workspace=EEFWorkspace(
                            tuple(args.actor_eef_min), tuple(args.actor_eef_max)
                        ),
                        confirmation_samples=args.actor_eef_confirmations,
                    )
            decisions = RLTDecisionWriter(
                args.decision_output,
                dimensions=dimensions,
                stage1_checkpoint=str(args.stage1_checkpoint),
                task=args.task,
                workflow_mode=args.workflow_mode,
                side=args.side,
                collection_phase=args.rollout_phase,
                sent_action_steps=sent_action_steps,
            )
            decisions.file.attrs["rollout_speed_scale"] = float(args.rollout_speed)
            # Allows the offline RECAP reward pass to align decision rows to
            # the authoritative 30 Hz camera archive without filename guesses.
            decisions.file.attrs["raw_episode_path"] = str(args.raw_output.resolve())
            decisions.file.attrs["sent_action_hz"] = float(args.chunk_hz)
            decisions.file.attrs["execution_profile"] = args.smooth_profile
            decisions.file.attrs["action_space"] = action_space
            decisions.file.attrs["eef_ik_backend"] = (
                "pink" if action_space == "eef_target" else "none"
            )
            decisions.file.attrs["planner_output_hz"] = 200.0
            decisions.file.attrs["planner_limits_applied_at_hz"] = (
                float(args.chunk_hz) if args.smooth_profile == "edge_v4" else 0.0
            )
            decisions.file.attrs["actor_eef_gate_enabled"] = bool(actor_gate is not None)
            decisions.file.attrs["actor_takeover_decisions"] = int(
                args.actor_takeover_decisions
            )
            decisions.file.attrs["actor_max_decisions"] = int(args.actor_max_decisions)
            decisions.file.attrs["q_select_k"] = int(args.q_select)
            if actor_gate is not None:
                decisions.file.attrs["actor_gate_kind"] = (
                    "fixed_decision_count"
                    if args.actor_takeover_decisions > 0
                    else "grasped_fixture_eef_v1"
                )
                decisions.file.attrs["actor_eef_gate_min_xyz_m"] = args.actor_eef_min
                decisions.file.attrs["actor_eef_gate_max_xyz_m"] = args.actor_eef_max
                decisions.file.attrs["actor_eef_gate_confirmations"] = int(
                    args.actor_eef_confirmations
                )
            LOG.info(
                "RLT server episode begun actor_enabled=%s generation=%s updates=%s",
                actor_ready,
                episode_status.get("actor_generation"),
                episode_status.get("actor_total_updates"),
            )
            # Begin recording at the current measured pose.  A startup init
            # hold is an explicit opt-in CLI operation and is not part of the
            # normal rollout path; arbitrary valid current poses are allowed.
            # If a caller explicitly requested the hold, keep it out of both
            # raw archives and replay data as before.
            recorder = _start_raw_recorder(args)
            recorder_marker = _wait_for_raw_recorder(recorder, args.recorder_ready_file)
            decisions.file.attrs["raw_recorder_first_frame_index"] = int(
                recorder_marker["raw_frame_index"]
            )
            decisions.file.attrs["raw_recorder_first_frame_time_ns"] = int(
                recorder_marker["pc_timestamp_ns"]
            )
            decisions.file.attrs["raw_recorder_first_bridge_sequence"] = int(
                recorder_marker["bridge_sequence"]
            )
            decisions.file.flush()
            LOG.info("raw recorder ready marker=%s", recorder_marker)
            deadline = time.monotonic() + args.max_seconds
            teach_seen = False
            actor_takeover_seen = False
            while not STOP_REQUESTED and time.monotonic() < deadline:
                if args.result_file.exists():
                    outcome = read_result_outcome(args.result_file)
                    break
                observation = robot.observe()
                if not observation.ready:
                    LOG.warning("observation not ready: %s", observation.raw.get("missing_or_stale"))
                    time.sleep(0.02)
                    continue
                observation_time_ns = time.time_ns()
                if observation.teach_active or observation.effective_mode == TEACH:
                    teach_seen = True
                    if args.workflow_mode == 1:
                        LOG.error("Teach appeared in Mode 1; episode must be aborted")
                        outcome = "discard"
                        break
                elif teach_seen or observation.effective_mode == RESYNC_REQUIRED:
                    LOG.info("Teach intervention ended; waiting for terminal label without policy resume")
                    while not STOP_REQUESTED and not args.result_file.exists():
                        time.sleep(0.05)
                    outcome = (
                        read_result_outcome(args.result_file)
                        if args.result_file.exists()
                        else "discard"
                    )
                    break

                inference = rlt_client.predict(
                    state=robot.model_eef_state(observation=observation),
                    images=observation.image_bytes,
                    cameras=cameras,
                    task=args.task,
                )
                gate_open = (
                    actor_gate.update(observation.puppet_state)
                    if actor_gate
                    else True
                )
                actor_takeover = _actor_control_allowed(
                    gate_open,
                    actor_decisions_executed,
                    args.actor_max_decisions,
                )
                first_actor_takeover = actor_takeover and not actor_takeover_seen

                # A terminal click intentionally returns the bridge to SAFE immediately.
                # Re-check after inference so that this normal stop is not mistaken for a
                # failed command and so an unexecuted chunk is not added to replay.
                requested_outcome = _requested_outcome(args.result_file)
                if requested_outcome is not None:
                    outcome = requested_outcome
                    break

                if observation.teach_active:
                    human_raw = np.repeat(
                        robot.model_eef_state(
                            role="master", observation=observation
                        )[None, :],
                        dimensions.action_steps,
                        axis=0,
                    )
                    executed = rlt_client.normalize_action_chunk(human_raw)
                    raw_chunk = human_raw
                    source = "human"
                else:
                    if actor_takeover:
                        executed = inference["policy_actions"]
                        raw_chunk = inference["raw_policy_chunk"]
                        source = str(inference["action_source"])
                    else:
                        executed = inference["reference_actions"]
                        # A short actor (10 ticks) must not shorten the BC
                        # reference before the phase gate permits takeover.
                        raw_chunk = inference["raw_reference_full_chunk"]
                        source = "reference"
                # Keep the decision/replay stream on the policy clock. The
                # selected PC-side smooth profile runs before two-point chunks
                # are interpolated by the bridge at 200 Hz.
                # ``raw_chunk`` is absolute model-frame EEF for relative-EEF
                # Stage 1 checkpoints, or physical joint targets for joint
                # checkpoints.  Retiming happens in that semantic space, then
                # EEF waypoints are converted to joint targets via Pink IK
                # (joint waypoints pass through); Edge V4 computes the physical
                # transition and the bridge interpolates at 200 Hz.
                action_chunk = resample_action_chunk(raw_chunk, args.rollout_speed)
                joint_targets: list[np.ndarray] = []
                for target in action_chunk:
                    if action_space == "eef_target":
                        joint_target, _ = robot.model_eef_target_to_joint_target(
                            target, observation=observation, pink_ticks=RLT_EEF_PINK_TICKS
                        )
                    else:
                        joint_target = np.asarray(target, dtype=np.float64)
                    joint_targets.append(joint_target)
                decision_chunk = np.stack(joint_targets).astype(np.float32, copy=False)
                edge_start_action = decision_chunk[0].astype(np.float64, copy=True)
                if args.smooth_profile == "edge_v4":
                    # The reference -> actor boundary can be much larger than a
                    # normal chunk boundary. Give this first transition a
                    # longer connection window without changing the actor's
                    # trajectory after the offset has decayed.
                    min_edge_steps = 15 if first_actor_takeover else 3
                    edge_blend_steps, edge_gap = adaptive_edge_blend_steps(
                        v4_last_command,
                        edge_start_action,
                        min_steps=min_edge_steps,
                        max_steps=min(30, len(decision_chunk)),
                        max_joint_step=0.08,
                        action_hz=args.chunk_hz,
                        velocity_limits=DEFAULT_POLICY_VELOCITY_LIMITS,
                        acceleration_limits=DEFAULT_POLICY_ACCELERATION_LIMITS,
                    )
                else:
                    edge_blend_steps, edge_gap = 0, 0.0
                action_chunk_completed = not args.send_actions or source == "human"
                if args.send_actions and source != "human":
                    sent_wire_steps = 0
                    try:
                        for wire_step, raw_target in enumerate(decision_chunk):
                            current = robot.observe()
                            if current.teach_active or current.effective_mode in (
                                TEACH, RESYNC_REQUIRED
                            ):
                                raise ControlNotReady("teach_active")
                            if args.smooth_profile == "edge_v4":
                                requested_target, _ = edge_blend_action(
                                    v4_last_command,
                                    edge_start_action,
                                    raw_target,
                                    step=wire_step,
                                    blend_steps=edge_blend_steps,
                                )
                                # Edge V4's joint v/a gate is the sole PC-side
                                # physical shaper. The bridge interpolates the
                                # already shaped command pair at 200 Hz.
                                assert v4_limiter is not None
                                command_target = v4_limiter.step(requested_target)
                            else:
                                command_target = raw_target.astype(np.float64, copy=False)
                            for side, target in auxiliary_holds.items():
                                hold_result = robot.send_action_for_side(
                                    side, target, observation=current
                                )
                                # A transient control-result feedback hiccup on
                                # an idle auxiliary arm must not discard a good
                                # recording: the hold target is a static pose,
                                # so retry the same command, and a still-timing-
                                # out hold degrades to a warning instead of a
                                # fatal rejection. Rejections for any other
                                # reason (safety, teach, mode) stay fatal, and
                                # the commanded right arm's action chunk keeps
                                # its strict rejection.
                                hold_retries = 0
                                while (
                                    not hold_result.get("accepted", False)
                                    and hold_result.get("reason")
                                    == "control_result_timeout"
                                    and hold_retries < 2
                                ):
                                    hold_retries += 1
                                    LOG.warning(
                                        "%s hold lost control result; retry %d/2",
                                        side,
                                        hold_retries,
                                    )
                                    time.sleep(0.01)
                                    hold_result = robot.send_action_for_side(
                                        side, target, observation=current
                                    )
                                if not hold_result.get("accepted", False):
                                    if (
                                        hold_result.get("reason")
                                        == "control_result_timeout"
                                    ):
                                        LOG.warning(
                                            "%s hold keeps timing out on control "
                                            "result; continuing (static hold pose)",
                                            side,
                                        )
                                    else:
                                        _raise_for_rejected_action(
                                            f"{side} hold", hold_result
                                        )
                            result = robot.send_action_chunk(
                                [v4_last_command, command_target],
                                waypoint_hz=args.chunk_hz,
                                observation=current,
                            )
                            if not result.get("accepted", False):
                                _raise_for_rejected_action("action chunk", result)
                            v4_last_command = command_target.astype(np.float64, copy=True)
                            sent_wire_steps += 1
                            tick_deadline = time.monotonic() + 1.0 / args.chunk_hz
                            while time.monotonic() < tick_deadline and not STOP_REQUESTED:
                                if args.result_file.exists():
                                    break
                                # Clamp to zero: the deadline can lapse while the
                                # result-file stat (external drive latency) is
                                # in flight, which would otherwise make
                                # time.sleep() raise ValueError on a negative
                                # length.
                                time.sleep(
                                    max(0.0, min(0.005, tick_deadline - time.monotonic()))
                                )
                    except ControlNotReady as exc:
                        requested_outcome = _requested_outcome(args.result_file)
                        if requested_outcome is not None:
                            outcome = requested_outcome
                            LOG.info("terminal label received while bridge entered SAFE")
                            break
                        current = robot.observe()
                        if _is_mode2_intervention_boundary(
                            args.workflow_mode,
                            current,
                            rejection_reason=str(exc),
                        ):
                            teach_seen = True
                            LOG.info(
                                "Teach took ownership while policy command was in flight; "
                                "dropping the unexecuted chunk and continuing Mode 2 recording"
                            )
                            continue
                        raise
                    action_chunk_completed = sent_wire_steps == len(decision_chunk)
                if source == "actor" and action_chunk_completed:
                    actor_decisions_executed += 1
                    actor_takeover_seen = True
                # Before the actor phase gate opens, the physical command is
                # a full 30-step BC chunk whereas Stage 2 tensors are 10-step.
                # Do not write a false 10-step "executed" target for that
                # command. The raw recorder remains the authoritative complete
                # trace; decision/replay rows begin at actor takeover.
                record_decision = not (
                    args.rollout_phase == "actor"
                    and not actor_takeover
                    and source == "reference"
                )
                if record_decision:
                    decisions.append(
                        z_rl=inference["z_rl"],
                        state=inference["state"],
                        reference_actions=inference["reference_actions"],
                        policy_actions=inference["policy_actions"],
                        executed_actions=executed,
                        sent_actions_raw=decision_chunk,
                        action_source=source,
                        is_intervention=source == "human",
                        bridge_sequence=int(observation.raw.get("sequence", -1)),
                        time_ns=observation_time_ns,
                    )
                else:
                    LOG.info("reference prefix executed as full native chunk; not added to Stage 2 replay")
                q_sel = inference.get("q_select")
                if isinstance(q_sel, dict):
                    q_select_records.append(
                        {"decision": decisions.count, "source": source, **q_sel}
                    )
                    if q_sel.get("q"):
                        LOG.info(
                            "q_select k=%s chosen=%s q_best=%.4f q_spread=%.4f",
                            q_sel.get("k"),
                            q_sel.get("chosen"),
                            float(max(q_sel["q"])),
                            float(q_sel.get("q_spread", 0.0)),
                        )
                LOG.info(
                    "decision=%d source=%s actor_ready=%s actor_takeover=%s "
                    "actor_budget=%d/%s eef_xyz=%s "
                    "chunk_seconds=%.3f action_sent=%s executor=%s edge_steps=%d edge_gap=%.4f",
                    decisions.count,
                    source,
                    actor_ready,
                    actor_takeover,
                    actor_decisions_executed,
                    (
                        "unlimited"
                        if args.actor_max_decisions <= 0
                        else str(args.actor_max_decisions)
                    ),
                    (
                        None
                        if actor_gate is None or actor_gate.last_position is None
                        else np.round(actor_gate.last_position, 4).tolist()
                    ),
                    len(decision_chunk) / args.chunk_hz,
                    bool(args.send_actions and source != "human"),
                    args.smooth_profile,
                    edge_blend_steps,
                    edge_gap,
                )
            else:
                outcome = "discard"
            if decisions is not None:
                decisions.file.attrs["actor_decisions_executed"] = int(
                    actor_decisions_executed
                )
    except BaseException:
        outcome = "discard"
        raise
    finally:
        if control_started:
            try:
                with CobotBridgeClient(args.endpoint, timeout_s=2.0) as client:
                    client.set_output(False)
                    client.set_mode(0)
            except Exception:
                LOG.exception("failed to return bridge to SAFE")
        if server_episode_started:
            try:
                with PersistentRLTPolicyClient(
                    args.policy_server_endpoint, timeout_s=3.0
                ) as rlt_client:
                    rlt_client.end_episode()
            except Exception:
                LOG.exception("failed to end RLT server episode")
        if decisions is not None and q_select_records:
            try:
                sidecar = decisions.path.with_name(
                    decisions.path.name + ".q_select.json"
                )
                sidecar.write_text(json.dumps(q_select_records, indent=1))
                LOG.info(
                    "Q-selection sidecar written: %s (%d records)",
                    sidecar,
                    len(q_select_records),
                )
            except Exception:
                LOG.exception("failed to write Q-selection sidecar")
        if decisions is not None and (outcome not in ("success", "failure") or STOP_REQUESTED):
            decisions.abort()
            if not args.result_file.exists():
                args.result_file.write_text("discard")
        elif decisions is not None:
            decisions.close(outcome=outcome)
        if recorder is not None:
            _wait_recorder(recorder)

    if outcome in ("success", "failure") and args.rollout_phase == "reference":
        LOG.info(
            "reference-only diagnostic saved outcome=%s raw=%s decisions=%s; "
            "replay/training deliberately skipped",
            outcome,
            args.raw_output,
            args.decision_output,
        )
    elif outcome in ("success", "failure"):
        if args.workflow_mode == 2:
            # The model action space follows the Stage 1 checkpoint:
            # relative-EEF checkpoints need the FK/relative-EEF round-trip
            # from the recorded joint commands; joint checkpoints consume the
            # raw 30 Hz joint commands as-is (already in model space before
            # normalization).
            raw_joint_chunk_to_model_actions = None
            if action_space == "eef_target":
                from .convert_lerobot_joint_to_eef import _joint_to_eef
                from .piper_eef_metrics import PiperForwardKinematics
                from lerobot.policies.smolvla.relative_eef import chunk_to_relative

                kinematics = PiperForwardKinematics()

                def raw_joint_chunk_to_model_actions(chunk: np.ndarray, anchor: np.ndarray) -> np.ndarray:
                    valid_chunk = np.ones(len(chunk), dtype=bool)
                    target_eef = _joint_to_eef(chunk, valid_chunk, kinematics)
                    anchor_eef = _joint_to_eef(
                        np.asarray(anchor, dtype=np.float32).reshape(1, 7),
                        np.ones(1, dtype=bool),
                        kinematics,
                    )[0]
                    return chunk_to_relative(anchor_eef, target_eef)

            with PersistentRLTPolicyClient(
                args.policy_server_endpoint, timeout_s=args.policy_server_timeout
            ) as rlt_client:
                rlt_client.begin_episode(
                    workflow_mode=2,
                    allow_actor=False,
                    actor_exploration_std=0.0,
                )
                try:
                    repair = repair_mode2_human_action_chunks(
                        args.decision_output,
                        args.raw_output,
                        normalize_action_chunk=rlt_client.normalize_action_chunk,
                        raw_action_to_model_chunk=raw_joint_chunk_to_model_actions,
                        action_fps=args.chunk_hz,
                    )
                finally:
                    rlt_client.end_episode()
            LOG.info("Mode 2 human chunks repaired from raw 30 Hz commands: %s", repair)
        pipeline_pid = _launch_post_episode(args)
        if args.stage2_output_dir is None or pipeline_pid is None:
            # Either no pipeline was configured, or it deliberately skipped
            # (cache-free lineage); build the replay bundle inline either way.
            metadata = build_replay_bundle(args.decision_output, args.replay_output)
            LOG.info("replay bundle ready: %s", json.dumps(metadata, sort_keys=True))
    else:
        _quarantine_aborted_episode(args)
        LOG.info("episode aborted; no replay bundle saved")
    if outcome in ("success", "failure"):
        args.recorder_ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
