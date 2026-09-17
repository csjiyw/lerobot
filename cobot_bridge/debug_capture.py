"""Low-priority, always-on debug capture for the Cobot Magic dashboard.

The worker has its own bridge connection and never sends control commands.  It
keeps sampling at a deliberately modest rate, while the dashboard marks the
beginning and end of every motion job.  Only the most recently finalized
session is retained, outside ``raw_datasets``, so it can never be converted or
used for training by accident.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .client import BridgeError, CobotBridgeClient
from .record_raw import RawEpisodeWriter
from .remote_robot import CobotMagicRemoteRobot, RemoteObservation


LOG = logging.getLogger("cobot_bridge.debug_capture")
LATEST_RAW_NAME = "latest.raw.h5"
LATEST_METADATA_NAME = "latest.json"


@dataclass
class DebugSession:
    job_id: str
    motion_kind: str
    task: str
    side: str
    cameras: tuple[str, ...]
    policy_id: str = ""
    policy_decision_trace: str = ""
    started_at_ns: int = field(default_factory=time.time_ns)


@dataclass
class _ActiveCapture:
    session: DebugSession
    partial_path: Path
    writer: RawEpisodeWriter
    process: Any | None = None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _nearest_sequence_rows(source: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest raw-frame rows and signed bridge-sequence differences."""

    source = np.asarray(source, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    rows = np.full(query.shape, -1, dtype=np.int64)
    errors = np.full(query.shape, np.iinfo(np.int64).min, dtype=np.int64)
    if source.size == 0 or not np.all(np.diff(source) >= 0):
        return rows, errors
    upper = np.searchsorted(source, query, side="left")
    lower = np.clip(upper - 1, 0, source.size - 1)
    upper = np.clip(upper, 0, source.size - 1)
    choose_upper = np.abs(source[upper] - query) < np.abs(source[lower] - query)
    rows = np.where(choose_upper, upper, lower).astype(np.int64, copy=False)
    errors = source[rows] - query
    return rows, errors


def _attach_policy_decision_trace(raw: h5py.File, trace_path: Path | None) -> dict[str, Any]:
    """Merge exact rollout decisions into a completed passive debug record."""

    if trace_path is None:
        return {"status": "not_requested", "count": 0}
    if not trace_path.is_file():
        return {"status": "missing", "count": 0, "path": str(trace_path)}
    try:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("format") != "cobot_magic_policy_decisions_v1":
                raise ValueError(f"line {line_number}: unsupported trace format")
            rows.append(value)
        if not rows:
            return {"status": "empty", "count": 0, "path": str(trace_path)}

        required = (
            "time_ns", "bridge_time_ns", "bridge_sequence", "model_state",
            "model_native_action_chunk", "model_absolute_action_chunk",
            "rollout_action_chunk", "model_output_semantics", "action_space",
            "consumed_actions",
        )
        missing = [key for key in required if any(key not in row for row in rows)]
        if missing:
            raise ValueError(f"trace is missing fields: {missing}")
        state = np.asarray([row["model_state"] for row in rows], dtype=np.float32)
        native = np.asarray([row["model_native_action_chunk"] for row in rows], dtype=np.float32)
        absolute = np.asarray([row["model_absolute_action_chunk"] for row in rows], dtype=np.float32)
        rollout = np.asarray([row["rollout_action_chunk"] for row in rows], dtype=np.float32)
        if (
            state.shape != (len(rows), 7)
            or native.ndim != 3
            or native.shape[0] != len(rows)
            or native.shape[2:] != (7,)
            or absolute.shape != native.shape
            or rollout.shape != native.shape
            or not np.isfinite(state).all()
            or not np.isfinite(native).all()
            or not np.isfinite(absolute).all()
            or not np.isfinite(rollout).all()
        ):
            raise ValueError("trace has invalid state or action-chunk shapes")
        semantics = {str(row["model_output_semantics"]) for row in rows}
        action_spaces = {str(row["action_space"]) for row in rows}
        if len(semantics) != 1 or len(action_spaces) != 1:
            raise ValueError("trace changes action semantics or action space within one rollout")
        if "frames" not in raw or "sequence" not in raw["frames"]:
            raise ValueError("debug record is missing frames/sequence for decision alignment")
        if "policy_decisions" in raw:
            del raw["policy_decisions"]
        decisions = raw.create_group("policy_decisions")
        sequence = np.asarray([row["bridge_sequence"] for row in rows], dtype=np.int64)
        raw_rows, sequence_error = _nearest_sequence_rows(
            np.asarray(raw["frames/sequence"][:], dtype=np.int64), sequence
        )
        decisions.attrs.update(
            {
                "format": "cobot_magic_policy_decisions_v1",
                "model_output_semantics": semantics.pop(),
                "action_space": action_spaces.pop(),
                "trace_path": str(trace_path),
            }
        )
        datasets = {
            "time_ns": np.asarray([row["time_ns"] for row in rows], dtype=np.int64),
            "bridge_time_ns": np.asarray([row["bridge_time_ns"] for row in rows], dtype=np.int64),
            "bridge_sequence": sequence,
            "raw_frame_index": raw_rows,
            "raw_sequence_error": sequence_error,
            "model_state": state,
            "model_native_action_chunk": native,
            "model_absolute_action_chunk": absolute,
            "rollout_action_chunk": rollout,
            "consumed_actions": np.asarray([row["consumed_actions"] for row in rows], dtype=np.int32),
            "latency_ms": np.asarray(
                [np.nan if row.get("latency_ms") is None else row["latency_ms"] for row in rows],
                dtype=np.float32,
            ),
            "rtc_prefix_steps": np.asarray(
                [-1 if row.get("rtc_prefix_steps") is None else row["rtc_prefix_steps"] for row in rows],
                dtype=np.int32,
            ),
        }
        for name, values in datasets.items():
            decisions.create_dataset(name, data=values, chunks=True)
        return {
            "status": "attached",
            "count": len(rows),
            "path": str(trace_path),
            "action_steps": int(native.shape[1]),
            "matched_raw_frames": int((raw_rows >= 0).sum()),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        LOG.warning("policy decision trace was not attached: %s", exc)
        return {"status": "invalid", "count": 0, "path": str(trace_path), "error": str(exc)}


def publish_completed_debug_record(
    partial_path: Path,
    debug_root: Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Atomically replace the retained debug record after validating it.

    The old latest record remains intact whenever a session was too short or
    capture failed.  ``partial_path`` is expected to be a fully closed HDF5.
    """

    with h5py.File(partial_path, "r+") as raw:
        if not bool(raw.attrs.get("completed", False)):
            raise RuntimeError("debug capture did not complete")
        frames = int(raw.attrs.get("num_frames", 0))
        if frames < 1:
            raise RuntimeError("debug capture has no frames")
        raw.attrs["debug_record"] = True
        raw.attrs["debug_sampled_fps"] = int(metadata["sampled_fps"])
        raw.attrs["debug_motion_kind"] = str(metadata["motion_kind"])
        raw.attrs["debug_job_id"] = str(metadata["job_id"])
        requested_trace = metadata.get("policy_decision_trace")
        trace_summary = _attach_policy_decision_trace(
            raw, None if not requested_trace else Path(str(requested_trace))
        )
        raw.attrs["policy_decision_trace_status"] = str(trace_summary["status"])
        raw.attrs["policy_decision_count"] = int(trace_summary["count"])
        raw.flush()

    debug_root.mkdir(parents=True, exist_ok=True)
    latest = debug_root / LATEST_RAW_NAME
    os.replace(partial_path, latest)
    _atomic_write_json(
        debug_root / LATEST_METADATA_NAME,
        {
            **metadata,
            "policy_decisions": trace_summary,
            "path": str(latest),
            "frames": frames,
            "published_at_ns": time.time_ns(),
        },
    )
    return latest


class DebugCaptureWorker:
    """One independent 15 Hz observer with one rolling on-disk session."""

    def __init__(
        self,
        *,
        endpoint: str,
        debug_root: Path,
        fps: int = 15,
        cameras: tuple[str, ...] = ("front", "right"),
        side: str = "right",
    ) -> None:
        if fps < 1:
            raise ValueError("debug capture fps must be positive")
        self.endpoint = endpoint
        self.debug_root = Path(debug_root)
        self.fps = int(fps)
        self.default_cameras = tuple(cameras)
        self.default_side = side
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: DebugSession | None = None
        self._active: _ActiveCapture | None = None
        self._processes: dict[str, Any] = {}
        self._last_error = ""
        self._last_publish: dict[str, Any] | None = None
        self._last_observation_ns = 0
        self._last_sequence = -1
        self._observation_count = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="cobot-debug-capture",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)

    def begin(self, session: DebugSession) -> None:
        """Queue a new motion boundary without blocking motion startup."""

        self.start()
        with self._lock:
            if self._pending is not None or self._active is not None:
                raise RuntimeError("a debug capture session is already active")
            self._pending = session

    def attach_process(self, job_id: str, process: Any) -> None:
        with self._lock:
            self._processes[job_id] = process
            if self._pending is not None and self._pending.job_id == job_id:
                return
            elif self._active is not None and self._active.session.job_id == job_id:
                self._active.process = process

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if self._pending is not None and self._pending.job_id == job_id:
                self._pending = None
                self._processes.pop(job_id, None)
                return
            if self._active is not None and self._active.session.job_id == job_id:
                active = self._active
                self._active = None
                self._processes.pop(job_id, None)
            else:
                return
        self._close_active(active, reason="start_failed", publish=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self._active.session if self._active is not None else self._pending
            return {
                "enabled": True,
                "fps": self.fps,
                "cameras": list(self.default_cameras),
                "state": "recording" if active is not None else "idle",
                "job_id": None if active is None else active.job_id,
                "last_error": self._last_error,
                "latest": self._last_publish,
                "last_observation_ns": self._last_observation_ns,
                "last_sequence": self._last_sequence,
                "observations": self._observation_count,
            }

    def _take_pending(self) -> DebugSession | None:
        with self._lock:
            if self._pending is None or self._active is not None:
                return None
            session = self._pending
            self._pending = None
            return session

    def _start_active(self, session: DebugSession) -> None:
        inflight = self.debug_root / ".inflight"
        inflight.mkdir(parents=True, exist_ok=True)
        partial = inflight / f"{session.job_id}_{uuid.uuid4().hex}.raw.h5.partial"
        writer = RawEpisodeWriter(
            partial,
            side=session.side,
            cameras=session.cameras,
            task=session.task,
            fps=self.fps,
            flush_frames=max(1, self.fps),
            policy_id=session.policy_id,
            collection_type="REPLAY",
        )
        with self._lock:
            # ``attach_process`` may have run while the writer was opening.
            process = self._processes.pop(session.job_id, None)
            self._active = _ActiveCapture(
                session=session,
                partial_path=partial,
                writer=writer,
                process=process,
            )
        LOG.info("debug capture session started: job=%s task=%s", session.job_id, session.task)

    def _active_for_observation(self) -> _ActiveCapture | None:
        with self._lock:
            return self._active

    def _end_if_process_finished(self) -> None:
        with self._lock:
            active = self._active
            if active is None or active.process is None or active.process.poll() is None:
                return
            self._active = None
            self._processes.pop(active.session.job_id, None)
        self._close_active(active, reason="motion_process_finished", publish=True)

    def _close_active(self, active: _ActiveCapture, *, reason: str, publish: bool) -> None:
        try:
            active.writer.close(completed=publish, stop_reason=reason, outcome="unknown")
            if not publish:
                active.partial_path.unlink(missing_ok=True)
                return
            metadata = {
                "job_id": active.session.job_id,
                "motion_kind": active.session.motion_kind,
                "task": active.session.task,
                "side": active.session.side,
                "cameras": list(active.session.cameras),
                "policy_id": active.session.policy_id,
                "policy_decision_trace": active.session.policy_decision_trace,
                "sampled_fps": self.fps,
                "started_at_ns": active.session.started_at_ns,
                "stop_reason": reason,
            }
            latest = publish_completed_debug_record(
                active.partial_path, self.debug_root, metadata=metadata
            )
            with self._lock:
                self._last_publish = {**metadata, "path": str(latest), "frames": active.writer.count}
                self._last_error = ""
            LOG.info("debug capture finalized: frames=%d output=%s", active.writer.count, latest)
        except Exception as exc:
            active.partial_path.unlink(missing_ok=True)
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("debug capture finalization failed")

    def _run(self) -> None:
        period = 1.0 / self.fps
        client: CobotBridgeClient | None = None
        robot: CobotMagicRemoteRobot | None = None
        robot_signature: tuple[str, tuple[str, ...]] | None = None
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                session = self._take_pending()
                if session is not None:
                    self._start_active(session)
                active = self._active_for_observation()
                wanted_side = active.session.side if active is not None else self.default_side
                wanted_cameras = active.session.cameras if active is not None else self.default_cameras
                signature = (wanted_side, wanted_cameras)
                try:
                    if client is None or robot is None or signature != robot_signature:
                        if client is not None:
                            client.close()
                        client = CobotBridgeClient(self.endpoint, timeout_s=0.75)
                        robot = CobotMagicRemoteRobot(
                            client, side=wanted_side, images=wanted_cameras
                        )
                        robot_signature = signature
                    observation = robot.observe()
                    if active is not None and observation.ready and all(
                        name in observation.image_bytes for name in active.session.cameras
                    ):
                        active.writer.append(observation)
                    elif active is not None:
                        active.writer.skipped += 1
                    with self._lock:
                        self._last_error = ""
                        self._last_observation_ns = time.time_ns()
                        self._last_sequence = int(observation.raw.get("sequence", -1))
                        self._observation_count += 1
                except (BridgeError, OSError, ValueError) as exc:
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
                    LOG.debug("debug capture observation unavailable: %s", exc)
                    if client is not None:
                        client.close()
                    client = None
                    robot = None
                    robot_signature = None
                self._end_if_process_finished()
                self._stop.wait(max(0.0, period - (time.perf_counter() - started)))
        finally:
            if client is not None:
                client.close()
            with self._lock:
                active = self._active
                self._active = None
                self._pending = None
                self._processes.clear()
            if active is not None:
                # A dashboard shutdown has no trustworthy task-end boundary.
                # Keep the previous complete record rather than replacing it
                # with an interrupted capture.
                self._close_active(active, reason="dashboard_shutdown", publish=False)
