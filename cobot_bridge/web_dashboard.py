"""Local web dashboard for Cobot Magic data collection and policy workflows.

The dashboard exposes only fixed, validated commands. It binds to localhost by
default and never enables real robot motion unless the request contains an
explicit confirmation flag.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .client import CobotBridgeClient, request_robot_safe_stop
from .cobot_mjpeg_cam_server import BOUNDARY as MJPEG_BOUNDARY
from .cobot_mjpeg_cam_server import BridgeJpegPump
from .debug_capture import DebugCaptureWorker, DebugSession
from .dashboard_events import DashboardEvents
from .lerobot_replay import export_episode_to_raw
from .move_init_pose import DEFAULT_JOINT_SPEED, DEFAULT_MIN_SECONDS
from .policy_client import PersistentPolicyClient, PolicyServerError
from .rlt_policy_client import PersistentRLTPolicyClient, RLTPolicyServerError
from .rlt_stage2_data import (
    ACTION_CHUNK_TIMEBASE,
    ACTOR_OUTPUT_SEMANTICS,
    DIRECT_ACTOR_OUTPUT_SEMANTICS,
    STEAM_OFFLINE_ACTOR_STAGE,
    SUPPORTED_ACTOR_CHECKPOINT_STAGES,
    SUPPORTED_ACTOR_OUTPUT_SEMANTICS,
)


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
GROOT_EEF_POLICY_ID = "external/groot_5482b_eef_90000"
DEFAULT_GROOT_EEF_CHECKPOINT = Path(
    "/home/fuxin/models/jyw/Longwu0622/"
    "5482b62ee3ff498a92f139f0788c0544_head_eef_joint/STEP1-eef/checkpoint-90000"
)
DEFAULT_GROOT_BACKBONE = Path("/home/fuxin/models/GR00T/nvidia/Cosmos-Reason2-2B")
MOTION_JOB_KINDS = {
    "replay_execute",
    "policy_execute",
    "rlt_policy_execute",
    "rlt_online_execute",
    "init_pose_execute",
}
RECORD_JOB_KINDS = {"record", "record_with_init"}
STARTUP_JOB_KINDS = {"robot_startup", "robot_self_check", "device_restart"}
CAMERA_NAMES = ("front", "left", "right")
ARM_ROLES = ("master", "puppet")
ARM_SIDES = ("left", "right")


@dataclass(frozen=True)
class DashboardConfig:
    workspace: Path
    raw_root: Path
    dataset_root: Path
    checkpoint_root: Path
    run_root: Path
    python: Path
    lerobot_replay_root: Path = Path.home() / "datasets" / "lerobot_v21"
    debug_root: Path = Path("debug_rollouts")
    debug_capture_enabled: bool = True
    rlt_root: Path = Path("../RL-Token-SmolVLA")
    endpoint: str = "tcp://192.168.250.2:5555"
    policy_server_endpoint: str = "tcp://127.0.0.1:5560"
    rlt_policy_server_endpoint: str = "tcp://127.0.0.1:5561"
    robot_policy_server_endpoint: str = "tcp://192.168.250.2:5560"
    robot_policy_server_bind: str = "tcp://0.0.0.0:5560"
    robot_policy_server_local_endpoint: str = "tcp://127.0.0.1:5560"
    robot_ssh_host: str = "agilex@192.168.250.2"
    robot_workspace: Path = Path("/home/agilex/smolvla_runtime")
    robot_control_workspace: Path = Path("/home/agilex/lerobot")
    robot_python: Path = Path("/home/agilex/miniforge3/envs/smolvla_robot/bin/python")
    groot_python: Path = Path("/home/fuxin/anaconda3/envs/groot_n1d7_web/bin/python")
    groot_eef_checkpoint_path: Path | None = None
    groot_backbone_path: Path | None = None
    api_token: str = ""
    pc_preview_fps: int = 15
    robot_sudo_password_file: Path | None = None

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        workspace = Path(__file__).resolve().parents[1]
        project_root = workspace.parent
        return cls(
            workspace=workspace,
            raw_root=Path(os.environ.get("COBOT_RAW_ROOT", project_root / "raw_datasets")),
            dataset_root=Path(os.environ.get("COBOT_DATASET_ROOT", project_root / "datasets")),
            checkpoint_root=Path(
                os.environ.get("COBOT_CHECKPOINT_ROOT", workspace / "outputs")
            ),
            run_root=Path(os.environ.get("COBOT_WEB_RUN_ROOT", project_root / "web_runs")),
            python=Path(os.environ.get("COBOT_PYTHON", sys.executable)),
            lerobot_replay_root=Path(
                os.environ.get(
                    "COBOT_LEROBOT_REPLAY_ROOT",
                    Path.home() / "datasets" / "lerobot_v21",
                )
            ).expanduser(),
            debug_root=Path(
                os.environ.get("COBOT_DEBUG_CAPTURE_ROOT", project_root / "debug_rollouts")
            ),
            debug_capture_enabled=os.environ.get("COBOT_DEBUG_CAPTURE_ENABLED", "1")
            .strip()
            .lower()
            not in ("0", "false", "no", "off"),
            rlt_root=Path(
                os.environ.get("COBOT_RLT_ROOT", project_root / "RL-Token-SmolVLA")
            ),
            endpoint=os.environ.get("COBOT_BRIDGE_ENDPOINT", "tcp://192.168.250.2:5555"),
            policy_server_endpoint=os.environ.get(
                "COBOT_POLICY_SERVER_ENDPOINT", "tcp://127.0.0.1:5560"
            ),
            rlt_policy_server_endpoint=os.environ.get(
                "COBOT_RLT_POLICY_SERVER_ENDPOINT", "tcp://127.0.0.1:5561"
            ),
            robot_policy_server_endpoint=os.environ.get(
                "COBOT_ROBOT_POLICY_SERVER_ENDPOINT", "tcp://192.168.250.2:5560"
            ),
            robot_policy_server_bind=os.environ.get(
                "COBOT_ROBOT_POLICY_SERVER_BIND", "tcp://0.0.0.0:5560"
            ),
            robot_policy_server_local_endpoint=os.environ.get(
                "COBOT_ROBOT_POLICY_SERVER_LOCAL_ENDPOINT", "tcp://127.0.0.1:5560"
            ),
            robot_ssh_host=os.environ.get(
                "COBOT_ROBOT_SSH_HOST", "agilex@192.168.250.2"
            ),
            robot_workspace=Path(
                os.environ.get("COBOT_ROBOT_POLICY_WORKSPACE", "/home/agilex/smolvla_runtime")
            ),
            robot_control_workspace=Path(
                os.environ.get("COBOT_ROBOT_CONTROL_WORKSPACE", "/home/agilex/lerobot")
            ),
            robot_python=Path(
                os.environ.get(
                    "COBOT_ROBOT_POLICY_PYTHON",
                    "/home/agilex/miniforge3/envs/smolvla_robot/bin/python",
                )
            ),
            groot_python=Path(
                os.environ.get(
                    "COBOT_GROOT_PYTHON",
                    "/home/fuxin/anaconda3/envs/groot_n1d7_web/bin/python",
                )
            ),
            groot_eef_checkpoint_path=(
                Path(
                    os.environ.get(
                        "COBOT_GROOT_EEF_CHECKPOINT",
                        str(DEFAULT_GROOT_EEF_CHECKPOINT),
                    )
                ).expanduser()
                if os.environ.get(
                    "COBOT_GROOT_EEF_CHECKPOINT", str(DEFAULT_GROOT_EEF_CHECKPOINT)
                ).strip()
                else None
            ),
            groot_backbone_path=(
                Path(
                    os.environ.get(
                        "COBOT_GROOT_BACKBONE_PATH", str(DEFAULT_GROOT_BACKBONE)
                    )
                ).expanduser()
                if os.environ.get(
                    "COBOT_GROOT_BACKBONE_PATH", str(DEFAULT_GROOT_BACKBONE)
                ).strip()
                else None
            ),
            api_token=os.environ.get("COBOT_WEB_TOKEN", ""),
            pc_preview_fps=max(1, min(60, int(os.environ.get("COBOT_PC_PREVIEW_FPS", "15")))),
            robot_sudo_password_file=(
                Path(os.environ["COBOT_ROBOT_SUDO_PASSWORD_FILE"])
                if os.environ.get("COBOT_ROBOT_SUDO_PASSWORD_FILE")
                else None
            ),
        )


@dataclass
class ManagedJob:
    id: str
    kind: str
    command: list[str]
    log_path: Path
    started_at_ns: int
    process: subprocess.Popen[bytes]
    metadata: dict[str, Any]
    stop_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        returncode = self.process.poll()
        if returncode is None:
            state = "stopping" if self.stop_requested else "running"
        else:
            state = "completed" if returncode == 0 else "failed"
        metadata = dict(self.metadata)
        phase_file = metadata.get("phase_file")
        if phase_file:
            try:
                metadata["phase"] = Path(phase_file).read_text().strip()
            except OSError:
                metadata.setdefault("phase", "queued")
        result_file = metadata.get("result_file")
        if result_file:
            try:
                metadata["result"] = Path(result_file).read_text().strip()
                metadata["result_submitted"] = True
            except OSError:
                metadata["result_submitted"] = False
        pipeline_status_file = metadata.get("pipeline_status_file")
        if pipeline_status_file:
            metadata["pipeline"] = _read_rlt_pipeline_status(Path(pipeline_status_file))
        if metadata.get("startup_report_file"):
            try:
                metadata["startup_report"] = json.loads(Path(metadata["startup_report_file"]).read_text())
            except (OSError, ValueError):
                metadata["startup_report"] = {"phase": "queued"}
        return {
            "id": self.id,
            "kind": self.kind,
            "state": state,
            "pid": self.process.pid,
            "returncode": returncode,
            "started_at_ns": self.started_at_ns,
            "log_path": str(self.log_path),
            "command": self.command,
            "metadata": metadata,
        }


class JobManager:
    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._jobs: dict[str, ManagedJob] = {}
        self._lock = threading.Lock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [job.snapshot() for job in self._jobs.values()]
        return sorted(jobs, key=lambda value: value["started_at_ns"], reverse=True)

    def process(self, job_id: str) -> subprocess.Popen[bytes]:
        """Return the managed process so a passive observer can watch it."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.process

    def start(
        self,
        kind: str,
        command: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            active = [job for job in self._jobs.values() if job.process.poll() is None]
            if kind in STARTUP_JOB_KINDS and any(job.kind in STARTUP_JOB_KINDS for job in active):
                raise RuntimeError("a robot startup/check/recovery job is already running")
            if kind in {"robot_startup", "robot_self_check"} and any(
                job.kind in RECORD_JOB_KINDS or self._has_active_motion(job) for job in active
            ):
                raise RuntimeError("finish recording and motion before robot startup/check")
            if (kind in MOTION_JOB_KINDS or kind in RECORD_JOB_KINDS) and any(
                job.kind in STARTUP_JOB_KINDS for job in active
            ):
                raise RuntimeError("wait for robot startup/check/recovery to finish")
            if kind in RECORD_JOB_KINDS and any(job.kind in RECORD_JOB_KINDS for job in active):
                raise RuntimeError("a recorder job is already running")
            if kind == "policy_server" and any(job.kind == "policy_server" for job in active):
                raise RuntimeError("a policy server job is already running")
            if kind == "rlt_policy_server" and any(
                job.kind == "rlt_policy_server" for job in active
            ):
                raise RuntimeError("an RLT policy server job is already running")
            if kind == "rlt_policy_server" and any(
                job.kind == "policy_server" and job.metadata.get("compute", "pc") == "pc"
                for job in active
            ):
                raise RuntimeError("stop the normal PC model server before RLT server")
            if (
                kind == "policy_server"
                and (metadata or {}).get("compute", "pc") == "pc"
                and any(job.kind == "rlt_policy_server" for job in active)
            ):
                raise RuntimeError("stop the RLT server before normal PC model server")
            if kind == "device_restart" and any(job.kind == "device_restart" for job in active):
                raise RuntimeError("a device restart is already running")
            if kind == "rlt_shadow" and any(job.kind == "rlt_shadow" for job in active):
                raise RuntimeError("an RLT shadow job is already running")
            if kind == "rlt_shadow" and any(
                (
                    job.kind == "policy_server"
                    and job.metadata.get("compute", "pc") == "pc"
                )
                or job.kind == "rlt_policy_server"
                for job in active
            ):
                raise RuntimeError("stop the GPU model server before loading RLT shadow")
            active_motion = any(self._has_active_motion(job) for job in active)
            if (kind in MOTION_JOB_KINDS or kind == "record_with_init") and active_motion:
                raise RuntimeError("a real-motion policy/replay job is already running")

            job_id = uuid.uuid4().hex[:12]
            self.config.run_root.mkdir(parents=True, exist_ok=True)
            log_path = self.config.run_root / f"{job_id}_{kind}.log"
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.config.workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            job = ManagedJob(
                id=job_id,
                kind=kind,
                command=command,
                log_path=log_path,
                started_at_ns=time.time_ns(),
                process=process,
                metadata=dict(metadata or {}),
            )
            self._jobs[job_id] = job
            return job.snapshot()

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.process.poll() is not None:
                return job.snapshot()
            job.stop_requested = True
            process = job.process

        if job.kind in MOTION_JOB_KINDS:
            safety = request_robot_safe_stop(self.config.endpoint, timeout_s=0.5)
            with self._lock:
                job.metadata["stop_safety"] = safety
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        threading.Thread(target=self._finish_stop, args=(process,), daemon=True).start()
        return job.snapshot()

    @staticmethod
    def _finish_stop(process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=8.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def stop_motion_jobs(self) -> None:
        for job in self.list():
            is_initializing_record = (
                job["kind"] == "record_with_init"
                and job.get("metadata", {}).get("phase") in ("queued", "initializing")
            )
            if (
                (job["kind"] in MOTION_JOB_KINDS or is_initializing_record)
                and job["state"] in ("running", "stopping")
            ):
                try:
                    self.stop(job["id"])
                except KeyError:
                    pass

    @staticmethod
    def _has_active_motion(job: ManagedJob) -> bool:
        if job.kind in MOTION_JOB_KINDS:
            return True
        if job.kind != "record_with_init":
            return False
        phase_file = job.metadata.get("phase_file")
        if not phase_file:
            return True
        try:
            return Path(phase_file).read_text().strip() != "recording"
        except OSError:
            return True

    def log(self, job_id: str, tail_bytes: int = 64_000) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            path = job.log_path
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - tail_bytes))
            return stream.read().decode("utf-8", errors="replace")


class DeviceMonitor:
    """Read-only bridge probe plus a small cache for single dashboard frames."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._images: dict[str, bytes] = {}
        self._image_types: dict[str, str] = {}
        self._joint_cache: dict[str, Any] | None = None
        self._joint_cached_at = 0.0
        self._joint_revision = 0

    @staticmethod
    def _media_type(encoding: str) -> str:
        value = encoding.lower()
        if "png" in value:
            return "image/png"
        return "image/jpeg"

    def probe(self, skip_images: set[str] | None = None) -> dict[str, Any]:
        # Live MJPEG is supplied by the dedicated, process-wide preview relay.
        # Do not pull those JPEG bytes again in this overview probe; bridge
        # metadata still reports camera freshness for the device display.
        skip = set(skip_images or ())
        wanted = [name for name in CAMERA_NAMES if name not in skip]
        try:
            with CobotBridgeClient(self.config.endpoint, timeout_s=1.2) as client:
                ping = client.ping()
                raw, images = client.get_observation(
                    sides=ARM_SIDES,
                    images=wanted,
                )
            devices = _device_status(raw, images)
            with self._lock:
                for name, data in images.items():
                    metadata = raw.get("images", {}).get(name) or {}
                    self._images[name] = data
                    self._image_types[name] = self._media_type(str(metadata.get("encoding", "jpeg")))
            control = raw.get("control", {})
            return {
                "online": True,
                "allow_control": bool(ping.get("allow_control", False)),
                "ready": bool(raw.get("ready", False)),
                "mode": control.get("effective_mode"),
                "mode_id": control.get("effective_mode_id"),
                "teach_active": bool(control.get("teach_active", False)),
                "emergency_latched": bool(control.get("emergency_latched", False)),
                "output_enabled": bool(control.get("output_enabled", False)),
                "epoch": control.get("epoch"),
                "devices": devices,
            }
        except Exception as exc:
            return {
                "online": False,
                "error": str(exc),
                "devices": _device_status({}, {}),
            }

    def image(self, name: str) -> tuple[bytes, str] | None:
        with self._lock:
            data = self._images.get(name)
            if data is None:
                return None
            return data, self._image_types.get(name, "image/jpeg")

    def control_status(self) -> dict[str, Any]:
        """Lightweight discrete status only, independent of overview and JPEGs."""
        try:
            with CobotBridgeClient(self.config.endpoint, timeout_s=0.3) as client:
                ping = client.ping()
                raw, _ = client.get_observation(sides=ARM_SIDES, images=())
            control = raw.get('control') or {}
            devices = _device_status(raw, {})
            # Same observation as the status probe. Cache for low-rate display;
            # browsers must not issue another robot request to read joint angles.
            with self._lock:
                self._joint_revision += 1
                self._joint_cached_at = time.monotonic()
                self._joint_cache = {
                    'revision': self._joint_revision,
                    'bridge_time_ns': raw.get('bridge_time_ns'),
                    'arms': {k: {n: v[n] for n in ('position', 'fresh', 'age_ms')}
                             for k, v in devices['arms'].items()},
                }
            # Remove all continuously varying fields: joint positions, frame
            # counters, timestamps, ages and JPEG sizes must not generate events.
            return {
                'online': True, 'allow_control': bool(ping.get('allow_control', False)) and all(
                    control.get(k) is not None for k in ('effective_mode', 'teach_active', 'output_enabled', 'emergency_latched')),
                'mode': control.get('effective_mode'), 'mode_id': control.get('effective_mode_id'),
                'teach_active': control.get('teach_active'),
                'output_enabled': control.get('output_enabled'),
                'emergency_latched': control.get('emergency_latched'),
                'epoch': control.get('epoch'), 'ready': bool(raw.get('ready')),
                'devices': {'ok': devices['ok'], 'ok_count': devices['ok_count'], 'total': 7,
                            'arms': {k: {n: v[n] for n in ('ok', 'fresh')} for k, v in devices['arms'].items()},
                            'cameras': {k: {n: v[n] for n in ('ok', 'fresh', 'has_frame')} for k, v in devices['cameras'].items()}},
            }
        except Exception:
            return {'online': False, 'devices': {'ok': False, 'ok_count': 0, 'total': 7,
                    'arms': {f'{r}_{s}': {'ok': False, 'fresh': False} for r in ARM_ROLES for s in ARM_SIDES},
                    'cameras': {n: {'ok': False, 'fresh': False, 'has_frame': False} for n in CAMERA_NAMES}}}

    def joint_state(self) -> dict[str, Any]:
        """Read local latest-value cache only; no bridge I/O or directory scan."""
        with self._lock:
            if self._joint_cache is None:
                return {'revision': 0, 'age_ms': None, 'arms': {}}
            elapsed_ms = max(0., (time.monotonic() - self._joint_cached_at) * 1000)
            return {**self._joint_cache, 'age_ms': round(elapsed_ms, 1),
                    'arms': {k: {**v, 'age_ms': round(float(v['age_ms'] or 0) + elapsed_ms, 1),
                                  'fresh': bool(v['fresh'] and elapsed_ms <= 1500)}
                             for k, v in self._joint_cache['arms'].items()}}


class DashboardMjpegRelay:
    """One PC-side bridge JPEG puller shared by all Dashboard browsers.

    This deliberately reuses :class:`BridgeJpegPump`, the same latest-frame
    cache used for the AGX loopback preview. The Dashboard serves its output
    on the PC's own origin, avoiding WebRTC and duplicate per-browser ZMQ
    image pulls.
    """

    def __init__(self, config: DashboardConfig) -> None:
        self._pump = BridgeJpegPump(config.endpoint, fps=config.pc_preview_fps)
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._pump.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._pump.stop()
            self._started = False

    def stream(self, name: str):
        """Yield newly observed JPEGs only; slow clients never build a queue."""
        self.start()
        slot = self._pump.slots[name]
        version = 0
        while True:
            data, stamp_ns, version = slot.wait_after(version, 1.0)
            if not data:
                continue
            header = (
                f"--{MJPEG_BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"X-Source-Stamp-Ns: {stamp_ns}\r\n\r\n"
            ).encode("ascii")
            yield header + data + b"\r\n"


class RecordRequest(BaseModel):
    collection: str = "cobot_magic_task"
    task: str = Field(min_length=1, max_length=300)
    side: Literal["left", "right"] = "right"
    cameras: list[Literal["front", "left", "right"]] = Field(
        default_factory=lambda: ["front", "right"]
    )
    fps: int = Field(default=30, ge=1, le=60)
    max_seconds: float = Field(default=3600.0, gt=0, le=14400)
    stop_on_teach_release: bool = True
    policy_id: str = Field(default="", max_length=200)
    collection_type: Literal["IL", "RL"] = "IL"

    @field_validator("collection")
    @classmethod
    def valid_collection(cls, value: str) -> str:
        if not NAME_PATTERN.fullmatch(value):
            raise ValueError("use letters, numbers, dot, underscore or dash")
        return value


class InitAndRecordRequest(RecordRequest):
    confirm_motion: bool = False


class ConvertRequest(BaseModel):
    raw_id: str
    dataset_name: str
    fps: int | None = Field(default=None, ge=1, le=60)
    action_source: Literal[
        "master_state",
        "puppet_command",
        "reference_action",
        "policy_action",
        "human_action",
        "executed_action",
        "puppet_state_t3_arm_cmd_gripper",
    ] = "executed_action"
    use_videos: bool = True
    trim_stationary_holds: bool = False

    @field_validator("dataset_name")
    @classmethod
    def valid_dataset_name(cls, value: str) -> str:
        if not NAME_PATTERN.fullmatch(value):
            raise ValueError("use letters, numbers, dot, underscore or dash")
        return value


class ReplayRequest(BaseModel):
    raw_id: str
    side: Literal["left", "right"] = "right"
    source: Literal["master_position", "puppet_position", "puppet_command"] = "puppet_command"
    mode: Literal["puppet", "sync"] = "sync"
    waypoint_hz: int = Field(default=30, ge=1, le=200)
    speed_scale: float = Field(default=0.5, gt=0, le=1.0)
    loop: bool = False
    execute: bool = False
    confirm_motion: bool = False


class LerobotReplayRequest(BaseModel):
    dataset_id: str
    episode_index: int = Field(default=0, ge=0)
    side: Literal["left", "right"] = "right"
    waypoint_hz: int = Field(default=30, ge=1, le=200)
    speed_scale: float = Field(default=0.5, gt=0, le=1.0)
    alignment_speed: float = Field(default=0.05, gt=0, le=0.12)
    alignment_min_seconds: float = Field(default=5.0, gt=0, le=30.0)
    execute: bool = False
    confirm_motion: bool = False


class PolicyRequest(BaseModel):
    task: str = Field(min_length=1, max_length=300)
    side: Literal["left", "right"] = "right"
    fps: float = Field(default=15.0, gt=0, le=60)
    duration: float = Field(default=60.0, gt=0, le=3600)
    mode: Literal[1, 2] = 1
    action_space: Literal["joint", "eef_delta", "eef_target"] = "joint"
    eef_ik_backend: Literal["pink", "sdk"] = "pink"
    runner: Literal[
        "standard", "smooth_v2", "smooth_v3", "edge_v4", "eef_v5", "guided_rtc"
    ] = "standard"
    n_action_steps: int | None = Field(default=None, ge=1, le=50)
    execute: bool = False
    confirm_motion: bool = False


class PolicyServerRequest(BaseModel):
    policy_id: str
    device: str = "cuda:0"
    compute: Literal["pc", "robot"] = "pc"
    inference_chunking: Literal["original", "guided_rtc"] = "original"
    num_steps: Literal[5, 10, 20] | None = None
    residual_actor: str | None = None
    z_encoder: str | None = None


class FinalizeRecordRequest(BaseModel):
    outcome: Literal["success", "failure"]


class InterventionPhaseRequest(BaseModel):
    phase: Literal["calibrate", "rescue", "retreat"]


class TeachLabel(BaseModel):
    index: int
    phase: Literal["calibrate", "rescue", "retreat"]


class TeachLabelsRequest(BaseModel):
    labels: list[TeachLabel]


class InitPoseRequest(BaseModel):
    execute: bool = False
    confirm_motion: bool = False


class RobotStartupRequest(BaseModel):
    check_only: bool = False


class RLTCheckpointRequest(BaseModel):
    checkpoint_id: str


class RLTPolicyServerRequest(RLTCheckpointRequest):
    device: str = Field(default="cuda:0", min_length=1, max_length=32)
    actor_checkpoint_id: str | None = None


class RLTRollbackRequest(RLTCheckpointRequest):
    target_updates: Literal[30000] = 30000


class RLTShadowRequest(RLTCheckpointRequest):
    task: str = Field(min_length=1, max_length=300)
    side: Literal["left", "right"] = "right"
    fps: float = Field(default=30.0, gt=0, le=30)
    duration: float = Field(default=0.0, ge=0, le=3600)
    device: str = Field(default="cuda:0", min_length=1, max_length=32)


class RLTOnlineRequest(RLTCheckpointRequest):
    task: str = Field(min_length=1, max_length=300)
    workflow_mode: Literal[1, 2]
    rollout_phase: Literal["actor", "reference"] = "actor"
    collection: str = "cobot_magic_rlt"
    raw_fps: int = Field(default=30, ge=1, le=30)
    chunk_hz: float = Field(default=30.0, gt=0, le=30)
    rollout_speed: float = Field(default=0.5, gt=0, le=1.0)
    smooth_profile: Literal["edge_v4", "raw", "smooth_v3", "smooth_v2"] = "edge_v4"
    actor_eef_gate: bool = True
    max_seconds: float = Field(default=3600.0, gt=0, le=14400)
    actor_exploration_std: float = Field(default=0.02, ge=0, le=0.1)
    q_select_k: int = Field(default=0, ge=0, le=64)
    actor_max_decisions: int = Field(default=0, ge=0, le=3)
    execute: bool = False
    confirm_motion: bool = False

    @field_validator("collection")
    @classmethod
    def valid_collection(cls, value: str) -> str:
        if not NAME_PATTERN.fullmatch(value):
            raise ValueError("use letters, numbers, dot, underscore or dash")
        return value


def _within(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("path id must be relative")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes configured root") from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _raw_files(config: DashboardConfig, *, paths: list[Path] | None = None) -> list[dict[str, Any]]:
    if not config.raw_root.exists():
        return []
    output = []
    for path in sorted(config.raw_root.rglob("*.h5") if paths is None else paths, reverse=True):
        relative = path.relative_to(config.raw_root)
        entry: dict[str, Any] = {
            "id": relative.as_posix(),
            "collection": relative.parent.as_posix(),
            "episode": relative.name,
            "size": path.stat().st_size,
            "modified_at_ns": path.stat().st_mtime_ns,
        }
        try:
            with h5py.File(path, "r") as raw:
                entry.update(
                    {
                        "format": str(raw.attrs.get("format", "")),
                        "frames": int(raw.attrs.get("num_frames", 0)),
                        "fps": int(raw.attrs.get("fps", 0)),
                        "task": str(raw.attrs.get("task", "")),
                        "scene_layout": str(raw.attrs.get("scene_layout", "unlabeled")),
                        "scene_layout_label": str(
                            raw.attrs.get("scene_layout_label", "Unlabeled")
                        ),
                        "task_setup": str(raw.attrs.get("task_setup", "unlabeled")),
                        "task_setup_label": str(
                            raw.attrs.get("task_setup_label", "Unlabeled")
                        ),
                        "task_objective": str(
                            raw.attrs.get("task_objective", raw.attrs.get("task", ""))
                        ),
                        "outcome": str(raw.attrs.get("outcome", "unknown")),
                        "collection_type": str(raw.attrs.get("collection_type", "UNKNOWN")),
                        "completed": bool(raw.attrs.get("completed", False)),
                    }
                )
        except OSError as exc:
            entry["error"] = str(exc)
        output.append(entry)
    return output


def _dataset_files(config: DashboardConfig) -> list[dict[str, Any]]:
    if not config.dataset_root.exists():
        return []
    output = []
    for info_path in sorted(config.dataset_root.rglob("meta/info.json"), reverse=True):
        root = info_path.parent.parent
        try:
            info = json.loads(info_path.read_text())
            output.append(
                {
                    "id": root.relative_to(config.dataset_root).as_posix(),
                    "frames": int(info.get("total_frames", 0)),
                    "episodes": int(info.get("total_episodes", 0)),
                    "fps": int(info.get("fps", 0)),
                    "robot_type": info.get("robot_type", ""),
                }
            )
        except (OSError, ValueError, TypeError) as exc:
            output.append(
                {"id": root.relative_to(config.dataset_root).as_posix(), "error": str(exc)}
            )
    return output


def _lerobot_replay_dataset_files(config: DashboardConfig) -> list[dict[str, Any]]:
    root = config.lerobot_replay_root
    if not root.exists():
        return []
    output = []
    for info_path in sorted(root.rglob("meta/info.json")):
        dataset_root = info_path.parent.parent
        try:
            info = json.loads(info_path.read_text())
            feature = info.get("features", {}).get("action_joint", {})
            if not isinstance(feature, dict):
                continue
            names = feature.get("names", {})
            if isinstance(names, dict):
                names = names.get("motors")
            if not isinstance(names, list) or not names:
                continue
            required_right = [*(f"r-j{joint}" for joint in range(1, 7)), "right_gripper"]
            if not all(name in names for name in required_right):
                continue
            output.append(
                {
                    "id": dataset_root.relative_to(root).as_posix(),
                    "frames": int(info.get("total_frames", 0)),
                    "episodes": int(info.get("total_episodes", 0)),
                    "fps": int(info.get("fps", 0)),
                    "robot_type": str(info.get("robot_type", "")),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    return output


def _policy_files(config: DashboardConfig) -> list[dict[str, Any]]:
    output = []
    if config.checkpoint_root.exists():
        for config_path in sorted(config.checkpoint_root.rglob("config.json"), reverse=True):
            root = config_path.parent
            model = root / "model.safetensors"
            try:
                policy_config = json.loads(config_path.read_text())
            except (OSError, ValueError):
                policy_config = {}
            architectures = policy_config.get("architectures", [])
            is_groot = (
                (root / "model.safetensors.index.json").is_file()
                and isinstance(architectures, list)
                and any(str(name).startswith("Gr00t") for name in architectures)
            )
            if not model.exists() and not is_groot:
                continue
            relative = root.relative_to(config.checkpoint_root)
            parts = relative.parts
            if "checkpoints" in parts:
                checkpoint_index = parts.index("checkpoints")
                run_id = Path(*parts[:checkpoint_index]).as_posix()
                checkpoint_name = (
                    parts[checkpoint_index + 1]
                    if checkpoint_index + 1 < len(parts)
                    else relative.name
                )
            else:
                run_id = (
                    relative.parent.as_posix()
                    if relative.parent.as_posix() != "."
                    else relative.name
                )
                checkpoint_name = relative.name
            output.append(
                {
                    "id": relative.as_posix(),
                    "type": "groot" if is_groot else policy_config.get("type", "unknown"),
                    "run_id": run_id,
                    "checkpoint": checkpoint_name,
                    "step": int(checkpoint_name) if checkpoint_name.isdigit() else None,
                }
            )

    external_groot = config.groot_eef_checkpoint_path
    if external_groot is not None:
        external_groot = external_groot.expanduser().resolve()
        config_path = external_groot / "config.json"
        try:
            policy_config = json.loads(config_path.read_text())
        except (OSError, ValueError):
            policy_config = {}
        architectures = policy_config.get("architectures", [])
        is_groot = (
            external_groot.is_dir()
            and (external_groot / "model.safetensors.index.json").is_file()
            and (external_groot / "processor_config.json").is_file()
            and isinstance(architectures, list)
            and any(str(name).startswith("Gr00t") for name in architectures)
        )
        if is_groot and not any(
            Path(item["id"]).as_posix() == GROOT_EEF_POLICY_ID for item in output
        ):
            output.append(
                {
                    "id": GROOT_EEF_POLICY_ID,
                    "type": "groot",
                    "run_id": "external/groot_5482b_eef",
                    "checkpoint": external_groot.name,
                    "step": 90000,
                }
            )
    return output


def _resolve_policy_checkpoint(config: DashboardConfig, policy_id: str) -> Path:
    if policy_id == GROOT_EEF_POLICY_ID:
        checkpoint = config.groot_eef_checkpoint_path
        if checkpoint is None:
            raise FileNotFoundError("external GR00T EEF checkpoint is not configured")
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return resolved
    return _within(config.checkpoint_root, policy_id)


def _read_rlt_pipeline_status(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {"state": "not_started"}
    progress_file = payload.get("trainer_progress_file")
    if progress_file:
        try:
            payload["trainer_progress"] = json.loads(Path(progress_file).read_text())
        except (OSError, ValueError, TypeError):
            payload["trainer_progress"] = {"state": "not_started"}
    return payload


def _latest_rlt_pipeline_for_checkpoint(
    config: DashboardConfig, checkpoint_path: Path
) -> dict[str, Any]:
    expected = str(checkpoint_path.expanduser().resolve())
    try:
        candidates = sorted(
            config.run_root.glob("rlt_*.pipeline.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return {"state": "not_started"}
    for path in candidates:
        payload = _read_rlt_pipeline_status(path)
        actual = payload.get("stage1_checkpoint")
        if actual and str(Path(actual).expanduser().resolve()) == expected:
            payload["status_file"] = str(path.resolve())
            return payload
    return {"state": "not_started"}


def _rlt_progress_for_checkpoint(config: DashboardConfig, checkpoint_path: Path) -> dict[str, Any]:
    run_root = checkpoint_path.parent
    replay_root = config.rlt_root / "replay" / run_root.name
    stage2_output = run_root.with_name(f"{run_root.name}_stage2")
    stage2_status: dict[str, Any] = {}
    try:
        stage2_status = json.loads((stage2_output / "status.json").read_text())
    except (OSError, ValueError, TypeError):
        pass
    latest = stage2_output / "latest.pt"
    rollback_30k = stage2_output / "checkpoint_00030000.pt"
    actor_ready = bool(
        latest.is_file()
        and stage2_status.get("stage") == "rlt_stage2"
        and stage2_status.get("actor_output_semantics") in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
        and stage2_status.get("action_chunk_timebase") == ACTION_CHUNK_TIMEBASE
        and str(Path(stage2_status.get("stage1_checkpoint", "")).expanduser().resolve())
        == str(checkpoint_path.resolve())
        # Operator policy (2026-08-25): ready_for_online no longer gates
        # actor availability (auto-disable removed in rlt_policy_server);
        # the gate state is still reported via ``ready_for_online`` below.
        and (
            int(stage2_status.get("total_updates", 0)) > 0
            or (
                stage2_status.get("initialization")
                in ("il_ground_truth_actor_bc", "il_ground_truth_residual_actor_bc")
                and int(stage2_status.get("bc_steps", 0)) > 0
            )
        )
    )
    return {
        "actor_ready": actor_ready,
        # Held-out one-step MSE remains visible in the Actor metadata but does
        # not gate real rollout; runtime behavior is validated by controlled
        # robot rollouts instead.
        "ready_for_online": actor_ready,
        "total_updates": int(stage2_status.get("total_updates", 0)),
        "latest_exists": latest.is_file(),
        "active_checkpoint": str(stage2_status.get("checkpoint", "")),
        "rollback_30k_available": rollback_30k.is_file(),
        "rollback_30k_active": int(stage2_status.get("total_updates", 0)) == 30000,
        "replay_root": str(replay_root),
        "pipeline": _latest_rlt_pipeline_for_checkpoint(config, checkpoint_path),
    }


def _rlt_actor_files(config: DashboardConfig, checkpoint_path: Path) -> list[dict[str, Any]]:
    """List small-metadata-selected Stage2 actors compatible with one Stage1.

    ``status.json`` is used for inventory so the dashboard does not deserialize
    every 50 MB actor/critic checkpoint on each refresh. The selected tensor
    file is still strictly validated by the start endpoint before it is used.
    """

    checkpoint_root = (config.rlt_root / "checkpoints").resolve()
    expected_stage1 = str(checkpoint_path.resolve())
    canonical_online_root = checkpoint_path.parent.with_name(
        f"{checkpoint_path.parent.name}_stage2"
    ).resolve()
    output: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for status_path in checkpoint_root.rglob("status.json"):
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        try:
            actual_stage1 = str(
                Path(status.get("stage1_checkpoint", "")).expanduser().resolve()
            )
        except (OSError, TypeError, ValueError):
            continue
        if (
            status.get("stage") != "rlt_stage2"
            or status.get("actor_output_semantics") not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
            or status.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE
            or actual_stage1 != expected_stage1
        ):
            continue
        updates = int(status.get("total_updates", 0))
        pre_rl = bool(
            updates == 0
            and status.get("initialization")
            in ("il_ground_truth_actor_bc", "il_ground_truth_residual_actor_bc")
            and int(status.get("bc_steps", 0)) > 0
        )
        warmup_only = bool(
            status_path.parent.resolve() == canonical_online_root
            and status.get("training_phase") == "warmup"
            and updates == 30000
        )
        # Use the exact ``latest.pt`` path whenever a run publishes one. The
        # resident policy server loads that path too; listing a pre-RL seed as
        # ``checkpoint_00000000.pt`` creates a false UI/server mismatch even
        # when both files contain the same selected warmup Actor.
        latest_value = status.get("latest")
        latest_path = Path(str(latest_value)).expanduser() if latest_value else None
        selected_value = (
            latest_value
            if latest_path is not None and latest_path.is_file()
            else status.get("checkpoint")
        )
        if not selected_value:
            selected_value = status.get("checkpoint")
        try:
            selected = Path(str(selected_value)).expanduser().resolve()
            selected.relative_to(checkpoint_root)
        except (OSError, TypeError, ValueError):
            continue
        if selected in seen or not selected.is_file() or selected.suffix != ".pt":
            continue
        seen.add(selected)
        validation = status.get("validation_gt_bc_loss")
        if validation is None:
            heldout = status.get("heldout_validation")
            validation = heldout.get("gt_bc_loss") if isinstance(heldout, dict) else None
        output_semantics = str(status.get("actor_output_semantics", ""))
        bc_steps = int(status.get("bc_steps", status.get("best_step", 0)))
        output.append(
            {
                "id": selected.relative_to(checkpoint_root).as_posix(),
                "path": str(selected),
                "ready": True,
                "kind": (
                    "pre_rl_bc"
                    if pre_rl
                    else "warmup_only" if warmup_only else "online_rl"
                ),
                "label": (
                    (
                        f"Pre-RL Residual Actor BC · {bc_steps} steps"
                        if output_semantics == ACTOR_OUTPUT_SEMANTICS
                        else f"Pre-RL Direct Actor BC · {bc_steps} steps"
                    )
                    if pre_rl
                    else "Warm-up-only Actor · 30k"
                    if warmup_only
                    else (
                        (
                            "Direct RL Actor · "
                            if output_semantics.startswith("direct")
                            else "Residual RL Actor · "
                        )
                        + (
                            status_path.parent.name.split("_stage2_", 1)[1]
                            if "_stage2_" in status_path.parent.name
                            else status_path.parent.name.removeprefix("cobot_magic_073rel_rlt_stage2_")
                        )
                    )
                ),
                "total_updates": updates,
                "bc_steps": bc_steps,
                "validation_gt_bc_loss": (
                    float(validation) if validation is not None else None
                ),
                "follows_latest": selected.name == "latest.pt",
                "modified_at_ns": selected.stat().st_mtime_ns,
            }
        )

    # Offline sweeps are never allowed to replace the canonical ``latest.pt``.
    # They become selectable only through an explicit, small manifest written
    # after the sweep has completed and its checkpoints have been validated.
    for inventory_path in checkpoint_root.rglob("actor_inventory.json"):
        try:
            inventory = json.loads(inventory_path.read_text())
            actual_stage1 = str(
                Path(inventory.get("stage1_checkpoint", "")).expanduser().resolve()
            )
        except (OSError, ValueError, TypeError):
            continue
        if (
            inventory.get("format") != "cobot_magic_rlt_actor_inventory_v1"
            or inventory.get("actor_output_semantics") not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
            or inventory.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE
            or inventory.get("publish_scope") != "manual_evaluation_only"
            or actual_stage1 != expected_stage1
        ):
            continue
        candidates = inventory.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("checkpoint"):
                continue
            try:
                selected = (inventory_path.parent / str(candidate["checkpoint"])).resolve()
                selected.relative_to(checkpoint_root)
                updates = int(candidate["total_updates"])
            except (OSError, TypeError, ValueError, KeyError):
                continue
            if selected in seen or not selected.is_file() or selected.suffix != ".pt":
                continue
            seen.add(selected)
            output.append(
                {
                    "id": selected.relative_to(checkpoint_root).as_posix(),
                    "path": str(selected),
                    "ready": True,
                    "kind": "offline_eval",
                    "label": str(candidate.get("label") or f"Offline Actor · {updates} updates"),
                    "total_updates": updates,
                    "bc_steps": 0,
                    "validation_gt_bc_loss": (
                        float(candidate["validation_gt_bc_loss"])
                        if candidate.get("validation_gt_bc_loss") is not None
                        else None
                    ),
                    "follows_latest": False,
                    "modified_at_ns": selected.stat().st_mtime_ns,
                }
            )

    # Steam is an offline actor export rather than a normal Stage 2 trainer
    # publication: it has no status.json/actor_inventory.json.  Discover only
    # explicitly named Steam files so a dashboard refresh never deserializes
    # every large actor/critic checkpoint in the tree.
    import torch

    for candidate in sorted(checkpoint_root.rglob("*.pt")):
        if candidate in seen or "steam" not in str(candidate).lower():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            actual_stage1 = str(
                Path(payload.get("stage1_checkpoint", "")).expanduser().resolve()
            )
        except (OSError, TypeError, ValueError):
            continue
        if (
            payload.get("stage") != STEAM_OFFLINE_ACTOR_STAGE
            or payload.get("actor_output_semantics") not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
            or payload.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE
            or actual_stage1 != expected_stage1
            or not isinstance(payload.get("actor_state_dict"), dict)
        ):
            continue
        seen.add(candidate)
        is_best = candidate.name == "actor_steam_best.pt"
        steam_label = (
            "STEAM Offline Actor · best"
            if is_best
            else f"STEAM Offline Actor · {candidate.stem.removeprefix('actor_steam_')}"
        )
        output.append(
            {
                "id": candidate.relative_to(checkpoint_root).as_posix(),
                "path": str(candidate),
                "ready": True,
                "kind": "offline_steam",
                "label": f"{steam_label} · deterministic · no async training",
                "total_updates": int(payload.get("step", payload.get("total_updates", 0))),
                "bc_steps": 0,
                "validation_gt_bc_loss": None,
                "follows_latest": is_best,
                "selection_priority": 0 if is_best else 1,
                "read_only": True,
                "inference_only": True,
                "checkpoint_stage": STEAM_OFFLINE_ACTOR_STAGE,
                "modified_at_ns": candidate.stat().st_mtime_ns,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            {"pre_rl_bc": 0, "warmup_only": 1, "offline_steam": 2, "offline_eval": 3}.get(
                item["kind"], 3
            ),
            int(item.get("selection_priority", 1)),
            -int(item["total_updates"]),
            -int(item["modified_at_ns"]),
        ),
    )


def _rlt_stage1_files(config: DashboardConfig) -> list[dict[str, Any]]:
    """List validated-by-construction Stage 1 artifacts without loading large tensors."""

    checkpoint_root = config.rlt_root / "checkpoints"
    if not checkpoint_root.exists():
        return []
    output: list[dict[str, Any]] = []
    for manifest_path in sorted(checkpoint_root.rglob("stage1_config.json"), reverse=True):
        run_root = manifest_path.parent
        checkpoint_path = run_root / "best_checkpoint.pt"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, TypeError) as exc:
            output.append(
                {
                    "id": run_root.relative_to(checkpoint_root).as_posix(),
                    "ready": False,
                    "error": f"invalid stage1_config.json: {exc}",
                }
            )
            continue
        prefix_stage1 = bool(
            manifest.get("stage") == "rlt_stage1_joint"
            and manifest.get("rlt_architecture") == "rlinf_prefix_v1"
        )
        legacy_stage1 = manifest.get("stage") == "rlt_stage1"
        if not prefix_stage1 and not legacy_stage1:
            continue
        history_path = run_root / "loss_history.json"
        final_loss: float | None = None
        final_step: int | None = None
        try:
            history = json.loads(history_path.read_text())
            if isinstance(history, list) and history:
                final_loss = float(history[-1]["loss"])
                final_step = int(history[-1]["step"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
        embedding_shapes = (
            {"prefix": manifest.get("prefix_shape", [])}
            if prefix_stage1
            else dict(manifest.get("embedding_shapes", {}))
        )
        smolvla_path = (
            run_root / "vla_best"
            if prefix_stage1
            else Path(str(manifest.get("smolvla_path", ""))).expanduser().resolve()
        )
        stage2_progress = _rlt_progress_for_checkpoint(config, checkpoint_path)
        actor_checkpoints = _rlt_actor_files(config, checkpoint_path)
        output.append(
            {
                "id": checkpoint_path.relative_to(checkpoint_root).as_posix(),
                "run_id": run_root.relative_to(checkpoint_root).as_posix(),
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "ready": checkpoint_path.is_file() and smolvla_path.is_dir(),
                "architecture": (
                    "rlinf_prefix_v1"
                    if prefix_stage1
                    else "legacy_action_conditioned_v2"
                ),
                "inference_only": bool(legacy_stage1),
                "checkpoint_exists": checkpoint_path.is_file(),
                "smolvla_exists": smolvla_path.is_dir(),
                "smolvla_path": str(smolvla_path),
                "dataset_root": str(manifest.get("dataset_root", "")),
                "dataset_frames": int(manifest.get("dataset_frames", 0)),
                "dataset_episodes": int(manifest.get("dataset_episodes", 0)),
                "dataset_fps": float(manifest.get("dataset_fps", 0.0)),
                "embedding_shapes": embedding_shapes,
                "rlt_hidden_dim": int(
                    manifest.get("rlt_config", {}).get("rlt_hidden_dim", 0)
                ),
                "chunk_size": int(manifest.get("rlt_config", {}).get("chunk_size", 0)),
                "final_step": final_step,
                "final_loss": final_loss,
                "stage2": stage2_progress,
                "actor_checkpoints": actor_checkpoints,
                "modified_at_ns": checkpoint_path.stat().st_mtime_ns
                if checkpoint_path.is_file()
                else manifest_path.stat().st_mtime_ns,
            }
        )
    return sorted(output, key=lambda item: int(item.get("modified_at_ns", 0)), reverse=True)


def _relative_id(root: Path, value: str | None) -> str | None:
    if not value:
        return None
    try:
        return Path(value).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _policy_endpoint(config: DashboardConfig, compute: str) -> str:
    return (
        config.robot_policy_server_endpoint
        if compute == "robot"
        else config.policy_server_endpoint
    )


def _policy_id_from_path(
    config: DashboardConfig, compute: str, policy_path: str | None
) -> str | None:
    if compute == "pc" and policy_path and config.groot_eef_checkpoint_path is not None:
        try:
            if (
                Path(policy_path).expanduser().resolve()
                == config.groot_eef_checkpoint_path.expanduser().resolve()
            ):
                return GROOT_EEF_POLICY_ID
        except OSError:
            pass
    root = (
        config.robot_workspace / "checkpoints"
        if compute == "robot"
        else config.checkpoint_root
    )
    return _relative_id(root, policy_path)


def _tcp_endpoint_in_use(endpoint: str) -> bool:
    """True when something already accepts TCP connections on a ``tcp://host:port``.

    A just-stopped server can keep its ZMQ socket for a moment, and a server
    started outside the dashboard holds it while the job manager knows nothing.
    Either way the replacement server fails to bind and exits, which used to look
    like a job that started and immediately completed.
    """

    match = re.fullmatch(r"tcp://([^:/]+):(\d+)", endpoint.strip())
    if match is None:
        return False
    try:
        with socket.create_connection((match.group(1), int(match.group(2))), timeout=0.25):
            return True
    except OSError:
        return False


def _policy_server_status(config: DashboardConfig, manager: JobManager) -> dict[str, Any]:
    active = next(
        (
            job
            for job in manager.list()
            if job["kind"] == "policy_server" and job["state"] in ("running", "stopping")
        ),
        None,
    )
    active_compute = str((active or {}).get("metadata", {}).get("compute", "pc"))
    computes = [active_compute] if active else ["pc", "robot"]
    errors: dict[str, str] = {}
    for compute in computes:
        endpoint = _policy_endpoint(config, compute)
        client = PersistentPolicyClient(endpoint, timeout_s=0.15)
        try:
            ping = client.ping()
        except (PolicyServerError, OSError, ValueError) as exc:
            errors[compute] = str(exc)
            continue
        finally:
            client.close()
        if ping.get("ready", False):
            return {
                "ready": True,
                "state": "ready",
                "compute": compute,
                "endpoint": endpoint,
                "job_id": active["id"] if active else None,
                "policy_id": _policy_id_from_path(config, compute, ping.get("policy_path")),
                **ping,
            }
    return {
        "ready": False,
        "state": "loading" if active and active["state"] == "running" else "offline",
        "compute": active_compute if active else None,
        "endpoint": _policy_endpoint(config, active_compute),
        "job_id": active["id"] if active else None,
        "error": errors.get(active_compute) or next(iter(errors.values()), "model server offline"),
    }


# Actor loss weights for online Stage-2 updates.  Kept in a small JSON file
# (RL-Token-SmolVLA/rlt_actor_weights.json) instead of being hard-coded here so
# experiments can retune the q/bc/il mix by editing the file — no code change
# and no dashboard restart needed (read fresh on every rollout launch).
DEFAULT_RLT_ACTOR_WEIGHTS = {
    "q_weight": 1.0,
    "bc_weight": 5.0,
    "il_anchor_weight": 30.0,
    "intervention_bc_weight": 1.0,
    # Actor output semantics / reference-anchor settings.  NOTE (2026-08-26):
    # these are FALLBACKS ONLY.  When a rollout selects an online RL actor, the
    # rollout handler overrides q/bc/il/iv weights, output semantics and the
    # reference-anchor settings from that actor's checkpoint payload, so
    # switching lineages is a pure UI operation (select actor + Start).  This
    # file still supplies ``action_stats`` (not stored in checkpoint payloads)
    # and the fallbacks for older checkpoints without the new fields.
    "actor_output_semantics": "residual",
    "ref_anchor_weight": 0.5,
    "ref_anchor_mask": "none",
    "action_stats": None,
}
RLT_ACTOR_WEIGHTS_FILENAME = "rlt_actor_weights.json"


def _actor_takeover_decisions_for_frame(
    actor_start_step: int | None,
    reference_action_steps: int | None,
    rollout_speed: float,
) -> int:
    """Convert a 30 Hz Actor frame gate into complete reference chunks."""

    if (
        actor_start_step is None
        or actor_start_step <= 0
        or reference_action_steps is None
        or reference_action_steps <= 0
        or not math.isfinite(rollout_speed)
        or rollout_speed <= 0
    ):
        return 0
    executed_reference_steps = max(
        2, int(round(reference_action_steps / rollout_speed))
    )
    return int(math.ceil(actor_start_step / executed_reference_steps))


RLT_ACTOR_TAKEOVER_DECISION_OVERRIDES = {
    # The v7 Actor was trained with a 720-frame handover annotation, but its
    # deployment rollout should hand control over earlier reference decisions
    # for rollout timing experiments.
    "cobot_magic_smolvla_il2trimrel80k_rlt_stage1_rlinf960_ref30_actor30_5k_bs64_stage2_actor20_speed05_seq_m26q1_ilfakebc_v1v4q1_utd5_g720_loose_v1_frozen83_v7": 9,
}


def _actor_takeover_decisions_for_checkpoint(
    checkpoint: Path | None,
    actor_start_step: int | None,
    reference_action_steps: int | None,
    rollout_speed: float,
) -> int:
    """Apply a deployment-only handover override for a selected Actor."""

    if checkpoint is not None:
        override = RLT_ACTOR_TAKEOVER_DECISION_OVERRIDES.get(checkpoint.parent.name)
        if override is not None:
            return int(override)
    return _actor_takeover_decisions_for_frame(
        actor_start_step, reference_action_steps, rollout_speed
    )


def _load_rlt_actor_weights(config: DashboardConfig) -> dict[str, Any]:
    weights = dict(DEFAULT_RLT_ACTOR_WEIGHTS)
    try:
        payload = json.loads((config.rlt_root / RLT_ACTOR_WEIGHTS_FILENAME).read_text())
        for key in weights:
            if key in payload:
                # float keys are the loss weights; the rest are strings/paths
                # (semantics, anchor mask, stats path) passed through verbatim.
                weights[key] = (
                    float(payload[key])
                    if isinstance(weights[key], float)
                    else payload[key]
                )
    except (OSError, ValueError, TypeError):
        pass
    return weights


def _rlt_policy_server_status(config: DashboardConfig, manager: JobManager) -> dict[str, Any]:
    active = next(
        (
            job
            for job in manager.list()
            if job["kind"] == "rlt_policy_server"
            and job["state"] in ("running", "stopping")
        ),
        None,
    )
    client = PersistentRLTPolicyClient(
        config.rlt_policy_server_endpoint, timeout_s=1.5 if active else 0.1
    )
    try:
        ping = client.ping()
    except (RLTPolicyServerError, OSError, ValueError) as exc:
        return {
            "ready": False,
            "state": "loading" if active and active["state"] == "running" else "offline",
            "endpoint": config.rlt_policy_server_endpoint,
            "job_id": active["id"] if active else None,
            "checkpoint_id": (active or {}).get("metadata", {}).get("checkpoint_id"),
            "error": str(exc),
        }
    finally:
        client.close()
    if not ping.get("ready", False):
        return {
            "ready": False,
            "state": "loading" if active else "offline",
            "endpoint": config.rlt_policy_server_endpoint,
            "job_id": active["id"] if active else None,
            **ping,
        }
    return {
        "ready": True,
        "state": "ready",
        "endpoint": config.rlt_policy_server_endpoint,
        "job_id": active["id"] if active else None,
        "checkpoint_id": _relative_id(
            config.rlt_root / "checkpoints", ping.get("stage1_checkpoint")
        ),
        "actor_checkpoint_id": _relative_id(
            config.rlt_root / "checkpoints", ping.get("stage2_checkpoint")
        ),
        **ping,
    }


def _capabilities(config: DashboardConfig) -> dict[str, dict[str, Any]]:
    rlt_scripts = config.rlt_root / "scripts"
    return {
        "record": {"ready": True},
        "convert": {"ready": True},
        "replay": {"ready": True},
        "lerobot_policy": {"ready": True},
        "rlt_stage1": {
            "ready": (rlt_scripts / "train_rlt_stage1.py").is_file(),
            "reason": "adapted Stage 1 trainer is unavailable",
        },
        "rlt_stage2": {
            "ready": all(
                (config.workspace / "cobot_bridge" / name).is_file()
                for name in (
                    "rlt_online.py",
                    "rlt_policy_client.py",
                    "rlt_policy_server.py",
                    "rlt_stage2_data.py",
                    "rlt_stage2_train.py",
                )
            ),
            "reason": "Cobot Magic online Stage 2 adapter with asynchronous trainer",
        },
        "rlt_policy": {
            "ready": (config.workspace / "cobot_bridge" / "rlt_shadow.py").is_file(),
            "reason": "read-only RLT shadow runner is unavailable",
        },
    }


def _device_status(raw: dict[str, Any], images: dict[str, bytes]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for role in ARM_ROLES:
        role_states = raw.get(role, {})
        for side in ARM_SIDES:
            key = f"{role}_{side}"
            sample = role_states.get(side) if isinstance(role_states, dict) else None
            position = sample.get("position", []) if isinstance(sample, dict) else []
            fresh = bool(sample and sample.get("fresh", False))
            arms[key] = {
                "role": role,
                "side": side,
                "ok": fresh and len(position) == 7,
                "fresh": fresh,
                "age_ms": sample.get("age_ms") if isinstance(sample, dict) else None,
                "position": [round(float(value), 4) for value in position[:7]],
            }

    cameras: dict[str, dict[str, Any]] = {}
    image_meta = raw.get("images", {})
    for name in CAMERA_NAMES:
        metadata = image_meta.get(name) if isinstance(image_meta, dict) else None
        fresh = bool(metadata and metadata.get("fresh", False))
        # The bridge reports freshness metadata for every camera even when the
        # frame bytes were not requested (live cameras are served via WebRTC),
        # so "a frame exists on the bridge" is a link-health statement on its own.
        has_frame = bool(images.get(name)) or fresh
        cameras[name] = {
            "ok": has_frame,
            "fresh": fresh,
            "has_frame": has_frame,
            "age_ms": metadata.get("age_ms") if isinstance(metadata, dict) else None,
            "stamp_ns": metadata.get("stamp_ns") if isinstance(metadata, dict) else None,
            "encoding": metadata.get("encoding") if isinstance(metadata, dict) else None,
            "bytes": (
                metadata.get("bytes", 0)
                if isinstance(metadata, dict)
                else len(images.get(name, b""))
            ),
        }

    ok_count = sum(device["ok"] for device in arms.values()) + sum(
        device["ok"] for device in cameras.values()
    )
    return {
        "ok": ok_count == 7,
        "ok_count": ok_count,
        "total": 7,
        "sequence": raw.get("sequence"),
        "arms": arms,
        "cameras": cameras,
        "missing_or_stale": list(raw.get("missing_or_stale", [])),
    }


def create_app(
    config: DashboardConfig | None = None,
    manager: JobManager | None = None,
    device_monitor: DeviceMonitor | None = None,
    preview_relay: DashboardMjpegRelay | None = None,
) -> FastAPI:
    config = config or DashboardConfig.from_env()
    manager = manager or JobManager(config)
    device_monitor = device_monitor or DeviceMonitor(config)
    preview_relay = preview_relay or DashboardMjpegRelay(config)
    debug_capture = DebugCaptureWorker(
        endpoint=config.endpoint,
        debug_root=config.debug_root,
        fps=15,
        cameras=("front", "right"),
        side="right",
    )
    static_root = Path(__file__).with_name("web_static")
    app = FastAPI(title="Cobot Magic Control Center", version="1")
    app.state.dashboard_config = config
    app.state.job_manager = manager
    app.state.device_monitor = device_monitor
    app.state.preview_relay = preview_relay
    app.state.debug_capture = debug_capture
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    # The Stage-1 inventory walks the complete RLT checkpoint tree and also
    # reads each matching Stage-2 status.  On this workstation that can take
    # several seconds.  It must never run in the browser's high-frequency
    # overview polling path, otherwise a slow filesystem scan makes the whole
    # dashboard appear frozen.  Refresh it in the background and serve the
    # last complete snapshot meanwhile.
    stage1_cache: dict[str, Any] = {
        "items": [],
        "refreshed_at": 0.0,
        "refreshing": False,
    }
    stage1_cache_lock = threading.Lock()

    def _refresh_stage1_cache() -> None:
        try:
            items = _rlt_stage1_files(config)
            with stage1_cache_lock:
                stage1_cache["items"] = items
                stage1_cache["refreshed_at"] = time.monotonic()
        finally:
            with stage1_cache_lock:
                stage1_cache["refreshing"] = False

    def stage1_for_overview() -> list[dict[str, Any]]:
        with stage1_cache_lock:
            stale = time.monotonic() - float(stage1_cache["refreshed_at"]) > 15.0
            if stale and not stage1_cache["refreshing"]:
                stage1_cache["refreshing"] = True
                threading.Thread(target=_refresh_stage1_cache, daemon=True).start()
            return list(stage1_cache["items"])

    def authorize(x_cobot_token: str | None) -> None:
        if config.api_token and x_cobot_token != config.api_token:
            raise HTTPException(status_code=401, detail="invalid dashboard token")

    runtime_events = DashboardEvents(
        manager.list,
        lambda: _policy_server_status(config, manager),
        lambda: _rlt_policy_server_status(config, manager),
        lambda output: _raw_files(config, paths=[Path(output)])[0],
        robot_status=getattr(device_monitor, 'control_status', None),
    )
    app.router.on_startup.append(runtime_events.start)
    app.router.on_shutdown.append(runtime_events.stop)
    app.state.runtime_events = runtime_events

    @app.get("/api/events")
    async def runtime_event_stream(
        request: Request,
        x_cobot_token: str | None = Header(default=None),
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        authorize(x_cobot_token)
        try:
            cursor = max(0, int(last_event_id)) if last_event_id else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from exc

        def encode(event: dict[str, Any]) -> str:
            return ((f"id: {event['id']}\n" if 'id' in event else '') + f"event: {event['event']}\n"
                    f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n")

        async def stream():
            nonlocal cursor
            heartbeat = time.monotonic()
            robot_revision = None
            robot_heartbeat = 0.0
            joint_revision = None
            joint_sent_at = 0.0
            if cursor is None:
                initial = runtime_events.snapshot()
                cursor = initial["id"]
                yield encode(initial)
            while not await request.is_disconnected():
                # Latest-only priority lane, sent before task/model history.
                # No SSE id: reconnects must not replay stale control transitions.
                robot_event = runtime_events.robot_event_after(robot_revision)
                if robot_event and (robot_event['event'] == 'robot_status_changed'
                                    or time.monotonic() - robot_heartbeat >= 1):
                    robot_revision = robot_event['data']['revision']
                    robot_heartbeat = time.monotonic()
                    yield encode(robot_event)
                if time.monotonic() - joint_sent_at >= 0.5:
                    joint_reader = getattr(device_monitor, 'joint_state', None)
                    joints = joint_reader() if joint_reader else None
                    if joints and joints['revision'] != joint_revision:
                        joint_revision = joints['revision']
                        joint_sent_at = time.monotonic()
                        yield encode({'event': 'joint_state', 'data': joints})
                events = runtime_events.after(cursor)
                if events is None:
                    event = runtime_events.snapshot()
                    cursor = event["id"]
                    yield encode(event)
                else:
                    for event in events:
                        cursor = event["id"]
                        yield encode(event)
                if time.monotonic() - heartbeat >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        })

    @app.get('/api/robot/joints')
    def robot_joint_cache(x_cobot_token: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(x_cobot_token)
        reader = getattr(device_monitor, 'joint_state', None)
        return reader() if reader else {'revision': 0, 'age_ms': None, 'arms': {}}

    if config.debug_capture_enabled:
        app.router.on_startup.append(debug_capture.start)
    app.router.on_shutdown.append(preview_relay.stop)
    app.router.on_shutdown.append(debug_capture.stop)

    def start_job(
        kind: str,
        command: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return manager.start(kind, command, metadata)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def start_motion_job(
        kind: str,
        command: list[str],
        metadata: dict[str, Any],
        *,
        task: str,
        side: str,
        cameras: tuple[str, ...],
        policy_id: str = "",
    ) -> dict[str, Any]:
        """Launch control immediately while a passive recorder observes it.

        Debug capture has no control-plane dependency: its own bridge client
        samples at 15 Hz and failures never prevent an explicitly confirmed
        robot command from starting.
        """

        capture_id: str | None = None
        details: dict[str, Any] = {
            "enabled": bool(config.debug_capture_enabled),
            "sampled_fps": 15,
            "root": str(config.debug_root),
            "latest": str(config.debug_root / "latest.raw.h5"),
        }
        if config.debug_capture_enabled:
            capture_id = f"debug_{uuid.uuid4().hex[:12]}"
            try:
                policy_decision_trace = str(metadata.get("policy_decision_trace", ""))
                debug_capture.begin(
                    DebugSession(
                        job_id=capture_id,
                        motion_kind=kind,
                        task=task,
                        side=side,
                        cameras=cameras,
                        policy_id=policy_id,
                        policy_decision_trace=policy_decision_trace,
                    )
                )
                details["capture_id"] = capture_id
                details["state"] = "starting"
                if policy_decision_trace:
                    # The runner writes once per inference chunk.  It is
                    # merged into the finalized debug H5 only after motion is
                    # finished, so the passive recorder stays independent.
                    command.extend(("--policy-decision-trace", policy_decision_trace))
                    details["policy_decision_trace"] = policy_decision_trace
            except RuntimeError as exc:
                # A debug capture must never block a real-motion command. The
                # active session will still retain the latest complete task.
                capture_id = None
                details.update({"state": "unavailable", "reason": str(exc)})
        else:
            details["state"] = "disabled"
        metadata = {**metadata, "debug_capture": details}
        try:
            result = start_job(kind, command, metadata)
        except Exception:
            if capture_id is not None:
                debug_capture.cancel(capture_id)
            raise
        if capture_id is not None:
            try:
                debug_capture.attach_process(capture_id, manager.process(result["id"]))
            except (KeyError, AttributeError) as exc:
                # The process is already running. Losing this optional
                # observer attachment is diagnostic-only, never control
                # critical; leave the error visible in the job metadata.
                details.update({"state": "unavailable", "reason": str(exc)})
            else:
                details["state"] = "recording"
        return result

    def prepare_record(request: RecordRequest) -> tuple[Path, Path, list[str]]:
        collection = config.raw_root / request.collection
        collection.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = collection / f"episode_{stamp}_{time.time_ns() % 1_000_000:06d}.raw.h5"
        stop_file = output.with_suffix(output.suffix + ".result")
        arguments = [
            "--endpoint",
            config.endpoint,
            "--output",
            str(output),
            "--task",
            request.task,
            "--side",
            request.side,
            "--cameras",
            *request.cameras,
            "--fps",
            str(request.fps),
            "--max-seconds",
            str(request.max_seconds),
            "--policy-id",
            request.policy_id,
            "--collection-type",
            request.collection_type,
            "--stop-file",
            str(stop_file),
        ]
        if request.stop_on_teach_release:
            arguments.append("--stop-on-teach-release")
        return output, stop_file, arguments

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            static_root / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/overview")
    async def overview(
        x_cobot_token: str | None = Header(default=None),
        live_cameras: str = Query(default=""),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        live_set = {name for name in live_cameras.split(",") if name in CAMERA_NAMES}
        # Keep injected/read-only monitors usable in unit tests and lightweight
        # diagnostics that do not implement the optional live-preview hint.
        probe = (
            asyncio.to_thread(device_monitor.probe, skip_images=live_set)
            if live_set
            else asyncio.to_thread(device_monitor.probe)
        )
        (
            bridge,
            raws,
            datasets,
            lerobot_replay_datasets,
            policies,
            policy_server,
            rlt_policy_server,
        ) = await asyncio.gather(
            probe,
            asyncio.to_thread(_raw_files, config),
            asyncio.to_thread(_dataset_files, config),
            asyncio.to_thread(_lerobot_replay_dataset_files, config),
            asyncio.to_thread(_policy_files, config),
            asyncio.to_thread(_policy_server_status, config, manager),
            asyncio.to_thread(_rlt_policy_server_status, config, manager),
        )
        rlt_stage1 = stage1_for_overview()
        capabilities = _capabilities(config)
        stage2_ready = bool(capabilities["rlt_stage2"]["ready"] and rlt_stage1)
        active_rlt = next(
            (
                job
                for job in manager.list()
                if job["kind"] == "rlt_online_execute"
                and job["state"] in ("running", "stopping")
            ),
            None,
        )
        return {
            "bridge": bridge,
            "raws": raws,
            "datasets": datasets,
            "lerobot_replay_datasets": lerobot_replay_datasets,
            "policies": policies,
            "policy_server": policy_server,
            "debug_capture": debug_capture.snapshot(),
            "rlt": {
                "stage1_checkpoints": rlt_stage1,
                "policy_server": rlt_policy_server,
                "online_stage2_ready": stage2_ready,
                "online_stage2_reason": (
                    "Cobot Magic Stage 2 adapter ready; real motion still requires explicit confirmation"
                    if stage2_ready
                    else "Stage 1 checkpoint or Cobot Magic Stage 2 adapter is unavailable"
                ),
                "active_online_job": active_rlt,
            },
            "jobs": manager.list(),
            "capabilities": capabilities,
            "paths": {
                "raw_root": str(config.raw_root),
                "dataset_root": str(config.dataset_root),
                "lerobot_replay_root": str(config.lerobot_replay_root),
                "checkpoint_root": str(config.checkpoint_root),
                "rlt_root": str(config.rlt_root),
                "policy_server_endpoint": config.policy_server_endpoint,
                "rlt_policy_server_endpoint": config.rlt_policy_server_endpoint,
                "robot_policy_server_endpoint": config.robot_policy_server_endpoint,
                "robot_policy_workspace": str(config.robot_workspace),
                "robot_control_workspace": str(config.robot_control_workspace),
                "run_root": str(config.run_root),
                "debug_capture_root": str(config.debug_root),
            },
        }

    @app.post("/api/policy-server/start")
    def start_policy_server(
        request: PolicyServerRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.compute == "pc":
            rlt_status = _rlt_policy_server_status(config, manager)
            if rlt_status.get("ready") or rlt_status.get("state") == "loading":
                raise HTTPException(
                    status_code=409,
                    detail="stop the RLT rollout server before loading a normal PC model server",
                )
        try:
            policy_path = _resolve_policy_checkpoint(config, request.policy_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = _policy_server_status(config, manager)
        if status["ready"]:
            if (
                status.get("policy_id") == request.policy_id
                and status.get("compute") == request.compute
                and status.get("inference_chunking", "original") == request.inference_chunking
                and (request.num_steps is None or status.get("num_steps") == request.num_steps)
            ):
                return {**status, "already_running": True}
            raise HTTPException(
                status_code=409,
                detail="another model or denoising-step setting is already loaded; stop the model server first",
            )
        try:
            selected_config = json.loads((policy_path / "config.json").read_text())
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid policy config: {exc}") from exc
        architectures = selected_config.get("architectures", [])
        is_groot = isinstance(architectures, list) and any(
            str(name).startswith("Gr00t") for name in architectures
        )
        is_smolvla = selected_config.get("type") == "smolvla"
        if request.num_steps is not None and not is_smolvla:
            raise HTTPException(
                status_code=400,
                detail="Denoising-step override is supported only for SmolVLA checkpoints",
            )
        if is_groot and request.compute == "robot":
            raise HTTPException(
                status_code=400,
                detail="GR00T 3B does not fit the robot GPU; select PC GPU",
            )
        if request.inference_chunking == "guided_rtc":
            if request.compute != "pc":
                raise HTTPException(
                    status_code=400,
                    detail="Guided RTC initial rollout is available on PC compute only",
                )
            if not is_smolvla:
                raise HTTPException(
                    status_code=400,
                    detail="Guided RTC requires a local SmolVLA flow-matching checkpoint",
                )
        if request.compute == "robot":
            command = [
                str(config.python),
                "-m",
                "cobot_bridge.remote_policy_server",
                "--ssh-host",
                config.robot_ssh_host,
                "--local-workspace",
                str(config.workspace),
                "--remote-workspace",
                str(config.robot_workspace),
                "--local-policy-path",
                str(policy_path),
                "--policy-id",
                request.policy_id,
                "--remote-python",
                str(config.robot_python),
                "--bind",
                config.robot_policy_server_bind,
                "--device",
                request.device,
                "--cameras",
                "front",
                "right",
            ]
            if request.num_steps is not None:
                command.extend(("--num-steps", str(request.num_steps)))
        elif is_groot:
            command = [
                str(config.groot_python),
                "-m",
                "cobot_bridge.groot_policy_server",
                "--bind",
                config.policy_server_endpoint,
                "--model-path",
                str(policy_path),
                "--device",
                request.device,
                "--cameras",
                "front",
                "right",
            ]
            if config.groot_backbone_path is not None:
                command.extend(("--backbone-path", str(config.groot_backbone_path)))
        else:
            command = [
                str(config.python),
                "-m",
                "cobot_bridge.policy_server",
                "--bind",
                config.policy_server_endpoint,
                "--policy-path",
                str(policy_path),
                "--device",
                request.device,
                "--cameras",
                "front",
                "right",
            ]
            if request.inference_chunking == "guided_rtc":
                command.extend(
                    (
                        "--inference-chunking",
                        "guided_rtc",
                        "--rtc-execution-horizon",
                        "1",
                    )
                )
            if request.num_steps is not None:
                command.extend(("--num-steps", str(request.num_steps)))
            residual_dirs: dict[str, str] = {}
            for flag, raw in (("--residual-actor", request.residual_actor), ("--z-encoder", request.z_encoder)):
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = (config.workspace / path).resolve()
                if not path.is_dir():
                    raise HTTPException(status_code=400, detail=f"{flag} directory not found: {path}")
                command.extend((flag, str(path)))
                residual_dirs[flag.lstrip("-")] = str(path)
            if bool(request.residual_actor) != bool(request.z_encoder):
                raise HTTPException(status_code=400, detail="residual_actor and z_encoder must be provided together")
        return start_job(
            "policy_server",
            command,
            {
                "policy_id": request.policy_id,
                "compute": request.compute,
                "endpoint": _policy_endpoint(config, request.compute),
                "inference_chunking": request.inference_chunking,
                "num_steps": request.num_steps,
                "residual_actor": request.residual_actor if request.compute == "pc" else None,
                "z_encoder": request.z_encoder if request.compute == "pc" else None,
            },
        )

    @app.post("/api/policy-server/stop")
    def stop_policy_server(x_cobot_token: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(x_cobot_token)
        status = _policy_server_status(config, manager)
        endpoint = str(status.get("endpoint") or config.policy_server_endpoint)
        client = PersistentPolicyClient(endpoint, timeout_s=1.0)
        try:
            result = client.shutdown()
            return {**result, "endpoint": endpoint, "compute": status.get("compute")}
        except (PolicyServerError, OSError, ValueError) as exc:
            active = next(
                (
                    job
                    for job in manager.list()
                    if job["kind"] == "policy_server"
                    and job["state"] in ("running", "stopping")
                ),
                None,
            )
            if active:
                return manager.stop(active["id"])
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            client.close()

    @app.get("/api/cameras/{name}")
    def camera_frame(
        name: Literal["front", "left", "right"],
        x_cobot_token: str | None = Header(default=None),
    ) -> Response:
        authorize(x_cobot_token)
        cached = device_monitor.image(name)
        if cached is None:
            raise HTTPException(status_code=503, detail=f"camera frame unavailable: {name}")
        data, media_type = cached
        return Response(
            content=data,
            media_type=media_type,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/live/mjpeg")
    def live_mjpeg(
        camera: Literal["front", "left", "right"] = Query(),
        token: str = Query(default=""),
        x_cobot_token: str | None = Header(default=None),
    ) -> StreamingResponse:
        # An <img> cannot attach the dashboard's header token. The optional
        # same-origin query token keeps authenticated previews working without
        # exposing a second unauthenticated endpoint.
        authorize(x_cobot_token or token)
        return StreamingResponse(
            preview_relay.stream(camera),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs")
    def jobs(x_cobot_token: str | None = Header(default=None)) -> list[dict[str, Any]]:
        authorize(x_cobot_token)
        return manager.list()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str, x_cobot_token: str | None = Header(default=None)) -> str:
        authorize(x_cobot_token)
        try:
            return manager.log(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown job") from exc

    @app.delete("/api/jobs/{job_id}")
    def stop_job(job_id: str, x_cobot_token: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(x_cobot_token)
        try:
            return manager.stop(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown job") from exc

    def resolve_rlt_checkpoint(checkpoint_id: str) -> Path:
        try:
            checkpoint = _within(config.rlt_root / "checkpoints", checkpoint_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise HTTPException(status_code=400, detail="RLT checkpoint must be a .pt file")
        return checkpoint

    def resolve_rlt_actor_checkpoint(checkpoint: Path, actor_checkpoint_id: str) -> Path:
        try:
            actor_checkpoint = _within(
                config.rlt_root / "checkpoints", actor_checkpoint_id
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not actor_checkpoint.is_file() or actor_checkpoint.suffix != ".pt":
            raise HTTPException(status_code=400, detail="Actor checkpoint must be a .pt file")

        import torch

        try:
            payload = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"cannot load Actor checkpoint: {exc}"
            ) from exc
        if (
            payload.get("stage") not in SUPPORTED_ACTOR_CHECKPOINT_STAGES
            or payload.get("actor_output_semantics") not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
            or payload.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE
            or str(Path(payload.get("stage1_checkpoint", "")).expanduser().resolve())
            != str(checkpoint.resolve())
            or not isinstance(payload.get("actor_state_dict"), dict)
        ):
            raise HTTPException(
                status_code=400,
                detail="Actor checkpoint failed Stage 1 and output-semantics compatibility checks",
            )
        return actor_checkpoint

    def rlt_stage2_output(checkpoint: Path) -> Path:
        run_name = checkpoint.parent.name
        return checkpoint.parent.with_name(f"{run_name}_stage2")

    def atomic_copy(source: Path, destination: Path) -> None:
        temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
        try:
            with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                while chunk := input_stream.read(1024 * 1024):
                    output_stream.write(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @app.post("/api/rlt-stage2/rollback-30k")
    def rollback_rlt_stage2(
        request: RLTRollbackRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Atomically restore the validated 30k actor without sending robot actions."""

        authorize(x_cobot_token)
        checkpoint = resolve_rlt_checkpoint(request.checkpoint_id)
        stage2_output = rlt_stage2_output(checkpoint)
        target = stage2_output / f"checkpoint_{request.target_updates:08d}.pt"
        latest = stage2_output / "latest.pt"
        if not target.is_file():
            raise HTTPException(status_code=404, detail="validated 30k Stage 2 checkpoint is unavailable")
        if any(
            job["kind"] == "rlt_online_execute" and job["state"] in ("running", "stopping")
            for job in manager.list()
        ):
            raise HTTPException(status_code=409, detail="finish or abort the active RLT episode first")
        server = _rlt_policy_server_status(config, manager)
        if server.get("active_episode", False):
            raise HTTPException(status_code=409, detail="the RLT server still owns an active episode")
        probe = device_monitor.probe()
        if probe.get("online", False) and (
            probe.get("teach_active", False) or probe.get("output_enabled", False)
        ):
            raise HTTPException(
                status_code=409,
                detail="rollback requires Teach off and robot output off",
            )

        import torch

        try:
            payload = torch.load(target, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"cannot load 30k checkpoint: {exc}") from exc
        if (
            payload.get("stage") != "rlt_stage2"
            or payload.get("actor_output_semantics") not in SUPPORTED_ACTOR_OUTPUT_SEMANTICS
            or payload.get("action_chunk_timebase") != ACTION_CHUNK_TIMEBASE
            or int(payload.get("total_updates", -1)) != request.target_updates
            or str(Path(payload.get("stage1_checkpoint", "")).expanduser().resolve())
            != str(checkpoint.resolve())
        ):
            raise HTTPException(status_code=400, detail="30k checkpoint failed Stage 2 compatibility checks")

        stage2_output.mkdir(parents=True, exist_ok=True)
        lock_path = stage2_output / ".train.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HTTPException(status_code=409, detail="Stage 2 training is currently running") from exc

            previous_updates = 0
            try:
                previous_status = json.loads((stage2_output / "status.json").read_text())
                previous_updates = int(previous_status.get("total_updates", 0))
            except (OSError, ValueError, TypeError):
                previous_status = {}

            trained_values = payload.get("trained_replay_names")
            if trained_values is None:
                trained_values = payload.get("trained_replay_paths", [])
            trained_replay_names = {Path(str(value)).name for value in trained_values}
            replay_root = config.rlt_root / "replay" / checkpoint.parent.name
            extra_replays = (
                sorted(
                    path
                    for path in replay_root.glob("*.npz")
                    if path.name not in trained_replay_names
                )
                if trained_replay_names and replay_root.is_dir()
                else []
            )
            quarantine = None
            quarantined_files: list[str] = []
            if extra_replays:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                quarantine = replay_root.with_name(
                    f"{replay_root.name}_rollback_after_30000_{stamp}_{uuid.uuid4().hex[:6]}"
                )
                quarantine.mkdir(parents=True)
                for replay in extra_replays:
                    stem = replay.name.removesuffix(".npz")
                    for source in (
                        replay_root / f"{stem}.decisions.h5",
                        replay,
                        replay_root / f"{stem}.npz.json",
                    ):
                        if source.is_file():
                            os.replace(source, quarantine / source.name)
                            quarantined_files.append(source.name)
                atomic_json(
                    quarantine / "rollback_manifest.json",
                    {
                        "stage1_checkpoint": str(checkpoint.resolve()),
                        "target_checkpoint": str(target.resolve()),
                        "target_updates": request.target_updates,
                        "previous_updates": previous_updates,
                        "quarantined_replays": [path.name for path in extra_replays],
                        "quarantined_files": quarantined_files,
                        "created_at_ns": time.time_ns(),
                    },
                )
            if previous_updates == request.target_updates and latest.is_file():
                already_active = True
                backup = None
            else:
                already_active = False
                backup = None
                if latest.is_file():
                    archive = stage2_output / "rollback_archive"
                    archive.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    backup = archive / f"latest_before_rollback_{previous_updates:08d}_{stamp}.pt"
                    atomic_copy(latest, backup)
                atomic_copy(target, latest)

            validation = dict(payload.get("actor_validation", {}))
            status = {
                "stage": "rlt_stage2",
                "actor_output_semantics": ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
                "checkpoint": str(target.resolve()),
                "latest": str(latest.resolve()),
                "stage1_checkpoint": str(checkpoint.resolve()),
                "buffer_transitions": int(payload.get("buffer_transitions", 0)),
                "new_transitions": int((payload.get("history") or [{}])[-1].get("new_transitions", 0)),
                "updates_this_run": int((payload.get("history") or [{}])[-1].get("updates", 0)),
                "total_updates": request.target_updates,
                "actor_updates_this_run": int(payload.get("actor_updates_this_run", 0)),
                "last_metrics": dict(payload.get("last_metrics", {})),
                "actor_validation": validation,
                "actor_warnings": list(payload.get("actor_warnings", [])),
                "training_phase": str(payload.get("training_phase", "unknown")),
                "ready_for_online": True,
                "actor_exploration_std": float(payload.get("actor_exploration_std", 0.0)),
                "actor_action_scale": list(payload.get("actor_action_scale", [])),
                "rollback_from_total_updates": previous_updates,
                "rolled_back_at_ns": time.time_ns(),
            }
            atomic_json(stage2_output / "status.json", status)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        reloaded = False
        loaded_updates = None
        if server.get("ready", False) and server.get("checkpoint_id") == request.checkpoint_id:
            try:
                with PersistentRLTPolicyClient(
                    config.rlt_policy_server_endpoint, timeout_s=30.0
                ) as client:
                    boundary = client.begin_episode(
                        workflow_mode=1,
                        allow_actor=False,
                        actor_exploration_std=0.0,
                    )
                    ended = client.end_episode()
                reloaded = bool(boundary.get("actor_reloaded", False))
                loaded_updates = int(ended.get("actor_total_updates", 0))
            except (RLTPolicyServerError, OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"30k was published, but the resident server did not reload it: {exc}",
                ) from exc
        return {
            "rolled_back": not already_active,
            "already_active": already_active,
            "target_updates": request.target_updates,
            "previous_updates": previous_updates,
            "checkpoint": str(target.resolve()),
            "backup": str(backup.resolve()) if backup else None,
            "server_reloaded": reloaded,
            "server_loaded_updates": loaded_updates,
            "quarantined_replays": len(extra_replays),
            "quarantine": str(quarantine.resolve()) if quarantine else None,
            "motion": False,
        }

    @app.post("/api/rlt-policy-server/start")
    def start_rlt_policy_server(
        request: RLTPolicyServerRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        checkpoint = resolve_rlt_checkpoint(request.checkpoint_id)
        probe = device_monitor.probe()
        if not probe.get("online", False):
            raise HTTPException(status_code=503, detail="robot bridge is offline")
        if not probe.get("devices", {}).get("ok", False):
            raise HTTPException(
                status_code=409, detail="all four arms and three cameras must be online"
            )
        normal_server = _policy_server_status(config, manager)
        if normal_server.get("compute") == "pc" and (
            normal_server.get("ready") or normal_server.get("state") == "loading"
        ):
            raise HTTPException(
                status_code=409, detail="stop the normal PC model server first"
            )
        if any(
            job["kind"] in ("rlt_shadow", "rlt_online_execute")
            and job["state"] in ("running", "stopping")
            for job in manager.list()
        ):
            raise HTTPException(
                status_code=409, detail="stop RLT shadow/rollout before changing the server"
            )
        stage2_latest = (
            resolve_rlt_actor_checkpoint(checkpoint, request.actor_checkpoint_id)
            if request.actor_checkpoint_id
            else rlt_stage2_output(checkpoint) / "latest.pt"
        )
        if not request.actor_checkpoint_id:
            if not stage2_latest.is_file():
                raise HTTPException(status_code=404, detail="Stage 2 Actor checkpoint is unavailable")
            # Keep old API callers compatible, but apply the same strict check.
            stage2_latest = resolve_rlt_actor_checkpoint(
                checkpoint,
                stage2_latest.resolve().relative_to(
                    (config.rlt_root / "checkpoints").resolve()
                ).as_posix(),
            )
        # Re-check after resolving the requested Actor so a future status
        # implementation can safely return an idempotent result.
        status = _rlt_policy_server_status(config, manager)
        requested_actor_id = stage2_latest.relative_to(
            (config.rlt_root / "checkpoints").resolve()
        ).as_posix()
        if status.get("ready"):
            if (
                status.get("checkpoint_id") == request.checkpoint_id
                and status.get("actor_checkpoint_id") == requested_actor_id
            ):
                return {**status, "already_running": True}
            raise HTTPException(
                status_code=409,
                detail="another Stage 1 or Actor is loaded; stop the RLT server first",
            )
        if not status.get("job_id") and _tcp_endpoint_in_use(config.rlt_policy_server_endpoint):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{config.rlt_policy_server_endpoint} is still held by a process this "
                    "dashboard does not track, so a new server could not bind it. "
                    "Wait a few seconds and retry, or find the holder with "
                    f"ss -ltnp 'sport = :{config.rlt_policy_server_endpoint.rsplit(':', 1)[-1]}'."
                ),
            )
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.rlt_policy_server",
            "--bind",
            config.rlt_policy_server_endpoint,
            "--stage1-checkpoint",
            str(checkpoint),
            "--stage2-checkpoint",
            str(stage2_latest),
            "--device",
            request.device,
            "--cameras",
            "front",
            "right",
        ]
        return start_job(
            "rlt_policy_server",
            command,
            {
                "checkpoint_id": request.checkpoint_id,
                "stage2_checkpoint": str(stage2_latest),
                "actor_checkpoint_id": requested_actor_id,
                "device": request.device,
                "endpoint": config.rlt_policy_server_endpoint,
                "motion": False,
            },
        )

    @app.post("/api/rlt-policy-server/stop")
    def stop_rlt_policy_server(
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if any(
            job["kind"] == "rlt_online_execute"
            and job["state"] in ("running", "stopping")
            for job in manager.list()
        ):
            raise HTTPException(
                status_code=409,
                detail="finish or abort the active RLT episode before stopping its server",
            )
        status = _rlt_policy_server_status(config, manager)
        if status.get("active_episode", False):
            raise HTTPException(
                status_code=409,
                detail="finish or abort the active RLT episode before stopping its server",
            )
        client = PersistentRLTPolicyClient(config.rlt_policy_server_endpoint, timeout_s=1.0)
        try:
            return client.shutdown()
        except (RLTPolicyServerError, OSError, ValueError) as exc:
            active = next(
                (
                    job
                    for job in manager.list()
                    if job["kind"] == "rlt_policy_server"
                    and job["state"] in ("running", "stopping")
                ),
                None,
            )
            if active:
                return manager.stop(active["id"])
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            client.close()

    @app.post("/api/jobs/rlt-validate")
    def validate_rlt_checkpoint(
        request: RLTCheckpointRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        checkpoint = resolve_rlt_checkpoint(request.checkpoint_id)
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.rlt_validate",
            "--rlt-checkpoint",
            str(checkpoint),
        ]
        return start_job(
            "rlt_validate",
            command,
            {"checkpoint_id": request.checkpoint_id, "motion": False},
        )

    @app.post("/api/jobs/rlt-shadow")
    def start_rlt_shadow(
        request: RLTShadowRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        checkpoint = resolve_rlt_checkpoint(request.checkpoint_id)
        diagnostic_output = config.run_root / f"rlt_shadow_{uuid.uuid4().hex[:12]}.jsonl"
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.rlt_shadow",
            "--endpoint",
            config.endpoint,
            "--rlt-checkpoint",
            str(checkpoint),
            "--task",
            request.task,
            "--side",
            request.side,
            "--fps",
            str(request.fps),
            "--duration",
            str(request.duration),
            "--device",
            request.device,
            "--cameras",
            "front",
            "right",
            "--diagnostic-output",
            str(diagnostic_output),
        ]
        return start_job(
            "rlt_shadow",
            command,
            {
                "checkpoint_id": request.checkpoint_id,
                "task": request.task,
                "side": request.side,
                "fps": request.fps,
                "duration": request.duration,
                "device": request.device,
                "diagnostic_output": str(diagnostic_output),
                "motion": False,
                "send_actions": False,
            },
        )

    @app.post("/api/jobs/rlt-online")
    def start_rlt_online(
        request: RLTOnlineRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.rollout_phase == "reference" and request.workflow_mode != 1:
            raise HTTPException(
                status_code=400, detail="reference-only rollout requires workflow Mode 1"
            )
        if not request.execute or not request.confirm_motion:
            raise HTTPException(
                status_code=400,
                detail="Online RLT real motion requires execute=true and confirm_motion=true",
            )
        checkpoint = resolve_rlt_checkpoint(request.checkpoint_id)
        probe = device_monitor.probe()
        if not probe.get("online", False):
            raise HTTPException(status_code=503, detail="robot bridge is offline")
        if not probe.get("allow_control", False):
            raise HTTPException(status_code=409, detail="robot bridge does not allow control")
        if not probe.get("devices", {}).get("ok", False):
            raise HTTPException(status_code=409, detail="all four arms and three cameras must be online")
        if probe.get("teach_active", False):
            raise HTTPException(status_code=409, detail="release Teach before Online RLT")
        if probe.get("emergency_latched", False):
            raise HTTPException(status_code=409, detail="click EXIT STOP before Online RLT")
        rlt_server = _rlt_policy_server_status(config, manager)
        if not rlt_server.get("ready", False):
            raise HTTPException(status_code=409, detail="start the RLT rollout server first")
        if rlt_server.get("checkpoint_id") != request.checkpoint_id:
            raise HTTPException(
                status_code=409,
                detail="selected Stage 1 does not match the RLT rollout server",
            )
        if rlt_server.get("active_episode", False):
            raise HTTPException(status_code=409, detail="the RLT server already owns an episode")
        if (
            rlt_server.get("action_space") == "eef_target"
            and request.smooth_profile not in ("edge_v4", "raw")
        ):
            raise HTTPException(
                status_code=400,
                detail="relative-EEF RLT supports Edge V4 or Raw (Pink IK then bridge interpolation)",
            )

        progress = _rlt_progress_for_checkpoint(config, checkpoint)
        # The persistent server may intentionally use a compatible Actor from a
        # non-canonical directory (for example an Actor-BC or offline-evaluation
        # checkpoint selected in the UI).  In that case ``<stage1>_stage2/latest.pt``
        # is not the source of truth.  Validate the exact Actor locked into the
        # resident server instead of incorrectly rejecting it because the
        # canonical Stage 2 alias is absent.
        actor_enabled = request.rollout_phase == "actor"
        server_actor_path: Path | None = None
        if actor_enabled:
            server_actor_id = rlt_server.get("actor_checkpoint_id")
            if server_actor_id:
                server_actor_path = resolve_rlt_actor_checkpoint(
                    checkpoint, str(server_actor_id)
                )
            elif not progress["actor_ready"]:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "actor is not ready: train Stage 2 once from available replay; "
                        f"published updates={progress['total_updates']}"
                    ),
                )

        collection = config.raw_root / request.collection
        collection.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        token = f"{stamp}_{time.time_ns() % 1_000_000:06d}"
        stem = f"rlt_{request.rollout_phase}_mode{request.workflow_mode}_{token}"
        raw_output = collection / f"{stem}.raw.h5"
        result_file = raw_output.with_suffix(raw_output.suffix + ".result")
        run_name = checkpoint.parent.name
        replay_root = config.rlt_root / "replay" / run_name
        pipeline_status = config.run_root / f"{stem}.pipeline.json"
        pipeline_log = config.run_root / f"{stem}.pipeline.log"
        stage2_output = rlt_stage2_output(checkpoint)
        actor_read_only = bool(actor_enabled and rlt_server.get("actor_read_only", False))
        actor_lineage_missing = False
        # The persistent server adopts an actor at BEGIN_EPISODE, so immediately
        # after server startup its ping may not yet expose the checkpoint stage.
        # Inspect the exact actor locked into the server now; otherwise a Steam
        # rollout could be incorrectly wired to the asynchronous trainer before
        # the server reaches its episode boundary.
        actor_payload: dict[str, Any] | None = None
        if actor_enabled and server_actor_path is not None:
            import torch

            try:
                actor_payload = torch.load(
                    server_actor_path, map_location="cpu", weights_only=False
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot inspect the selected Actor checkpoint: {exc}",
                ) from exc
            actor_read_only = actor_read_only or (
                isinstance(actor_payload, dict)
                and actor_payload.get("stage") == STEAM_OFFLINE_ACTOR_STAGE
            )
            # Online learning must continue the exact selected actor lineage,
            # rather than silently creating the historical canonical TD3 path.
            stage2_output = server_actor_path.parent
            try:
                actor_status = json.loads((stage2_output / "status.json").read_text())
                replay_root = Path(str(actor_status["replay_root"])).expanduser().resolve()
            except (OSError, ValueError, TypeError, KeyError):
                # A freshly BC-warmed Direct Actor has no prior Stage-2 replay
                # lineage yet. It must still be executable for controlled
                # evaluation/data collection; only its asynchronous update is
                # disabled until a lineage is explicitly initialized.
                actor_lineage_missing = True
        replay_output = replay_root / f"{stem}.npz"
        decision_output = replay_root / f"{stem}.decisions.h5"
        inference_only = bool(
            actor_read_only
            or actor_lineage_missing
            or
            next(
                (
                    item.get("inference_only", False)
                    for item in _rlt_stage1_files(config)
                    if item.get("id") == request.checkpoint_id
                ),
                False,
            )
        )
        archive_root = config.dataset_root / "rlt_archive" / stem
        archive_repo_id = f"local/{stem}"
        recap_checkpoint = config.rlt_root / "checkpoints/recap_value_smolvla_073rel/value_head_best.pt"
        if actor_enabled and not inference_only and not recap_checkpoint.is_file():
            raise HTTPException(status_code=409, detail="frozen RECAP value checkpoint is unavailable")
        actor_validation_cache = None
        if actor_payload is not None:
            init = actor_payload.get("actor_init_checkpoint")
            if init:
                candidate = Path(str(init)).expanduser().resolve().parent / "actor_bc_features.npz"
                if candidate.is_file():
                    actor_validation_cache = candidate
            if actor_validation_cache is None:
                # Zero-residual actors can be initialized directly from the
                # reference VLA while still using a phase-specific BC-only
                # cache.  New checkpoints persist it explicitly; for a
                # compatible older artifact also recover it from held-out
                # validation lineage.
                cache_value = actor_payload.get("actor_validation_cache")
                if not cache_value:
                    heldout = actor_payload.get("heldout_validation")
                    if isinstance(heldout, dict):
                        cache_value = heldout.get("cache")
                if cache_value:
                    candidate = Path(str(cache_value)).expanduser().resolve()
                    if candidate.is_file() and candidate.with_suffix(
                        candidate.suffix + ".json"
                    ).is_file():
                        actor_validation_cache = candidate
        if actor_enabled and not inference_only and actor_validation_cache is None:
            # Actor-BC-free actors (zero-init VLA passthrough) legitimately
            # carry no IL validation cache. The trainer only requires the cache
            # when the actor's own recorded recipe enables the IL anchor
            # (rlt_stage2_train raises unless il_anchor_weight > 0), so gate on
            # the recorded recipe instead of on the artifact's mere presence.
            il_anchor_needed = True
            if isinstance(actor_payload, dict):
                _recorded_il = actor_payload.get("il_anchor_weight")
                if isinstance(_recorded_il, (int, float)):
                    il_anchor_needed = float(_recorded_il) > 0.0
            if il_anchor_needed:
                raise HTTPException(
                    status_code=409,
                    detail="selected actor has no IL validation cache; automatic online updates are disabled",
                )
        actor_weights = _load_rlt_actor_weights(config)
        # The selected actor's checkpoint records the exact recipe it was
        # trained with; derive the asynchronous-training flags from it so the
        # UI selection alone switches lineages.  rlt_actor_weights.json only
        # provides fallbacks for fields older checkpoints do not record.
        if actor_payload is not None:
            for _key in (
                "q_weight",
                "bc_weight",
                "il_anchor_weight",
                "intervention_bc_weight",
            ):
                _value = actor_payload.get(_key)
                if isinstance(_value, (int, float)):
                    actor_weights[_key] = float(_value)
            _sem = str(actor_payload.get("actor_output_semantics", ""))
            if _sem:
                actor_weights["actor_output_semantics"] = (
                    "direct" if _sem.startswith("direct") else "residual"
                )
            _aw = actor_payload.get("ref_anchor_weight")
            actor_weights["ref_anchor_weight"] = max(
                0.5, float(_aw) if _aw is not None else 0.5
            )
            actor_weights["ref_anchor_mask"] = "none"
            _stats = actor_payload.get("action_stats_path") or actor_weights.get(
                "action_stats"
            )
            actor_weights["action_stats"] = str(_stats) if _stats else None
            if (
                actor_weights["ref_anchor_mask"] != "none"
                and actor_weights["action_stats"] is None
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "selected actor uses a reference-box anchor but no action "
                        f"stats path is available; set action_stats in {RLT_ACTOR_WEIGHTS_FILENAME}"
                    ),
                )
        actor_start_step = None
        if isinstance(actor_payload, dict):
            recorded_start = actor_payload.get("actor_start_step")
            if isinstance(recorded_start, (int, float)):
                actor_start_step = int(recorded_start)
        takeover_decisions = (
            _actor_takeover_decisions_for_checkpoint(
                server_actor_path,
                actor_start_step,
                int(rlt_server.get("reference_action_steps", 0) or 0),
                float(request.rollout_speed),
            )
            if actor_enabled
            else 0
        )
        # Actor rollouts must pass through the PC-side velocity/acceleration
        # shaper. A raw actor chunk can reproduce the takeover speed jump even
        # when the selected UI profile is intended for a reference trial.
        rollout_profile = "edge_v4" if actor_enabled else request.smooth_profile
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.rlt_online",
            "--endpoint",
            config.endpoint,
            "--policy-server-endpoint",
            config.rlt_policy_server_endpoint,
            "--stage1-checkpoint",
            str(checkpoint),
            "--task",
            request.task,
            "--workflow-mode",
            str(request.workflow_mode),
            "--rollout-phase",
            request.rollout_phase,
            "--side",
            "right",
            "--cameras",
            "front",
            "right",
            "--raw-output",
            str(raw_output),
            "--decision-output",
            str(decision_output),
            "--replay-output",
            str(replay_output),
            "--result-file",
            str(result_file),
            "--raw-fps",
            str(request.raw_fps),
            "--chunk-hz",
            str(request.chunk_hz),
            "--rollout-speed",
            str(request.rollout_speed),
            "--smooth-profile",
            rollout_profile,
            "--max-seconds",
            str(request.max_seconds),
            "--actor-exploration-std",
            str(
                0.0
                if request.rollout_phase == "reference" or actor_read_only
                else request.actor_exploration_std
            ),
            "--q-select",
            str(int(request.q_select_k) if not actor_read_only else 0),
            "--actor-max-decisions",
            str(request.actor_max_decisions if actor_enabled else 0),
            "--send-actions",
            "--confirm-motion",
        ]
        command.append(
            "--actor-eef-gate" if request.actor_eef_gate else "--no-actor-eef-gate"
        )
        # Current rollout calibration: expose Actor after six complete
        # reference decisions.
        command.extend(("--actor-takeover-decisions", "6"))
        if actor_enabled and not inference_only:
            command.extend(
                [
                    "--stage2-output-dir",
                    str(stage2_output),
                    "--archive-root",
                    str(archive_root),
                    "--archive-repo-id",
                    archive_repo_id,
                    "--trainer-device",
                    "cuda:1",
                    "--pipeline-status-file",
                    str(pipeline_status),
                    "--pipeline-log",
                    str(pipeline_log),
                    "--recap-checkpoint",
                    str(recap_checkpoint),
                    "--recap-device",
                    "cuda:1",
                    "--il-anchor-weight",
                    str(actor_weights["il_anchor_weight"]),
                    "--q-weight",
                    str(actor_weights["q_weight"]),
                    "--bc-weight",
                    str(actor_weights["bc_weight"]),
                    "--intervention-bc-weight",
                    str(actor_weights.get("intervention_bc_weight", 1.0)),
                    "--actor-output-semantics",
                    str(actor_weights.get("actor_output_semantics", "residual")),
                    "--ref-anchor-weight",
                    str(actor_weights.get("ref_anchor_weight", 0.0)),
                    "--ref-anchor-mask",
                    str(actor_weights.get("ref_anchor_mask", "none")),
                ]
            )
            if actor_weights.get("action_stats"):
                command.extend(["--action-stats", str(actor_weights["action_stats"])])
            if actor_validation_cache is not None:
                command.extend(
                    ["--actor-validation-cache", str(actor_validation_cache)]
                )
        if actor_enabled:
            command.append("--enable-actor")
        result = start_motion_job(
            "rlt_online_execute",
            command,
            {
                "motion": True,
                "workflow_mode": request.workflow_mode,
                "rollout_phase": request.rollout_phase,
                "checkpoint_id": request.checkpoint_id,
                "raw_output": str(raw_output),
                "decision_output": str(decision_output),
                "replay_output": str(replay_output),
                "result_file": str(result_file),
                "stage2_output": str(stage2_output),
                "archive_root": str(archive_root),
                "pipeline_status_file": str(pipeline_status),
                "pipeline_log": str(pipeline_log),
                "rollout_server_endpoint": config.rlt_policy_server_endpoint,
                "rollout_device": str(rlt_server.get("device", "cuda:0")),
                "actor_generation": int(rlt_server.get("actor_generation", 0)),
                "trainer_device": "cuda:1",
                "actor_enabled": actor_enabled,
                "actor_read_only": actor_read_only,
                "actor_eef_gate": bool(request.actor_eef_gate),
                "actor_start_step": actor_start_step,
                "actor_takeover_decisions": takeover_decisions,
                "actor_max_decisions": (
                    request.actor_max_decisions if actor_enabled else 0
                ),
                "smooth_profile": rollout_profile,
                "rollout_speed": float(request.rollout_speed),
                "training_requested": bool(actor_enabled and not inference_only),
                "inference_only": inference_only,
                "actor_lineage_missing": actor_lineage_missing,
                "actor_weights": actor_weights,
            },
            task=request.task,
            side="right",
            cameras=("front", "right"),
            policy_id=request.checkpoint_id,
        )
        result["actor_enabled"] = actor_enabled
        result["rollout_phase"] = request.rollout_phase
        result["inference_only"] = inference_only
        result["training_requested"] = bool(actor_enabled and not inference_only)
        return result

    def _rlt_result(job_id: str, outcome: str) -> dict[str, Any]:
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") != "rlt_online_execute":
            raise HTTPException(status_code=404, detail="active Online RLT episode not found")
        if job.get("state") != "running":
            raise HTTPException(status_code=409, detail="Online RLT episode is not running")
        result_value = (job.get("metadata") or {}).get("result_file")
        if not result_value:
            raise HTTPException(status_code=500, detail="Online RLT result file is missing")
        result_path = Path(result_value)
        if result_path.exists():
            raise HTTPException(status_code=409, detail="episode result was already submitted")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_name(result_path.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(outcome)
        os.replace(temporary, result_path)
        try:
            with CobotBridgeClient(config.endpoint, timeout_s=2.0) as client:
                client.set_output(False)
                client.set_mode(0)
        except Exception:
            pass
        return {
            "job_id": job_id,
            "outcome": outcome,
            "pipeline": (
                "discard"
                if outcome == "discard"
                else "replay+train+archive queued"
            ),
        }

    @app.post("/api/jobs/{job_id}/finalize-rlt")
    def finalize_rlt(
        job_id: str,
        request: FinalizeRecordRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        return _rlt_result(job_id, request.outcome)

    @app.post("/api/jobs/{job_id}/abort-rlt")
    def abort_rlt(
        job_id: str,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        return _rlt_result(job_id, "discard")

    @app.post("/api/jobs/{job_id}/intervention-phase")
    def intervention_phase(
        job_id: str,
        request: InterventionPhaseRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Append an intervention-phase mark beside the episode's result file.

        The dashboard records one JSON line per phase switch into
        ``<raw-stem>.phases.jsonl`` next to the episode ``.result`` file.
        Offline reconstruction/provenance attributes each human transition to
        the latest mark at or before its own ``time_ns``; unmarked (legacy)
        rows keep the "unknown" phase instead of a fabricated default.
        """
        authorize(x_cobot_token)
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") != "rlt_online_execute":
            raise HTTPException(status_code=404, detail="active Online RLT episode not found")
        if job.get("state") != "running":
            raise HTTPException(status_code=409, detail="Online RLT episode is not running")
        result_value = (job.get("metadata") or {}).get("result_file")
        if not result_value:
            raise HTTPException(status_code=500, detail="Online RLT result file is missing")
        result_path = Path(result_value)
        stem = result_path.name
        if stem.endswith(".result"):
            stem = stem[: -len(".result")]
        sidecar = result_path.parent / f"{stem}.phases.jsonl"
        record = {"t_ns": time.time_ns(), "phase": request.phase}
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return {"ok": True, "path": str(sidecar), **record}

    def _rlt_sidecar_and_segments(job: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
        metadata = job.get("metadata") or {}
        raw_value = metadata.get("raw_output")
        decision_value = metadata.get("decision_output")
        if not raw_value:
            raise HTTPException(status_code=500, detail="Online RLT raw output path is missing")
        sidecar = Path(f"{raw_value}.phases.jsonl")
        if not decision_value or not Path(decision_value).is_file():
            raise HTTPException(status_code=409, detail="decision stream not available yet")
        import h5py

        with h5py.File(decision_value, "r") as stream:
            human = stream["is_intervention"][:].astype(bool)
            times = stream["time_ns"][:].astype("int64")
        segments: list[dict[str, Any]] = []
        index, total = 0, len(human)
        while index < total:
            if not human[index]:
                index += 1
                continue
            end = index
            while end < total and human[end]:
                end += 1
            t0 = int(times[index])
            t1 = int(times[end]) if end < total else int(times[-1])
            segments.append({
                "index": len(segments),
                "t0_ns": t0,
                "t1_ns": t1,
                "start_s": round((t0 - int(times[0])) / 1e9, 1) if total else 0.0,
                "dur_s": round((t1 - t0) / 1e9, 1),
            })
            index = end
        return sidecar, segments

    @app.get("/api/jobs/{job_id}/teach-segments")
    def teach_segments(
        job_id: str,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Detected Teach intervention segments for post-episode phase labels."""
        authorize(x_cobot_token)
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") != "rlt_online_execute":
            raise HTTPException(status_code=404, detail="Online RLT episode not found")
        _, segments = _rlt_sidecar_and_segments(job)
        return {"job_id": job_id, "state": job.get("state"), "segments": segments}

    @app.post("/api/jobs/{job_id}/teach-labels")
    def teach_labels(
        job_id: str,
        request: TeachLabelsRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Store interval phase labels for teach segments (replaces prior set).

        Labels are interval records appended to the episode's phases sidecar;
        point marks written live are preserved.  Attribution happens when
        replay is loaded, so labelling may occur any time before training.
        """
        authorize(x_cobot_token)
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") != "rlt_online_execute":
            raise HTTPException(status_code=404, detail="Online RLT episode not found")
        sidecar, segments = _rlt_sidecar_and_segments(job)
        by_index = {segment["index"]: segment for segment in segments}
        intervals = []
        for label in request.labels:
            segment = by_index.get(label.index)
            if segment is None:
                raise HTTPException(status_code=409, detail=f"teach segment {label.index} no longer exists")
            intervals.append({"t0_ns": segment["t0_ns"], "t1_ns": segment["t1_ns"], "phase": label.phase, "src": "teach_labels"})
        preserved: list[str] = []
        if sidecar.is_file():
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if "t0_ns" in item and "t1_ns" in item:
                    continue  # replaced wholesale by the new label set
                preserved.append(json.dumps(item))
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        temporary = sidecar.with_name(sidecar.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text("".join(line + "\n" for line in [*preserved, *(json.dumps(record) for record in intervals)]))
        os.replace(temporary, sidecar)
        return {"ok": True, "labeled": len(intervals), "segments": len(segments), "path": str(sidecar)}

    @app.post("/api/jobs/record")
    def record(
        request: RecordRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        output, stop_file, arguments = prepare_record(request)
        command = [str(config.python), "-m", "cobot_bridge.record_raw", *arguments]
        result = start_motion_job(
            "record",
            command,
            {
                "collection_type": request.collection_type,
                "output": str(output),
                "result_file": str(stop_file),
            },
            task=request.task,
            side=request.side,
            cameras=tuple(request.cameras),
            policy_id=request.policy_id,
        )
        result["output"] = str(output)
        return result

    @app.post("/api/jobs/init-and-record")
    def init_and_record(
        request: InitAndRecordRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if not request.confirm_motion:
            raise HTTPException(status_code=400, detail="real motion requires confirm_motion=true")
        if request.collection_type != "IL":
            raise HTTPException(status_code=400, detail="init-and-record is only available for IL")
        probe = device_monitor.probe()
        if not probe.get("online", False):
            raise HTTPException(status_code=503, detail="robot bridge is offline")
        if not probe.get("allow_control", False):
            raise HTTPException(status_code=409, detail="robot bridge does not allow control")
        if not probe.get("devices", {}).get("ok", False):
            raise HTTPException(status_code=409, detail="all four arms and three cameras must be online")
        if probe.get("teach_active", False):
            raise HTTPException(status_code=409, detail="release Teach before moving to init")
        if probe.get("emergency_latched", False):
            raise HTTPException(status_code=409, detail="click EXIT STOP before moving to init")

        output, stop_file, record_arguments = prepare_record(request)
        phase_file = output.with_suffix(output.suffix + ".phase")
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.init_then_record",
            "--endpoint",
            config.endpoint,
            "--phase-file",
            str(phase_file),
            "--stop-file",
            str(stop_file),
            "--",
            *record_arguments,
        ]
        result = start_motion_job(
            "record_with_init",
            command,
            {
                "collection_type": "IL",
                "output": str(output),
                "result_file": str(stop_file),
                "phase_file": str(phase_file),
                "phase": "queued",
            },
            task=request.task,
            side=request.side,
            cameras=tuple(request.cameras),
            policy_id=request.policy_id or "human-il",
        )
        result["output"] = str(output)
        return result

    @app.post("/api/jobs/{job_id}/finalize-record")
    def finalize_record(
        job_id: str,
        request: FinalizeRecordRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") not in RECORD_JOB_KINDS:
            raise HTTPException(status_code=404, detail="active recorder not found")
        if job.get("state") != "running":
            raise HTTPException(status_code=409, detail="recorder is not running")
        metadata = job.get("metadata") or {}
        if metadata.get("phase") in ("queued", "initializing"):
            raise HTTPException(status_code=409, detail="recording has not started yet")
        collection_type = metadata.get("collection_type")
        if collection_type not in ("IL", "RL"):
            raise HTTPException(status_code=400, detail="unknown recorder collection type")
        result_file_value = metadata.get("result_file")
        if not result_file_value:
            raise HTTPException(status_code=500, detail="recorder result file is missing")
        result_file = Path(result_file_value)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        if result_file.exists():
            raise HTTPException(status_code=409, detail="recorder result was already submitted")
        temporary = result_file.with_name(result_file.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(request.outcome)
        os.replace(temporary, result_file)
        if collection_type == "RL":
            manager.stop_motion_jobs()
        return {
            "saving": True,
            "job_id": job_id,
            "collection_type": collection_type,
            "outcome": request.outcome,
        }

    @app.post("/api/jobs/{job_id}/abort-record")
    def abort_record(
        job_id: str,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        job = next((item for item in manager.list() if item["id"] == job_id), None)
        if job is None or job.get("kind") not in RECORD_JOB_KINDS:
            raise HTTPException(status_code=404, detail="active recorder not found")
        if job.get("state") != "running":
            raise HTTPException(status_code=409, detail="recorder is not running")
        metadata = job.get("metadata") or {}
        collection_type = metadata.get("collection_type")
        if collection_type not in ("IL", "RL"):
            raise HTTPException(status_code=400, detail="unknown recorder collection type")
        result_file_value = metadata.get("result_file")
        if not result_file_value:
            raise HTTPException(status_code=500, detail="recorder result file is missing")
        result_file = Path(result_file_value)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        if result_file.exists():
            raise HTTPException(status_code=409, detail="recorder result was already submitted")
        temporary = result_file.with_name(result_file.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text("discard")
        os.replace(temporary, result_file)
        if collection_type == "RL":
            manager.stop_motion_jobs()
        return {
            "discarding": True,
            "job_id": job_id,
            "collection_type": collection_type,
        }

    @app.post("/api/jobs/convert")
    def convert(
        request: ConvertRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        try:
            raw_path = _within(config.raw_root, request.raw_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        root = config.dataset_root / request.dataset_name
        if root.exists():
            raise HTTPException(status_code=409, detail=f"dataset already exists: {root}")
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.convert_raw_to_lerobot",
            "--raw",
            str(raw_path),
            "--repo-id",
            f"local/{request.dataset_name}",
            "--root",
            str(root),
            "--action-source",
            request.action_source,
            "--use-videos" if request.use_videos else "--no-use-videos",
        ]
        if request.fps is not None:
            command.extend(("--fps", str(request.fps)))
        if request.trim_stationary_holds:
            command.append("--trim-stationary-holds")
        result = start_job("convert", command)
        result["output"] = str(root)
        return result

    @app.post("/api/jobs/replay")
    def replay(
        request: ReplayRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.execute and not request.confirm_motion:
            raise HTTPException(status_code=400, detail="real motion requires confirm_motion=true")
        if request.loop and request.mode != "sync":
            raise HTTPException(status_code=400, detail="loop replay requires sync mode")
        try:
            raw_path = _within(config.raw_root, request.raw_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.replay_raw",
            "--raw",
            str(raw_path),
            "--endpoint",
            config.endpoint,
            "--side",
            request.side,
            "--source",
            request.source,
            "--mode",
            request.mode,
            "--waypoint-hz",
            str(request.waypoint_hz),
            "--speed-scale",
            str(request.speed_scale),
        ]
        if request.loop:
            command.append("--loop")
        if request.execute:
            command.append("--execute")
        if request.execute:
            try:
                with h5py.File(raw_path, "r") as raw:
                    replay_task = str(raw.attrs.get("task", f"replay: {raw_path.stem}"))
            except OSError:
                replay_task = f"replay: {raw_path.stem}"
            return start_motion_job(
                "replay_execute",
                command,
                {"raw": str(raw_path), "source": request.source, "mode": request.mode},
                task=replay_task,
                side=request.side,
                cameras=("front", request.side),
                policy_id=f"replay:{raw_path.name}",
            )
        return start_job("replay_dry_run", command)

    @app.post("/api/jobs/replay-lerobot")
    def replay_lerobot(
        request: LerobotReplayRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.execute and not request.confirm_motion:
            raise HTTPException(status_code=400, detail="real motion requires confirm_motion=true")
        try:
            dataset_path = _within(config.lerobot_replay_root, request.dataset_id)
            exported = config.run_root / "lerobot_replay" / f"{uuid.uuid4().hex}.raw.h5"
            replay = export_episode_to_raw(
                dataset_path,
                request.episode_index,
                exported,
                side=request.side,
            )
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        command = [
            str(config.python),
            "-m",
            "cobot_bridge.replay_raw",
            "--raw",
            replay["path"],
            "--endpoint",
            config.endpoint,
            "--side",
            request.side,
            "--source",
            "puppet_command",
            "--mode",
            "puppet",
            "--waypoint-hz",
            str(request.waypoint_hz),
            "--speed-scale",
            str(request.speed_scale),
            "--alignment-speed",
            str(request.alignment_speed),
            "--alignment-min-seconds",
            str(request.alignment_min_seconds),
            "--no-trim-stationary-holds",
        ]
        metadata = {
            "source_type": "lerobot",
            "dataset_id": request.dataset_id,
            "episode_index": request.episode_index,
            "source_feature": replay["source_feature"],
            "joint_dimensions": replay["joint_dimensions"],
            "frames": replay["frames"],
            "fps": replay["fps"],
            "speed_scale": request.speed_scale,
            "alignment_speed": request.alignment_speed,
            "exported_replay_h5": replay["path"],
            "mode": "puppet",
        }
        if not request.execute:
            return start_job("replay_dry_run", command, metadata)
        command.append("--execute")
        return start_motion_job(
            "replay_execute",
            command,
            metadata,
            task=replay["task"],
            side=request.side,
            cameras=("front", request.side),
            policy_id=f"lerobot:{Path(request.dataset_id).name}:ep{request.episode_index}",
        )

    @app.post("/api/jobs/policy")
    def policy(
        request: PolicyRequest, x_cobot_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.execute and not request.confirm_motion:
            raise HTTPException(status_code=400, detail="real motion requires confirm_motion=true")
        if request.action_space in ("eef_delta", "eef_target") and request.runner not in (
            "standard", "eef_v5", "guided_rtc"
        ):
            raise HTTPException(
                status_code=400,
                detail="EEF action spaces require Standard, EEF V5, or Guided RTC runner",
            )
        server = _policy_server_status(config, manager)
        if not server["ready"]:
            raise HTTPException(status_code=409, detail="model server is not ready")
        compute = str(server.get("compute", "pc"))
        server_chunking = str(server.get("inference_chunking", "original"))
        if request.runner == "guided_rtc":
            if server_chunking != "guided_rtc":
                raise HTTPException(
                    status_code=409,
                    detail="start the model server with Guided RTC before selecting Guided RTC rollout",
                )
            if compute != "pc":
                raise HTTPException(status_code=409, detail="Guided RTC rollout currently requires PC compute")
            if request.action_space == "eef_delta":
                raise HTTPException(
                    status_code=409,
                    detail="Guided RTC terminal continuity requires absolute joint or eef_target actions",
                )
            is_relative_eef = server.get("relative_eef_actions")
            expected_action_space = "eef_target" if is_relative_eef else "joint"
            if request.action_space != expected_action_space:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "this checkpoint declares relative EEF actions; select eef_target"
                        if is_relative_eef
                        else "this checkpoint declares absolute joint actions; select joint"
                    ),
                )
            if request.n_action_steps is not None:
                raise HTTPException(
                    status_code=409,
                    detail="Guided RTC executes the complete native action chunk; clear Actions consumed",
                )
        elif server_chunking == "guided_rtc":
            raise HTTPException(
                status_code=409,
                detail="this model server is in Guided RTC mode; select Guided RTC rollout or stop/reload it",
            )
        if request.runner == "eef_v5" and (
            request.action_space != "eef_target"
            or request.eef_ik_backend != "pink"
            or compute != "pc"
        ):
            raise HTTPException(
                status_code=409,
                detail="EEF V5 requires PC compute, absolute eef_target actions, and Pink IK",
            )
        if (
            compute == "robot"
            and request.action_space in ("eef_delta", "eef_target")
            and request.eef_ik_backend == "pink"
        ):
            raise HTTPException(
                status_code=409,
                detail="Pink IK is installed on the PC only; select PC compute or use the legacy SDK backend",
            )
        rollout_token = uuid.uuid4().hex[:12]
        diagnostic_output: Path | None = None
        policy_decision_trace: Path | None = None
        if (
            request.execute
            and compute == "pc"
            and config.debug_capture_enabled
            and request.runner in ("standard", "eef_v5", "guided_rtc")
        ):
            policy_decision_trace = (
                config.debug_root / ".inflight" / f"policy_{rollout_token}.jsonl"
            )
        if compute == "robot":
            if request.runner in ("smooth_v2", "smooth_v3", "edge_v4"):
                diagnostic_output = (
                    config.robot_workspace / "diagnostics" / f"smooth_{rollout_token}.jsonl"
                )
            command = [
                str(config.python),
                "-m",
                "cobot_bridge.remote_rollout",
                "--ssh-host",
                config.robot_ssh_host,
                "--remote-workspace",
                str(config.robot_workspace),
                "--remote-python",
                str(config.robot_python),
                "--session",
                f"cobot_rollout_{rollout_token}",
                "--endpoint",
                config.endpoint,
                "--policy-server-endpoint",
                config.robot_policy_server_local_endpoint,
                "--task",
                request.task,
                "--side",
                request.side,
                "--fps",
                str(request.fps),
                "--duration",
                str(request.duration),
                "--mode",
                str(request.mode),
                "--action-space",
                request.action_space,
                "--eef-ik-backend",
                request.eef_ik_backend,
                "--runner",
                request.runner,
                "--cameras",
                "front",
                "right",
            ]
        else:
            if request.runner in ("smooth_v2", "smooth_v3", "edge_v4"):
                diagnostic_output = config.run_root / f"smooth_{rollout_token}.jsonl"
            runner_module = (
                {
                    "standard": "cobot_bridge.run_lerobot_policy",
                    "smooth_v2": "cobot_bridge.run_lerobot_policy_smooth",
                    "smooth_v3": "cobot_bridge.run_lerobot_policy_smooth_v3",
                    "edge_v4": "cobot_bridge.run_lerobot_policy_edge_v4",
                    "eef_v5": "cobot_bridge.run_lerobot_policy_eef_v5",
                    "guided_rtc": "cobot_bridge.run_lerobot_policy",
                }[request.runner]
            )
            command = [
                str(config.python),
                "-m",
                runner_module,
                "--endpoint",
                config.endpoint,
                "--task",
                request.task,
                "--side",
                request.side,
                "--fps",
                str(request.fps),
                "--duration",
                str(request.duration),
                "--mode",
                str(request.mode),
                "--action-space",
                request.action_space,
                "--eef-ik-backend",
                request.eef_ik_backend,
                "--policy-server-endpoint",
                str(server.get("endpoint") or config.policy_server_endpoint),
                "--cameras",
                "front",
                "right",
            ]
            if request.runner == "guided_rtc":
                command.append("--guided-rtc-terminal")
        if diagnostic_output is not None:
            command.extend(("--diagnostic-output", str(diagnostic_output)))
        if request.n_action_steps is not None:
            command.extend(("--n-action-steps", str(request.n_action_steps)))
        if request.execute:
            # Preserve the rollout contract used by the trained policies:
            # verify and hold the Ready pose after taking control, then reset
            # local rollout state before the first policy chunk is consumed.
            command.extend(("--startup-init-hold", "--send-actions"))
        metadata = {
            "policy_id": server.get("policy_id"),
            "compute": compute,
            "mode": request.mode,
            "action_space": request.action_space,
            "eef_ik_backend": request.eef_ik_backend,
            "runner": request.runner,
            "inference_chunking": server_chunking,
            "diagnostic_output": (
                None if diagnostic_output is None else str(diagnostic_output)
            ),
            "n_action_steps": request.n_action_steps or "model_chunk",
        }
        if policy_decision_trace is not None:
            metadata["policy_decision_trace"] = str(policy_decision_trace)
        if request.execute:
            return start_motion_job(
                "policy_execute",
                command,
                metadata,
                task=request.task,
                side=request.side,
                cameras=("front", "right"),
                policy_id=str(server.get("policy_id") or ""),
            )
        return start_job("policy_dry_run", command, metadata)

    @app.post("/api/jobs/init-pose")
    def init_pose(
        request: InitPoseRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if request.execute and not request.confirm_motion:
            raise HTTPException(status_code=400, detail="real motion requires confirm_motion=true")
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.move_init_pose",
            "--endpoint",
            config.endpoint,
            "--joint-speed",
            str(DEFAULT_JOINT_SPEED),
            "--min-seconds",
            str(DEFAULT_MIN_SECONDS),
            "--waypoint-hz",
            "20",
        ]
        if request.execute:
            command.append("--execute")
        return start_job("init_pose_execute" if request.execute else "init_pose_dry_run", command)

    @app.post("/api/emergency-safe")
    async def emergency_safe(
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        manager.stop_motion_jobs()

        def command_safe() -> dict[str, Any]:
            with CobotBridgeClient(config.endpoint, timeout_s=1.0) as client:
                emergency = client.emergency_safe()
            return {"emergency": emergency}

        try:
            result = await asyncio.to_thread(command_safe)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"safe": True, **result}

    @app.post("/api/system/start-robot")
    async def start_robot(
        request: RobotStartupRequest,
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        if any(j["state"] in ("running", "stopping") and (
            j["kind"] in RECORD_JOB_KINDS | MOTION_JOB_KINDS | STARTUP_JOB_KINDS
        ) for j in manager.list()):
            raise HTTPException(status_code=409, detail="请先结束录制、运动或其他设备维护任务")
        probe = await asyncio.to_thread(device_monitor.probe)
        if not request.check_only and (probe.get("teach_active") or probe.get("output_enabled")):
            raise HTTPException(status_code=409, detail="请先关闭 Teach 和运动输出")
        report = config.run_root / f"robot_startup_{uuid.uuid4().hex[:12]}.json"
        command = [str(config.python), "-m", "cobot_bridge.ensure_robot",
                   "--ssh-host", config.robot_ssh_host,
                   "--remote-workspace", str(config.robot_control_workspace),
                   "--endpoint", config.endpoint, "--report", str(report)]
        if request.check_only:
            command.append("--check-only")
        if config.robot_sudo_password_file is not None:
            command.extend(["--sudo-password-file", str(config.robot_sudo_password_file)])
        return start_job("robot_self_check" if request.check_only else "robot_startup", command,
                         {"startup_report_file": str(report), "check_only": request.check_only})

    @app.post("/api/system/restart-devices")
    async def restart_devices(
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)
        probe = await asyncio.to_thread(device_monitor.probe)
        if not probe.get("online", False):
            raise HTTPException(
                status_code=503,
                detail="robot bridge is offline; Teach state cannot be verified safely",
            )
        if probe.get("teach_active", False):
            raise HTTPException(status_code=409, detail="release Teach before restarting devices")

        active_record = next(
            (
                job
                for job in manager.list()
                if job["kind"] in RECORD_JOB_KINDS
                and job["state"] in ("running", "stopping")
            ),
            None,
        )
        if active_record is not None:
            raise HTTPException(
                status_code=409,
                detail="finish or abort the active recording before restarting devices",
            )

        manager.stop_motion_jobs()
        command = [
            str(config.python),
            "-m",
            "cobot_bridge.restart_devices",
            "--ssh-host",
            config.robot_ssh_host,
            "--remote-workspace",
            str(config.robot_control_workspace),
            "--endpoint",
            config.endpoint,
        ]
        if config.robot_sudo_password_file is not None:
            command.extend(["--sudo-password-file", str(config.robot_sudo_password_file)])
        return start_job(
            "device_restart",
            command,
            {"ssh_host": config.robot_ssh_host, "endpoint": config.endpoint},
        )

    @app.post("/api/clear-emergency")
    async def clear_emergency(
        x_cobot_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_cobot_token)

        def command_clear() -> dict[str, Any]:
            with CobotBridgeClient(config.endpoint, timeout_s=1.0) as client:
                return client.clear_emergency()

        try:
            result = await asyncio.to_thread(command_clear)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not result.get("accepted", False):
            raise HTTPException(status_code=409, detail=result.get("reason", "clear rejected"))
        return {"cleared": True, "result": result}

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("cobot_bridge.web_dashboard:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
