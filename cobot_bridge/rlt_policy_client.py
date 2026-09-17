"""Client protocol for the persistent Cobot Magic RLT rollout server."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import zmq


RLT_POLICY_PROTOCOL = "cobot-magic-rlt-policy"
RLT_POLICY_VERSION = 1
RLT_POLICY_COMMANDS = (
    "PING",
    "BEGIN_EPISODE",
    "END_EPISODE",
    "PREDICT",
    "NORMALIZE_ACTION",
    "SHUTDOWN",
)


class RLTPolicyServerError(RuntimeError):
    """The persistent RLT policy server rejected or malformed a request."""


def encode_rlt_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_rlt_json(frame: bytes) -> dict[str, Any]:
    try:
        value = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RLTPolicyServerError(f"invalid RLT policy JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RLTPolicyServerError("RLT policy JSON root must be an object")
    return value


def make_rlt_request(command: str, **payload: Any) -> dict[str, Any]:
    command = str(command).upper()
    if command not in RLT_POLICY_COMMANDS:
        raise RLTPolicyServerError(f"unknown RLT policy command: {command}")
    return {
        "protocol": RLT_POLICY_PROTOCOL,
        "version": RLT_POLICY_VERSION,
        "request_id": uuid.uuid4().hex,
        "command": command,
        **payload,
    }


def validate_rlt_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        payload.get("protocol") != RLT_POLICY_PROTOCOL
        or payload.get("version") != RLT_POLICY_VERSION
    ):
        raise RLTPolicyServerError("incompatible RLT policy request")
    if payload.get("command") not in RLT_POLICY_COMMANDS:
        raise RLTPolicyServerError(f"unknown RLT policy command: {payload.get('command')}")
    if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
        raise RLTPolicyServerError("missing RLT policy request_id")
    return dict(payload)


def make_rlt_response(
    request: Mapping[str, Any], *, ok: bool, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "protocol": RLT_POLICY_PROTOCOL,
        "version": RLT_POLICY_VERSION,
        "request_id": request.get("request_id"),
        "command": request.get("command"),
        "ok": bool(ok),
    }
    if result is not None:
        response["result"] = result
    if error is not None:
        response["error"] = str(error)
    return response


def _finite_array(result: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(result.get(name, []), dtype=np.float32)
    if values.shape != shape or not np.isfinite(values).all():
        raise RLTPolicyServerError(f"invalid {name}: expected {shape}, got {values.shape}")
    return values


class PersistentRLTPolicyClient:
    """One REQ client; create one instance per RLT collection process."""

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        timeout_ms = max(1, int(self.timeout_s * 1000))
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        self._context.term()

    def __enter__(self) -> "PersistentRLTPolicyClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _request(self, command: str, *, frames: Sequence[bytes] = (), **payload: Any) -> Any:
        request = make_rlt_request(command, **payload)
        assert self._socket is not None
        try:
            self._socket.send_multipart([encode_rlt_json(request), *frames])
            response_frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            self._connect()
            raise RLTPolicyServerError(
                f"timeout waiting for {command} from {self.endpoint}"
            ) from exc
        if len(response_frames) != 1:
            raise RLTPolicyServerError("RLT policy server returned unexpected multipart response")
        response = decode_rlt_json(response_frames[0])
        if (
            response.get("protocol") != RLT_POLICY_PROTOCOL
            or response.get("version") != RLT_POLICY_VERSION
            or response.get("request_id") != request["request_id"]
            or response.get("command") != request["command"]
        ):
            raise RLTPolicyServerError("incompatible RLT policy response")
        if not response.get("ok", False):
            raise RLTPolicyServerError(str(response.get("error", "RLT policy request failed")))
        return response.get("result")

    def ping(self) -> dict[str, Any]:
        return dict(self._request("PING", client_time_ns=time.time_ns()) or {})

    def begin_episode(
        self,
        *,
        workflow_mode: int,
        allow_actor: bool,
        actor_exploration_std: float,
        q_select_k: int = 0,
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "BEGIN_EPISODE",
                workflow_mode=int(workflow_mode),
                allow_actor=bool(allow_actor),
                actor_exploration_std=float(actor_exploration_std),
                q_select_k=int(q_select_k),
            )
            or {}
        )

    def end_episode(self) -> dict[str, Any]:
        return dict(self._request("END_EPISODE") or {})

    def shutdown(self) -> dict[str, Any]:
        return dict(self._request("SHUTDOWN") or {})

    def normalize_action_chunk(self, raw_actions: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_actions, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1:] != (7,) or not np.isfinite(raw).all():
            raise RLTPolicyServerError(f"invalid raw action chunk: {raw.shape}")
        result = self._request("NORMALIZE_ACTION", raw_actions=raw.tolist())
        if not isinstance(result, dict):
            raise RLTPolicyServerError("invalid NORMALIZE_ACTION result")
        return _finite_array(result, "normalized_actions", raw.shape)

    def predict(
        self,
        *,
        state: Sequence[float],
        images: Mapping[str, bytes],
        cameras: Sequence[str],
        task: str,
        robot_type: str = "cobot_magic_piper",
    ) -> dict[str, Any]:
        selected = [str(name) for name in cameras]
        missing = [name for name in selected if name not in images]
        if missing:
            raise RLTPolicyServerError(f"missing RLT policy images: {missing}")
        result = self._request(
            "PREDICT",
            frames=[bytes(images[name]) for name in selected],
            state=[float(value) for value in state],
            cameras=selected,
            task=str(task),
            robot_type=str(robot_type),
        )
        if not isinstance(result, dict):
            raise RLTPolicyServerError("invalid PREDICT result")
        dimensions = dict(result.get("dimensions") or {})
        z_dim = int(dimensions.get("z_dim", 0))
        state_dim = int(dimensions.get("state_dim", 0))
        action_steps = int(dimensions.get("action_steps", 0))
        action_dim = int(dimensions.get("action_dim", 0))
        if min(z_dim, state_dim, action_steps, action_dim) <= 0:
            raise RLTPolicyServerError(f"invalid RLT dimensions: {dimensions}")
        parsed = dict(result)
        parsed["z_rl"] = _finite_array(result, "z_rl", (z_dim,))
        parsed["state"] = _finite_array(result, "state", (state_dim,))
        parsed["reference_actions"] = _finite_array(
            result, "reference_actions", (action_steps, action_dim)
        )
        parsed["policy_actions"] = _finite_array(
            result, "policy_actions", (action_steps, action_dim)
        )
        parsed["executed_actions"] = _finite_array(
            result, "executed_actions", (action_steps, action_dim)
        )
        parsed["raw_action_chunk"] = _finite_array(
            result, "raw_action_chunk", (action_steps, action_dim)
        )
        parsed["raw_reference_chunk"] = _finite_array(
            result, "raw_reference_chunk", (action_steps, action_dim)
        )
        reference_action_steps = int(result.get("reference_action_steps", action_steps))
        if reference_action_steps < action_steps:
            raise RLTPolicyServerError(
                "reference_action_steps cannot be shorter than the execution horizon"
            )
        parsed["raw_reference_full_chunk"] = _finite_array(
            result,
            "raw_reference_full_chunk",
            (reference_action_steps, action_dim),
        )
        parsed["raw_policy_chunk"] = _finite_array(
            result, "raw_policy_chunk", (action_steps, action_dim)
        )
        source = str(result.get("action_source", ""))
        if source not in ("reference", "actor"):
            raise RLTPolicyServerError(f"invalid action_source: {source!r}")
        return parsed
