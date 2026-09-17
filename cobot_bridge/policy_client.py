"""Client for a persistent PC-side or robot-side Cobot Magic policy server."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import zmq


POLICY_PROTOCOL = "cobot-magic-policy"
POLICY_VERSION = 1
POLICY_COMMANDS = ("PING", "RESET", "PREDICT", "PREDICT_CHUNK", "SHUTDOWN")


class PolicyServerError(RuntimeError):
    """The persistent policy server rejected or malformed a request."""


def encode_policy_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_policy_json(frame: bytes) -> dict[str, Any]:
    try:
        value = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyServerError(f"invalid policy JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyServerError("policy JSON root must be an object")
    return value


def make_policy_request(command: str, **payload: Any) -> dict[str, Any]:
    command = str(command).upper()
    if command not in POLICY_COMMANDS:
        raise PolicyServerError(f"unknown policy command: {command}")
    return {
        "protocol": POLICY_PROTOCOL,
        "version": POLICY_VERSION,
        "request_id": uuid.uuid4().hex,
        "command": command,
        **payload,
    }


def validate_policy_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("protocol") != POLICY_PROTOCOL or payload.get("version") != POLICY_VERSION:
        raise PolicyServerError("incompatible policy request")
    if payload.get("command") not in POLICY_COMMANDS:
        raise PolicyServerError(f"unknown policy command: {payload.get('command')}")
    if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
        raise PolicyServerError("missing policy request_id")
    return dict(payload)


def make_policy_response(
    request: Mapping[str, Any], *, ok: bool, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "protocol": POLICY_PROTOCOL,
        "version": POLICY_VERSION,
        "request_id": request.get("request_id"),
        "command": request.get("command"),
        "ok": bool(ok),
    }
    if result is not None:
        response["result"] = result
    if error is not None:
        response["error"] = str(error)
    return response


class PersistentPolicyClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 20.0) -> None:
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

    def _request(self, command: str, *, frames: Sequence[bytes] = (), **payload: Any) -> Any:
        request = make_policy_request(command, **payload)
        assert self._socket is not None
        try:
            self._socket.send_multipart([encode_policy_json(request), *frames])
            response_frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            self._connect()
            raise PolicyServerError(f"timeout waiting for {command} from {self.endpoint}") from exc
        if len(response_frames) != 1:
            raise PolicyServerError("policy server returned unexpected multipart response")
        response = decode_policy_json(response_frames[0])
        if (
            response.get("protocol") != POLICY_PROTOCOL
            or response.get("version") != POLICY_VERSION
            or response.get("request_id") != request["request_id"]
            or response.get("command") != request["command"]
        ):
            raise PolicyServerError("incompatible policy response")
        if not response.get("ok", False):
            raise PolicyServerError(str(response.get("error", "policy request failed")))
        return response.get("result")

    def ping(self) -> dict[str, Any]:
        result = self._request("PING", client_time_ns=time.time_ns())
        return dict(result or {})

    def reset(self) -> dict[str, Any]:
        result = self._request("RESET")
        return dict(result or {})

    def shutdown(self) -> dict[str, Any]:
        result = self._request("SHUTDOWN")
        return dict(result or {})

    def predict(
        self,
        *,
        state: Sequence[float],
        images: Mapping[str, bytes],
        cameras: Sequence[str],
        task: str,
        robot_type: str = "cobot_magic_piper",
        action_space: str = "joint",
    ) -> np.ndarray:
        selected = [str(name) for name in cameras]
        missing = [name for name in selected if name not in images]
        if missing:
            raise PolicyServerError(f"missing policy images: {missing}")
        result = self._request(
            "PREDICT",
            frames=[bytes(images[name]) for name in selected],
            state=[float(value) for value in state],
            cameras=selected,
            task=str(task),
            robot_type=str(robot_type),
            action_space=str(action_space),
        )
        if not isinstance(result, dict):
            raise PolicyServerError("invalid PREDICT result")
        action = np.asarray(result.get("action", []), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise PolicyServerError(f"invalid policy action shape/value: {action}")
        return action

    def predict_chunk(
        self,
        *,
        state: Sequence[float],
        images: Mapping[str, bytes],
        cameras: Sequence[str],
        task: str,
        robot_type: str = "cobot_magic_piper",
        action_space: str = "joint",
    ) -> np.ndarray:
        return self.predict_chunk_details(
            state=state,
            images=images,
            cameras=cameras,
            task=task,
            robot_type=robot_type,
            action_space=action_space,
        )["action_chunk"]

    @staticmethod
    def _parse_chunk_result(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise PolicyServerError("invalid PREDICT_CHUNK result")
        action_chunk = np.asarray(result.get("action_chunk", []), dtype=np.float32)
        if (
            action_chunk.ndim != 2
            or action_chunk.shape[1:] != (7,)
            or not np.isfinite(action_chunk).all()
        ):
            raise PolicyServerError(f"invalid policy action chunk shape/value: {action_chunk.shape}")
        native = np.asarray(
            result.get("model_native_action_chunk", action_chunk), dtype=np.float32
        )
        if native.shape != action_chunk.shape or not np.isfinite(native).all():
            raise PolicyServerError(
                "invalid model native action chunk shape/value: "
                f"{native.shape}, expected {action_chunk.shape}"
            )
        semantics = str(result.get("model_output_semantics", "absolute_joint"))
        if semantics not in ("absolute_joint", "absolute_eef", "relative_eef"):
            raise PolicyServerError(f"unsupported model output semantics: {semantics!r}")
        return {
            "action_chunk": action_chunk,
            "model_native_action_chunk": native,
            "model_output_semantics": semantics,
            "latency_ms": result.get("latency_ms"),
            "rtc_prefix_steps": result.get("rtc_prefix_steps"),
        }

    def predict_chunk_details(
        self,
        *,
        state: Sequence[float],
        images: Mapping[str, bytes],
        cameras: Sequence[str],
        task: str,
        robot_type: str = "cobot_magic_piper",
        action_space: str = "joint",
    ) -> dict[str, Any]:
        """Return robot-space and checkpoint-native outputs for one chunk."""

        selected = [str(name) for name in cameras]
        missing = [name for name in selected if name not in images]
        if missing:
            raise PolicyServerError(f"missing policy images: {missing}")
        result = self._request(
            "PREDICT_CHUNK",
            frames=[bytes(images[name]) for name in selected],
            state=[float(value) for value in state],
            cameras=selected,
            task=str(task),
            robot_type=str(robot_type),
            action_space=str(action_space),
        )
        return self._parse_chunk_result(result)
