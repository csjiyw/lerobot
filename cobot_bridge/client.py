"""Synchronous ZeroMQ client used by collection and RLT processes on the PC."""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import zmq

from .protocol import (
    CAMERAS,
    REQUESTABLE_MODES,
    SIDES,
    ProtocolError,
    decode_json,
    encode_json,
    make_request,
    unpack_observation_response,
    validate_action,
    validate_action_chunk,
    validate_names,
    validate_response,
)


class BridgeError(RuntimeError):
    """The bridge rejected a request or returned malformed data."""


class BridgeTimeout(BridgeError):
    """The bridge did not reply before the configured timeout."""


class CobotBridgeClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 2.0, token: str = "") -> None:
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.token = token
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

    def __enter__(self) -> "CobotBridgeClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _request(self, command: str, *, observation: bool = False, **payload: Any) -> Any:
        request = make_request(command, **payload)
        if self.token:
            request["token"] = self.token
        assert self._socket is not None
        try:
            self._socket.send_multipart([encode_json(request)])
            frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            self._connect()
            raise BridgeTimeout(f"timeout waiting for {command} from {self.endpoint}") from exc

        try:
            if observation:
                return unpack_observation_response(frames, request)
            if len(frames) != 1:
                raise ProtocolError("unexpected_multipart_response")
            response = validate_response(decode_json(frames[0]), request)
        except ProtocolError as exc:
            raise BridgeError(str(exc)) from exc
        if not response["ok"]:
            raise BridgeError(str(response.get("error", "request_failed")))
        return response.get("result")

    def ping(self) -> dict[str, Any]:
        result = self._request("PING", client_time_ns=time.time_ns())
        if not isinstance(result, dict):
            raise BridgeError("invalid_ping_response")
        return result

    def get_observation(
        self,
        *,
        sides: Iterable[str] = ("left",),
        images: Iterable[str] = CAMERAS,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        selected_sides = validate_names(sides, SIDES, "side")
        selected_images = validate_names(images, CAMERAS, "camera")
        return self._request(
            "GET_OBSERVATION",
            observation=True,
            sides=selected_sides,
            images=selected_images,
        )

    def set_mode(self, mode: int) -> dict[str, Any]:
        mode = int(mode)
        if mode not in REQUESTABLE_MODES:
            raise BridgeError(f"invalid_requestable_mode:{mode}")
        result = self._request("SET_MODE", mode=mode)
        return dict(result or {})

    def set_output(self, enabled: bool) -> dict[str, Any]:
        result = self._request("SET_OUTPUT", enabled=bool(enabled))
        return dict(result or {})

    def emergency_safe(self) -> dict[str, Any]:
        result = self._request("EMERGENCY_SAFE")
        return dict(result or {})

    def clear_emergency(self) -> dict[str, Any]:
        result = self._request("CLEAR_EMERGENCY")
        return dict(result or {})

    def resync(self) -> dict[str, Any]:
        result = self._request("RESYNC")
        return dict(result or {})

    def set_action(
        self,
        side: str,
        action: Sequence[float],
        *,
        epoch: int,
        sequence: int | None = None,
        alignment_speed: float | None = None,
        action_space: str = "joint",
    ) -> dict[str, Any]:
        selected_side = validate_names([side], SIDES, "side")[0]
        values = validate_action(action, action_space=action_space)
        payload: dict[str, Any] = {
            "side": selected_side,
            "action": values,
            "epoch": int(epoch),
            "action_space": str(action_space),
        }
        if sequence is not None:
            payload["sequence"] = int(sequence)
        if alignment_speed is not None:
            payload["alignment_speed"] = float(alignment_speed)
        result = self._request("SET_ACTION", **payload)
        return dict(result or {})

    def set_action_chunk(
        self,
        side: str,
        actions: Sequence[Sequence[float]],
        *,
        epoch: int,
        sequence: int,
        waypoint_hz: float,
        action_space: str = "joint",
    ) -> dict[str, Any]:
        selected_side = validate_names([side], SIDES, "side")[0]
        values = validate_action_chunk(actions, action_space=action_space)
        result = self._request(
            "SET_ACTION_CHUNK",
            side=selected_side,
            actions=values,
            epoch=int(epoch),
            sequence=int(sequence),
            waypoint_hz=float(waypoint_hz),
            action_space=str(action_space),
        )
        return dict(result or {})


def request_robot_safe_stop(endpoint: str, *, timeout_s: float = 1.5) -> dict[str, Any]:
    """Best-effort output-off + SAFE using a fresh socket for each request.

    Stop buttons may interrupt a runner while its REQ socket is waiting for a
    response. A new short-lived socket avoids inheriting that REQ state and
    disables physical output before the runner is signalled to exit.
    """

    results: dict[str, Any] = {}
    errors: list[str] = []
    for key, action in (("output_disabled", "output"), ("mode_safe", "safe")):
        try:
            with CobotBridgeClient(endpoint, timeout_s=timeout_s) as client:
                response = client.set_output(False) if action == "output" else client.set_mode(0)
            results[key] = response
            if not response.get("accepted", False):
                errors.append(f"{key} rejected: {response}")
        except Exception as exc:  # Keep trying the independent SAFE request.
            errors.append(f"{key} failed: {type(exc).__name__}: {exc}")
    results["safe"] = not errors
    if errors:
        results["errors"] = errors
    return results
