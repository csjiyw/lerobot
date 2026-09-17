"""Wire protocol shared by the AGX bridge and the PC client.

The first ZeroMQ multipart frame is always UTF-8 JSON. Observation replies may
append one JPEG frame per requested camera. ROS messages never cross the wire.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "cobot-magic-bridge"
VERSION = 1
JOINT_COUNT = 7
SIDES = ("left", "right")
CAMERAS = ("front", "left", "right")
ACTION_SPACES = ("joint", "eef_target")
REQUESTABLE_MODES = (0, 1, 2)
COMMANDS = {
    "PING",
    "GET_OBSERVATION",
    "SET_MODE",
    "SET_OUTPUT",
    "EMERGENCY_SAFE",
    "CLEAR_EMERGENCY",
    "RESYNC",
    "SET_ACTION",
    "SET_ACTION_CHUNK",
}


class ProtocolError(RuntimeError):
    """Raised for malformed or incompatible bridge messages."""


def encode_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_json(frame: bytes) -> dict[str, Any]:
    try:
        value = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid_json: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("json_root_must_be_object")
    return value


def make_request(command: str, **payload: Any) -> dict[str, Any]:
    command = str(command).upper()
    if command not in COMMANDS:
        raise ProtocolError(f"unknown_command:{command}")
    return {
        "protocol": PROTOCOL,
        "version": VERSION,
        "request_id": uuid.uuid4().hex,
        "command": command,
        **payload,
    }


def validate_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("protocol") != PROTOCOL:
        raise ProtocolError("protocol_mismatch")
    if payload.get("version") != VERSION:
        raise ProtocolError("version_mismatch")
    command = str(payload.get("command", "")).upper()
    if command not in COMMANDS:
        raise ProtocolError(f"unknown_command:{command}")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("missing_request_id")
    return dict(payload)


def make_response(
    request: Mapping[str, Any], *, ok: bool, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "request_id": request.get("request_id"),
        "command": request.get("command"),
        "ok": bool(ok),
    }
    if result is not None:
        response["result"] = result
    if error is not None:
        response["error"] = str(error)
    return response


def validate_response(response: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    if response.get("protocol") != PROTOCOL or response.get("version") != VERSION:
        raise ProtocolError("incompatible_response")
    if response.get("request_id") != request.get("request_id"):
        raise ProtocolError("request_id_mismatch")
    if response.get("command") != request.get("command"):
        raise ProtocolError("command_mismatch")
    if not isinstance(response.get("ok"), bool):
        raise ProtocolError("missing_ok")
    return dict(response)


def validate_names(values: Iterable[Any], allowed: Sequence[str], field: str) -> list[str]:
    result: list[str] = []
    for value in values:
        name = str(value).lower()
        if name not in allowed:
            raise ProtocolError(f"invalid_{field}:{name}")
        if name not in result:
            result.append(name)
    return result


def validate_action(values: Any, *, action_space: str = "joint") -> list[float]:
    action_space = str(action_space).lower()
    if action_space not in ACTION_SPACES:
        raise ProtocolError(f"invalid_action_space:{action_space}")
    if not isinstance(values, (list, tuple)):
        raise ProtocolError("action_must_be_list")
    result = [float(value) for value in values]
    if len(result) != JOINT_COUNT:
        raise ProtocolError(f"expected_{JOINT_COUNT}_action_values")
    if not all(math.isfinite(value) for value in result):
        raise ProtocolError("non_finite_action")
    if action_space == "eef_target":
        # EEF targets on the wire are SDK/Piper link6 poses in
        # meters/radians/meters. The PC policy boundary is responsible for
        # converting a model TCP such as gripper_center to link6. These are
        # protocol sanity bounds, not motion limits; the ROS arbiter remains
        # the physical safety owner.
        if any(abs(value) > 2.0 for value in result[:3]):
            raise ProtocolError("eef_position_out_of_protocol_range")
        if any(abs(value) > 4.0 * math.pi for value in result[3:6]):
            raise ProtocolError("eef_rotation_out_of_protocol_range")
        if not 0.0 <= result[6] <= 0.2:
            raise ProtocolError("eef_gripper_out_of_protocol_range")
    return result


def validate_action_chunk(
    values: Any, *, max_actions: int = 4096, action_space: str = "joint"
) -> list[list[float]]:
    if not isinstance(values, (list, tuple)):
        raise ProtocolError("actions_must_be_list")
    if not values:
        raise ProtocolError("action_chunk_empty")
    if len(values) > max_actions:
        raise ProtocolError(f"action_chunk_too_long:{len(values)}>{max_actions}")
    return [validate_action(action, action_space=action_space) for action in values]


def interpolate_action_chunk(
    actions: Sequence[Sequence[float]], elapsed_s: float, waypoint_hz: float
) -> tuple[list[float], int, bool]:
    """Linearly interpolate fixed-rate waypoints for a high-rate scheduler."""
    if not actions:
        raise ProtocolError("action_chunk_empty")
    if waypoint_hz <= 0 or not math.isfinite(waypoint_hz):
        raise ProtocolError("invalid_waypoint_hz")
    phase = max(0.0, float(elapsed_s)) * float(waypoint_hz)
    left = min(int(math.floor(phase)), len(actions) - 1)
    if left >= len(actions) - 1:
        return [float(value) for value in actions[-1]], left, True
    alpha = phase - left
    result = [
        float(actions[left][joint])
        + alpha * (float(actions[left + 1][joint]) - float(actions[left][joint]))
        for joint in range(JOINT_COUNT)
    ]
    return result, left, False


def pack_observation_response(
    request: Mapping[str, Any], observation: dict[str, Any], images: Mapping[str, bytes]
) -> list[bytes]:
    descriptors = observation.setdefault("images", {})
    frames: list[bytes] = []
    for name in CAMERAS:
        if name not in images:
            continue
        data = bytes(images[name])
        descriptor = descriptors.setdefault(name, {})
        descriptor["frame"] = len(frames) + 1
        descriptor["size"] = len(data)
        frames.append(data)
    header = make_response(request, ok=True, result=observation)
    return [encode_json(header), *frames]


def unpack_observation_response(
    frames: Sequence[bytes], request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not frames:
        raise ProtocolError("empty_response")
    response = validate_response(decode_json(frames[0]), request)
    if not response["ok"]:
        raise ProtocolError(str(response.get("error", "request_failed")))
    observation = response.get("result")
    if not isinstance(observation, dict):
        raise ProtocolError("observation_missing")
    image_data: dict[str, bytes] = {}
    descriptors = observation.get("images", {})
    if not isinstance(descriptors, dict):
        raise ProtocolError("invalid_image_descriptors")
    for name, descriptor in descriptors.items():
        if descriptor is None:
            continue
        if not isinstance(descriptor, dict):
            raise ProtocolError(f"invalid_image_descriptor:{name}")
        frame_index = descriptor.get("frame")
        if frame_index is None:
            continue
        if not isinstance(frame_index, int) or frame_index <= 0 or frame_index >= len(frames):
            raise ProtocolError(f"invalid_image_frame:{name}")
        data = bytes(frames[frame_index])
        if descriptor.get("size") != len(data):
            raise ProtocolError(f"image_size_mismatch:{name}")
        image_data[str(name)] = data
    return observation, image_data
