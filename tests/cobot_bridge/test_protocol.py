from __future__ import annotations

import pytest

from cobot_bridge.protocol import (
    ProtocolError,
    make_request,
    pack_observation_response,
    unpack_observation_response,
    validate_action,
    validate_action_chunk,
    validate_request,
    interpolate_action_chunk,
)


def test_observation_multipart_round_trip():
    request = make_request("GET_OBSERVATION", sides=["left"], images=["front", "left"])
    validate_request(request)
    observation = {
        "ready": True,
        "images": {
            "front": {"encoding": "jpeg"},
            "left": {"encoding": "jpeg"},
        },
    }
    frames = pack_observation_response(
        request,
        observation,
        {"front": b"front-jpeg", "left": b"left-jpeg"},
    )
    decoded, images = unpack_observation_response(frames, request)
    assert decoded["ready"] is True
    assert images == {"front": b"front-jpeg", "left": b"left-jpeg"}


@pytest.mark.parametrize(
    "value, reason",
    [
        ([0.0] * 6, "expected_7_action_values"),
        ([0.0] * 6 + [float("nan")], "non_finite_action"),
        ("bad", "action_must_be_list"),
    ],
)
def test_invalid_action(value, reason):
    with pytest.raises(ProtocolError, match=reason):
        validate_action(value)


def test_protocol_version_is_required():
    request = make_request("PING")
    request["version"] = 999
    with pytest.raises(ProtocolError, match="version_mismatch"):
        validate_request(request)


@pytest.mark.parametrize("command", ["EMERGENCY_SAFE", "CLEAR_EMERGENCY"])
def test_emergency_commands_are_in_protocol(command):
    assert validate_request(make_request(command))["command"] == command


def test_action_chunk_validation_and_interpolation():
    actions = validate_action_chunk([[0.0] * 7, [1.0] * 7])
    midpoint, index, finished = interpolate_action_chunk(actions, 0.025, 20.0)
    assert midpoint == [0.5] * 7
    assert index == 0
    assert finished is False
    final, index, finished = interpolate_action_chunk(actions, 0.05, 20.0)
    assert final == [1.0] * 7
    assert index == 1
    assert finished is True
