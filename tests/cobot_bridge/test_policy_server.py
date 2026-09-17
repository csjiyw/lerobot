from __future__ import annotations

import cv2
import numpy as np
import pytest

from cobot_bridge.policy_client import (
    POLICY_PROTOCOL,
    PersistentPolicyClient,
    PolicyServerError,
    decode_policy_json,
    encode_policy_json,
    make_policy_request,
    make_policy_response,
    validate_policy_request,
)
from cobot_bridge.policy_server import _decode_jpeg, _policy_key


def test_policy_protocol_round_trip():
    request = make_policy_request("PREDICT", state=[0.0] * 7, cameras=["front", "right"])

    decoded = validate_policy_request(decode_policy_json(encode_policy_json(request)))
    response = make_policy_response(decoded, ok=True, result={"action": [0.0] * 7})

    assert decoded["protocol"] == POLICY_PROTOCOL
    assert response["request_id"] == request["request_id"]
    assert response["result"]["action"] == [0.0] * 7


def test_policy_protocol_rejects_unknown_command():
    with pytest.raises(PolicyServerError, match="unknown"):
        make_policy_request("MOVE_ROBOT")


def test_policy_protocol_supports_server_shutdown():
    request = make_policy_request("SHUTDOWN")

    decoded = validate_policy_request(decode_policy_json(encode_policy_json(request)))

    assert decoded["command"] == "SHUTDOWN"


def test_policy_protocol_supports_full_chunk_prediction():
    request = make_policy_request(
        "PREDICT_CHUNK", state=[0.0] * 7, cameras=["front", "right"]
    )

    decoded = validate_policy_request(decode_policy_json(encode_policy_json(request)))

    assert decoded["command"] == "PREDICT_CHUNK"


def test_policy_chunk_details_preserve_native_relative_eef_output():
    result = PersistentPolicyClient._parse_chunk_result(
        {
            "action_chunk": [[1.0] * 7, [2.0] * 7],
            "model_native_action_chunk": [[0.1] * 7, [0.2] * 7],
            "model_output_semantics": "relative_eef",
            "latency_ms": 10.0,
            "rtc_prefix_steps": 1,
        }
    )

    assert result["action_chunk"].shape == (2, 7)
    assert result["model_native_action_chunk"].shape == (2, 7)
    assert result["model_output_semantics"] == "relative_eef"
    assert result["action_chunk"][0, 0] == pytest.approx(1.0)
    assert result["model_native_action_chunk"][0, 0] == pytest.approx(0.1)


def test_policy_server_decodes_jpeg_as_rgb():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok

    decoded = _decode_jpeg(jpeg.tobytes())

    assert decoded.shape == rgb.shape
    assert float(decoded[..., 0].mean()) > 230
    assert float(decoded[..., 2].mean()) < 10
    assert _policy_key("front") == "observation.images.camera_f"
    assert _policy_key("right") == "observation.images.camera_r"
