from __future__ import annotations

import threading

import zmq

from cobot_bridge.client import CobotBridgeClient
from cobot_bridge.protocol import (
    decode_json,
    encode_json,
    make_response,
    pack_observation_response,
    validate_request,
)


def _fake_server(context: zmq.Context, endpoint: str, ready: threading.Event) -> None:
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    ready.set()
    for _ in range(5):
        request = validate_request(decode_json(socket.recv_multipart()[0]))
        if request["command"] == "PING":
            response = [encode_json(make_response(request, ok=True, result={"pong": True}))]
        elif request["command"] == "GET_OBSERVATION":
            response = pack_observation_response(
                request,
                {
                    "ready": True,
                    "control": {"epoch": 4, "effective_mode_id": 2},
                    "images": {"front": {"encoding": "jpeg"}},
                },
                {"front": b"jpeg"},
            )
        else:
            response = [
                encode_json(
                    make_response(
                        request,
                        ok=True,
                        result={"accepted": False, "reason": "dry_run"},
                    )
                )
            ]
        socket.send_multipart(response)
    socket.close(0)


def test_client_ping_observation_and_action():
    context = zmq.Context()
    probe = context.socket(zmq.REP)
    port = probe.bind_to_random_port("tcp://127.0.0.1")
    probe.close(0)
    endpoint = f"tcp://127.0.0.1:{port}"
    ready = threading.Event()
    thread = threading.Thread(target=_fake_server, args=(context, endpoint, ready))
    thread.start()
    assert ready.wait(2.0)

    with CobotBridgeClient(endpoint, timeout_s=2.0) as client:
        assert client.ping()["pong"] is True
        observation, images = client.get_observation(sides=("left",), images=("front",))
        assert observation["control"]["epoch"] == 4
        assert images == {"front": b"jpeg"}
        result = client.set_action("left", [0.0] * 7, epoch=4, sequence=1)
        assert result == {"accepted": False, "reason": "dry_run"}
        assert client.emergency_safe() == {"accepted": False, "reason": "dry_run"}
        assert client.clear_emergency() == {"accepted": False, "reason": "dry_run"}

    thread.join(timeout=2.0)
    context.term()
    assert not thread.is_alive()
