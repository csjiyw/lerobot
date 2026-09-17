#!/usr/bin/env python3
"""Local 7D RLT IPC server with a NumPy-version-independent wire format.

The default hold mode is communication-only.  This process does not import
ROS and never publishes a robot command.  The wire sequence is:
UPDATE multipart -> ACK -> WAIT -> SNAPSHOT multipart.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from typing import Any

import numpy as np
import zmq


LOG = logging.getLogger("rlt_ipc_server")
STOP = False
UPDATE_KEYS = ("state", "cam_front", "cam_side")
SNAPSHOT_KEYS = ("state", "action", "cam_front", "cam_side")


def stop_handler(_signal: int, _frame: Any) -> None:
    global STOP
    STOP = True


def pack_arrays(
    message_type: bytes,
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> list[bytes]:
    metadata: dict[str, dict[str, Any]] = {}
    payload: list[bytes] = []
    for key in keys:
        array = np.ascontiguousarray(arrays[key])
        metadata[key] = {"shape": list(array.shape), "dtype": array.dtype.str}
        payload.append(array.tobytes(order="C"))
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return [message_type, header, *payload]


class SevenDServer:
    def __init__(self, host: str, port: int, action_steps: int, mode: str) -> None:
        self.mode = mode
        self.action_steps = action_steps
        self.config = {
            "state": ((1, 7), np.dtype(np.float32)),
            "action": ((action_steps, 7), np.dtype(np.float32)),
            "cam_front": ((1, 3, 224, 224), np.dtype(np.uint8)),
            "cam_side": ((1, 3, 224, 224), np.dtype(np.uint8)),
        }
        self.data = {
            key: np.zeros(shape, dtype=dtype)
            for key, (shape, dtype) in self.config.items()
        }
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.socket.bind(f"tcp://{host}:{port}")
        LOG.info(
            "bound tcp://%s:%d mode=%s action_shape=%s publishers=none",
            host,
            port,
            mode,
            self.data["action"].shape,
        )

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()

    def decode_update(self, frames: list[bytes]) -> dict[str, np.ndarray]:
        if len(frames) != 2 + len(UPDATE_KEYS) or frames[0] != b"UPDATE":
            raise ValueError("malformed UPDATE multipart message")
        metadata = json.loads(frames[1].decode("utf-8"))
        decoded: dict[str, np.ndarray] = {}
        for index, key in enumerate(UPDATE_KEYS):
            descriptor = metadata[key]
            shape = tuple(int(value) for value in descriptor["shape"])
            dtype = np.dtype(descriptor["dtype"])
            expected_shape, expected_dtype = self.config[key]
            if shape != expected_shape or dtype != expected_dtype:
                raise ValueError(
                    f"{key} metadata {(shape, dtype)} != "
                    f"{(expected_shape, expected_dtype)}"
                )
            raw = frames[2 + index]
            expected_bytes = int(np.prod(shape)) * dtype.itemsize
            if len(raw) != expected_bytes:
                raise ValueError(f"{key} bytes {len(raw)} != {expected_bytes}")
            decoded[key] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return decoded

    def make_action(self) -> None:
        if self.mode == "hold":
            self.data["action"][...] = np.repeat(
                self.data["state"], self.action_steps, axis=0
            )
        elif self.mode == "zero":
            self.data["action"].fill(0.0)
        else:
            raise ValueError(f"unsupported mode {self.mode!r}")

    def serve(self, max_cycles: int) -> int:
        cycles = 0
        finish_after_snapshot = False
        try:
            while not STOP:
                try:
                    frames = self.socket.recv_multipart()
                except zmq.Again:
                    continue
                if not frames:
                    self.socket.send_multipart([b"ERR", b"empty message"])
                    continue

                message_type = frames[0]
                if message_type == b"UPDATE":
                    try:
                        updates = self.decode_update(frames)
                        for key, value in updates.items():
                            self.data[key][...] = value
                        self.make_action()
                    except (KeyError, TypeError, ValueError) as exc:
                        LOG.error("rejecting UPDATE: %s", exc)
                        self.socket.send_multipart(
                            [b"ERR", str(exc).encode("utf-8")]
                        )
                        continue
                    cycles += 1
                    LOG.info(
                        "frame=%d state_shape=%s state_mean=%.6f "
                        "front_mean=%.3f side_mean=%.3f",
                        cycles,
                        self.data["state"].shape,
                        float(self.data["state"].mean()),
                        float(self.data["cam_front"].mean()),
                        float(self.data["cam_side"].mean()),
                    )
                    self.socket.send_multipart([b"ACK"])
                    if max_cycles > 0 and cycles >= max_cycles:
                        finish_after_snapshot = True
                elif message_type == b"WAIT":
                    self.socket.send_multipart(
                        pack_arrays(b"SNAPSHOT", self.data, SNAPSHOT_KEYS)
                    )
                    if finish_after_snapshot:
                        time.sleep(0.5)
                        break
                else:
                    self.socket.send_multipart(
                        [b"ERR", f"unknown type {message_type!r}".encode("utf-8")]
                    )
        finally:
            self.close()
        LOG.info("stopped after %d frame(s)", cycles)
        return cycles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7999)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--mode", choices=("hold", "zero"), default="hold")
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    SevenDServer(args.host, args.port, args.action_steps, args.mode).serve(
        args.max_cycles
    )


if __name__ == "__main__":
    main()
