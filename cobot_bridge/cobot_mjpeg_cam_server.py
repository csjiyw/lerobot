"""AGX-local, low-latency JPEG preview for the Cobot dashboard.

The dashboard itself is served by the PC, but when it is opened in the AGX
touchscreen browser its camera ``<img>`` elements resolve ``127.0.0.1`` on the
AGX.  This service deliberately binds only that loopback address:

    bridge JPEG cache (ZMQ) -> local HTTP multipart JPEG -> AGX browser

It is a display-only path.  It never publishes ROS commands and never opens a
raw image subscription.  Each browser client receives only a newly observed
JPEG, avoiding the H.264 / WebRTC jitter buffer that can grow latency when a
frame is briefly delayed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from aiohttp import web

from .client import CobotBridgeClient


LOG = logging.getLogger("mjpeg_cam")
CAMERAS = ("front", "left", "right")
BOUNDARY = "cobot-frame"


@dataclass
class LatestJpeg:
    """A latest-only JPEG slot shared by the bridge pump and HTTP clients."""

    data: bytes | None = None
    stamp_ns: int = 0
    version: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)

    def put(self, data: bytes, stamp_ns: int) -> None:
        with self.condition:
            # The bridge can be polled faster than a camera publishes.  Do
            # not generate duplicate multipart frames for the same source
            # sample, otherwise a browser is free to build a display queue.
            if stamp_ns > 0 and stamp_ns == self.stamp_ns:
                return
            self.data = data
            self.stamp_ns = stamp_ns
            self.version += 1
            self.condition.notify_all()

    def wait_after(self, version: int, timeout_s: float) -> tuple[bytes | None, int, int]:
        with self.condition:
            self.condition.wait_for(lambda: self.version > version, timeout=timeout_s)
            return self.data, self.stamp_ns, self.version


class BridgeJpegPump:
    """Poll the bridge's existing three-camera JPEG cache at a fixed cadence."""

    def __init__(self, endpoint: str, fps: int) -> None:
        self.endpoint = endpoint
        self.period_s = 1.0 / fps
        self.slots = {name: LatestJpeg() for name in CAMERAS}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="mjpeg-bridge-pump", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        client = CobotBridgeClient(self.endpoint, timeout_s=0.8)
        next_tick = time.monotonic()
        last_error_log = 0.0
        pulls = 0
        try:
            while not self.stop_event.is_set():
                try:
                    observation, images = client.get_observation(
                        sides=("left",), images=CAMERAS
                    )
                    image_meta = observation.get("images", {})
                    for name in CAMERAS:
                        data = images.get(name)
                        metadata = image_meta.get(name) if isinstance(image_meta, dict) else None
                        stamp_ns = int(metadata.get("stamp_ns", 0)) if isinstance(metadata, dict) else 0
                        if data:
                            self.slots[name].put(bytes(data), stamp_ns)
                    pulls += 1
                    if pulls == 1:
                        LOG.info(
                            "bridge JPEG pump ready: %s",
                            {name: len(images.get(name, b"")) for name in CAMERAS},
                        )
                except Exception as exc:  # noqa: BLE001 - preview must retry independently
                    now = time.monotonic()
                    if now - last_error_log >= 5.0:
                        LOG.warning("bridge JPEG pull failed: %s", exc)
                        last_error_log = now

                next_tick += self.period_s
                delay = next_tick - time.monotonic()
                if delay <= 0:
                    next_tick = time.monotonic()
                else:
                    self.stop_event.wait(delay)
        finally:
            client.close()


class MjpegServer:
    def __init__(self, *, bridge_endpoint: str, port: int, fps: int) -> None:
        self.port = port
        self.pump = BridgeJpegPump(bridge_endpoint, fps)

    async def mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.query.get("camera", "")
        slot = self.pump.slots.get(name)
        if slot is None:
            raise web.HTTPBadRequest(text=f"unknown camera {name!r}")

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        version = 0
        try:
            while True:
                data, stamp_ns, version = await asyncio.to_thread(slot.wait_after, version, 1.0)
                if not data:
                    continue
                header = (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(data)}\r\n"
                    f"X-Source-Stamp-Ns: {stamp_ns}\r\n\r\n"
                ).encode("ascii")
                await response.write(header + data + b"\r\n")
        except (asyncio.CancelledError, ConnectionError, RuntimeError):
            pass
        finally:
            try:
                await response.write_eof()
            except (ConnectionError, RuntimeError):
                pass
        return response

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                name: {"stamp_ns": slot.stamp_ns, "version": slot.version}
                for name, slot in self.pump.slots.items()
            }
        )

    def run(self) -> None:
        self.pump.start()
        app = web.Application()
        app.router.add_get("/mjpeg", self.mjpeg)
        app.router.add_get("/health", self.health)
        LOG.info("AGX-local JPEG preview on 127.0.0.1:%d", self.port)
        try:
            web.run_app(app, host="127.0.0.1", port=self.port, print=False)
        finally:
            self.pump.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.fps <= 0 or args.fps > 60:
        parser.error("--fps must be in [1, 60]")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    MjpegServer(
        bridge_endpoint=args.bridge_endpoint,
        port=args.port,
        fps=args.fps,
    ).run()


if __name__ == "__main__":
    main()
