"""Low-latency WebRTC camera view for Cobot Magic.

The preview path intentionally bypasses bridge/ZMQ image transport:

    RealSense RGB -> ROS ``sensor_msgs/Image`` -> NVENC H.264 -> WebRTC

Policy rollout, recording, and robot-state transport keep using the bridge.
This avoids the previous ``JPEG -> RGB -> software H.264`` preview transcode.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterator

import aiortc.codecs as aiortc_codecs
import aiortc.rtcrtpsender as aiortc_rtcrtpsender
import aioice.ice
import av
import numpy as np
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.codecs.h264 import H264Encoder
from aiortc.mediastreams import MediaStreamTrack
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

LOG = logging.getLogger("webrtc_cam")

CAMERA_ORDER = ("front", "left", "right")
CAMERA_TOPICS = {
    "front": "/camera_f/color/image_raw",
    "left": "/camera_l/color/image_raw",
    "right": "/camera_r/color/image_raw",
}


@dataclass
class LatestFrame:
    """A thread-safe latest raw RGB sample, keyed by ROS source timestamp."""

    data: bytes | None = None
    width: int = 0
    height: int = 0
    stamp_ns: int = 0
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def put(self, data: bytes, *, width: int, height: int, stamp_ns: int) -> None:
        assert self._lock is not None
        with self._lock:
            self.data = data
            self.width = int(width)
            self.height = int(height)
            self.stamp_ns = int(stamp_ns)

    def get(self) -> tuple[bytes | None, int, int, int]:
        assert self._lock is not None
        with self._lock:
            return self.data, self.width, self.height, self.stamp_ns


class NvencH264Encoder(H264Encoder):
    """aiortc's H.264 RTP packetizer with NVIDIA NVENC encoding."""

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        # aiortc continuously retunes its sender bitrate from RTCP feedback.
        # The stock H264 encoder recreates libx264 on a >10% change, but an
        # NVENC recreation produces a visible keyframe gap and made three
        # browser tracks look like they were repeatedly disconnecting.  NVENC
        # accepts a bitrate update in its existing session; recreate only for
        # an actual video-format change.
        if self.codec and (
            frame.width != self.codec.width or frame.height != self.codec.height
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        frame.pict_type = (
            av.video.frame.PictureType.I
            if force_keyframe
            else av.video.frame.PictureType.NONE
        )
        if self.codec is None:
            codec = av.CodecContext.create("h264_nvenc", "w")
            codec.width = frame.width
            codec.height = frame.height
            codec.bit_rate = self.target_bitrate
            codec.pix_fmt = "yuv420p"
            codec.framerate = Fraction(30, 1)
            codec.time_base = Fraction(1, 30)
            codec.options = {
                "preset": "p1",
                "tune": "ll",
                "rc": "cbr",
                "zerolatency": "1",
                # Without ``delay=0`` this FFmpeg/NVENC build buffers the
                # first camera frame until a later encode/flush call.
                "delay": "0",
                "bf": "0",
            }
            codec.profile = "baseline"
            self.codec = codec
            LOG.info(
                "NVENC H.264 initialized: %dx%d @ %d bps",
                codec.width,
                codec.height,
                codec.bit_rate,
            )
        else:
            self.codec.bit_rate = self.target_bitrate

        bitstream = b"".join(bytes(packet) for packet in self.codec.encode(frame))
        if bitstream:
            yield from self._split_bitstream(bitstream)


def _install_nvenc_h264_encoder() -> None:
    """Patch aiortc locally, without editing its installed package."""

    if getattr(aiortc_codecs, "_cobot_nvenc_h264", False):
        return
    original_get_encoder = aiortc_codecs.get_encoder

    def get_encoder(codec: Any) -> Any:
        if codec.mimeType.lower() == "video/h264":
            return NvencH264Encoder()
        return original_get_encoder(codec)

    aiortc_codecs.get_encoder = get_encoder
    # RTCRtpSender imported get_encoder by value when its module loaded.
    aiortc_rtcrtpsender.get_encoder = get_encoder
    aiortc_codecs._cobot_nvenc_h264 = True


def _restrict_ice_to_host(host: str) -> None:
    """Advertise only the direct robot-LAN address for WebRTC media.

    The AGX also has Wi-Fi and Internet-facing interfaces.  Advertising every
    address makes a dual-homed PC probe and sometimes select a less stable
    Wi-Fi pair even when its robot cable is connected.  Signaling remains on
    its normal HTTP port; this limits only UDP media candidates.
    """

    original_get_host_addresses = aioice.ice.get_host_addresses

    def get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        addresses = original_get_host_addresses(use_ipv4, use_ipv6)
        if use_ipv4 and host in addresses:
            return [host]
        return addresses

    aioice.ice.get_host_addresses = get_host_addresses


class RosCameraSource(Node):
    """Direct subscriber for the RealSense raw RGB topics."""

    def __init__(self, slots: dict[str, LatestFrame]) -> None:
        super().__init__("cobot_webrtc_camera_source")
        self._slots = slots
        self._invalid_encoding_logged: set[str] = set()
        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image,
                topic,
                lambda message, camera=name: self._callback(camera, message),
                qos_profile_sensor_data,
            )

    def _callback(self, name: str, message: Image) -> None:
        if message.encoding.lower() != "rgb8":
            if name not in self._invalid_encoding_logged:
                self.get_logger().error(
                    "%s expected rgb8 but received %s; preview frame skipped",
                    name,
                    message.encoding,
                )
                self._invalid_encoding_logged.add(name)
            return
        expected = int(message.width) * int(message.height) * 3
        if int(message.step) != int(message.width) * 3 or len(message.data) != expected:
            self.get_logger().error(
                "%s received non-contiguous rgb8 (%d bytes, step=%d, expected=%d)",
                name,
                len(message.data),
                message.step,
                expected,
            )
            return
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if stamp_ns <= 0:
            self.get_logger().error("%s frame has no ROS timestamp; preview frame skipped", name)
            return
        self._slots[name].put(
            bytes(message.data),
            width=int(message.width),
            height=int(message.height),
            stamp_ns=stamp_ns,
        )


class CameraTrack(MediaStreamTrack):
    """One H.264/NVENC WebRTC track, paced by source camera timestamps."""

    kind = "video"

    def __init__(self, name: str, slot: LatestFrame, *, fps: int, max_w: int, max_h: int) -> None:
        super().__init__()
        self.name = name
        self.slot = slot
        self.fps = fps
        self.max_w = max_w
        self.max_h = max_h
        self._last_stamp_ns = 0
        self._seq = 0
        self._logged_size = False

    async def recv(self) -> av.VideoFrame:
        while True:
            data, width, height, stamp_ns = self.slot.get()
            if data is None or stamp_ns <= self._last_stamp_ns:
                await asyncio.sleep(0.003)
                continue
            self._last_stamp_ns = stamp_ns
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            if width > self.max_w or height > self.max_h:
                scale = min(self.max_w / width, self.max_h / height)
                new_w = max(2, int(round(width * scale // 2)) * 2)
                new_h = max(2, int(round(height * scale // 2)) * 2)
                frame = frame.reformat(width=new_w, height=new_h)
            if not self._logged_size:
                LOG.info("camera %s raw RGB %dx%d -> H.264 NVENC", self.name, width, height)
                self._logged_size = True
            frame.pts = self._seq
            frame.time_base = Fraction(1, self.fps)
            self._seq += 1
            return frame


class WebRtcCamServer:
    def __init__(
        self, *, port: int, fps: int, max_w: int, max_h: int, ice_host: str
    ) -> None:
        self.port = port
        self.fps = fps
        self.max_w = max_w
        self.max_h = max_h
        self.ice_host = ice_host
        self.slots = {name: LatestFrame() for name in CAMERA_ORDER}

    async def _index(self, _request: web.Request) -> web.Response:
        return web.Response(text=_PAGE, content_type="text/html")

    async def _signal(self, request: web.Request) -> web.WebSocketResponse:
        name = request.query.get("camera", "")
        if name not in self.slots:
            return web.Response(status=400, text=f"unknown camera {name!r}")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = CameraTrack(
            name,
            self.slots[name],
            fps=self.fps,
            max_w=self.max_w,
            max_h=self.max_h,
        )
        sender = peer.addTrack(track)
        transceiver = next(item for item in peer.getTransceivers() if item.sender is sender)
        h264 = [
            capability
            for capability in RTCRtpSender.getCapabilities("video").codecs
            if capability.mimeType.lower() == "video/h264"
        ]
        if not h264:
            await peer.close()
            return web.Response(status=500, text="aiortc has no H.264 capability")
        transceiver.setCodecPreferences(h264)
        try:
            async for message in ws:
                if message.type == web.WSMsgType.TEXT:
                    # Chrome protects host addresses in SDP with an mDNS
                    # ``*.local`` candidate.  This PC has several NICs, and
                    # the robot resolves that name to its unrelated USB NIC
                    # (192.168.25.199) while Chrome sends media from the
                    # direct robot LAN (192.168.250.1).  Use the already
                    # authenticated signaling peer address for *host* mDNS
                    # candidates, which makes the SDP and UDP source agree.
                    # Do not change normal IP, srflx, or relay candidates.
                    peer_host = request.remote
                    offer_sdp = message.data
                    rewritten = 0
                    if peer_host:
                        rewritten_lines: list[str] = []
                        for line in offer_sdp.splitlines(keepends=True):
                            fields = line.rstrip("\r\n").split()
                            if (
                                len(fields) >= 8
                                and fields[0].startswith("a=candidate:")
                                and fields[4].endswith(".local")
                                and fields[6] == "typ"
                                and fields[7] == "host"
                            ):
                                fields[4] = peer_host
                                line = " ".join(fields) + ("\r\n" if line.endswith("\r\n") else "\n")
                                rewritten += 1
                            rewritten_lines.append(line)
                        offer_sdp = "".join(rewritten_lines)
                    if rewritten:
                        LOG.info(
                            "camera %s rewrote %d browser mDNS ICE candidate(s) to signaling peer %s",
                            name,
                            rewritten,
                            peer_host,
                        )
                    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
                    await peer.setRemoteDescription(offer)
                    answer = await peer.createAnswer()
                    await peer.setLocalDescription(answer)
                    deadline = asyncio.get_running_loop().time() + 5.0
                    while (
                        peer.iceGatheringState != "complete"
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.01)
                    await ws.send_str(peer.localDescription.sdp)
                elif message.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        finally:
            await peer.close()
            if not ws.closed:
                await ws.close()
        return ws

    def run(self) -> None:
        import rclpy

        _install_nvenc_h264_encoder()
        _restrict_ice_to_host(self.ice_host)
        rclpy.init()
        source = RosCameraSource(self.slots)
        executor = SingleThreadedExecutor()
        executor.add_node(source)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/signal", self._signal)
        LOG.info(
            "WebRTC camera server on 0.0.0.0:%d (direct ROS RGB -> NVENC H.264; "
            "ice_host=%s fps=%d max=%dx%d)",
            self.port,
            self.ice_host,
            self.fps,
            self.max_w,
            self.max_h,
        )
        try:
            web.run_app(app, host="0.0.0.0", port=self.port, print=False)
        finally:
            executor.shutdown()
            source.destroy_node()
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Cobot Live</title></head>
<body><div id="state">connecting...</div><script>
const camera = new URLSearchParams(location.search).get("camera") || "front";
const video = document.createElement("video"); video.autoplay = true; video.playsInline = true;
video.style.cssText = "width:100%;max-width:960px;background:#000"; document.body.append(video);
const pc = new RTCPeerConnection({iceServers: []});
pc.ontrack = (event) => { video.srcObject = event.streams[0]; document.querySelector('#state').textContent = camera + ' live'; };
const ws = new WebSocket(`ws://${location.host}/signal?camera=${encodeURIComponent(camera)}`);
ws.onopen = async () => { pc.addTransceiver('video', {direction:'recvonly'}); const offer=await pc.createOffer(); await pc.setLocalDescription(offer); ws.send(offer.sdp); };
ws.onmessage = async (event) => { await pc.setRemoteDescription({type:'answer', sdp:event.data}); };
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-endpoint",
        default=None,
        help="deprecated compatibility option; preview reads ROS directly",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ice-host", default="192.168.250.2")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--max-height", type=int, default=480)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.bridge_endpoint:
        LOG.info("ignoring deprecated --bridge-endpoint=%s", args.bridge_endpoint)
    if os.environ.get("WEbrtc_DEBUG", "").lower() in ("1", "true"):
        for logger_name in ("aiortc", "aioice"):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    WebRtcCamServer(
        port=args.port,
        fps=args.fps,
        max_w=args.max_width,
        max_h=args.max_height,
        ice_host=args.ice_host,
    ).run()


if __name__ == "__main__":
    main()
