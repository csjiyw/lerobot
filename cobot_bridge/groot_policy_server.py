"""Serve a Cobot Magic GR00T checkpoint through the existing rollout protocol."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import zmq

from .policy_client import (
    PolicyServerError,
    decode_policy_json,
    encode_policy_json,
    make_policy_response,
    validate_policy_request,
)
from .protocol import CAMERAS


LOG = logging.getLogger("cobot_bridge.groot_policy_server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5560")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=("front", "right"))
    return parser


def _decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PolicyServerError("invalid policy JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _format_observation(
    state: Sequence[float],
    images: Mapping[str, np.ndarray],
    *,
    cameras: Sequence[str],
    task: str,
    pose_state_key: str,
) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise PolicyServerError("state must contain seven finite values")
    selected = tuple(str(name) for name in cameras)
    if selected != ("front", "right"):
        raise PolicyServerError(f"GR00T checkpoint requires cameras front/right, got {selected}")

    video: dict[str, np.ndarray] = {}
    for name in selected:
        if name not in images:
            raise PolicyServerError(f"missing GR00T image: {name}")
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise PolicyServerError(f"invalid {name} RGB image: {image.shape} {image.dtype}")
        video[name] = image[None, None, ...]

    return {
        "video": video,
        "state": {
            pose_state_key: values[:6][None, None, ...],
            "gripper": values[6:][None, None, ...],
        },
        "language": {"annotation.human.task_description": [[str(task)]]},
    }


def _merge_action_chunk(actions: Mapping[str, Any], *, pose_action_key: str) -> np.ndarray:
    pose = np.asarray(actions.get(pose_action_key), dtype=np.float32)
    gripper = np.asarray(actions.get("gripper"), dtype=np.float32)
    if pose.ndim != 3 or pose.shape[0] != 1 or pose.shape[2] != 6:
        raise PolicyServerError(
            f"GR00T returned invalid {pose_action_key} chunk: {pose.shape}"
        )
    if gripper.ndim != 3 or gripper.shape[0] != 1 or gripper.shape[2] != 1:
        raise PolicyServerError(f"GR00T returned invalid gripper chunk: {gripper.shape}")
    if pose.shape[1] != gripper.shape[1]:
        raise PolicyServerError(
            "GR00T pose/gripper horizon mismatch: "
            f"{pose.shape[1]} != {gripper.shape[1]}"
        )
    chunk = np.concatenate((pose[0], gripper[0]), axis=-1)
    if not np.isfinite(chunk).all():
        raise PolicyServerError("GR00T returned non-finite actions")
    return chunk.astype(np.float32, copy=False)


def _predict_chunk(
    policy: Any, observation: dict[str, Any], *, pose_action_key: str
) -> np.ndarray:
    actions, _ = policy.get_action(observation)
    return _merge_action_chunk(actions, pose_action_key=pose_action_key)


def _resolve_pose_key(modality: Mapping[str, Any], group: str) -> str:
    """Return the single six-DoF stream used alongside the gripper stream.

    Older dashboard checkpoints use ``arm``; the N1.7 relative-EEF checkpoint
    uses ``eef_pose``.  The transport protocol is the same seven-value robot
    state in both cases, so the adapter derives the checkpoint key instead of
    rewriting model output or applying a second relative-to-absolute transform.
    """
    keys = tuple(str(key) for key in modality[group].modality_keys)
    pose_keys = tuple(key for key in keys if key != "gripper")
    if len(pose_keys) != 1 or "gripper" not in keys:
        raise RuntimeError(
            f"unsupported GR00T {group} schema {keys}; expected one pose key plus gripper"
        )
    return pose_keys[0]


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    modality = policy.get_modality_config()
    configured_cameras = tuple(args.cameras)
    expected_cameras = tuple(modality["video"].modality_keys)
    if configured_cameras != expected_cameras:
        raise RuntimeError(
            f"checkpoint camera schema is {expected_cameras}, server configured {configured_cameras}"
        )
    pose_state_key = _resolve_pose_key(modality, "state")
    pose_action_key = _resolve_pose_key(modality, "action")
    chunk_size = len(modality["action"].delta_indices)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    LOG.info(
        "GR00T policy server ready bind=%s device=%s cameras=%s chunk=%d "
        "state=%s action=%s model=%s",
        args.bind,
        args.device,
        configured_cameras,
        chunk_size,
        pose_state_key,
        pose_action_key,
        args.model_path,
    )
    shutdown_requested = False
    try:
        while not shutdown_requested:
            frames = socket.recv_multipart()
            request: dict[str, Any] = {}
            try:
                if not frames:
                    raise PolicyServerError("empty policy request")
                request = validate_policy_request(decode_policy_json(frames[0]))
                command = request["command"]
                if command == "PING":
                    result = {
                        "ready": True,
                        "policy_type": "groot",
                        "device": str(args.device),
                        "cameras": list(configured_cameras),
                        "policy_path": str(args.model_path),
                        "chunk_size": chunk_size,
                        "n_action_steps": chunk_size,
                        # ``Gr00tPolicy`` converts the checkpoint's relative
                        # EEF representation back to absolute EEF targets.
                        # The dashboard uses this flag to require the EEF V5
                        # (Pink IK) execution path rather than a joint runner.
                        "relative_eef_actions": pose_action_key == "eef_pose",
                        "action_space": "eef_target"
                        if pose_action_key == "eef_pose"
                        else "joint_target",
                        "server_time_ns": time.time_ns(),
                    }
                elif command == "RESET":
                    policy.reset()
                    result = {"reset": True}
                elif command == "SHUTDOWN":
                    shutdown_requested = True
                    result = {"shutting_down": True}
                else:
                    cameras = tuple(str(name) for name in request.get("cameras", ()))
                    if cameras != configured_cameras:
                        raise PolicyServerError(
                            f"camera mismatch: request={cameras} configured={configured_cameras}"
                        )
                    if len(frames) != len(cameras) + 1:
                        raise PolicyServerError(
                            f"expected {len(cameras)} JPEG frames, received {len(frames) - 1}"
                        )
                    images = {
                        name: _decode_jpeg(data)
                        for name, data in zip(cameras, frames[1:], strict=True)
                    }
                    observation = _format_observation(
                        request.get("state", ()),
                        images,
                        cameras=cameras,
                        task=str(request.get("task", "")),
                        pose_state_key=pose_state_key,
                    )
                    started = time.perf_counter()
                    chunk = _predict_chunk(
                        policy, observation, pose_action_key=pose_action_key
                    )
                    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    if command == "PREDICT_CHUNK":
                        result = {
                            "action_chunk": chunk.tolist(),
                            "model_native_action_chunk": chunk.tolist(),
                            "model_output_semantics": "absolute_eef"
                            if pose_action_key == "eef_pose"
                            else "absolute_joint",
                            "chunk_size": int(chunk.shape[0]),
                            "latency_ms": latency_ms,
                        }
                    else:
                        result = {
                            "action": chunk[0].tolist(),
                            "model_output_semantics": "absolute_eef"
                            if pose_action_key == "eef_pose"
                            else "absolute_joint",
                            "latency_ms": latency_ms,
                        }
                response = make_policy_response(request, ok=True, result=result)
            except Exception as exc:
                LOG.exception("GR00T policy request failed")
                response = make_policy_response(request, ok=False, error=str(exc))
            socket.send_multipart([encode_policy_json(response)])
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
