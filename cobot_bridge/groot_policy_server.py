"""Serve a Cobot Magic GR00T checkpoint through the existing rollout protocol."""

from __future__ import annotations

import argparse
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

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
    parser.add_argument(
        "--backbone-path",
        default=None,
        help="Optional local Cosmos/Qwen backbone path overriding the training-host path.",
    )
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
    camera_key_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise PolicyServerError("state must contain seven finite values")
    selected = tuple(str(name) for name in cameras)
    if selected != ("front", "right"):
        raise PolicyServerError(f"GR00T checkpoint requires cameras front/right, got {selected}")

    video: dict[str, np.ndarray] = {}
    key_map = camera_key_map or {name: name for name in selected}
    for name in selected:
        if name not in images:
            raise PolicyServerError(f"missing GR00T image: {name}")
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise PolicyServerError(f"invalid {name} RGB image: {image.shape} {image.dtype}")
        model_key = key_map.get(name)
        if not model_key:
            raise PolicyServerError(f"no GR00T video feature mapping for camera {name}")
        video[model_key] = image[None, None, ...]

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


def _resolve_camera_key_map(
    model_cameras: Sequence[str], runtime_cameras: Sequence[str]
) -> dict[str, str]:
    """Map bridge camera names to the checkpoint's video modality keys."""
    runtime = tuple(str(name) for name in runtime_cameras)
    model = tuple(str(name) for name in model_cameras)
    if runtime != ("front", "right"):
        raise RuntimeError(f"GR00T server requires runtime cameras front/right, got {runtime}")
    if set(model) == set(runtime):
        mapping = {name: name for name in runtime}
    elif set(model) == {"cam_high", "cam_right_wrist"}:
        mapping = {"front": "cam_high", "right": "cam_right_wrist"}
    else:
        raise RuntimeError(
            "unsupported GR00T camera schema: "
            f"checkpoint={model}, runtime={runtime}; expected front/right or "
            "cam_high/cam_right_wrist"
        )
    if set(mapping.values()) != set(model):
        raise RuntimeError(f"incomplete GR00T camera mapping: {mapping} -> {model}")
    return mapping


def _resolve_checkpoint_embodiment_tag(model_path: str | Path) -> str:
    """Read the run's tag from saved training config, then verify processor support."""
    model_dir = Path(model_path).expanduser()
    processor_config = model_dir / "processor_config.json"
    if not processor_config.is_file():
        processor_config = model_dir / "processor" / "processor_config.json"
    if not processor_config.is_file():
        raise FileNotFoundError(f"GR00T processor config not found under {model_dir}")
    with processor_config.open() as stream:
        processor = json.load(stream)
    supported = set(
        processor.get("processor_kwargs", {}).get("modality_configs", {}).keys()
    )

    run_config = model_dir / "experiment_cfg" / "conf.yaml"
    if run_config.is_file():
        try:
            import yaml

            with run_config.open() as stream:
                config = yaml.safe_load(stream) or {}
            datasets = config.get("data", {}).get("datasets", [])
            tags = [str(item.get("embodiment_tag", "")) for item in datasets]
            selected = next((tag for tag in tags if tag in supported), None)
            if selected:
                return selected
        except (OSError, TypeError, ValueError):
            LOG.warning("Could not parse saved GR00T experiment config %s", run_config)

    # Older exports may not include experiment_cfg. Keep the server's historic
    # NEW_EMBODIMENT behavior when its processor explicitly supports that tag.
    for candidate in ("new_embodiment", "cobotmagic"):
        if candidate in supported:
            return candidate
    raise RuntimeError(
        "Could not identify a supported embodiment tag from checkpoint metadata; "
        f"processor tags include {sorted(supported)}"
    )


def _load_policy(model_path: str, *, device: str, backbone_path: str | None):
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers import AutoModel, AutoProcessor

    from gr00t.data.embodiment_tags import EmbodimentTag

    tag_value = _resolve_checkpoint_embodiment_tag(model_path)
    try:
        embodiment_tag: Any = EmbodimentTag.resolve(tag_value)
    except ValueError:
        # Some project-specific tags (e.g. cobotmagic) are valid processor keys
        # but intentionally have no public EmbodimentTag enum member.
        embodiment_tag = Enum(
            "CheckpointEmbodimentTag", {"SELECTED": tag_value}
        ).SELECTED

    if not backbone_path:
        return Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=True,
        ), tag_value

    local_backbone = Path(backbone_path).expanduser().resolve()
    if not local_backbone.is_dir():
        raise FileNotFoundError(f"local GR00T backbone does not exist: {local_backbone}")
    original_model_loader = AutoModel.from_pretrained
    original_processor_loader = AutoProcessor.from_pretrained

    def load_model(path, *args, **kwargs):
        kwargs["model_name"] = str(local_backbone)
        return original_model_loader(path, *args, **kwargs)

    def load_processor(path, *args, **kwargs):
        kwargs["model_name"] = str(local_backbone)
        kwargs["transformers_loading_kwargs"] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        return original_processor_loader(path, *args, **kwargs)

    # Checkpoints copied from another machine embed the original absolute
    # backbone path. Override only that path during loading; checkpoint weights,
    # modality config, embodiment id, and normalization stats remain untouched.
    with patch.object(AutoModel, "from_pretrained", new=staticmethod(load_model)):
        with patch.object(
            AutoProcessor, "from_pretrained", new=staticmethod(load_processor)
        ):
            policy = Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=device,
                strict=True,
            )
    return policy, tag_value


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    policy, embodiment_tag = _load_policy(
        args.model_path, device=args.device, backbone_path=args.backbone_path
    )
    modality = policy.get_modality_config()
    configured_cameras = tuple(args.cameras)
    expected_cameras = tuple(modality["video"].modality_keys)
    camera_key_map = _resolve_camera_key_map(expected_cameras, configured_cameras)
    pose_state_key = _resolve_pose_key(modality, "state")
    pose_action_key = _resolve_pose_key(modality, "action")
    chunk_size = len(modality["action"].delta_indices)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    LOG.info(
        "GR00T policy server ready bind=%s device=%s cameras=%s chunk=%d "
        "state=%s action=%s cameras=%s->%s embodiment=%s model=%s",
        args.bind,
        args.device,
        configured_cameras,
        chunk_size,
        pose_state_key,
        pose_action_key,
        configured_cameras,
        expected_cameras,
        embodiment_tag,
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
                        "embodiment_tag": embodiment_tag,
                        "device": str(args.device),
                        "cameras": list(configured_cameras),
                        "model_cameras": list(expected_cameras),
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
                        camera_key_map=camera_key_map,
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
