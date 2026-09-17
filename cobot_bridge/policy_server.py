"""Persistent local SmolVLA inference server for repeated robot rollouts."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np
import torch
import zmq

from .policy_client import (
    PolicyServerError,
    decode_policy_json,
    encode_policy_json,
    make_policy_response,
    validate_policy_request,
)
from .protocol import CAMERAS
from .run_lerobot_policy import _load_policy


LOG = logging.getLogger("cobot_bridge.policy_server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5560")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=("front", "right"))
    parser.add_argument(
        "--inference-chunking",
        choices=("original", "guided_rtc"),
        default="original",
        help="original chunk sampling or Guided RTC for flow-matching SmolVLA",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="Guided RTC overlap/execution horizon in policy-rate actions",
    )
    parser.add_argument(
        "--vlm-model-path",
        help="local SmolVLM config/processor directory; skip loading duplicate base weights",
    )
    return parser


def _decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PolicyServerError("invalid policy JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _policy_key(name: str) -> str:
    return f"observation.images.camera_{name[0] if name != 'front' else 'f'}"


def _predict_full_chunk(
    observation: dict[str, np.ndarray],
    *,
    policy,
    device: torch.device,
    preprocessor,
    postprocessor,
    task: str,
    robot_type: str,
    prev_chunk_left_over: torch.Tensor | None = None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
) -> np.ndarray:
    from lerobot.policies.utils import prepare_observation_for_inference

    # RTC computes a vector-Jacobian product inside each denoise step.  It can
    # temporarily enable autograd under ``no_grad``, but never under
    # ``inference_mode``.  The no-leftover first chunk remains pure inference.
    inference_context = nullcontext() if prev_chunk_left_over is not None else torch.inference_mode()
    with (
        inference_context,
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and bool(getattr(policy.config, "use_amp", False))
        else nullcontext(),
    ):
        batch = prepare_observation_for_inference(observation, device, task, robot_type)
        batch = preprocessor(batch)
        action_chunk = policy.predict_action_chunk(
            batch,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        processed = [postprocessor(action_chunk[:, index, :]) for index in range(action_chunk.shape[1])]
        action_chunk = torch.stack(processed, dim=1).squeeze(0)
    return action_chunk.detach().cpu().numpy().astype(np.float32, copy=False)


def _processor_uses_relative_eef_actions(preprocessor) -> bool:
    """Whether the loaded processor defines chunk-relative EEF action semantics."""
    return any(
        getattr(step, "_registry_name", None) == "smolvla_relative_eef_chunk"
        for step in preprocessor.steps
    )


def _restore_relative_eef_chunk(preprocessor, state: np.ndarray, action_chunk: np.ndarray) -> np.ndarray:
    """Restore processor-relative EEF actions to absolute model-frame targets."""
    if not _processor_uses_relative_eef_actions(preprocessor):
        return action_chunk
    from lerobot.policies.smolvla.relative_eef import relative_chunk_to_absolute

    return relative_chunk_to_absolute(state, action_chunk)


def _configure_guided_rtc(policy, *, execution_horizon: int) -> None:
    """Enable the vendored SmolVLA RTC sampler on this resident policy."""

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if not hasattr(policy, "init_rtc_processor") or not hasattr(policy, "config"):
        raise RuntimeError("Guided RTC is supported only by local flow-matching SmolVLA policies")
    chunk_size = int(getattr(policy.config, "chunk_size", 0))
    if execution_horizon < 1 or execution_horizon >= chunk_size:
        raise ValueError(
            "rtc_execution_horizon must be in [1, chunk_size); "
            f"got {execution_horizon} for chunk_size={chunk_size}"
        )
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=int(execution_horizon),
    )
    policy.init_rtc_processor()


def _normalize_rtc_leftover(
    preprocessor,
    *,
    state: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert committed physical targets into the loaded model's action space.

    Relative-EEF checkpoints are anchored to *this inference observation*, not
    the observation that originally produced the targets.  Stats stay wholly
    inside the loaded checkpoint processor.
    """

    values = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != (7,) or not np.isfinite(values).all():
        raise PolicyServerError("stored RTC targets must be a finite (steps, 7) array")
    if len(values) < 1:
        raise PolicyServerError("stored RTC targets must not be empty")
    if _processor_uses_relative_eef_actions(preprocessor):
        from lerobot.policies.smolvla.relative_eef import chunk_to_relative

        values = chunk_to_relative(state, values)
    from lerobot.processor import NormalizerProcessorStep

    normalizers = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    if len(normalizers) != 1:
        raise RuntimeError(f"expected one checkpoint normalizer, found {len(normalizers)}")
    normalized = normalizers[0]._normalize_action(
        torch.as_tensor(values, dtype=torch.float32, device=device), inverse=False
    )
    return normalized.unsqueeze(0)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy, preprocessor, postprocessor = _load_policy(args)
    if args.inference_chunking == "guided_rtc":
        _configure_guided_rtc(policy, execution_horizon=args.rtc_execution_horizon)
    from lerobot.utils.control_utils import predict_action

    device = torch.device(policy.config.device)
    configured_cameras = tuple(args.cameras)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    LOG.info(
        "policy server ready bind=%s device=%s cameras=%s policy=%s chunking=%s",
        args.bind,
        device,
        configured_cameras,
        args.policy_path,
        args.inference_chunking,
    )
    shutdown_requested = False
    # Owned entirely by the resident server.  It is intentionally saved after
    # model-space restoration, so the next request can re-anchor a relative
    # EEF target to its fresh EEF observation or normalize an absolute-joint
    # target with the exact same checkpoint processor.
    rtc_previous_terminal: np.ndarray | None = None
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
                        "device": str(device),
                        "cameras": list(configured_cameras),
                        "policy_path": str(args.policy_path),
                        "chunk_size": int(policy.config.chunk_size),
                        "n_action_steps": int(policy.config.n_action_steps),
                        "inference_chunking": args.inference_chunking,
                        "relative_eef_actions": _processor_uses_relative_eef_actions(preprocessor),
                        "rtc": (
                            {
                                "execution_horizon": int(args.rtc_execution_horizon),
                                "prefix_attention_schedule": "EXP",
                                "max_guidance_weight": 10.0,
                                "terminal_available": rtc_previous_terminal is not None,
                            }
                            if args.inference_chunking == "guided_rtc"
                            else None
                        ),
                        "server_time_ns": time.time_ns(),
                    }
                elif command == "RESET":
                    policy.reset()
                    rtc_previous_terminal = None
                    result = {"reset": True, "rtc_terminal_cleared": True}
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
                    state = np.asarray(request.get("state", []), dtype=np.float32)
                    if state.shape != (7,) or not np.isfinite(state).all():
                        raise PolicyServerError("state must contain seven finite values")
                    observation: dict[str, np.ndarray] = {"observation.state": state}
                    for name, data in zip(cameras, frames[1:], strict=True):
                        observation[_policy_key(name)] = _decode_jpeg(data)
                    started = time.perf_counter()
                    if command == "PREDICT_CHUNK":
                        if "rtc_leftover_targets" in request or "rtc_inference_delay" in request:
                            raise PolicyServerError(
                                "RTC continuity is server-owned; clients must not supply prior actions or delay"
                            )
                        rtc_leftover = None
                        rtc_delay = None
                        if args.inference_chunking == "guided_rtc" and rtc_previous_terminal is not None:
                            rtc_leftover = _normalize_rtc_leftover(
                                preprocessor,
                                state=state,
                                targets=rtc_previous_terminal[None, :],
                                device=device,
                            )
                            rtc_delay = 0
                        # Preserve the checkpoint-native output separately
                        # from the restored robot-space target.  For a
                        # relative-EEF policy these differ: the former is the
                        # relative model action and the latter is the absolute
                        # EEF target consumed by the IK boundary.
                        native_values = _predict_full_chunk(
                            observation,
                            policy=policy,
                            device=device,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            task=str(request.get("task", "")),
                            robot_type=str(request.get("robot_type", "cobot_magic_piper")),
                            prev_chunk_left_over=rtc_leftover,
                            inference_delay=rtc_delay,
                            execution_horizon=(
                                int(args.rtc_execution_horizon)
                                if args.inference_chunking == "guided_rtc"
                                else None
                            ),
                        )
                        if (
                            native_values.ndim != 2
                            or native_values.shape[1:] != (7,)
                            or not np.isfinite(native_values).all()
                        ):
                            raise PolicyServerError(
                                "policy returned invalid native action chunk: "
                                f"{native_values.shape}"
                            )
                        values = _restore_relative_eef_chunk(
                            preprocessor, state, native_values
                        )
                        if values.ndim != 2 or values.shape[1:] != (7,) or not np.isfinite(values).all():
                            raise PolicyServerError(f"policy returned invalid action chunk: {values.shape}")
                        if args.inference_chunking == "guided_rtc":
                            rtc_previous_terminal = values[-1].astype(np.float32, copy=True)
                        result = {
                            "action_chunk": values.tolist(),
                            "model_native_action_chunk": native_values.tolist(),
                            "model_output_semantics": (
                                "relative_eef"
                                if _processor_uses_relative_eef_actions(preprocessor)
                                else "absolute_joint"
                            ),
                            "chunk_size": int(values.shape[0]),
                            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                            "rtc_prefix_steps": 0 if rtc_leftover is None else int(rtc_leftover.shape[1]),
                        }
                    else:
                        if args.inference_chunking == "guided_rtc":
                            raise PolicyServerError("Guided RTC requires PREDICT_CHUNK")
                        if _processor_uses_relative_eef_actions(preprocessor):
                            raise PolicyServerError(
                                "PREDICT is unsupported for a relative-EEF policy; use PREDICT_CHUNK"
                            )
                        action = predict_action(
                            observation=observation,
                            policy=policy,
                            device=device,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            use_amp=bool(getattr(policy.config, "use_amp", False)),
                            task=str(request.get("task", "")),
                            robot_type=str(request.get("robot_type", "cobot_magic_piper")),
                        )
                        if isinstance(action, torch.Tensor):
                            values = action.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                        else:
                            values = np.asarray(action, dtype=np.float32).reshape(-1)
                        if values.shape != (7,) or not np.isfinite(values).all():
                            raise PolicyServerError(f"policy returned invalid action: {values}")
                        result = {
                            "action": values.tolist(),
                            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        }
                response = make_policy_response(request, ok=True, result=result)
            except Exception as exc:
                LOG.exception("policy request failed")
                response = make_policy_response(request, ok=False, error=str(exc))
            socket.send_multipart([encode_policy_json(response)])
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
