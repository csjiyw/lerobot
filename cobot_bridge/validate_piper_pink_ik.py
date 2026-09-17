"""Offline FK/IK validation on a recorded Cobot Magic raw trajectory.

The tool never opens ROS, ZMQ or a robot command interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .piper_pink_ik import PiperIKSettings, PiperPinkIK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--state", choices=("puppet_position", "master_position"), default="puppet_position")
    parser.add_argument("--end-effector-frame", default="link6")
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--max-joint-velocity", type=float, default=0.45)
    parser.add_argument(
        "--max-joint-acceleration",
        type=float,
        default=0.0,
        help="Pink-side hard acceleration limit; 0 leaves this to the robot arbiter",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--maximum-frames", type=int, default=0)
    return parser


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_stride < 1 or args.maximum_frames < 0:
        raise ValueError("frame-stride must be positive and maximum-frames non-negative")
    with h5py.File(args.raw.expanduser().resolve(), "r") as raw:
        states = np.asarray(raw[f"frames/{args.state}"][:, :6], dtype=float)
        dataset_hz = float(raw.attrs.get("fps", 30.0))
    states = states[:: args.frame_stride]
    dataset_hz /= args.frame_stride
    if args.maximum_frames:
        states = states[: args.maximum_frames]
    if len(states) < 2:
        raise ValueError("trajectory must contain at least two selected frames")

    settings = PiperIKSettings(
        control_hz=args.control_hz,
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
    )
    ik = PiperPinkIK(
        args.urdf,
        end_effector_frame=args.end_effector_frame,
        settings=settings,
    )
    ik.reset(states[0])
    predicted = states[0].copy()
    if args.control_hz < dataset_hz:
        raise ValueError("control-hz must be at least the selected dataset rate")
    substep_counts: list[int] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    joint_errors: list[float] = []
    minimum_singular_values: list[float] = []
    conditions: list[float] = []
    velocities: list[float] = []
    near_singular = 0
    failures: list[dict[str, object]] = []

    for frame_index, measured in enumerate(states[1:], start=1):
        target_pose = ik.forward_kinematics(measured)
        ik.set_posture_reference(predicted)
        # Keep the two time bases exact over the whole trajectory.  For the
        # normal 30 Hz -> 200 Hz path this produces a 7, 6, 7 tick pattern,
        # rather than incorrectly treating seven ticks per frame as 210 Hz.
        previous_tick = round((frame_index - 1) * args.control_hz / dataset_hz)
        current_tick = round(frame_index * args.control_hz / dataset_hz)
        substeps = current_tick - previous_tick
        substep_counts.append(substeps)
        try:
            step = None
            for _ in range(substeps):
                step = ik.step(target_pose)
            assert step is not None
        except Exception as exc:
            failures.append({"frame": frame_index, "error": f"{type(exc).__name__}: {exc}"})
            ik.reset(measured)
            predicted = measured.copy()
            continue
        predicted = step.joints
        diagnostics = step.diagnostics
        position_errors.append(diagnostics.position_error_m)
        orientation_errors.append(diagnostics.orientation_error_rad)
        joint_errors.append(float(np.max(np.abs(predicted - measured))))
        minimum_singular_values.append(diagnostics.minimum_singular_value)
        conditions.append(diagnostics.jacobian_condition)
        velocities.append(diagnostics.maximum_joint_velocity_rad_s)
        near_singular += int(diagnostics.near_singularity)

    summary = {
        "motion": False,
        "raw": str(args.raw.expanduser().resolve()),
        "urdf": str(args.urdf.expanduser().resolve()),
        "state": args.state,
        "frames": int(len(states)),
        "dataset_hz": dataset_hz,
        "control_hz": args.control_hz,
        "ik_substeps_per_frame": {
            "min": min(substep_counts),
            "max": max(substep_counts),
            "mean": float(np.mean(substep_counts)),
        },
        "position_error_mm": {
            "p50": 1_000.0 * percentile(position_errors, 50),
            "p95": 1_000.0 * percentile(position_errors, 95),
            "max": 1_000.0 * max(position_errors, default=float("nan")),
        },
        "orientation_error_deg": {
            "p50": float(np.degrees(percentile(orientation_errors, 50))),
            "p95": float(np.degrees(percentile(orientation_errors, 95))),
            "max": float(np.degrees(max(orientation_errors, default=float("nan")))),
        },
        "joint_reconstruction_error_rad": {
            "p50": percentile(joint_errors, 50),
            "p95": percentile(joint_errors, 95),
            "max": max(joint_errors, default=float("nan")),
        },
        "jacobian": {
            "minimum_singular_value_min": min(minimum_singular_values, default=float("nan")),
            "condition_p95": percentile(conditions, 95),
            "condition_max": max(conditions, default=float("nan")),
            "near_singular_frames": near_singular,
        },
        "maximum_ik_velocity_rad_s": max(velocities, default=float("nan")),
        "solver_failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
