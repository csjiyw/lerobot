#!/usr/bin/env python3
from pathlib import Path
from typing import Dict


ARM_JOINT_LIMITS = {
    "joint1": (-2.618, 2.618),
    "joint2": (0.0, 3.14),
    "joint3": (-2.697, 0.0),
    "joint4": (-1.832, 1.832),
    "joint5": (-1.22, 1.22),
    "joint6": (-3.14, 3.14),
}

GRIPPER_LIMIT = {
    "joint7": (0.0, 0.08),
}

ARM_JOINT_ORDER = [f"joint{i}" for i in range(1, 7)]


def print_joint_limits():
    print("\n" + "=" * 100)
    print(f"{'关节名称':<12} {'类型':<18} {'最小值 (lower)':<18} {'最大值 (upper)':<18} {'单位':<15}")
    print("=" * 100)

    for joint_name in ARM_JOINT_ORDER:
        lower, upper = ARM_JOINT_LIMITS[joint_name]
        print(f"{joint_name:<12} {'Revolute':<18} {lower:<18.4f} {upper:<18.4f} {'弧度 (rad)':<15}")

    lower, upper = GRIPPER_LIMIT["joint7"]
    print(f"{'joint7':<12} {'Prismatic (Gripper)':<18} {lower:<18.4f} {upper:<18.4f} {'米 (m)':<15}")

    print("=" * 100)


def find_position_datasets(h5_file):
    position_datasets = []
    action_datasets = []
    state_datasets = []

    def find_positions(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset) and "position" in name.lower():
            position_datasets.append(name)
            if "action" in name.lower():
                action_datasets.append(name)
            elif "state" in name.lower():
                state_datasets.append(name)

    h5_file.visititems(find_positions)
    return position_datasets, action_datasets, state_datasets


def split_arm_and_gripper_datasets(h5_file, dataset_paths, max_frames=None):
    """
    固定规则：
    - 12 cols -> arm (left6 + right6)
    - 2 cols  -> gripper (left + right)
    """
    arm_ds = None
    gripper_ds = None

    for ds_path in dataset_paths:
        data = h5_file[ds_path][:]
        if max_frames:
            data = data[:max_frames]

        if data.ndim != 2:
            continue
        if data.shape[1] == 12 and arm_ds is None:
            arm_ds = ds_path
        elif data.shape[1] == 2 and gripper_ds is None:
            gripper_ds = ds_path

    return arm_ds, gripper_ds


def iter_arm_columns():
    for i in range(12):
        side = "RIGHT" if i >= 6 else "LEFT"
        joint_idx = i % 6
        joint_name = ARM_JOINT_ORDER[joint_idx]
        yield side, joint_name, i


def iter_gripper_columns():
    yield "LEFT", "joint7", 0
    yield "RIGHT", "joint7", 1


def check_range(data_col, lower, upper):
    import numpy as np
    violations = int(np.sum((data_col < lower) | (data_col > upper)))
    violation_pct = (violations / len(data_col)) * 100 if len(data_col) > 0 else 0.0
    return {
        "min": float(np.min(data_col)),
        "max": float(np.max(data_col)),
        "mean": float(np.mean(data_col)),
        "violations": violations,
        "violation_pct": violation_pct,
    }


def batch_check_h5_files(h5_dir: str, max_frames: int = None,
                         compare_action_state: bool = True,
                         output_csv: str = None):
    import h5py
    import numpy as np
    import csv

    h5_dir = Path(h5_dir)
    h5_files = sorted(h5_dir.rglob("*.h5"))
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        return

    print(f"\n{'='*120}")
    print(f"BATCH ANALYSIS: Found {len(h5_files)} H5 files in {h5_dir}")
    print(f"{'='*120}")

    stats = {}
    for joint in ARM_JOINT_ORDER + ["joint7"]:
        for side in ["LEFT", "RIGHT"]:
            stats[(joint, side)] = {
                "state_min": [], "state_max": [], "state_mean": [],
                "action_min": [], "action_max": [], "action_mean": [],
                "state_violations": [], "action_violations": [],
                "tracking_error": [],
                "files_with_state_viol": 0,
                "files_with_action_viol": 0,
                "files_processed": 0,
            }

    total_files_processed = 0

    for idx, h5_path in enumerate(h5_files):
        print(f"\n[{idx+1}/{len(h5_files)}] Processing: {h5_path.name}", end=" ")

        try:
            with h5py.File(h5_path, "r") as f:
                _, action_datasets, state_datasets = find_position_datasets(f)
                if not action_datasets or not state_datasets:
                    print("⊘ Skipped (no action/state position datasets)")
                    continue

                action_arm_ds, action_gripper_ds = split_arm_and_gripper_datasets(f, action_datasets, max_frames)
                state_arm_ds, state_gripper_ds = split_arm_and_gripper_datasets(f, state_datasets, max_frames)

                if state_arm_ds is None or action_arm_ds is None:
                    print("⊘ Skipped (missing 12-col arm dataset)")
                    continue

                action_arm = f[action_arm_ds][:]
                state_arm = f[state_arm_ds][:]
                if max_frames:
                    action_arm = action_arm[:max_frames]
                    state_arm = state_arm[:max_frames]

                for side, joint_name, col_idx in iter_arm_columns():
                    lower, upper = ARM_JOINT_LIMITS[joint_name]
                    s = state_arm[:, col_idx]
                    a = action_arm[:, col_idx]

                    s_info = check_range(s, lower, upper)
                    a_info = check_range(a, lower, upper)

                    key = (joint_name, side)
                    stats[key]["state_min"].append(s_info["min"])
                    stats[key]["state_max"].append(s_info["max"])
                    stats[key]["state_mean"].append(s_info["mean"])
                    stats[key]["action_min"].append(a_info["min"])
                    stats[key]["action_max"].append(a_info["max"])
                    stats[key]["action_mean"].append(a_info["mean"])
                    stats[key]["state_violations"].append(s_info["violations"])
                    stats[key]["action_violations"].append(a_info["violations"])
                    stats[key]["tracking_error"].append(float(np.mean(np.abs(s - a))))
                    stats[key]["files_processed"] += 1

                    if s_info["violations"] > 0:
                        stats[key]["files_with_state_viol"] += 1
                    if a_info["violations"] > 0:
                        stats[key]["files_with_action_viol"] += 1

                if state_gripper_ds is not None and action_gripper_ds is not None:
                    action_gripper = f[action_gripper_ds][:]
                    state_gripper = f[state_gripper_ds][:]
                    if max_frames:
                        action_gripper = action_gripper[:max_frames]
                        state_gripper = state_gripper[:max_frames]

                    lower, upper = GRIPPER_LIMIT["joint7"]
                    for side, joint_name, col_idx in iter_gripper_columns():
                        s = state_gripper[:, col_idx]
                        a = action_gripper[:, col_idx]

                        s_info = check_range(s, lower, upper)
                        a_info = check_range(a, lower, upper)

                        key = (joint_name, side)
                        stats[key]["state_min"].append(s_info["min"])
                        stats[key]["state_max"].append(s_info["max"])
                        stats[key]["state_mean"].append(s_info["mean"])
                        stats[key]["action_min"].append(a_info["min"])
                        stats[key]["action_max"].append(a_info["max"])
                        stats[key]["action_mean"].append(a_info["mean"])
                        stats[key]["state_violations"].append(s_info["violations"])
                        stats[key]["action_violations"].append(a_info["violations"])
                        stats[key]["tracking_error"].append(float(np.mean(np.abs(s - a))))
                        stats[key]["files_processed"] += 1

                        if s_info["violations"] > 0:
                            stats[key]["files_with_state_viol"] += 1
                        if a_info["violations"] > 0:
                            stats[key]["files_with_action_viol"] += 1

                total_files_processed += 1
                print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n\n{'='*140}")
    print("AGGREGATED JOINT ANALYSIS")
    print(f"{'='*140}")
    print(f"Total files processed: {total_files_processed}\n")

    print(f"{'Joint':<10} | {'Side':<5} | {'State Range':<22} | {'Action Range':<22} | {'Limit Range':<22} | {'Files w/ Viol (S/A)':<18} | {'Mean Track Err':<15}")
    print("=" * 140)

    for joint in ARM_JOINT_ORDER + ["joint7"]:
        for side in ["LEFT", "RIGHT"]:
            s = stats[(joint, side)]
            if s["files_processed"] == 0:
                continue

            if joint == "joint7":
                lower, upper = GRIPPER_LIMIT[joint]
            else:
                lower, upper = ARM_JOINT_LIMITS[joint]

            row = (
                f"{joint:<10} | {side:<5} | "
                f"[{min(s['state_min']):<8.4f}, {max(s['state_max']):<8.4f}] | "
                f"[{min(s['action_min']):<8.4f}, {max(s['action_max']):<8.4f}] | "
                f"[{lower:<8.4f}, {upper:<8.4f}] | "
                f"{s['files_with_state_viol']}/{s['files_with_action_viol']:<16} | "
                f"{np.mean(s['tracking_error']):<15.6f}"
            )
            print(row)

    print("=" * 140)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Joint limit checker")
    parser.add_argument("--h5", type=str, help="single h5 path")
    parser.add_argument("--batch-dir", type=str, help="directory of h5 files")
    parser.add_argument("--max-frames", type=int, default=1000, help="0 means all")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--output-csv", type=str)
    args = parser.parse_args()

    print_joint_limits()
    max_frames = args.max_frames if args.max_frames > 0 else None

    if args.batch_dir:
        batch_check_h5_files(
            args.batch_dir,
            max_frames=max_frames,
            compare_action_state=not args.no_compare,
            output_csv=args.output_csv,
        )