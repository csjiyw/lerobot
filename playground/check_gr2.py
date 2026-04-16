from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/root/jyw_ws/lerobot_jpeg/playground/output_v3_test_0331")

DATASET_IDS = {
    "傅利叶GR-2": [
        "167b536f67db4de998f9ee4118611e54",
        "478f93055aaa48029d5e0f3b3be9c061",
        "6e275d421e394e03a3dd5cc0783867ef",
        "092926d5e59840a2935efc3a2515f293",
        "fd53bb497c464ccaa2a703bbd324ca77",
    ],
}

HAND_START = 14
HAND_END = 26

HAND_NAMES = [
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]

EXPECTED_MIN = np.array([
    -1.3, -1.3, -1.3, -1.3,  0.0, -1.3,
    -1.3, -1.3, -1.3, -1.3,  0.0, -1.3,
], dtype=np.float32)

EXPECTED_MAX = np.array([
     0.0,  0.0,  0.0,  0.0,  1.0,  0.0,
     0.0,  0.0,  0.0,  0.0,  1.0,  0.0,
], dtype=np.float32)


def explain_range(actual_min, actual_max, expected_min, expected_max, tol=1e-4):
    low_ok = actual_min >= expected_min - tol
    high_ok = actual_max <= expected_max + tol

    if low_ok and high_ok:
        if abs(actual_min - expected_min) <= tol and abs(actual_max - expected_max) <= tol:
            return "OK, range matches expected limit"
        return "OK, inside expected limit with tiny numeric error"

    msgs = []
    if not low_ok:
        msgs.append(f"min below limit by {expected_min - actual_min:.6f}")
    if not high_ok:
        msgs.append(f"max above limit by {actual_max - expected_max:.6f}")
    return "; ".join(msgs)


def find_parquet_files(dataset_path: Path):
    data_dir = dataset_path / "data"
    return sorted(data_dir.rglob("*.parquet"))


def load_column_from_parquets(parquet_files, column_name):
    arrays = []
    for f in parquet_files:
        table = pq.read_table(f, columns=[column_name])
        col = table[column_name]
        arr = np.asarray(col.to_pylist(), dtype=np.float32)
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def check_one_dataset(dataset_path: Path, dataset_name: str):
    parquet_files = find_parquet_files(dataset_path)
    assert parquet_files, f"No parquet files found under {dataset_path / 'data'}"

    print(f"\n{'#' * 140}")
    print(f"DATASET: {dataset_name}")
    print(f"PATH   : {dataset_path}")
    print(f"found {len(parquet_files)} parquet files")

    all_state = load_column_from_parquets(parquet_files, "observation.state")
    all_action = load_column_from_parquets(parquet_files, "action")

    print("all_state shape :", all_state.shape)
    print("all_action shape:", all_action.shape)

    hand_state = all_state[:, HAND_START:HAND_END]
    hand_action = all_action[:, HAND_START:HAND_END]

    state_min = hand_state.min(axis=0)
    state_max = hand_state.max(axis=0)
    action_min = hand_action.min(axis=0)
    action_max = hand_action.max(axis=0)

    print(f"\ntotal frames checked: {hand_state.shape[0]}\n")

    print("STATE RANGE CHECK")
    print("=" * 120)
    for j, name in enumerate(HAND_NAMES):
        explain = explain_range(state_min[j], state_max[j], EXPECTED_MIN[j], EXPECTED_MAX[j])
        print(
            f"joint_name: {name}\n"
            f"  expected_limit: [{EXPECTED_MIN[j]: .4f}, {EXPECTED_MAX[j]: .4f}]\n"
            f"  actual_range  : [{state_min[j]: .6f}, {state_max[j]: .6f}]\n"
            f"  explain       : {explain}\n"
        )

    print("ACTION RANGE CHECK")
    print("=" * 120)
    for j, name in enumerate(HAND_NAMES):
        explain = explain_range(action_min[j], action_max[j], EXPECTED_MIN[j], EXPECTED_MAX[j])
        print(
            f"joint_name: {name}\n"
            f"  expected_limit: [{EXPECTED_MIN[j]: .4f}, {EXPECTED_MAX[j]: .4f}]\n"
            f"  actual_range  : [{action_min[j]: .6f}, {action_max[j]: .6f}]\n"
            f"  explain       : {explain}\n"
        )

    return {
        "dataset_name": dataset_name,
        "frames": hand_state.shape[0],
        "state_min": state_min,
        "state_max": state_max,
        "action_min": action_min,
        "action_max": action_max,
    }


all_results = []

for robot_name, dataset_ids in DATASET_IDS.items():
    for ds_id in dataset_ids:
        dataset_path = ROOT / ds_id
        result = check_one_dataset(dataset_path, f"{robot_name} / {ds_id}")
        all_results.append(result)

# 汇总五个 dataset 的 overall min/max
overall_state_min = np.min(np.stack([x["state_min"] for x in all_results], axis=0), axis=0)
overall_state_max = np.max(np.stack([x["state_max"] for x in all_results], axis=0), axis=0)
overall_action_min = np.min(np.stack([x["action_min"] for x in all_results], axis=0), axis=0)
overall_action_max = np.max(np.stack([x["action_max"] for x in all_results], axis=0), axis=0)
overall_frames = sum(x["frames"] for x in all_results)

print(f"\n{'#' * 140}")
print("OVERALL SUMMARY ACROSS ALL 5 DATASETS")
print(f"total datasets checked: {len(all_results)}")
print(f"total frames checked  : {overall_frames}\n")

print("OVERALL STATE RANGE CHECK")
print("=" * 120)
for j, name in enumerate(HAND_NAMES):
    explain = explain_range(
        overall_state_min[j], overall_state_max[j],
        EXPECTED_MIN[j], EXPECTED_MAX[j]
    )
    print(
        f"joint_name: {name}\n"
        f"  expected_limit: [{EXPECTED_MIN[j]: .4f}, {EXPECTED_MAX[j]: .4f}]\n"
        f"  actual_range  : [{overall_state_min[j]: .6f}, {overall_state_max[j]: .6f}]\n"
        f"  explain       : {explain}\n"
    )

print("OVERALL ACTION RANGE CHECK")
print("=" * 120)
for j, name in enumerate(HAND_NAMES):
    explain = explain_range(
        overall_action_min[j], overall_action_max[j],
        EXPECTED_MIN[j], EXPECTED_MAX[j]
    )
    print(
        f"joint_name: {name}\n"
        f"  expected_limit: [{EXPECTED_MIN[j]: .4f}, {EXPECTED_MAX[j]: .4f}]\n"
        f"  actual_range  : [{overall_action_min[j]: .6f}, {overall_action_max[j]: .6f}]\n"
        f"  explain       : {explain}\n"
    )