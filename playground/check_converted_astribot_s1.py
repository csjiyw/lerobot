import json
from pathlib import Path

ROOT = Path("/mnt/sdc/output_0331")

EXPECTED = {
    "robot_type": "AstribotS1",
    "chunks_size": 1000,
    "fps": 30,
    "features": {
        "observation.images.head": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.hand_left": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.torso": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [25],
            "names": {
                "motors": [
                    "l_shoulder_pitch",
                    "l_shoulder_roll",
                    "l_shoulder_yaw",
                    "l_elbow_pitch",
                    "l_wrist_yaw",
                    "l_wrist_pitch",
                    "l_wrist_roll",
                    "r_shoulder_pitch",
                    "r_shoulder_roll",
                    "r_shoulder_yaw",
                    "r_elbow_pitch",
                    "r_wrist_yaw",
                    "r_wrist_pitch",
                    "r_wrist_roll",
                    "left_claw",
                    "right_claw",
                    "head_pan",
                    "head_tilt",
                    "ankle_pitch",
                    "knee_pitch",
                    "hip_pitch",
                    "hip_yaw",
                    "chassis_x*",
                    "chassis_y*",
                    "chassis_yaw*",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": [25],
            "names": {
                "motors": [
                    "l_shoulder_pitch",
                    "l_shoulder_roll",
                    "l_shoulder_yaw",
                    "l_elbow_pitch",
                    "l_wrist_yaw",
                    "l_wrist_pitch",
                    "l_wrist_roll",
                    "r_shoulder_pitch",
                    "r_shoulder_roll",
                    "r_shoulder_yaw",
                    "r_elbow_pitch",
                    "r_wrist_yaw",
                    "r_wrist_pitch",
                    "r_wrist_roll",
                    "left_claw",
                    "right_claw",
                    "head_pan",
                    "head_tilt",
                    "ankle_pitch",
                    "knee_pitch",
                    "hip_pitch",
                    "hip_yaw",
                    "chassis_x*",
                    "chassis_y*",
                    "chassis_yaw*",
                ]
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    },
}


def load_info(task_dir: Path):
    info_path = task_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    with open(info_path, "r") as f:
        return json.load(f)


def compare_dict(expected, actual, prefix=""):
    diffs = []

    if isinstance(expected, dict) and isinstance(actual, dict):
        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())

        for k in sorted(exp_keys - act_keys):
            diffs.append(f"{prefix}{k}: missing")


        for k in sorted(exp_keys & act_keys):
            diffs.extend(compare_dict(expected[k], actual[k], prefix + k + "."))
    else:
        if expected != actual:
            diffs.append(
                f"{prefix[:-1]}:\n"
                f"  expected={expected}\n"
                f"  actual  ={actual}"
            )

    return diffs


def extract_check_target(info):
    return {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "chunks_size": info.get("chunks_size"),
        "fps": info.get("fps"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "features": info.get("features"),
    }


def main():
    task_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    ok_tasks = []
    bad_tasks = []
    error_tasks = []

    for task_dir in task_dirs:
        try:
            info = load_info(task_dir)

            # 只检查 astribot_s1 风格输出
            if info.get("robot_type") != "AstribotS1":
                continue

            target = extract_check_target(info)
            diffs = compare_dict(EXPECTED, target)

            if diffs:
                bad_tasks.append((task_dir.name, diffs))
            else:
                ok_tasks.append(task_dir.name)

        except Exception as e:
            error_tasks.append((task_dir.name, str(e)))

    print("===== OK TASKS =====")
    for x in ok_tasks:
        print(x)

    print("\n===== BAD TASKS =====")
    for task_name, diffs in bad_tasks:
        print(f"\n{task_name}")
        for d in diffs:
            print(d)

    print("\n===== ERROR TASKS =====")
    for task_name, err in error_tasks:
        print(f"{task_name}: {err}")

    print("\n===== SUMMARY =====")
    print(f"ok    : {len(ok_tasks)}")
    print(f"bad   : {len(bad_tasks)}")
    print(f"error : {len(error_tasks)}")


if __name__ == "__main__":
    main()