import json
from pathlib import Path

ROOT = Path("/mnt/sdc/output_0331")

RENAME = {
    "l_gripper": "left_claw",
    "r_gripper": "right_claw",
}

def rename_motors(motors):
    return [RENAME.get(x, x) for x in motors]

for task_dir in sorted(ROOT.iterdir()):
    if not task_dir.is_dir():
        continue

    info_path = task_dir / "meta" / "info.json"
    if not info_path.exists():
        continue

    try:
        with open(info_path, "r") as f:
            info = json.load(f)

        if info.get("robot_type") != "AstribotS1":
            continue

        changed = False
        features = info.get("features", {})

        for key in ["observation.state", "action"]:
            motors = features.get(key, {}).get("names", {}).get("motors")
            if motors is None:
                continue

            new_motors = rename_motors(motors)
            if new_motors != motors:
                info["features"][key]["names"]["motors"] = new_motors
                changed = True

        if changed:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
            print(f"[UPDATED] {task_dir.name}")
        else:
            print(f"[SKIP] {task_dir.name}")

    except Exception as e:
        print(f"[ERROR] {task_dir.name}: {e}")