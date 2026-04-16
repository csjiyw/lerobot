import os
import subprocess

RCLONE_BIN="/root/jyw_ws/lerobot_jpeg/playground/rclone-v1.68.2-linux-amd64/rclone"
REMOTE_ROOT="huawei-cloud:openloong-apps-prod-private/data-collector-svc/align"

def download_task(task_id: str, local_task_dir: str, workers: int):
    remote_task_root = f"{REMOTE_ROOT}/{task_id}"
    os.makedirs(local_task_dir, exist_ok=True)

    # 先列一级子目录
    list_cmd = [
        RCLONE_BIN, "lsf",
        remote_task_root,
        "--dirs-only",
    ]
    list_result = subprocess.run(list_cmd, capture_output=True, text=True)

    if list_result.returncode != 0:
        raise RuntimeError(list_result.stderr)

    dir_names = [
        line.strip().rstrip("/")
        for line in list_result.stdout.splitlines()
        if line.strip()
    ]
    dir_names = sorted(dir_names)

    # 先测试前 5 个
    dirs_to_download = dir_names[:5]

    if not dirs_to_download:
        print(f"[WARNING] No subdirectories found for task {task_id}")
        return

    print(f"[INFO] Found {len(dir_names)} directories, dry-run first {len(dirs_to_download)}")

    copy_cmd = [
        RCLONE_BIN, "copy",
        remote_task_root,
        local_task_dir,
    ]

    for d in dirs_to_download:
        copy_cmd += ["--filter", f"+ {d}/**"]

    copy_cmd += [
        "--filter", "- **",
        "--transfers", str(workers),
        "--checkers", str(workers),
        # "--dry-run",
        "-P",
    ]

    print("[CMD]", " ".join(copy_cmd))
    subprocess.run(copy_cmd, check=True)

if __name__=="__main__":
    download_task("cc9e6f7a8c484a1f98d373295b844124","/tmp/test_rclone_download",4)