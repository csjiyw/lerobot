# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys
import threading
import traceback
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import diskcache
from openpyxl import load_workbook
from tqdm import tqdm

# ===================== TOP CONFIG =====================


# Force initialize cache (clear cached_task_id)
FORCE_INIT_CACHE = True  # Set to True to reset and start fresh

# 并行 task 数
PARALLEL_DOWNLOAD_TASKS = 1
PARALLEL_CVT_TASKS = 1

# 每个 task 内部资源
DOWNLOAD_WORKERS_PER_TASK = 4
CVT_WORKERS_PER_TASK = 6

# 背压：最多下载领先多少个 task
MAX_AHEAD = 2
SENTINEL = object()

# 路径相关
HOME = "/root/jyw_ws/lerobot_jpeg/playground"

RCLONE_BIN = f"{HOME}/rclone-v1.68.2-linux-amd64/rclone"
XLSX_IN = f"{HOME}/数据需求总表同步_转换列表.xlsx"

LOCAL_TMP_ROOT = "/mnt/fastdisk/tmp_download"
OUTPUT_ROOT = "/mnt/sdc/output_g1"
CACHE_DIR = "/root/_robot_g1_ckpt"

# 远端相关
REMOTE_ROOT = "huawei-cloud:openloong-apps-prod-private/data-collector-svc/align"

# Excel 列名
TASK_ID_COLUMN = "任务ID"
TASK_DSCP_COLUMN = "任务步骤描述-人话英文版"
DEVICE_COLUMN = "采集设备"

# 设备 -> robot name (used by base_converter)
DEVICE_TO_ROBOT = {
    "智元G1": "ginie1_all",
}

# Tasks to filter for
TARGET_TASKS = {
}

# Base converter script path
BASE_CONVERTER = f"{HOME}/convert/base_converter.py"


# ======================================================
ALLOWED_DEVICES = set(DEVICE_TO_ROBOT.keys())

def run(cmd, check=True):
    p = subprocess.Popen(cmd, start_new_session=False)
    ret = p.wait()
    if check and ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def get_robot_name(device_name: str) -> str:
    name = str(device_name).strip()
    if name not in DEVICE_TO_ROBOT:
        raise ValueError(f"Unsupported device: {device_name}")
    return DEVICE_TO_ROBOT[name]


# def download_task(task_id: str, local_task_dir: str, workers: int):
#     remote_task_root = f"{REMOTE_ROOT}/{task_id}"
#     os.makedirs(local_task_dir, exist_ok=True)

#     cmd = [
#         RCLONE_BIN, "copy",
#         remote_task_root,
#         local_task_dir,
#     ]
#     run(cmd, check=True)

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

def cvt_task(task_id: str, local_task_dir: str, task_dscp: str, device_name: str, workers: int, cache: diskcache.Cache):
    output_dir = f"{OUTPUT_ROOT}/{task_id}"
    robot_name = get_robot_name(device_name)

    cmd = [
        "python", BASE_CONVERTER,
        "--robot", robot_name,
        "--input", local_task_dir,
        "--output", output_dir,
        "--repo_id", task_id,
        "--task", str(task_dscp) if task_dscp is not None else "",
        "--workers", str(workers),
    ]
    print(f"device_name: {device_name}, robot_name: {robot_name}")
    print(f"task_dscp: {task_dscp}")

    # Inline subprocess call to capture stderr without crashing the pipeline
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        stderr_msg = result.stderr if result.stderr else "No stderr captured"
        cache.set(f"convert_failure_{task_id}", {
            "task_id": task_id,
            "device": device_name,
            "robot": robot_name,
            "cmd": cmd,
            "returncode": result.returncode,
            "traceback": "Subprocess failed",
            "stderr": stderr_msg,
        })
        print(f"[FAILURE CACHED] {task_id} - Reason: {stderr_msg.strip()[:100]}...")



def cleanup(local_task_dir: str):
    shutil.rmtree(local_task_dir, ignore_errors=True)


def main():
    cache = diskcache.Cache(CACHE_DIR)
    os.makedirs(LOCAL_TMP_ROOT, exist_ok=True)

    # Force init cache if requested
    if FORCE_INIT_CACHE:
        print(f"\n{'='*80}")
        print(f"[FORCE INIT] Clearing cache at: {CACHE_DIR}")
        try:
            cache.delete("cached_task_id")
            print(f"[FORCE INIT] Deleted cached_task_id")
            
            # Optionally clear all failure records
            keys_to_delete = [key for key in cache.keys() if key.startswith("convert_failure_")]
            for key in keys_to_delete:
                cache.delete(key)
            print(f"[FORCE INIT] Cleared {len(keys_to_delete)} failure records")
            
            print(f"[FORCE INIT] Cache initialized successfully")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"[FORCE INIT] Warning: {e}")
            print(f"[FORCE INIT] Continuing anyway...\n")

    wb = load_workbook(XLSX_IN)
    ws = wb.active

    headers = [cell.value for cell in ws[1]]
    col_idx = {str(name).strip(): i for i, name in enumerate(headers) if name is not None}

    required_cols = [TASK_ID_COLUMN, TASK_DSCP_COLUMN, DEVICE_COLUMN]
    for c in required_cols:
        if c not in col_idx:
            raise KeyError(f"Excel missing required column: {c}")

    task_id_col = col_idx[TASK_ID_COLUMN]
    task_dscp_col = col_idx[TASK_DSCP_COLUMN]
    device_col = col_idx[DEVICE_COLUMN]

    task_rows = []
    seen_task_ids = set()
    for row in ws.iter_rows(min_row=2):
        cell_val = row[task_id_col].value
        if cell_val is None:
            continue
        task_id = str(cell_val).strip()
        if not task_id:
            continue
        if task_id in seen_task_ids:
            continue

        # Filter for specific target tasks
        if TARGET_TASKS and task_id not in TARGET_TASKS:
            continue

        device_name = row[device_col].value
        device_name = str(device_name).strip() if device_name is not None else ""
        if device_name not in ALLOWED_DEVICES:
            continue

        seen_task_ids.add(task_id)
        task_dscp = row[task_dscp_col].value
        task_rows.append((task_id, task_dscp, device_name))

    # TEMP: Only process first 5 tasks for testing
    task_rows = task_rows[:5]
    print(f"[DEBUG] Limited to {len(task_rows)} tasks for testing")

    ready_q = Queue()
    slots = Queue(maxsize=PARALLEL_CVT_TASKS + MAX_AHEAD)

    def cvt_and_cleanup(task_id: str, local_dir: str, task_dscp: str, device_name: str, cvt_threads: int):
        try:
            cvt_task(task_id, local_dir, task_dscp, device_name, workers=cvt_threads, cache=cache)
            cache.set("cached_task_id", task_id)
            # cleanup(local_dir)
            print(f"[DONE] CVT {task_id}")
        except Exception:
            print(traceback.format_exc())
            os._exit(1)
        finally:
            slots.get()
            slots.task_done()
        return task_id

    def cvt_submitter(cache_obj):
        while True:
            item = ready_q.get()
            try:
                if item is SENTINEL:
                    return
                task_id, local_dir, task_dscp, device_name = item
                print("cvt get", task_id, device_name)
                cvt_and_cleanup(task_id, local_dir, task_dscp, device_name, CVT_WORKERS_PER_TASK)
            finally:
                ready_q.task_done()

    def submit_download(dl_pool: ThreadPoolExecutor, task_id: str, task_dscp: str, device_name: str):
        def _job():
            local_dir = os.path.join(LOCAL_TMP_ROOT, task_id)
            print(f"\n=== DOWNLOAD TASK {task_id} ({device_name}) ===")
            download_task(task_id, local_dir, workers=DOWNLOAD_WORKERS_PER_TASK)
            print(f"\n=== DOWNLOAD TASK FINISH {task_id} ({device_name}) ===")
            ready_q.put((task_id, local_dir, task_dscp, device_name))
        return dl_pool.submit(_job)

    cached_task_id = cache.get("cached_task_id")
    skip_mode = cached_task_id is not None
    seen_cached = False

    # Log cache status at start
    if skip_mode:
        print(f"\n{'='*80}")
        print(f"[RESUME MODE] Found cached task_id: {cached_task_id}")
        print(f"[RESUME MODE] Will skip tasks until after {cached_task_id}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"[FRESH START] No cached task_id found, processing from beginning")
        print(f"{'='*80}\n")

    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_TASKS) as dl_pool:
        t = threading.Thread(target=cvt_submitter, args=(cache,), daemon=True)
        t.start()

        download_futures = []

        for task_id, task_dscp, device_name in tqdm(task_rows, total=len(task_rows), miniters=1):
            if skip_mode:
                if not seen_cached:
                    if task_id == cached_task_id:
                        print(f"\n[SKIP MODE] Found cached task: {task_id}")
                        print(f"[SKIP MODE] Marking as seen, will resume from next task")
                        seen_cached = True
                    print(f"[SKIP MODE] Skipping task {task_id} (before cached task {cached_task_id})")
                    continue
                else:
                    if task_id == cached_task_id:
                        print(f"[SKIP MODE] Skipping duplicate cached task: {task_id}")
                        continue
                    print(f"\n[RESUME MODE] Resuming from task: {task_id}")
                    print(f"[RESUME MODE] Clearing skip_mode, cached_task_id = {cached_task_id}")
                    skip_mode = False
                    cached_task_id = None

            print("ready_q size:", ready_q.qsize())
            print("slots size:", slots.qsize())

            slots.put(1)
            download_futures.append(submit_download(dl_pool, task_id, task_dscp, device_name))

        for fut in as_completed(download_futures):
            fut.result()

        ready_q.join()
        ready_q.put(SENTINEL)
        t.join()

    # Log completion status
    print(f"\n{'='*80}")
    print(f"[COMPLETED] Processing finished")
    if skip_mode:
        print(f"[WARNING] Still in skip_mode - no tasks were processed after cached task")
    else:
        print(f"[SUCCESS] All eligible tasks processed")
    print(f"{'='*80}\n")


    wb.close()


if __name__ == "__main__":
    main()
