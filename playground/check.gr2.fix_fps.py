'''
change:
view downloading queue when download done or cvt done
resume job
''' 
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import sleep
import traceback
from openpyxl import load_workbook
from tqdm import tqdm

import os, threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import diskcache


# ====== CONFIG ======
DOWNLOAD_TASK_TXT = f"gr2_filtered_download_list.txt"
RCLONE_BIN = f"rclone-v1.68.2-linux-amd64/rclone"
LOCAL_TMP_ROOT = "/mnt/fastdisk/tmp_download"


# =====================

XLSX_IN  = f"task_and_description_checked.xlsx"




def parse_task_and_episode(remote_line: str):
    """
    支持：
      .../<task_id>/<episode_id>/      (目录)
      .../<task_id>/<episode_id>.h5   (文件)
    返回 (task_id, episode_name)
    """
    p = remote_line.strip()
    if not p:
        return None, None
    p2 = p.rstrip("/")
    parts = p2.split("/")
    if len(parts) < 2:
        return None, None
    task_id = parts[-2].strip().lower()
    episode_name = parts[-1].strip()  # 目录名 或 *.h5
    return task_id, episode_name

def get_task_root(remote_ep: str) -> str:
    """
    remote_ep:
      .../<task_id>/<episode_id>/      或 .../<task_id>/<episode_id>.h5
    return:
      .../<task_id>
    """
    p = remote_ep.strip().rstrip("/")
    parts = p.split("/")
    if len(parts) < 2:
        raise ValueError(f"bad remote_ep: {remote_ep}")
    return "/".join(parts[:-1])  # 去掉 episode 那层

def check_dataset_fps(info_path="meta/info.json",target_fps=30):
    info_path = Path(info_path)

    with open(info_path) as f:
        info = json.load(f)

    changed = False

    # dataset fps
    if info.get("fps") != target_fps:
        info["fps"] = target_fps
        changed = True

    # features fps
    for v in info.get("features", {}).values():
        if "fps" in v and v["fps"] != target_fps:
            v["fps"] = target_fps
            changed = True

        if "info" in v and "video.fps" in v["info"] and v["info"]["video.fps"] != target_fps:
            v["info"]["video.fps"] = target_fps
            changed = True

    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("fps fixed to 30")
    else:
        print("fps already 30")

def main():
    wb = load_workbook(XLSX_IN)
    ws = wb.active  # 或 wb["Sheet1"]

    
    # cvt_futures = []

    with open(DOWNLOAD_TASK_TXT, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # dl_pool 控制“并行下载多少个 task”

    if True:

        download_futures = []

        cur_task = None
        cur_task_root = None
        cur_eps = []


        def flush_group():
            nonlocal cur_task, cur_task_root, cur_eps

            if cur_task and cur_eps:
                # check exist
                # find 
                if not os.path.isdir(f"/mnt/sdc/output/{cur_task}"):
                    print(f"!!!!  {cur_task} not exist")
                else:
                    check_dataset_fps(f"/mnt/sdc/output/{cur_task}/meta/info.json")

            cur_task, cur_task_root, cur_eps = None, None, []

        
        with open(DOWNLOAD_TASK_TXT, "r", encoding="utf-8") as f:
            # for line in tqdm(f, total=total_lines, miniters=1):
            for line in f:
                s = line.strip()
                if not s:
                    continue
                task_id, ep_name = parse_task_and_episode(s)
                if not task_id:
                    continue
                

                if cur_task is not None and task_id != cur_task:
                    flush_group()


                if cur_task is None:
                    cur_task = task_id
                    cur_task_root = get_task_root(s)
                    cur_eps = []
                cur_eps.append(ep_name)



        flush_group()



if __name__ == "__main__":
    main()