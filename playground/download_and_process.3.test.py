'''
change:
view downloading queue when download done or cvt done
resume job
''' 
# -*- coding: utf-8 -*-
import os
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

# ====== 并行 task 数（你要的）======
PARALLEL_DOWNLOAD_TASKS = 1
PARALLEL_CVT_TASKS = 1 #max(1, os.cpu_count() - 4)  # speed up save image cost time

# ====== 每个 task 内部资源（你也要的）======
DOWNLOAD_WORKERS_PER_TASK = 8   # 传给 rclone --transfers/--checkers 等
CVT_THREADS_PER_TASK = 8        #  speed up episode 

# ====== 背压：最多下载领先多少个 task ======
MAX_AHEAD = 5
SENTINEL = object()

HOME="/root/jyw_ws/lerobot/playground"
# ====== CONFIG ======
DOWNLOAD_TASK_TXT = f"{HOME}/filtered_download_list.txt"
RCLONE_BIN = f"{HOME}/rclone-v1.68.2-linux-amd64/rclone"
LOCAL_TMP_ROOT = "/mnt/fastdisk/tmp_download"



MAX_EPISODES_PER_TASK = 999999
# =====================

XLSX_IN  = f"{HOME}/task_and_description_checked.xlsx"



def run(cmd, check=True):
    # print("CMD:", " ".join(cmd) if isinstance(cmd, list) else cmd)
    # return subprocess.run(cmd, check=check)
    p=subprocess.Popen(cmd,  start_new_session=False)
    p.wait()

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


def main():

    cache = diskcache.Cache("/mnt/sdc/output/_ckpt")

    os.makedirs(LOCAL_TMP_ROOT, exist_ok=True)

    wb = load_workbook(XLSX_IN)
    ws = wb.active  # 或 wb["Sheet1"]


    ready_q = Queue()
    
    # cvt_futures = []
    lock = threading.Lock()

    slots = Queue(maxsize=PARALLEL_CVT_TASKS + MAX_AHEAD)  # 磁盘上最多允许多少个task同时存在

    def cvt_and_cleanup(task_id: str, local_dir: str, task_dscp,cvt_threads: int):
        try:
            # cvt_task(task_id, local_dir, task_dscp,workers=cvt_threads)   # 你改一下签名
            print(f"cvt finish {slots.qsize()}")
            slots.get() 
            # cleanup(local_dir)
        except Exception as e:
            print(traceback.format_exc())
            os._exit(1)
        return task_id

    def cvt_submitter(cvt_pool: ProcessPoolExecutor,cache):
        while True:
            item = ready_q.get()
            try:
                if item is SENTINEL:
                    return
                task_id, local_dir,task_dscp = item
                print("cvt get",task_id)
                # fut = cvt_pool.submit(cvt_and_cleanup, task_id, local_dir, CVT_THREADS_PER_TASK)
                # with lock:
                #     cvt_futures.append(fut)
                
                cvt_and_cleanup(task_id, local_dir, task_dscp,CVT_THREADS_PER_TASK)
                # cache.set("cached_task_id",task_id)

            finally:
                ready_q.task_done()

    def submit_download(dl_pool: ThreadPoolExecutor, task_id, task_root, eps):

        def _job():
            local_dir = os.path.join(LOCAL_TMP_ROOT, task_id)
            # print(f"\n=== DOWNLOAD TASK {task_id} episodes={len(eps)} ===")
            # download_batch(task_root, local_dir, eps, workers=DOWNLOAD_WORKERS_PER_TASK)  # 你改一下签名
            # print(f"\n=== DOWNLOAD TASK FINISH {task_id} episodes={len(eps)} ===")
            for row in ws.iter_rows(min_row=2):  # 跳过表头
                a_val = str(row[0].value).strip() if row[0].value else ""
                if a_val.lower() == task_id.lower():
                    task_dscp = row[10].value   # K 列 = index 10
                    break

            ready_q.put((task_id, local_dir,task_dscp))  # 队列满则阻塞 => 限制领先量

        return dl_pool.submit(_job)

    with open(DOWNLOAD_TASK_TXT, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # dl_pool 控制“并行下载多少个 task”
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_TASKS) as dl_pool, \
         ThreadPoolExecutor(max_workers=PARALLEL_CVT_TASKS) as cvt_pool:

        t = threading.Thread(target=cvt_submitter, args=(cvt_pool,cache), daemon=True)
        t.start()

        download_futures = []

        cur_task = None
        cur_task_root = None
        cur_eps = []

        cached_task_id=None#cache.get("cached_task_id")

        skip_mode = cached_task_id is not None
        seen_cached = False  # 是否已经遇到过 cached_task_id（但仍要跳过它）

        def flush_group():
            nonlocal cur_task, cur_task_root, cur_eps
            # print pending cvt queue
            print("ready_q size:", ready_q.qsize())
            print("slots size:", slots.qsize())
            slots.put(1)

            if cur_task and cur_eps:
                download_futures.append(submit_download(dl_pool, cur_task, cur_task_root, cur_eps))
            cur_task, cur_task_root, cur_eps = None, None, []

        with open(DOWNLOAD_TASK_TXT, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, miniters=1):
          
                s = line.strip()
                if not s:
                    continue
                task_id, ep_name = parse_task_and_episode(s)
                if not task_id:
                    continue
                
                # skip logic
                if skip_mode:
                    if not seen_cached:
                        # 还没遇到 cached_task_id：一直跳
                        if task_id == cached_task_id:
                            seen_cached = True
                        continue
                    else:
                        # 已遇到 cached_task_id：仍然跳过 cached_task_id 这一段
                        if task_id == cached_task_id:
                            continue
                        # 一旦 task_id 变了（到了下一个 task），开始正常跑
                        skip_mode = False
                        cached_task_id = None
                        # 这里不要 continue，让当前行（下一个 task 的第一条）直接进入正常逻辑
                

                if cur_task is not None and task_id != cur_task:
                    flush_group()


                if cur_task is None:
                    cur_task = task_id
                    cur_task_root = get_task_root(s)
                    cur_eps = []
                cur_eps.append(ep_name)


        flush_group()

        # 等下载完成
        for fut in as_completed(download_futures):
            fut.result()

        # 等队列被消费并提交完
        ready_q.join()
        ready_q.put(SENTINEL)
        t.join()

        # 等转换完成
        # with lock:
        #     futures = list(cvt_futures)
        # for fut in as_completed(futures):
        #     tid = fut.result()
        #     print(f"[DONE] CVT {tid}")

if __name__ == "__main__":
    main()