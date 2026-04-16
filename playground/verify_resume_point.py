# -*- coding: utf-8 -*-
import os
import diskcache
from openpyxl import load_workbook

# ===================== CONFIG =====================
HOME = "/root/jyw_ws/lerobot_jpeg/playground"
XLSX_IN = f"{HOME}/数据需求总表同步_转换列表.xlsx"
OUTPUT_ROOT = "/mnt/sdc/output_0331"
CACHE_DIR = "/root/_robot2_ckpt"

TASK_ID_COLUMN = "任务ID"
DEVICE_COLUMN = "采集设备"
DEVICE_TO_ROBOT = {"傅利叶GR-2": "gr2", "星尘智能S1": "astribot_s1"}
ALLOWED_DEVICES = set(DEVICE_TO_ROBOT.keys())
# ==================================================

def get_filtered_tasks():
    wb = load_workbook(XLSX_IN)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    col_idx = {str(name).strip(): i for i, name in enumerate(headers) if name is not None}
    task_id_col, device_col = col_idx[TASK_ID_COLUMN], col_idx[DEVICE_COLUMN]
    
    task_rows = []
    seen = set()
    for row in ws.iter_rows(min_row=2):
        tid = str(row[task_id_col].value).strip() if row[task_id_col].value else None
        if not tid or tid in seen: continue
        dev = str(row[device_col].value).strip() if row[device_col].value else ""
        if dev not in ALLOWED_DEVICES: continue
        seen.add(tid)
        task_rows.append(tid)
    wb.close()
    return task_rows

def is_converted(tid):
    path = os.path.join(OUTPUT_ROOT, tid)
    return os.path.isdir(path) and len(os.listdir(path)) > 0

def main():
    cache = diskcache.Cache(CACHE_DIR)
    tasks = get_filtered_tasks()
    cached_id = cache.get("cached_task_id")
    
    if cached_id not in tasks:
        print(f"❌ cached_id {cached_id} not found in filtered Excel list!")
        return

    idx = tasks.index(cached_id)
    print(f"Target Task ID: {cached_id}")
    print(f"Position in list: Index [{idx}] of {len(tasks)} total tasks")
    print("-" * 60)

    # Look Backward for the real last success
    last_success_id = None
    last_success_idx = -1
    for i in range(idx, -1, -1):
        if is_converted(tasks[i]):
            last_success_id = tasks[i]
            last_success_idx = i
            break
    
    # Look Forward to check for the "Crash Streak"
    streak_count = 0
    for i in range(idx + 1, len(tasks)):
        if not is_converted(tasks[i]):
            streak_count += 1
        else:
            break

    print(f"Backward Analysis: Last actual success was {last_success_id} (Index {last_success_idx})")
    print(f"Forward Analysis: {streak_count} tasks following the cached_id are all failures/missing.")
    
    if streak_count == (len(tasks) - 1 - idx):
        print("\n🚨 CONFIRMED: Disk Space Crash.")
        print(f"Everything from Index {idx} to the end of the list failed.")
    else:
        print("\nℹ️  Not a total crash. Some tasks after the cached_id succeeded.")

    print("-" * 60)
    print(f"RECOMMENDATION: To resume perfectly, set cached_task_id to: {last_success_id}")

if __name__ == "__main__":
    main()