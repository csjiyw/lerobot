# -*- coding: utf-8 -*-
from openpyxl import load_workbook

import subprocess

from tqdm import trange


XLSX_IN  = r"task_and_description_checked.xlsx"
OUTPUT_TXT = "download_list.txt"
REMOTE_BASE = "huawei-cloud:openloong-apps-prod-private"

wb = load_workbook(XLSX_IN)
ws = wb.active  # 或 wb["Sheet1"]

# 假设：A列=任务名中文，B列=任务描述中文；新增 C列/D列英文
# ws["A1"] = "任务ID"

download_list=[]
# for r in trange(2, 7):
for r in trange(2, ws.max_row + 1):
    data_id = ws[f"A{r}"].value
    robot_class=ws[f"D{r}"].value
    task_name=ws[f"J{r}"].value
    task_description=ws[f"K{r}"].value
    # print(data_id)
    obs_location=ws[f"L{r}"].value
    if obs_location is None:
        continue
    elif obs_location =="data-collector-svc":
        obs_location+="/align"


    remote_path = f"{REMOTE_BASE}/{obs_location}/{data_id}"

    # print("Processing:", remote_path)

    cmd = [
        "./rclone", "lsf",
        remote_path,
        # "--dirs-only"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    episodes = result.stdout.splitlines()[:300]

    # print(episodes)

    for ep in episodes:
        ep = ep.strip()
        if ep:
            full_path = f"{remote_path}/{ep}"
            # print(full_path)
            download_list.append(full_path)
    # break

# 🔥 统一保存
with open(OUTPUT_TXT, "w") as f:
    for line in download_list:
        f.write(line + "\n")

print("Saved:", OUTPUT_TXT)
print("Total episodes:", len(download_list))



