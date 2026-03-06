# -*- coding: utf-8 -*-
from openpyxl import load_workbook

import subprocess


XLSX_IN  = r"/root/jyw_ws/task_and_description.en.xlsx"

TXT_IN  = "/root/jyw_ws/folders.txt"

# 1️⃣ 读取 folders.txt
with open(TXT_IN, "r") as f:
    folder_set = set(line.strip().rstrip("/") for line in f if line.strip())

with open("/root/jyw_ws/a2d_all_files.txt", "r") as f:
    a2d_folder2_set = set(line.strip().rstrip("/") for line in f if line.strip())

folder2_set = {line.split("/")[-1] for line in a2d_folder2_set}

id_to_folder = {}

for line in a2d_folder2_set:
    line = line.strip().rstrip("/")
    parts = line.split("/")

    if len(parts) >= 2:
        folder = parts[0]      # 顶层目录
        data_id = parts[-1]    # 最后一段

        id_to_folder[data_id] = folder

wb = load_workbook(XLSX_IN)
ws = wb.active  # 或 wb["Sheet1"]

# 假设：A列=任务名中文，B列=任务描述中文；新增 C列/D列英文
# ws["A1"] = "任务ID"
ws['L1']="obs_folder"

not_found_list=[]
for r in range(2, ws.max_row + 1):
    data_id = ws[f"A{r}"].value
    robot_class=ws[f"D{r}"].value
    task_name=ws[f"B{r}"].value
    # print(data_id)

    if data_id in folder_set:
        # print(f"{data_id}  ✅ FOUND")
        ws[f"L{r}"].value="data-collector-svc"
        ...
    else:
        if data_id in folder2_set:
            ws[f"L{r}"].value=id_to_folder[data_id]
            ...

        else:
            print(f"{data_id} x {robot_class} x {task_name} ❌ NOT FOUND")
            not_found_list.append(data_id)

print(len(not_found_list))
wb.save("task_and_description_checked.xlsx")
    # cmd = f"""
    # ./rclone lsf huawei-cloud:openloong-apps-prod-private/data-collector-svc \
    # --dirs-only --max-depth 2 \
    # | grep -E "^([^/]+/){{1}}{data_id}/$"
    # """

    # print("Running:", data_id)

    # result = subprocess.run(
    #     cmd,
    #     shell=True,
    #     capture_output=True,
    #     text=True
    # )

    # print("output",result.stdout)
    # print("error",result.stderr)

    # exit()



