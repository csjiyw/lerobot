#!/bin/bash

REMOTE="huawei-cloud:openloong-apps-prod-private"
OUTPUT="a2d_all_files.txt"

# 清空旧文件
> "$OUTPUT"

# 1️⃣ 找所有包含 a2d 的顶层目录
dirs=$(./rclone lsf "$REMOTE" --dirs-only | grep -i a2d)

# 2️⃣ 遍历目录
for dir in $dirs
do
    echo "Processing $dir"

    ./rclone lsf "$REMOTE/$dir"  2>/dev/null \
        | sed "s:^:$dir:" >> "$OUTPUT"
done

echo "Done. Saved to $OUTPUT"