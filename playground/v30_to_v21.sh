export TMPDIR=/mnt/fastdisk
export TEMP=/mnt/fastdisk
export TMP=/mnt/fastdisk
export HF_LEROBOT_HOME=/mnt/fastdisk/huggingface/lerobot
export HF_HOME=/mnt/fastdisk/huggingface

find /mnt/sdc/output_0331 -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
| xargs -I{} -P 16 bash -c '
    repo_id="$1"
    echo "[START] $repo_id"
    python cvt_30_to_21.py --repo-id "$repo_id" --root /mnt/sdc/output_0331
' _ {}