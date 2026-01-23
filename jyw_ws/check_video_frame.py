# torchcodec_check2.py
# 用 torchcodec 检查每个 mp4 的 num_frames，并验证最后一帧是否可取
#
# 用法：
#   python torchcodec_check2.py
#   python torchcodec_check2.py --pattern "/abs/path/observation.images.*/*/file-*.mp4"

import argparse
import glob
from torchcodec.decoders import VideoDecoder


def find_num_frames_by_probe(vd: VideoDecoder) -> int:
    """
    当 metadata 里没有 num_frames 时，
    用指数扩展 + 二分法探测最大合法 frame index（稳但慢一点）
    """
    lo, hi = 0, 1
    # 指数扩展找上界
    while True:
        try:
            vd.get_frames_at(indices=[hi])
            lo = hi
            hi *= 2
        except Exception:
            break
    # 二分找最后可用 index
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            vd.get_frames_at(indices=[mid])
            lo = mid
        except Exception:
            hi = mid
    return lo + 1  # 最大合法 index = lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default="/root/.cache/huggingface/lerobot/jyw/"
                "task_agx_fold_the_blue_cloth_towel_neatly_cfg_fps30_cam3_w224h224_abs14_"
                "trim_tail_test_merged/videos/observation.images.*/*/file-*.mp4",
        help="glob pattern for mp4 files",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern:\n{args.pattern}")

    print("=== torchcodec per-file check ===")
    for f in files:
        vd = VideoDecoder(f)

        # 尝试从 metadata 读取
        meta = (
            getattr(vd, "metadata", None)
            or getattr(vd, "stream_metadata", None)
            or getattr(vd, "video_metadata", None)
        )

        num_frames = None
        duration = None
        fps = None

        if meta is not None:
            num_frames = getattr(meta, "num_frames", None) or getattr(meta, "nb_frames", None)
            duration = getattr(meta, "duration_seconds", None) or getattr(meta, "duration", None)
            fps = getattr(meta, "fps", None) or getattr(meta, "avg_fps", None) or getattr(meta, "average_fps", None)

        # metadata 没给 num_frames，就探测
        if num_frames is None:
            num_frames = find_num_frames_by_probe(vd)

        # 验证最后一帧是否可取（复现/排除你训练里的越界问题）
        ok = True
        err = ""
        try:
            vd.get_frames_at(indices=[num_frames - 1])
        except Exception as e:
            ok = False
            err = str(e)

        if ok:
            print(f"{f}\tframes={num_frames}\tduration={duration}\tfps={fps}")
        else:
            print(f"{f}\tframes={num_frames}\tduration={duration}\tfps={fps}\tLAST_FRAME_FAIL: {err}")


if __name__ == "__main__":
    main()
