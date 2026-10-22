"""Split RLT raw HDF5 episodes into human-controlled (control_source==2) segments.

Each input ``<name>.raw.h5`` with ``<name>.raw.h5.result`` equal to ``success``
is split at the ``frames/control_source`` boundary into contiguous runs of
``control_source == 2`` (human teleoperation).  Runs shorter than
``--min-frames`` are dropped.  Every kept run is written to the output
directory as ``<stem>_segNN.raw.h5`` with the full raw schema sliced to the
run, so the standard ``convert_raw_to_lerobot.py`` pipeline (head-only trim,
joint->EEF, stats) can consume the segments unchanged.

Policy (control_source==1), idle/SAFE (==0) and unknown (==3) frames are
excluded entirely: they must not appear in the imitation dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

LOG = logging.getLogger("cobot_bridge.extract_rlt_human_segments")

CONTROL_HUMAN = 2


def read_result(path: Path) -> str:
    if not path.exists():
        return "missing"
    return path.read_text().strip().lower()


def find_human_runs(control_source: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Return [(lo, hi)) of contiguous control_source==2 runs with length >= min_frames."""

    source = np.asarray(control_source, dtype=np.int64)
    runs: list[tuple[int, int]] = []
    n = len(source)
    i = 0
    while i < n:
        if source[i] == CONTROL_HUMAN:
            j = i
            while j + 1 < n and source[j + 1] == CONTROL_HUMAN:
                j += 1
            if j - i + 1 >= min_frames:
                runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs


def write_segment(src: h5py.File, out_path: Path, lo: int, hi: int) -> None:
    """Copy the full raw schema, sliced to [lo, hi), plus updated attributes."""

    with h5py.File(out_path, "w") as dst:
        # Copy scalar attributes, adjusting the frame count.
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["num_frames"] = hi - lo
        if "closed_at_ns" in src.attrs:
            dst.attrs.pop("closed_at_ns")
        dst.attrs["closed_at_ns"] = src.attrs.get("closed_at_ns", 0)

        # frames/* : slice every dataset.
        src_frames = src["frames"]
        dst_frames = dst.create_group("frames")
        for name, ds in src_frames.items():
            if len(ds) == hi - lo or len(ds) == src.attrs.get("num_frames", 0) or len(ds) == len(next(iter(src_frames.values()))):
                dst_frames.create_dataset(
                    name,
                    data=ds[lo:hi][:],
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                # Episode-level scalar dataset (shape (,)); copy unchanged.
                dst_frames.create_dataset(name, data=ds[:], chunks=True)

        # images/* : slice jpeg/stamp_ns arrays.
        src_images = src["images"]
        dst_images = dst.create_group("images")
        for cam, cam_group in src_images.items():
            dst_cam = dst_images.create_group(cam)
            for name, ds in cam_group.items():
                if len(ds) == src.attrs.get("num_frames", 0):
                    dst_cam.create_dataset(name, data=ds[lo:hi][:], chunks=True, compression="gzip", compression_opts=4)
                else:
                    dst_cam.create_dataset(name, data=ds[:], chunks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", required=True, help="raw h5 files to scan")
    parser.add_argument("--out-dir", type=Path, required=True, help="where segment h5 files are written")
    parser.add_argument("--min-frames", type=int, default=30, help="minimum segment length in frames (30 = 1.0 s @30 fps)")
    parser.add_argument("--manifest", type=Path, default=None, help="optional JSON manifest output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"min_frames": args.min_frames, "episodes": []}
    total_segments = 0
    total_frames = 0

    for raw_path in args.raw:
        result = read_result(raw_path.with_name(raw_path.name + ".result"))
        entry: dict[str, Any] = {"raw": str(raw_path), "result": result}
        if result != "success":
            entry["status"] = "skipped-not-success"
            manifest["episodes"].append(entry)
            LOG.info("skip (result=%s): %s", result, raw_path.name)
            continue

        with h5py.File(raw_path, "r") as src:
            if not bool(src.attrs.get("completed", False)):
                entry["status"] = "skipped-not-completed"
                manifest["episodes"].append(entry)
                LOG.warning("skip (not completed): %s", raw_path.name)
                continue
            count = int(src.attrs.get("num_frames", 0))
            if count <= 0:
                entry["status"] = "skipped-empty"
                manifest["episodes"].append(entry)
                continue
            control_source = np.asarray(src["frames/control_source"][:])
            runs = find_human_runs(control_source, args.min_frames)
            stem = raw_path.stem
            seg_paths: list[str] = []
            seg_frames: list[int] = []
            for idx, (lo, hi) in enumerate(runs):
                out_path = args.out_dir / f"{stem}_seg{idx:02d}.raw.h5"
                write_segment(src, out_path, lo, hi)
                seg_paths.append(str(out_path))
                seg_frames.append(hi - lo)
                total_segments += 1
                total_frames += hi - lo
            entry["status"] = "ok"
            entry["raw_frames"] = count
            entry["num_segments"] = len(runs)
            entry["segment_frames"] = seg_frames
            entry["segments"] = seg_paths
            LOG.info(
                "ep=%s result=success frames=%d human_runs(>=%d)=%d kept_frames=%d",
                raw_path.name,
                count,
                args.min_frames,
                len(runs),
                sum(seg_frames),
            )
            manifest["episodes"].append(entry)

    manifest["total_segments"] = total_segments
    manifest["total_segment_frames"] = total_frames
    manifest_path = args.manifest or (args.out_dir / "segments.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    LOG.info(
        "done: episodes_scanned=%d segments=%d segment_frames=%d -> %s",
        len(args.raw),
        total_segments,
        total_frames,
        manifest_path,
    )


if __name__ == "__main__":
    main()
