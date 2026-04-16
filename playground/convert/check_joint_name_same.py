#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import h5py


def find_episodes(data_dir: Path):
    """Find all H5 episode files."""
    episodes = []

    # Format 1: subdirectories with h5 files
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            h5_files = list(subdir.glob("*.h5"))
            if h5_files:
                episodes.append(h5_files[0])

    # Format 2: h5 files directly in directory
    for h5_file in sorted(data_dir.glob("*.h5")):
        if h5_file not in episodes:
            episodes.append(h5_file)

    return episodes


def read_joint_names(h5_path: Path):
    """
    Read joint names from metadata.json -> data_info -> joint_info.

    Return format:
    {
        "state/arm": [...],
        "state/gripper": [...],
        "action/arm": [...],
        ...
    }
    """
    with h5py.File(h5_path, "r") as f:
        if "metadata.json" not in f:
            raise KeyError("missing metadata.json")

        metadata_raw = f["metadata.json"][()]
        metadata = json.loads(metadata_raw)

        joint_info = metadata.get("data_info", {}).get("joint_info", {})
        if not joint_info:
            raise KeyError("metadata.json missing data_info/joint_info")

        result = {}
        for k, v in joint_info.items():
            joint_names = v.get("joint_names", None)
            if joint_names is not None:
                result[k] = list(joint_names)

        return result


def compare_joint_dict(ref, cur):
    """
    Compare two joint-name dicts.

    Return list of mismatch messages.
    """
    msgs = []

    ref_keys = set(ref.keys())
    cur_keys = set(cur.keys())

    missing_keys = sorted(ref_keys - cur_keys)
    extra_keys = sorted(cur_keys - ref_keys)

    if missing_keys:
        msgs.append(f"missing keys: {missing_keys}")
    if extra_keys:
        msgs.append(f"extra keys: {extra_keys}")

    common_keys = sorted(ref_keys & cur_keys)
    for k in common_keys:
        if ref[k] != cur[k]:
            msgs.append(
                f'joint_names mismatch at "{k}"\n'
                f"  ref: {ref[k]}\n"
                f"  cur: {cur[k]}"
            )

    return msgs


def main():
    parser = argparse.ArgumentParser(
        description="Check whether all H5 files have the same joint names"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input data directory")
    args = parser.parse_args()

    episodes = find_episodes(args.input)
    if not episodes:
        print(f"No .h5 files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(episodes)} h5 files")

    ref_path = episodes[0]
    try:
        ref_joint_names = read_joint_names(ref_path)
    except Exception as e:
        print(f"[ERROR] failed to read reference file: {ref_path}")
        print(e)
        sys.exit(1)

    print(f"Reference file: {ref_path}")
    print("Reference joint keys:")
    for k in sorted(ref_joint_names.keys()):
        print(f"  {k}: {ref_joint_names[k]}")

    mismatch_count = 0
    error_count = 0

    for h5_path in episodes[1:]:
        try:
            cur_joint_names = read_joint_names(h5_path)
            diffs = compare_joint_dict(ref_joint_names, cur_joint_names)
            if diffs:
                mismatch_count += 1
                print(f"\n[MISMATCH] {h5_path}")
                for d in diffs:
                    print(d)
            else:
                # print(f"[OK] {h5_path}")
                ...
        except Exception as e:
            error_count += 1
            print(f"\n[ERROR] {h5_path}")
            print(e)

    print("\n" + "=" * 80)
    print(f"Total files     : {len(episodes)}")
    print(f"Mismatched files: {mismatch_count}")
    print(f"Error files     : {error_count}")

    if mismatch_count == 0 and error_count == 0:
        print("All h5 files have the same joint names.")


if __name__ == "__main__":
    main()