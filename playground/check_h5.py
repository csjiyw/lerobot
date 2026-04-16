#!/usr/bin/env python3
import h5py
import json
import os
import sys
from pathlib import Path

H5_DIR = "/mnt/fastdisk/tmp_download/245699c6916d441b996a3865b13a9cb9"

def main():
    h5_files = list(Path(H5_DIR).rglob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {H5_DIR}")
        return

    for h5_path in h5_files:
        print(f"\n{'='*60}")
        print(f"FILE: {h5_path}")
        print(f"{'='*60}")
        try:
            with h5py.File(h5_path, "r") as f:
                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  [DSET] {name:50s} shape={obj.shape} dtype={obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"  [GRP ] {name}")
                f.visititems(_visit)

                if "metadata.json" in f:
                    x = f["metadata.json"][()]
                    if isinstance(x, bytes):
                        x = x.decode("utf-8")
                    elif hasattr(x, "tobytes"):
                        x = x.tobytes().decode("utf-8")
                    try:
                        meta = json.loads(x)
                        print(f"  [META] frame_count={meta.get('time_align_info', {}).get('frame_count')}")
                    except json.JSONDecodeError:
                        print(f"  [META] (parse failed)")
        except Exception as e:
            print(f"  [ERR] {e}")

if __name__ == "__main__":
    main()
