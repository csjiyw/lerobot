#!/usr/bin/env python3
import sys
import json
import h5py
import numpy as np
from pathlib import Path

def walk_h5(f: h5py.File):
    """Print the H5 hierarchy + dataset shapes/types."""
    print("\n=== H5 STRUCTURE ===")
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DSET] {name:60s} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GRP ] {name}")
    f.visititems(_visit)
    print("====================\n")

def main():
    # --- Input Handling ---
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = input("Please enter the path to the .h5 file: ").strip()

    path = Path(h5_path)
    if not path.exists():
        print(f"Error: File not found at {h5_path}")
        sys.exit(1)

    try:
        with h5py.File(path, "r") as f:
            print(f"\nInspecting: {path}")
            walk_h5(f)

            # Handle metadata.json
            if "metadata.json" in f:
                x = f["metadata.json"][()]
                if isinstance(x, bytes):
                    x = x.decode("utf-8")
                elif hasattr(x, "tobytes"):
                    x = x.tobytes().decode("utf-8")
                
                try:
                    metadata = json.loads(x)
                    print("\n=== METADATA ===")
                    print(json.dumps(metadata, indent=2, ensure_ascii=False))
                    if "time_align_info" in metadata:
                        print(f"\nFrame Count: {metadata['time_align_info'].get('frame_count')}")
                except json.JSONDecodeError:
                    print("\n[!] Could not parse metadata.json as JSON.")
            else:
                print("\n[!] metadata.json not found in H5 file.")

            # --- Interactive Mode ---
            print("\n" + "="*50)
            print("INTERACTIVE EXPLORER")
            print("Enter a dataset key (e.g., 'cameras/head/color/data') to inspect its content.")
            print("Enter 'q' to quit.")
            print("="*50)

            while True:
                key = input("\nKey to read > ").strip()
                if key.lower() == 'q':
                    break
                if not key:
                    continue
                
                try:
                    if key in f:
                        obj = f[key]
                        if isinstance(obj, h5py.Dataset):
                            # Read first 5 elements as a sample
                            sample = obj[:5]
                            print(f"Type: Dataset | Shape: {obj.shape} | Dtype: {obj.dtype}")
                            print(f"Sample (first 5):\n{sample}")
                        elif isinstance(obj, h5py.Group):
                            print(f"Type: Group | Contains: {list(obj.keys())}")
                        else:
                            print(f"Type: {type(obj)} | Value: {obj}")
                    else:
                        print(f"Error: Key '{key}' not found in H5 file.")
                except Exception as e:
                    print(f"Error reading key '{key}': {e}")

    except Exception as e:
        print(f"An error occurred opening the file: {e}")

if __name__ == "__main__":
    main()
