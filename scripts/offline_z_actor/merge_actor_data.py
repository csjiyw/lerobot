#!/usr/bin/env python3
"""Merge current-environment and historical-anchor actor arrays."""
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cur", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-episode-offset", type=int, default=1000)
    a = args = ap.parse_args()
    cur = np.load(a.cur); old = np.load(a.old)
    out = {}
    for k in ("z", "state", "ref", "action", "human", "frame"):
        out[k] = np.concatenate([cur[k], old[k]], 0)
    out["episode"] = np.concatenate([cur["episode"], old["episode"] + a.old_episode_offset], 0)
    np.savez_compressed(a.out, **out)
    print(f"merged {len(out['z'])} rows -> {a.out} (human {int(out['human'].sum())})")

if __name__ == "__main__":
    main()
