"""Build immutable sparse-reward replay bundles from finalized decisions.

This utility is intentionally offline and does no feature inference.  It is
the second half of speed-aligned raw reconstruction: decisions already contain
the observation features and the inverse-retimed behavior chunks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .rlt_stage2_data import _atomic_json, build_replay_bundle, load_replay_bundle


def build_all(
    decision_root: Path,
    output_root: Path,
    *,
    pattern: str = "*.decisions.h5",
    trim_stationary: bool = True,
) -> dict:
    decision_root = Path(decision_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    decisions = sorted(decision_root.glob(pattern))
    if not decisions:
        raise FileNotFoundError(f"no decision streams matched {pattern!r} under {decision_root}")
    items = []
    try:
        for decision in decisions:
            output = output_root / decision.name.replace(".decisions.h5", ".npz")
            metadata = build_replay_bundle(
                decision, output, trim_stationary=trim_stationary
            )
            arrays, loaded = load_replay_bundle(output)
            if metadata != loaded:
                raise RuntimeError(f"metadata round trip failed for {output.name}")
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError(f"non-finite replay tensor in {output.name}")
            items.append(
                {
                    "decision": str(decision),
                    "replay": str(output),
                    "outcome": metadata["outcome"],
                    "transitions": int(metadata["transition_count"]),
                    "critic_valid": int(metadata["td_valid_transitions"]),
                    "human_action_steps": int(metadata.get("human_action_steps", 0)),
                }
            )
            print(json.dumps(items[-1], sort_keys=True), flush=True)
    except BaseException:
        _atomic_json(
            output_root / "manifest.failed.json",
            {"state": "failed", "created_at_ns": time.time_ns(), "items": items},
        )
        raise
    manifest = {
        "format": "cobot_magic_rlt_decision_replay_build_v1",
        "state": "completed",
        "created_at_ns": time.time_ns(),
        "decision_root": str(decision_root),
        "output_root": str(output_root),
        "trim_stationary": bool(trim_stationary),
        "episodes": len(items),
        "transitions": sum(item["transitions"] for item in items),
        "critic_valid": sum(item["critic_valid"] for item in items),
        "human_action_steps": sum(item["human_action_steps"] for item in items),
        "items": items,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.decisions.h5")
    parser.add_argument("--no-trim-stationary", action="store_true")
    args = parser.parse_args()
    manifest = build_all(
        args.decision_root,
        args.output_root,
        pattern=args.pattern,
        trim_stationary=not args.no_trim_stationary,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
