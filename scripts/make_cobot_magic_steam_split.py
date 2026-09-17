#!/usr/bin/env python3
"""Create deterministic episode-level STEAM train/eval manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def outcome(path: Path) -> str:
    with h5py.File(path, "r") as raw:
        return str(raw.attrs.get("outcome", "unknown")).strip().lower()


def split_every_fifth(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    eval_paths = paths[4::5]
    eval_set = set(eval_paths)
    return [path for path in paths if path not in eval_set], eval_paths


def write_manifest(path: Path, episodes: list[Path]) -> None:
    path.write_text("".join(f"{episode.resolve()}\n" for episode in episodes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--il-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    il = sorted(args.il_root.glob("*.raw.h5"))
    rollout = sorted(args.rollout_root.glob("*.raw.h5"))
    if not il or not rollout:
        raise RuntimeError(f"Missing raw episodes: il={len(il)} rollout={len(rollout)}")

    il_train, il_eval = split_every_fifth(il)
    rollout_by_outcome: dict[str, list[Path]] = {}
    for episode in rollout:
        rollout_by_outcome.setdefault(outcome(episode), []).append(episode)

    rollout_train: list[Path] = []
    rollout_eval: list[Path] = []
    for label in sorted(rollout_by_outcome):
        train, held_out = split_every_fifth(rollout_by_outcome[label])
        rollout_train.extend(train)
        rollout_eval.extend(held_out)
    rollout_train.sort()
    rollout_eval.sort()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "il_train": il_train,
        "il_eval": il_eval,
        "rollout_train": rollout_train,
        "rollout_eval": rollout_eval,
    }
    for name, episodes in splits.items():
        write_manifest(args.output_dir / f"{name}.txt", episodes)

    report = {
        "rule": "sorted episodes; every fifth episode held out; rollout stratified by outcome",
        "counts": {name: len(episodes) for name, episodes in splits.items()},
        "rollout_outcomes": {
            name: {
                label: sum(outcome(path) == label for path in episodes)
                for label in sorted(rollout_by_outcome)
            }
            for name, episodes in (
                ("rollout_train", rollout_train),
                ("rollout_eval", rollout_eval),
            )
        },
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
