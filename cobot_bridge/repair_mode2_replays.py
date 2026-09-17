"""Recover Mode 2 human action chunks from immutable 30 Hz raw episodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .rlt_policy_client import PersistentRLTPolicyClient
from .rlt_stage2_data import (
    build_replay_bundle,
    load_replay_bundle,
    repair_mode2_human_action_chunks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--policy-server-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--action-fps", type=float, default=30.0)
    parser.add_argument("--backup-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    replay_root = args.replay_root.expanduser().resolve()
    backup_root = args.backup_root.expanduser().resolve()
    backup_root.mkdir(parents=True, exist_ok=False)
    paths = sorted(replay_root.glob("rlt_actor_mode2_*.decisions.h5"))
    if not paths:
        raise ValueError(f"no Mode 2 decision streams under {replay_root}")

    manifest: list[dict] = []
    with PersistentRLTPolicyClient(args.policy_server_endpoint, timeout_s=30.0) as client:
        status = client.ping()
        if bool(status.get("active_episode", False)):
            raise RuntimeError("RLT policy server has an active episode")
        client.begin_episode(
            workflow_mode=2, allow_actor=False, actor_exploration_std=0.0
        )
        try:
            for decision_path in paths:
                stem = decision_path.name.removesuffix(".decisions.h5")
                raw_path = raw_root / f"{stem}.raw.h5"
                replay_path = replay_root / f"{stem}.npz"
                sidecar_path = replay_path.with_suffix(replay_path.suffix + ".json")
                required = (raw_path, decision_path, replay_path, sidecar_path)
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"{stem} is incomplete: {missing}")

                with tempfile.TemporaryDirectory(
                    prefix=f".{stem}.repair-", dir=replay_root
                ) as temporary_value:
                    temporary = Path(temporary_value)
                    repaired_decision = temporary / decision_path.name
                    repaired_replay = temporary / replay_path.name
                    shutil.copy2(decision_path, repaired_decision)
                    repair = repair_mode2_human_action_chunks(
                        repaired_decision,
                        raw_path,
                        normalize_action_chunk=client.normalize_action_chunk,
                        action_fps=args.action_fps,
                    )
                    metadata = build_replay_bundle(repaired_decision, repaired_replay)
                    arrays, loaded_metadata = load_replay_bundle(repaired_replay)
                    if metadata != loaded_metadata:
                        raise RuntimeError(f"{stem}: replay metadata round trip failed")
                    if not all(np.isfinite(values).all() for values in arrays.values()):
                        raise RuntimeError(f"{stem}: repaired replay contains non-finite values")
                    if int(metadata["intervention_transitions"]) <= 0:
                        raise RuntimeError(f"{stem}: repaired replay has no interventions")

                    for source in (decision_path, replay_path, sidecar_path):
                        shutil.copy2(source, backup_root / source.name)
                    os.replace(repaired_decision, decision_path)
                    os.replace(repaired_replay, replay_path)
                    os.replace(
                        repaired_replay.with_suffix(repaired_replay.suffix + ".json"),
                        sidecar_path,
                    )
                    row = {
                        "stem": stem,
                        "raw": str(raw_path),
                        "decision": str(decision_path),
                        "replay": str(replay_path),
                        "outcome": metadata["outcome"],
                        "transitions": int(metadata["transition_count"]),
                        "interventions": int(metadata["intervention_transitions"]),
                        "repair": repair,
                    }
                    manifest.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            client.end_episode()

    payload = {
        "format": "cobot_magic_mode2_repair_v1",
        "created_at_ns": time.time_ns(),
        "action_fps": float(args.action_fps),
        "episodes": manifest,
    }
    (backup_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "repaired_episodes": len(manifest),
                "backup_root": str(backup_root),
                "action_fps": float(args.action_fps),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
