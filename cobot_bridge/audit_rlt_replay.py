"""Audit legacy replay or materialize provenance-enriched copies offline.

Does not re-encode observations, alter actions/rewards or connect to a robot.
Unaligned human chunks are BC-only until reconstructed with matching next obs.
The source directory and checkpoints are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .rlt_replay_provenance import actor_training_mask
from .rlt_stage2_data import _atomic_json, load_replay_bundle


def audit_replay(source_root, *, output_root=None, actor_start_step=780):
    source_root = Path(source_root).resolve()
    paths = sorted(source_root.glob("*.npz"))
    if not paths:
        raise ValueError(f"no replay bundles under {source_root}")
    if output_root is not None:
        output_root = Path(output_root).resolve()
        # Never mix migrated and pre-existing data, or overwrite any bundle.
        output_root.mkdir(parents=True, exist_ok=False)
    totals = {name: 0 for name in (
        "transitions", "actor_eligible", "human_eligible", "actor_source_eligible",
        "critic_valid", "bc_only", "unknown_source", "success_episodes",
        "human_assisted_success_episodes", "actor_without_human_success_episodes",
        "reference_only_success_episodes", "unknown_autonomy_success_episodes",
        "phase_unknown_human", "phase_calibrate_human", "phase_rescue_human", "phase_retreat_human",
    )}
    phase_names = ("unknown", "calibrate", "rescue", "retreat")
    episodes = []
    for path in paths:
        arrays, meta = load_replay_bundle(path)
        mask = actor_training_mask(arrays, actor_start_step)
        source = arrays["action_source"].reshape(-1)
        human = arrays["is_intervention"].reshape(-1).astype(bool)
        valid = arrays["td_valid"].reshape(-1).astype(bool)
        totals["transitions"] += len(mask)
        totals["actor_eligible"] += int(mask.sum())
        totals["human_eligible"] += int((mask & human).sum())
        totals["actor_source_eligible"] += int((mask & (source == 1)).sum())
        totals["critic_valid"] += int(valid.sum())
        totals["bc_only"] += int((~valid).sum())
        totals["unknown_source"] += int((source == 3).sum())
        phases = arrays.get("intervention_phase")
        phase_counts = {name: 0 for name in phase_names}
        if phases is not None:
            phases = phases.reshape(-1).astype(np.int64)
            for code, name in enumerate(phase_names):
                phase_counts[name] = int((human & (phases == code)).sum())
                totals[f"phase_{name}_human"] += phase_counts[name]
        success = meta.get("outcome") == "success"
        # Counts include the terminal observation's source when available.
        assisted = human.any() or meta.get("action_source_counts", {}).get("human", 0) > 0
        if success:
            totals["success_episodes"] += 1
            if assisted:
                totals["human_assisted_success_episodes"] += 1
            elif (source == 3).any():
                totals["unknown_autonomy_success_episodes"] += 1
            elif (source == 1).any():
                totals["actor_without_human_success_episodes"] += 1
            else:
                totals["reference_only_success_episodes"] += 1
        row = {
            "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "outcome": meta.get("outcome"), "transitions": len(mask),
            "actor_eligible": int(mask.sum()), "critic_valid": int(valid.sum()),
            "transition_action_source_counts": {str(i): int((source == i).sum()) for i in range(4)},
            "human_phase_counts": phase_counts,
        }
        if output_root is not None:
            meta = {**meta, "provenance_copy_of": str(path), "provenance_source_sha256": row["sha256"]}
            destination = output_root / path.name
            with destination.open("xb") as stream:
                np.savez_compressed(stream, metadata=json.dumps(meta, sort_keys=True), **arrays)
            _atomic_json(destination.with_suffix(".npz.json"), meta)
        episodes.append(row)
    report = {
        "source_root": str(source_root), "output_root": str(output_root) if output_root else None,
        "actor_start_step": actor_start_step, "episode_count": len(episodes),
        "totals": totals, "episodes": episodes,
        "note": "BC-only is quarantine, not temporal reconstruction. Success labels include valid human-assisted completions; not an autonomous evaluation rate.",
    }
    if output_root is not None:
        _atomic_json(output_root / "provenance_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="New directory only; omit for read-only audit")
    parser.add_argument("--actor-start-step", type=int, default=780)
    args = parser.parse_args()
    report = audit_replay(args.replay_root, output_root=args.output_root, actor_start_step=args.actor_start_step)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
