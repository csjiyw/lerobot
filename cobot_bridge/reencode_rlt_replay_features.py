"""Re-encode repaired RLT decisions with a new Stage 1 encoder and VLA.

The source decision files already contain authoritative contiguous-30-Hz
executed action chunks. This offline command preserves those chunks and their
human/reference labels, while recomputing only observation-dependent z_rl,
state normalization, and reference actions for a new Stage 1 checkpoint. It
never connects to the robot bridge and cannot publish robot commands.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .reconstruct_rlt_replay import nearest_indices
from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import (
    ACTION_SOURCE_NAMES,
    ACTION_CHUNK_TIMEBASE,
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pattern", default="*.decisions.h5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-sequence-error", type=int, default=32)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help="Output action-chunk length override (e.g. 10 for the remote "
        "chunk10 RLT recipe). The source archive must have at least this many "
        "steps; executed chunks and reference actions are truncated to it.",
    )
    return parser


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def _source_names(source: h5py.File) -> list[str]:
    encoded = json.loads(str(source.attrs["action_source_names"]))
    by_id = {int(value): str(name) for name, value in encoded.items()}
    values = source["action_source"][:].astype(np.int64)
    if any(int(value) not in by_id for value in np.unique(values)):
        raise ValueError("source decisions contain an unknown action source")
    return [by_id[int(value)] for value in values]


def reencode_episode(
    source_path: Path,
    *,
    raw_path: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
    max_sequence_error: int,
    target_action_steps: int | None = None,
) -> dict[str, Any]:
    stem = source_path.name.removesuffix(".decisions.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    replay_path = output_root / f"{stem}.npz"
    if decision_path.exists() or replay_path.exists():
        if decision_path.exists() and replay_path.exists():
            return {"state": "SKIP", "episode": stem, "replay": str(replay_path)}
        raise FileExistsError(f"partial output exists for {stem}")

    writer: RLTDecisionWriter | None = None
    try:
        with h5py.File(source_path, "r") as source, h5py.File(raw_path, "r") as raw:
            if not bool(source.attrs.get("completed", False)):
                raise ValueError("source decision episode is incomplete")
            if not bool(raw.attrs.get("completed", False)):
                raise ValueError("raw episode is incomplete")
            if str(source.attrs.get("action_chunk_timebase", "")) != ACTION_CHUNK_TIMEBASE:
                raise ValueError("source decisions are not contiguous-30-Hz repaired data")
            runtime_dimensions = RLTDimensions(**runtime.dimensions)
            target_steps = (
                int(target_action_steps)
                if target_action_steps is not None
                else runtime_dimensions.action_steps
            )
            if int(source.attrs["action_steps"]) < target_steps:
                raise ValueError(
                    "cannot expand action chunks: target "
                    f"{target_steps} > source {int(source.attrs['action_steps'])}"
                )
            dimensions = RLTDimensions(
                z_dim=runtime_dimensions.z_dim,
                state_dim=runtime_dimensions.state_dim,
                action_steps=target_steps,
                action_dim=runtime_dimensions.action_dim,
            )
            if int(source.attrs["state_dim"]) != dimensions.state_dim:
                raise ValueError("source/new state dimensions differ")
            if int(source.attrs["action_dim"]) != dimensions.action_dim:
                raise ValueError("source/new action dimensions differ")

            cameras = tuple(json.loads(str(raw.attrs["cameras"])))
            if cameras != runtime.cameras:
                raise ValueError(f"camera mismatch: raw={cameras} runtime={runtime.cameras}")
            raw_sequence = raw["frames/sequence"][:].astype(np.int64)
            decision_sequence = source["bridge_sequence"][:].astype(np.int64)
            count = int(source["z_rl"].shape[0])
            observation_rows = nearest_indices(raw_sequence, decision_sequence)
            in_range = (decision_sequence >= raw_sequence[0]) & (
                decision_sequence <= raw_sequence[-1]
            )
            errors = np.abs(raw_sequence[observation_rows] - decision_sequence)
            if np.any(errors[in_range] > max_sequence_error):
                raise ValueError(
                    "decision/raw sequence error "
                    f"{int(errors[in_range].max())} exceeds "
                    f"{max_sequence_error}"
                )

            if count < 2 or observation_rows.shape != (count,):
                raise ValueError("source decisions have an invalid row count")
            selected = np.flatnonzero(in_range)
            if selected.size < 2:
                raise ValueError("fewer than two in-range source decisions remain")
            source_names = _source_names(source)
            policy_actions = source["policy_actions"][:].astype(np.float32)
            executed_actions = source["executed_actions"][:].astype(np.float32)
            if not np.isfinite(policy_actions).all() or not np.isfinite(executed_actions).all():
                raise ValueError("source actions contain non-finite values")

            writer = RLTDecisionWriter(
                decision_path,
                dimensions=dimensions,
                stage1_checkpoint=str(runtime.stage1_path),
                task=str(source.attrs["task"]),
                workflow_mode=int(source.attrs["workflow_mode"]),
                side=str(source.attrs.get("side", "right")),
                collection_phase=str(source.attrs.get("collection_phase", "actor")),
            )
            writer.file.attrs.update(
                {
                    "feature_reencode_source_decision": str(source_path),
                    "feature_reencode_raw": str(raw_path),
                    "behavior_chunk_source": "preserved_repaired_contiguous_30hz",
                    "source_decisions": count,
                    "dropped_sequence_out_of_range": int((~in_range).sum()),
                }
            )
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}
            state_raw = raw["frames/puppet_position"]
            raw_time = raw["frames/pc_timestamp_ns"]
            for number, source_row in enumerate(selected, 1):
                source_row = int(source_row)
                observation_row = observation_rows[source_row]
                observation_row = int(observation_row)
                inference = runtime.predict(
                    state=np.asarray(state_raw[observation_row], dtype=np.float32),
                    images={
                        name: _jpeg(dataset, observation_row)
                        for name, dataset in image_datasets.items()
                    },
                    task=str(source.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                source_name = source_names[source_row]
                policy_actions_row = policy_actions[source_row]
                executed_actions_row = executed_actions[source_row]
                if policy_actions_row.shape[0] > target_steps:
                    policy_actions_row = policy_actions_row[:target_steps]
                    executed_actions_row = executed_actions_row[:target_steps]
                reference_actions_row = np.asarray(
                    inference["reference_actions"], dtype=np.float32
                )
                if reference_actions_row.shape[0] > target_steps:
                    reference_actions_row = reference_actions_row[:target_steps]
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=reference_actions_row,
                    policy_actions=policy_actions_row,
                    executed_actions=executed_actions_row,
                    action_source=source_name,
                    is_intervention=source_name == "human",
                    bridge_sequence=int(raw_sequence[observation_row]),
                    time_ns=int(raw_time[observation_row]),
                )
                if number == 1 or number % 20 == 0:
                    print(
                        f"episode={stem} decision={number}/{selected.size}",
                        flush=True,
                    )
            outcome = str(source.attrs["outcome"])
            writer.close(outcome=outcome)
            writer = None
            replay_metadata = build_replay_bundle(decision_path, replay_path)
            return {
                "state": "DONE",
                "episode": stem,
                "source_decisions": count,
                "decisions": int(selected.size),
                "dropped_sequence_out_of_range": int((~in_range).sum()),
                "transitions": int(replay_metadata["transition_count"]),
                "outcome": outcome,
                "replay": str(replay_path),
            }
    except BaseException:
        if writer is not None:
            writer.abort()
        replay_path.unlink(missing_ok=True)
        replay_path.with_suffix(replay_path.suffix + ".json").unlink(missing_ok=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    source_root = args.source_replay_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    stage1 = args.stage1_checkpoint.expanduser().resolve()
    if args.max_sequence_error < 0:
        raise ValueError("max_sequence_error cannot be negative")
    source_paths = sorted(source_root.glob(args.pattern))
    if args.limit is not None:
        source_paths = source_paths[: args.limit]
    if not source_paths:
        raise FileNotFoundError(f"no source decisions under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = RLTPolicyRuntime(
        stage1_checkpoint=stage1,
        stage2_checkpoint=output_root / ".no_stage2_checkpoint.pt",
        device=args.device,
        cameras=("front", "right"),
    )
    runtime.begin_episode(workflow_mode=1, allow_actor=False, actor_exploration_std=0.0)
    results: list[dict[str, Any]] = []
    try:
        for source_path in source_paths:
            stem = source_path.name.removesuffix(".decisions.h5")
            raw_path = raw_root / f"{stem}.raw.h5"
            try:
                result = reencode_episode(
                    source_path,
                    raw_path=raw_path,
                    output_root=output_root,
                    runtime=runtime,
                    max_sequence_error=args.max_sequence_error,
                    target_action_steps=args.action_steps,
                )
            except BaseException as exc:
                result = {
                    "state": "FAIL",
                    "episode": stem,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(json.dumps(result, sort_keys=True), flush=True)
                results.append(result)
                if args.fail_fast:
                    raise
                continue
            print(json.dumps(result, sort_keys=True), flush=True)
            results.append(result)
    finally:
        runtime.end_episode()

    summary = {
        "stage1_checkpoint": str(stage1),
        "source_replay_root": str(source_root),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "total": len(results),
        "done": sum(item["state"] == "DONE" for item in results),
        "skip": sum(item["state"] == "SKIP" for item in results),
        "fail": sum(item["state"] == "FAIL" for item in results),
        "transitions": sum(int(item.get("transitions", 0)) for item in results),
        "results": results,
    }
    _atomic_json(output_root / "feature_reencode_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
