"""Re-encode historical RLT-online episodes on a contiguous 30 Hz timebase.

Old decision files remain useful for their observation sequence, source label,
and terminal outcome, but their action chunks used every second VLA output.
This command therefore re-runs frozen Stage 1 on the recorded observation and
reconstructs the behavior chunk from the authoritative raw 30 Hz command
stream.  It never connects to the bridge and cannot move the robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from .reconstruct_rlt_replay import nearest_indices
from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import (
    ACTION_SOURCE_NAMES,
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
)


LOG = logging.getLogger("cobot_bridge.reencode_rlt_online_replay")
DECISION_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r"\s+\[INFO\]\s+decision=(\d+)\s+source=(reference|actor|human)"
)


@dataclass(frozen=True)
class LoggedDecision:
    number: int
    time_ns: int
    source: str


def parse_online_decision_log(text: str, *, timezone: str) -> list[LoggedDecision]:
    zone = ZoneInfo(timezone)
    output: list[LoggedDecision] = []
    for timestamp, number, source in DECISION_PATTERN.findall(text):
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=zone
        )
        whole_seconds = int(parsed.replace(microsecond=0).timestamp())
        output.append(
            LoggedDecision(
                number=int(number),
                time_ns=whole_seconds * 1_000_000_000 + parsed.microsecond * 1_000,
                source=source,
            )
        )
    if not output or [item.number for item in output] != list(range(1, len(output) + 1)):
        raise ValueError("RLT online log has missing or unordered decision records")
    if any(right.time_ns <= left.time_ns for left, right in zip(output, output[1:])):
        raise ValueError("RLT online decision timestamps are not strictly increasing")
    return output


def contiguous_indices_at_times(
    timestamps_ns: np.ndarray,
    starts_ns: np.ndarray,
    *,
    action_steps: int,
    action_fps: float,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    starts = np.asarray(starts_ns, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size == 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("raw timestamps must be a non-empty increasing vector")
    if action_steps <= 0 or not np.isfinite(action_fps) or action_fps <= 0:
        raise ValueError("action_steps and action_fps must be positive")
    offsets = (
        np.arange(action_steps, dtype=np.float64) * 1e9 / float(action_fps)
    ).round().astype(np.int64)
    return np.stack(
        [nearest_indices(timestamps, start + offsets) for start in starts], axis=0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-decisions-root", type=Path, required=True)
    parser.add_argument("--job-log-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--action-fps", type=float, default=30.0)
    parser.add_argument("--pattern", default="rlt_*.raw.h5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-sequence-error", type=int, default=32)
    return parser


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _matching_log(raw_path: Path, log_root: Path) -> Path:
    matches = [
        path
        for path in log_root.glob("*_rlt_online_execute.log")
        if raw_path.name in path.read_text(errors="replace")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one RLT online log for {raw_path.name}, found {len(matches)}")
    return matches[0].resolve()


def _source_names(source: h5py.File) -> list[str]:
    encoded = json.loads(str(source.attrs["action_source_names"]))
    by_id = {int(value): str(name) for name, value in encoded.items()}
    values = source["action_source"][:].astype(np.int64)
    if any(int(value) not in by_id for value in np.unique(values)):
        raise ValueError("source decision stream has an unknown action source")
    return [by_id[int(value)] for value in values]


def _nearest_sequence_indices(
    raw_sequence: np.ndarray, decision_sequence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_values = np.asarray(raw_sequence, dtype=np.int64)
    decisions = np.asarray(decision_sequence, dtype=np.int64)
    indices = nearest_indices(raw_values, decisions)
    in_range = (decisions >= raw_values[0]) & (decisions <= raw_values[-1])
    errors = np.abs(raw_values[indices] - decisions)
    return indices, in_range, errors


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def reencode_episode(
    raw_path: Path,
    *,
    source_decision_path: Path,
    log_path: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
    timezone: str,
    action_fps: float,
    max_sequence_error: int,
) -> dict[str, Any]:
    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    replay_path = output_root / f"{stem}.npz"
    if decision_path.exists() or replay_path.exists():
        raise FileExistsError(f"output already exists for {stem}")
    logged = parse_online_decision_log(log_path.read_text(errors="replace"), timezone=timezone)
    dimensions = RLTDimensions(**runtime.dimensions)
    writer: RLTDecisionWriter | None = None
    runtime.begin_episode(workflow_mode=1, allow_actor=False, actor_exploration_std=0.0)
    try:
        with h5py.File(raw_path, "r") as raw, h5py.File(source_decision_path, "r") as source:
            if not bool(raw.attrs.get("completed", False)) or not bool(
                source.attrs.get("completed", False)
            ):
                raise ValueError("raw/source decision episode is incomplete")
            source_count = int(source["time_ns"].shape[0])
            if len(logged) not in (source_count, source_count + 1):
                raise ValueError(
                    f"decision/log count mismatch: source={source_count} log={len(logged)}"
                )
            logged = logged[:source_count]
            sources = _source_names(source)
            if [item.source for item in logged] != sources:
                raise ValueError("decision source labels disagree with persistent RLT log")

            cameras = tuple(json.loads(str(raw.attrs["cameras"])))
            if cameras != runtime.cameras:
                raise ValueError(f"camera mismatch: raw={cameras} runtime={runtime.cameras}")
            timestamps = raw["frames/pc_timestamp_ns"][:].astype(np.int64)
            raw_sequence = raw["frames/sequence"][:].astype(np.int64)
            observations, sequence_in_range, sequence_errors = _nearest_sequence_indices(
                raw_sequence, source["bridge_sequence"][:].astype(np.int64)
            )
            if np.any(sequence_errors[sequence_in_range] > max_sequence_error):
                raise ValueError(
                    "decision/raw sequence alignment exceeds limit: "
                    f"{int(sequence_errors[sequence_in_range].max())} > {max_sequence_error}"
                )

            action_starts = np.asarray([item.time_ns for item in logged], dtype=np.int64)
            action_rows = contiguous_indices_at_times(
                timestamps,
                action_starts,
                action_steps=dimensions.action_steps,
                action_fps=action_fps,
            )
            raw_human = raw["frames/is_human_intervention"][:].astype(bool)
            raw_policy = raw["frames/is_policy"][:].astype(bool)
            executed_valid = raw["frames/executed_action_valid"][:].astype(bool)
            executed_raw = raw["frames/executed_action"][:].astype(np.float32)
            if not np.isfinite(executed_raw[executed_valid]).all():
                raise ValueError("raw executed actions contain non-finite values")

            keep = sequence_in_range.copy()
            dropped_action_tail = 0
            for row, source_name in enumerate(sources):
                if source_name == "human":
                    start = int(observations[row])
                    if not raw_human[start]:
                        keep[row] = False
                        continue
                    segment_end = start
                    while segment_end + 1 < raw_human.size and raw_human[segment_end + 1]:
                        segment_end += 1
                    targets = timestamps[start] + (
                        np.arange(dimensions.action_steps, dtype=np.float64)
                        * 1e9
                        / float(action_fps)
                    ).round().astype(np.int64)
                    indices = nearest_indices(timestamps, targets)
                    action_rows[row] = np.clip(indices, start, segment_end)
                else:
                    end_time = action_starts[row] + int(
                        round((dimensions.action_steps - 1) * 1e9 / action_fps)
                    )
                    if action_starts[row] < timestamps[0] or end_time > timestamps[-1]:
                        keep[row] = False
                        dropped_action_tail += 1
                        continue
                indices = action_rows[row]
                expected_source = raw_human if source_name == "human" else raw_policy
                if not executed_valid[indices].all() or not expected_source[indices].all():
                    keep[row] = False

            selected = np.flatnonzero(keep)
            if selected.size < 2:
                raise ValueError("fewer than two fully aligned decisions remain")
            outcome = str(source.attrs.get("outcome", raw.attrs.get("outcome", "unknown")))
            if outcome not in ("success", "failure"):
                raise ValueError(f"invalid terminal outcome: {outcome}")
            workflow_mode = int(source.attrs.get("workflow_mode", 0))
            collection_phase = str(source.attrs.get("collection_phase", "actor"))
            writer = RLTDecisionWriter(
                decision_path,
                dimensions=dimensions,
                stage1_checkpoint=str(runtime.stage1_path),
                task=str(raw.attrs["task"]),
                workflow_mode=workflow_mode,
                side=str(raw.attrs["side"]),
                collection_phase=collection_phase,
            )
            writer.file.attrs.update(
                {
                    "historical_reencode_source_decision": str(source_decision_path),
                    "historical_reencode_log": str(log_path),
                    "behavior_chunk_source": "raw_executed_action_contiguous_30hz",
                    "behavior_policy_start": "rlt_online_action_sent_log_time",
                    "behavior_human_start": "teach_observation_bridge_sequence",
                }
            )
            state_raw = raw["frames/puppet_position"][:].astype(np.float32)
            sequence = raw["frames/sequence"]
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}
            for number, row in enumerate(selected, 1):
                observation_index = int(observations[row])
                inference = runtime.predict(
                    state=state_raw[observation_index],
                    images={
                        name: _jpeg(dataset, observation_index)
                        for name, dataset in image_datasets.items()
                    },
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                behavior = runtime.normalize_actions(executed_raw[action_rows[row]])
                source_name = sources[row]
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=inference["reference_actions"],
                    policy_actions=behavior,
                    executed_actions=behavior,
                    action_source=source_name,
                    is_intervention=source_name == "human",
                    bridge_sequence=int(sequence[observation_index]),
                    time_ns=int(timestamps[observation_index]),
                )
                if number == 1 or number % 20 == 0:
                    LOG.info("%s decision=%d/%d", stem, number, selected.size)
            writer.close(outcome=outcome)
            writer = None
        replay = build_replay_bundle(decision_path, replay_path)
        return {
            "raw": str(raw_path),
            "source_decision": str(source_decision_path),
            "log": str(log_path),
            "decision": str(decision_path),
            "replay": str(replay_path),
            "outcome": replay["outcome"],
            "source_decisions": source_count,
            "kept_decisions": int(selected.size),
            "transitions": int(replay["transition_count"]),
            "dropped_sequence_out_of_range": int((~sequence_in_range).sum()),
            "dropped_action_tail": int(dropped_action_tail),
            "dropped_source_or_validity": int(source_count - selected.size - (~sequence_in_range).sum() - dropped_action_tail),
            "median_sequence_error": float(np.median(sequence_errors[sequence_in_range])),
            "max_sequence_error": int(sequence_errors[sequence_in_range].max()),
        }
    except BaseException:
        if writer is not None:
            writer.abort()
        decision_path.unlink(missing_ok=True)
        replay_path.unlink(missing_ok=True)
        replay_path.with_suffix(replay_path.suffix + ".json").unlink(missing_ok=True)
        raise
    finally:
        runtime.end_episode()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw_root = args.raw_root.expanduser().resolve()
    source_root = args.source_decisions_root.expanduser().resolve()
    log_root = args.job_log_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(raw_root.glob(args.pattern))
    if args.limit is not None:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths:
        raise RuntimeError(f"no raw episodes match {args.pattern!r} under {raw_root}")
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=output_root / ".no_stage2_checkpoint.pt",
        device=args.device,
        cameras=("front", "right"),
    )
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, raw_path in enumerate(raw_paths, 1):
        stem = raw_path.name.removesuffix(".raw.h5")
        source = source_root / f"{stem}.decisions.h5"
        record: dict[str, Any]
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            replay_path = output_root / f"{stem}.npz"
            if replay_path.is_file() and replay_path.with_suffix(".npz.json").is_file():
                record = {"state": "SKIP", "raw": str(raw_path), "replay": str(replay_path)}
            else:
                record = {
                    "state": "DONE",
                    **reencode_episode(
                        raw_path,
                        source_decision_path=source,
                        log_path=_matching_log(raw_path, log_root),
                        output_root=output_root,
                        runtime=runtime,
                        timezone=args.timezone,
                        action_fps=args.action_fps,
                        max_sequence_error=args.max_sequence_error,
                    ),
                }
        except Exception as error:
            LOG.exception("failed to re-encode %s", raw_path.name)
            record = {"state": "FAIL", "raw": str(raw_path), "error": str(error)}
        results.append(record)
        done = sum(item["state"] == "DONE" for item in results)
        skipped = sum(item["state"] == "SKIP" for item in results)
        failed = sum(item["state"] == "FAIL" for item in results)
        elapsed = time.monotonic() - started
        rate = index / max(elapsed, 1e-9)
        manifest = {
            "stage1_checkpoint": str(runtime.stage1_path),
            "pattern": args.pattern,
            "total": len(raw_paths),
            "processed": index,
            "done": done,
            "skipped": skipped,
            "failed": failed,
            "episodes_per_second": rate,
            "eta_seconds": (len(raw_paths) - index) / max(rate, 1e-9),
            "results": results,
        }
        _atomic_json(output_root / "reencode_manifest.json", manifest)
        LOG.info(
            "progress=%d/%d DONE=%d SKIP=%d FAIL=%d eta=%.1fmin",
            index,
            len(raw_paths),
            done,
            skipped,
            failed,
            manifest["eta_seconds"] / 60,
        )
    if any(item["state"] == "FAIL" for item in results):
        raise RuntimeError("one or more historical RLT episodes failed to re-encode")


if __name__ == "__main__":
    main()
