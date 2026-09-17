"""Reconstruct RLT decision/replay bundles from labeled Mode-1 raw rollouts.

The regular dashboard rollout stores the actually executed 30 Hz command
stream but not the frozen policy's latent token or complete reference chunk.
This command aligns each raw episode with its rollout log, re-runs the frozen
Stage-1 reference policy on the recorded observations, and keeps the recorded
commands as the off-policy behavior actions.

Raw episodes are immutable.  Outputs are resumable and written to a separate
directory; no robot or bridge connection is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import (
    ACTION_SOURCE_NAMES,
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
)


LOG = logging.getLogger("cobot_bridge.reconstruct_rlt_replay")
CHUNK_LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r"\s+\[INFO\]\s+received action chunk size=(\d+) consuming=(\d+)"
)
LOG_TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[INFO\]"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--job-log-root",
        type=Path,
        help="persistent dashboard log directory used when /api/jobs is unavailable",
    )
    parser.add_argument(
        "--prefer-job-logs",
        action="store_true",
        help="Use persistent dashboard logs even when the live dashboard API is reachable.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--max-job-start-delta", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup-decisions-root",
        type=Path,
        help="Legacy warmup decision streams to re-encode with the new Stage 1.",
    )
    parser.add_argument(
        "--only-warmup",
        action="store_true",
        help="Skip standard rollout reconstruction and only re-encode warmups.",
    )
    parser.add_argument(
        "--decision-pattern",
        default="rlt_warmup_mode1_*.decisions.h5",
        help="Glob used under --warmup-decisions-root for exact-timestamp re-encoding.",
    )
    return parser


def parse_chunk_completion_times(
    text: str, *, timezone: str, expected_chunk_size: int = 30
) -> np.ndarray:
    zone = ZoneInfo(timezone)
    values: list[int] = []
    for timestamp, chunk_size, consumed in CHUNK_LOG_PATTERN.findall(text):
        if int(chunk_size) != expected_chunk_size or int(consumed) != expected_chunk_size:
            raise ValueError(
                f"unexpected rollout chunk: size={chunk_size} consumed={consumed}"
            )
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=zone)
        whole_seconds = int(parsed.replace(microsecond=0).timestamp())
        values.append(whole_seconds * 1_000_000_000 + parsed.microsecond * 1_000)
    result = np.asarray(values, dtype=np.int64)
    if len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError(f"need at least two ordered chunk timestamps, got {len(result)}")
    return result


def nearest_indices(timestamps_ns: np.ndarray, targets_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    targets = np.asarray(targets_ns, dtype=np.int64)
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    use_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    return np.where(use_left, left, right).astype(np.int64)


def decision_indices(
    timestamps_ns: np.ndarray,
    completion_times_ns: np.ndarray,
    *,
    source_fps: float,
    action_steps: int,
    action_stride: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return observation indices and per-decision behavior action indices."""

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    completions = np.asarray(completion_times_ns, dtype=np.int64)
    nominal_execution_ns = int(round(30 / source_fps * 1e9))
    observed_latencies = np.diff(completions) - nominal_execution_ns
    positive = observed_latencies[observed_latencies > 0]
    latency_ns = int(np.median(positive)) if len(positive) else int(0.1e9)
    latency_ns = int(np.clip(latency_ns, int(0.02e9), int(0.5e9)))
    observation_targets = completions - latency_ns
    offsets = (
        np.arange(action_steps, dtype=np.float64) * action_stride / source_fps * 1e9
    ).round().astype(np.int64)

    available = (observation_targets >= timestamps[0]) & (
        completions + offsets[-1] <= timestamps[-1]
    )
    completions = completions[available]
    observation_targets = observation_targets[available]
    if len(completions) < 2:
        raise ValueError("fewer than two complete decisions overlap the raw episode")
    observations = nearest_indices(timestamps, observation_targets)
    action_indices = np.stack(
        [nearest_indices(timestamps, completion + offsets) for completion in completions]
    )
    return observations, action_indices, latency_ns / 1e6


def fetch_jobs(dashboard_url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(dashboard_url.rstrip("/") + "/api/jobs", timeout=10) as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError("dashboard /api/jobs did not return a list")
    return payload


def first_log_timestamp_ns(text: str, *, timezone: str) -> int:
    match = LOG_TIMESTAMP_PATTERN.search(text)
    if match is None:
        raise ValueError("log contains no timestamped INFO record")
    parsed = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
        tzinfo=ZoneInfo(timezone)
    )
    whole_seconds = int(parsed.replace(microsecond=0).timestamp())
    return whole_seconds * 1_000_000_000 + parsed.microsecond * 1_000


def match_policy_log_offline(
    raw_path: Path,
    job_log_root: Path,
    *,
    timezone: str,
    max_start_delta_s: float,
) -> Path:
    """Recover a record/policy job pair from persistent dashboard logs."""

    record_matches: list[tuple[Path, str]] = []
    for path in job_log_root.glob("*_record.log"):
        text = path.read_text(errors="replace")
        if raw_path.name in text:
            record_matches.append((path, text))
    if len(record_matches) != 1:
        raise ValueError(
            f"expected one persistent record log for {raw_path.name}, "
            f"found {len(record_matches)}"
        )
    record_started_ns = first_log_timestamp_ns(record_matches[0][1], timezone=timezone)

    candidates: list[tuple[int, Path]] = []
    for path in job_log_root.glob("*_policy_execute.log"):
        text = path.read_text(errors="replace")
        if (
            "policy_server=" not in text
            or "'effective_mode': 'POLICY_PUPPET'" not in text
            or "received action chunk size=30 consuming=30" not in text
        ):
            continue
        started_ns = first_log_timestamp_ns(text, timezone=timezone)
        candidates.append((abs(started_ns - record_started_ns), path))
    if not candidates:
        raise ValueError(f"no persistent Mode-1 policy log found under {job_log_root}")
    delta_ns, policy_path = min(candidates, key=lambda item: item[0])
    if delta_ns / 1e9 > max_start_delta_s:
        raise ValueError(
            f"nearest persistent policy log for {raw_path.name} starts "
            f"{delta_ns / 1e9:.3f}s away"
        )
    return policy_path.resolve()


def match_policy_log(
    raw_path: Path, jobs: list[dict[str, Any]], *, max_start_delta_s: float
) -> Path:
    raw_resolved = str(raw_path.resolve())
    record = [
        job
        for job in jobs
        if job.get("kind") == "record"
        and str(job.get("metadata", {}).get("output", "")) == raw_resolved
    ]
    if len(record) != 1:
        raise ValueError(f"expected one record job for {raw_path.name}, found {len(record)}")
    started = int(record[0]["started_at_ns"])
    policies = [job for job in jobs if job.get("kind") == "policy_execute"]
    if not policies:
        raise ValueError("dashboard history contains no policy jobs")
    policy = min(policies, key=lambda item: abs(int(item["started_at_ns"]) - started))
    delta_s = abs(int(policy["started_at_ns"]) - started) / 1e9
    if delta_s > max_start_delta_s:
        raise ValueError(
            f"nearest policy job for {raw_path.name} starts {delta_s:.3f}s away"
        )
    metadata = policy.get("metadata", {})
    if metadata.get("runner") != "standard" or int(metadata.get("mode", 0)) != 1:
        raise ValueError(f"unsupported policy job metadata: {metadata}")
    command = list(policy.get("command", []))
    if "--fps" not in command or float(command[command.index("--fps") + 1]) != 30.0:
        raise ValueError("offline reconstruction currently requires a 30 FPS rollout")
    path = Path(str(policy["log_path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def reconstruct_episode(
    raw_path: Path,
    *,
    policy_log: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
    timezone: str,
    source_fps: float,
) -> dict[str, Any]:
    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    replay_path = output_root / f"{stem}.npz"
    if decision_path.exists() or replay_path.exists():
        raise FileExistsError(f"output already exists for {stem}")
    completions = parse_chunk_completion_times(policy_log.read_text(errors="replace"), timezone=timezone)
    dimensions = RLTDimensions(**runtime.dimensions)

    writer: RLTDecisionWriter | None = None
    runtime.begin_episode(workflow_mode=1, allow_actor=False, actor_exploration_std=0.0)
    try:
        with h5py.File(raw_path, "r") as raw:
            if not bool(raw.attrs.get("completed", False)):
                raise ValueError("raw episode is not completed")
            outcome = str(raw.attrs.get("outcome", "unknown"))
            if outcome not in ("success", "failure"):
                raise ValueError(f"invalid outcome: {outcome}")
            cameras = json.loads(str(raw.attrs["cameras"]))
            if tuple(cameras) != runtime.cameras:
                raise ValueError(f"camera mismatch: raw={cameras} runtime={runtime.cameras}")
            timestamps = raw["frames/pc_timestamp_ns"][:]
            observations, action_rows, latency_ms = decision_indices(
                timestamps,
                completions,
                source_fps=source_fps,
                action_steps=dimensions.action_steps,
                action_stride=int(runtime.config.action_stride),
            )
            valid = raw["frames/executed_action_valid"][:].astype(bool)
            policy = raw["frames/is_policy"][:].astype(bool)
            selected = action_rows.reshape(-1)
            if not valid[selected].all() or not policy[selected].all():
                raise ValueError("a reconstructed action sample is not a valid policy action")

            writer = RLTDecisionWriter(
                decision_path,
                stage1_checkpoint=runtime.stage1_path,
                task=str(raw.attrs["task"]),
                workflow_mode=1,
                collection_phase="warmup",
                side=str(raw.attrs["side"]),
                dimensions=dimensions,
            )
            # Load the compact numeric streams once.  Nearest-neighbour sampling
            # may legitimately select the same 30 Hz frame twice when recorder
            # timestamps jitter; h5py fancy indexing requires strictly increasing
            # unique indices, while NumPy preserves the intended sampling exactly.
            executed_raw = raw["frames/executed_action"][:]
            state_raw = raw["frames/puppet_position"][:]
            sequence = raw["frames/sequence"]
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}
            for decision_number, (observation_index, indices) in enumerate(
                zip(observations, action_rows, strict=True)
            ):
                inference = runtime.predict(
                    state=np.asarray(state_raw[observation_index], dtype=np.float32),
                    images={
                        name: _jpeg(dataset, int(observation_index))
                        for name, dataset in image_datasets.items()
                    },
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                behavior = runtime.normalize_actions(
                    np.asarray(executed_raw[indices], dtype=np.float32)
                )
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=inference["reference_actions"],
                    policy_actions=behavior,
                    executed_actions=behavior,
                    action_source="reference",
                    is_intervention=False,
                    bridge_sequence=int(sequence[observation_index]),
                    time_ns=int(timestamps[observation_index]),
                )
                if decision_number == 0 or (decision_number + 1) % 20 == 0:
                    LOG.info(
                        "%s decision=%d/%d", stem, decision_number + 1, len(observations)
                    )
            writer.close(outcome=outcome)
            writer = None
        replay = build_replay_bundle(
            decision_path,
            replay_path,
            trim_stationary=False,
        )
        return {
            "raw": str(raw_path),
            "policy_log": str(policy_log),
            "decision": str(decision_path),
            "replay": str(replay_path),
            "outcome": replay["outcome"],
            "decisions": replay["source_decisions"],
            "transitions": replay["transition_count"],
            "estimated_inference_latency_ms": latency_ms,
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


def reconstruct_warmup_episode(
    raw_path: Path,
    *,
    source_decision_path: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
) -> dict[str, Any]:
    """Re-encode a recorded legacy decision stream at its exact timestamps."""

    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    replay_path = output_root / f"{stem}.npz"
    if decision_path.exists() or replay_path.exists():
        raise FileExistsError(f"output already exists for {stem}")
    dimensions = RLTDimensions(**runtime.dimensions)
    writer: RLTDecisionWriter | None = None
    runtime.begin_episode(workflow_mode=1, allow_actor=False, actor_exploration_std=0.0)
    try:
        with h5py.File(raw_path, "r") as raw, h5py.File(source_decision_path, "r") as source:
            if not bool(raw.attrs.get("completed", False)) or not bool(
                source.attrs.get("completed", False)
            ):
                raise ValueError("warmup raw/decision source is not completed")
            outcome = str(source.attrs.get("outcome", "unknown"))
            if outcome not in ("success", "failure"):
                raise ValueError(f"invalid warmup outcome: {outcome}")
            cameras = json.loads(str(raw.attrs["cameras"]))
            if tuple(cameras) != runtime.cameras:
                raise ValueError(f"camera mismatch: raw={cameras} runtime={runtime.cameras}")
            source_times = source["time_ns"][:].astype(np.int64)
            timestamps = raw["frames/pc_timestamp_ns"][:].astype(np.int64)
            observations = nearest_indices(timestamps, source_times)
            timestamp_error_ms = np.abs(timestamps[observations] - source_times) / 1e6
            behavior = source["executed_actions"][:].astype(np.float32)
            workflow_mode = int(source.attrs.get("workflow_mode", 1))
            collection_phase = str(source.attrs.get("collection_phase", "warmup"))
            action_sources = source["action_source"][:].astype(np.int64)
            intervention = source["is_intervention"][:].astype(bool)
            source_names = {
                int(value): name for name, value in ACTION_SOURCE_NAMES.items()
            }
            expected = (
                len(source_times), dimensions.action_steps, dimensions.action_dim
            )
            if behavior.shape != expected or not np.isfinite(behavior).all():
                raise ValueError(f"invalid legacy warmup behavior chunks: {behavior.shape}")
            if action_sources.shape != (len(source_times),) or intervention.shape != (
                len(source_times),
            ):
                raise ValueError("invalid legacy action-source arrays")
            if any(int(value) not in source_names for value in np.unique(action_sources)):
                raise ValueError("legacy decision stream has unknown action source")
            aligned = timestamp_error_ms <= 20.0
            dropped_alignment_outliers = int((~aligned).sum())
            if dropped_alignment_outliers:
                if int(aligned.sum()) < 2:
                    raise ValueError(
                        "legacy decision/raw alignment leaves fewer than two decisions"
                    )
                LOG.warning(
                    "%s dropping %d/%d decision timestamp outliers (max %.3f ms)",
                    stem,
                    dropped_alignment_outliers,
                    len(aligned),
                    float(timestamp_error_ms.max()),
                )
                source_times = source_times[aligned]
                observations = observations[aligned]
                timestamp_error_ms = timestamp_error_ms[aligned]
                behavior = behavior[aligned]
                action_sources = action_sources[aligned]
                intervention = intervention[aligned]

            writer = RLTDecisionWriter(
                decision_path,
                stage1_checkpoint=runtime.stage1_path,
                task=str(raw.attrs["task"]),
                workflow_mode=workflow_mode,
                collection_phase=collection_phase,
                side=str(raw.attrs["side"]),
                dimensions=dimensions,
            )
            state_raw = raw["frames/puppet_position"][:]
            sequence = raw["frames/sequence"]
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}
            for decision_number, observation_index in enumerate(observations):
                inference = runtime.predict(
                    state=np.asarray(state_raw[observation_index], dtype=np.float32),
                    images={
                        name: _jpeg(dataset, int(observation_index))
                        for name, dataset in image_datasets.items()
                    },
                    task=str(raw.attrs["task"]),
                    robot_type="cobot_magic_piper",
                )
                writer.append(
                    z_rl=inference["z_rl"],
                    state=inference["state"],
                    reference_actions=inference["reference_actions"],
                    policy_actions=behavior[decision_number],
                    executed_actions=behavior[decision_number],
                    action_source=source_names[int(action_sources[decision_number])],
                    is_intervention=bool(intervention[decision_number]),
                    bridge_sequence=int(sequence[observation_index]),
                    time_ns=int(timestamps[observation_index]),
                )
                if decision_number == 0 or (decision_number + 1) % 20 == 0:
                    LOG.info(
                        "%s decision=%d/%d", stem, decision_number + 1, len(observations)
                    )
            writer.close(outcome=outcome)
            writer = None
        replay = build_replay_bundle(decision_path, replay_path, trim_stationary=False)
        return {
            "raw": str(raw_path),
            "source_decision": str(source_decision_path),
            "decision": str(decision_path),
            "replay": str(replay_path),
            "outcome": replay["outcome"],
            "decisions": replay["source_decisions"],
            "transitions": replay["transition_count"],
            "median_timestamp_error_ms": float(np.median(timestamp_error_ms)),
            "max_timestamp_error_ms": float(timestamp_error_ms.max()),
            "dropped_alignment_outliers": dropped_alignment_outliers,
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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.only_warmup and args.warmup_decisions_root is None:
        raise ValueError("--only-warmup requires --warmup-decisions-root")
    jobs: list[dict[str, Any]] | None = None
    if not args.only_warmup:
        if args.prefer_job_logs:
            if args.job_log_root is None:
                raise ValueError("--prefer-job-logs requires --job-log-root")
            LOG.info("using persistent dashboard logs from %s", args.job_log_root)
        else:
            try:
                jobs = fetch_jobs(args.dashboard_url)
                LOG.info("using live dashboard job history from %s", args.dashboard_url)
            except (OSError, ValueError, urllib.error.URLError) as error:
                if args.job_log_root is None:
                    raise RuntimeError(
                        "dashboard job history is unavailable; pass --job-log-root to use "
                        "persistent logs"
                    ) from error
                LOG.warning("dashboard job history unavailable; using %s", args.job_log_root)
    # Only ordinary Mode-1 rollout records have a persistent record/policy log
    # pair.  Legacy warmup and later Actor episodes are deliberately excluded.
    raw_paths = [] if args.only_warmup else sorted(raw_root.glob("episode_*.raw.h5"))
    if args.limit is not None:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths and not args.only_warmup:
        raise RuntimeError(f"no IL-policy Mode-1 rollout raw episodes under {raw_root}")

    runtime = RLTPolicyRuntime(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=output_root / ".no_stage2_checkpoint.pt",
        device=args.device,
        cameras=("front", "right"),
    )
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    for episode_number, raw_path in enumerate(raw_paths, 1):
        if jobs is not None:
            policy_log = match_policy_log(
                raw_path, jobs, max_start_delta_s=args.max_job_start_delta
            )
        else:
            policy_log = match_policy_log_offline(
                raw_path,
                args.job_log_root.expanduser().resolve(),
                timezone=args.timezone,
                max_start_delta_s=args.max_job_start_delta,
            )
        result = reconstruct_episode(
            raw_path,
            policy_log=policy_log,
            output_root=output_root,
            runtime=runtime,
            timezone=args.timezone,
            source_fps=args.source_fps,
        )
        completed.append(result)
        atomic_json(
            output_root / "offline_reconstruction_manifest.json",
            {
                "stage1_checkpoint": str(runtime.stage1_path),
                "raw_root": str(raw_root),
                "output_root": str(output_root),
                "completed": completed,
                "completed_episodes": len(completed),
                "target_episodes": len(raw_paths),
                "transitions": sum(int(item["transitions"]) for item in completed),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        LOG.info(
            "episode=%d/%d outcome=%s decisions=%d transitions=%d",
            episode_number,
            len(raw_paths),
            result["outcome"],
            result["decisions"],
            result["transitions"],
        )
    if args.warmup_decisions_root is not None:
        warmup_root = args.warmup_decisions_root.expanduser().resolve()
        source_paths = sorted(warmup_root.glob(args.decision_pattern))
        for source_path in source_paths:
            raw_path = raw_root / source_path.name.replace(".decisions.h5", ".raw.h5")
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            result = reconstruct_warmup_episode(
                raw_path,
                source_decision_path=source_path,
                output_root=output_root,
                runtime=runtime,
            )
            completed.append(result)
            LOG.info(
                "warmup outcome=%s decisions=%d transitions=%d",
                result["outcome"], result["decisions"], result["transitions"],
            )


if __name__ == "__main__":
    main()
