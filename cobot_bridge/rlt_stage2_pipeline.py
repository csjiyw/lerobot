"""Detached post-episode RLT pipeline: train on GPU 1 and archive on CPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .rlt_stage2_data import (
    DEFAULT_ACTOR_EXPLORATION_STD,
)


def should_train_stage2(rollout_phase: str, _legacy_status: dict[str, Any] | None = None) -> bool:
    """Every labeled RLT actor episode requests an actor-critic update."""

    if rollout_phase != "actor":
        raise ValueError(f"invalid rollout_phase: {rollout_phase!r}")
    return True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage1-checkpoint", type=Path, required=True)
    value.add_argument("--replay-root", type=Path, required=True)
    value.add_argument("--decision-output", type=Path, required=True)
    value.add_argument("--replay-output", type=Path, required=True)
    value.add_argument("--stage2-output-dir", type=Path, required=True)
    value.add_argument("--trainer-device", default="cuda:1")
    value.add_argument(
        "--actor-exploration-std", type=float, default=DEFAULT_ACTOR_EXPLORATION_STD
    )
    value.add_argument("--rollout-phase", choices=("actor",), required=True)
    value.add_argument("--raw", type=Path, required=True)
    value.add_argument("--task", required=True)
    value.add_argument("--recap-checkpoint", type=Path, required=True)
    value.add_argument("--recap-device", default="cuda:1")
    value.add_argument("--actor-validation-cache", type=Path, required=True)
    value.add_argument("--il-anchor-weight", type=float, default=30.0)
    value.add_argument("--q-weight", type=float, default=0.01)
    value.add_argument("--bc-weight", type=float, default=5.0)
    value.add_argument(
        "--intervention-bc-weight", type=float, default=1.0,
        help="Extra BC multiplier on is_intervention=1 transitions (human corrections).",
    )
    value.add_argument(
        "--actor-output-semantics", choices=("residual", "direct"), default="residual",
        help="Actor output semantics: residual (delta + reference) or direct "
        "(independent full action). Must match the selected actor lineage.",
    )
    value.add_argument(
        "--ref-anchor-weight", type=float, default=0.0,
        help="Weight of the reference-anchor (actor -> VLA reference) loss term.",
    )
    value.add_argument(
        "--ref-anchor-mask", choices=("none", "outside_box"), default="none",
        help="Apply the reference anchor only on transitions whose physical EEF "
        "state is outside the gate box (requires --action-stats).",
    )
    value.add_argument(
        "--action-stats", type=Path, default=None,
        help="Dataset meta/stats.json with 7-dim 'action' mean/std (for the box mask).",
    )
    value.add_argument("--archive-root", type=Path, required=True)
    value.add_argument("--archive-repo-id", required=True)
    value.add_argument("--fps", type=int, default=30)
    value.add_argument("--status-file", type=Path, required=True)
    return value


def atomic_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_archive_command(args: argparse.Namespace) -> list[str]:
    """Archive the complete actor rollout; policy holds/corrections are data."""

    return [
        sys.executable,
        "-m",
        "cobot_bridge.convert_raw_to_lerobot",
        "--raw",
        str(args.raw),
        "--repo-id",
        args.archive_repo_id,
        "--root",
        str(args.archive_root),
        "--action-source",
        "executed_action",
        "--fps",
        str(args.fps),
        "--use-videos",
    ]


def build_train_command(
    args: argparse.Namespace, trainer_progress_file: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cobot_bridge.rlt_stage2_train",
        "--stage1-checkpoint",
        str(args.stage1_checkpoint),
        "--replay-root",
        str(args.replay_root),
        "--output-dir",
        str(args.stage2_output_dir),
        "--device",
        args.trainer_device,
        "--training-phase",
        "actor",
        "--actor-exploration-std",
        str(args.actor_exploration_std),
        "--critic-target-mode",
        "mc",
        "--actor-validation-cache",
        str(args.actor_validation_cache),
        "--il-anchor-weight",
        str(args.il_anchor_weight),
        "--q-weight",
        str(args.q_weight),
        "--bc-weight",
        str(args.bc_weight),
        "--intervention-bc-weight",
        str(args.intervention_bc_weight),
        "--actor-output-semantics",
        args.actor_output_semantics,
        "--ref-anchor-weight",
        str(args.ref_anchor_weight),
        "--ref-anchor-mask",
        args.ref_anchor_mask,
    ]
    if args.action_stats is not None:
        command.append("--action-stats")
        command.append(str(args.action_stats))
    command.extend(["--progress-file", str(trainer_progress_file)])
    return command


def main() -> None:
    args = parser().parse_args()
    started_ns = time.time_ns()
    trainer_progress_file = args.status_file.with_name(
        args.status_file.name.removesuffix(".json") + ".trainer.json"
    )
    common = {
        "started_at_ns": started_ns,
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "replay_root": str(args.replay_root.resolve()),
        "decision_output": str(args.decision_output.resolve()),
        "replay_output": str(args.replay_output.resolve()),
        "stage2_output_dir": str(args.stage2_output_dir.resolve()),
        "archive_root": str(args.archive_root.resolve()),
        "trainer_device": args.trainer_device,
        "trainer_progress_file": str(trainer_progress_file.resolve()),
        "rollout_phase": args.rollout_phase,
    }
    atomic_status(args.status_file, {**common, "state": "starting"})
    args.stage2_output_dir.mkdir(parents=True, exist_ok=True)
    archive_command = build_archive_command(args)
    train_command = build_train_command(args, trainer_progress_file)
    recap_trace = args.archive_root / "recap_value_trace.npz"
    recap_command = [
        sys.executable,
        "-m",
        "cobot_bridge.recap_online_value",
        "--stage1-checkpoint",
        str(args.stage1_checkpoint),
        "--recap-checkpoint",
        str(args.recap_checkpoint),
        "--raw",
        str(args.raw),
        "--task",
        args.task,
        "--output",
        str(recap_trace),
        "--device",
        args.recap_device,
    ]
    replay_command = [
        sys.executable,
        "-m",
        "cobot_bridge.build_recap_dense_replay",
        "--decision-root",
        str(args.decision_output.parent),
        "--raw-root",
        str(args.raw.parent),
        "--output-root",
        str(args.replay_output.parent),
        "--value-trace",
        str(recap_trace),
        "--pattern",
        args.decision_output.name,
    ]
    archive = subprocess.Popen(archive_command)
    should_train = should_train_stage2(args.rollout_phase)
    common["training_requested"] = should_train
    train = None
    train_returncode = archive_returncode = None
    try:
        if should_train:
            lock_path = args.stage2_output_dir / ".train.lock"
            with lock_path.open("a+") as lock:
                atomic_status(
                    args.status_file,
                    {**common, "state": "waiting_for_trainer_lock", "archive_pid": archive.pid},
                )
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                recap = subprocess.run(recap_command)
                if recap.returncode != 0:
                    raise RuntimeError(f"RECAP scoring failed with return code {recap.returncode}")
                replay = subprocess.run(replay_command)
                if replay.returncode != 0:
                    raise RuntimeError(f"RECAP replay build failed with return code {replay.returncode}")
                train = subprocess.Popen(train_command)
                atomic_status(
                    args.status_file,
                    {
                        **common,
                        "state": "running",
                        "archive_pid": archive.pid,
                        "trainer_pid": train.pid,
                    },
                )
                train_returncode = train.wait()
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        archive_returncode = archive.wait()
    except BaseException:
        if train is not None and train.poll() is None:
            train.terminate()
        if archive.poll() is None:
            archive.terminate()
        atomic_status(
            args.status_file,
            {
                **common,
                "state": "failed",
                "finished_at_ns": time.time_ns(),
                "trainer_returncode": train_returncode,
                "archive_returncode": archive_returncode,
            },
        )
        raise
    trainer_ok = train_returncode == 0 if should_train else train_returncode is None
    state = "completed" if trainer_ok and archive_returncode == 0 else "failed"
    atomic_status(
        args.status_file,
        {
            **common,
            "state": state,
            "finished_at_ns": time.time_ns(),
            "trainer_returncode": train_returncode,
            "archive_returncode": archive_returncode,
        },
    )
    if state != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
