"""Copy a legacy replay episode into the explicit RLT raw-data schema.

The source file is never modified. For this replay-only migration, the sampled
replay command is both the reference action and policy action. The arbiter's
sampled puppet command is the executed action, and Teach master state is the
human action.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from .convert_raw_to_lerobot import derive_control_labels
from .record_raw import (
    CONTROL_HUMAN,
    CONTROL_POLICY,
    CONTROL_SOURCE_NAMES,
    FORMAT,
    OUTCOME_NAMES,
    OUTCOME_UNKNOWN,
    PHASE_HUMAN,
    PHASE_LOOP_TRANSITION,
    PHASE_NAMES,
    PHASE_POLICY,
    PHASE_TERMINAL,
    SUPPORTED_FORMATS,
    default_phase,
)


def _range(value: str) -> tuple[int, int]:
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("range must be START:STOP (STOP exclusive)") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("range requires 0 <= START < STOP")
    return start, stop


def _replace(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=values, maxshape=(None, *values.shape[1:]), chunks=True)


def migrate(
    source: Path,
    output: Path,
    *,
    loop_transitions: list[tuple[int, int]],
    episode_id: int,
    policy_id: str,
    outcome: str,
) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("source and output must differ; migration is intentionally non-destructive")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    try:
        with h5py.File(output, "r+") as raw:
            old_format = str(raw.attrs.get("format", ""))
            if old_format not in SUPPORTED_FORMATS:
                raise ValueError(f"unsupported raw format: {old_format!r}")
            frames = raw["frames"]
            count = len(frames["pc_timestamp_ns"])
            if count == 0:
                raise ValueError("raw episode is empty")

            teach = np.asarray(frames["teach_active"][:], dtype=np.uint8)
            modes = np.asarray(frames["effective_mode"][:], dtype=np.int64)
            labels = derive_control_labels(teach, modes)
            source_ids = np.asarray(labels["control_source"], dtype=np.int8)

            command = np.asarray(frames["puppet_command"][:], dtype=np.float32)
            command_valid = np.asarray(frames["puppet_command_valid"][:], dtype=bool)
            command_clean = np.where(command_valid[:, None], command, 0.0).astype(np.float32)
            master = np.asarray(frames["master_position"][:], dtype=np.float32)

            policy_valid = (source_ids == CONTROL_POLICY) & command_valid
            human_valid = source_ids == CONTROL_HUMAN
            if outcome == "unknown" and np.any(human_valid):
                outcome = "success"
            reference_action = np.where(policy_valid[:, None], command_clean, 0.0).astype(np.float32)
            policy_action = reference_action.copy()
            human_action = np.where(human_valid[:, None], master, 0.0).astype(np.float32)

            phase = np.asarray([default_phase(int(value)) for value in source_ids], dtype=np.int8)
            for start, stop in loop_transitions:
                if stop > count:
                    raise ValueError(f"loop transition {start}:{stop} exceeds {count} frames")
                if not np.all(source_ids[start:stop] == CONTROL_POLICY):
                    raise ValueError(f"loop transition {start}:{stop} is not entirely policy-owned")
                phase[start:stop] = PHASE_LOOP_TRANSITION
            phase[-1] = PHASE_TERMINAL

            cycle_id = np.full(count, -1, dtype=np.int32)
            cycle = 0
            in_transition = False
            for index, value in enumerate(phase):
                if value == PHASE_LOOP_TRANSITION:
                    if not in_transition:
                        cycle += 1
                    in_transition = True
                    cycle_id[index] = cycle
                elif value in (PHASE_POLICY, PHASE_HUMAN):
                    in_transition = False
                    cycle_id[index] = cycle
                else:
                    in_transition = False

            learning_mask = np.isin(phase, (PHASE_POLICY, PHASE_HUMAN)).astype(np.uint8)
            done = np.zeros(count, dtype=np.uint8)
            done[-1] = 1
            success = np.full(count, OUTCOME_UNKNOWN, dtype=np.int8)
            success[-1] = OUTCOME_NAMES[outcome]
            reward = np.zeros(count, dtype=np.float32)
            reward_valid = np.zeros(count, dtype=np.uint8)
            if outcome in ("success", "failure"):
                reward[-1] = float(outcome == "success")
                reward_valid[-1] = 1

            replacements = {
                **labels,
                "episode_id": np.full(count, episode_id, dtype=np.int64),
                "cycle_id": cycle_id,
                "phase": phase,
                "learning_mask": learning_mask,
                "reference_action": reference_action,
                "reference_action_valid": policy_valid.astype(np.uint8),
                "policy_action": policy_action,
                "policy_action_valid": policy_valid.astype(np.uint8),
                "human_action": human_action,
                "human_action_valid": human_valid.astype(np.uint8),
                "executed_action": command_clean,
                "executed_action_valid": command_valid.astype(np.uint8),
                "reward": reward,
                "reward_valid": reward_valid,
                "done": done,
                "success": success,
            }
            for name, values in replacements.items():
                _replace(frames, name, np.asarray(values))

            raw.attrs["migrated_from_format"] = old_format
            raw.attrs["format"] = FORMAT
            raw.attrs["episode_id"] = int(episode_id)
            raw.attrs["policy_id"] = policy_id
            raw.attrs["outcome"] = outcome
            raw.attrs["outcome_id"] = OUTCOME_NAMES[outcome]
            raw.attrs["control_source_names"] = json.dumps(CONTROL_SOURCE_NAMES, sort_keys=True)
            raw.attrs["phase_names"] = json.dumps(PHASE_NAMES, sort_keys=True)
            raw.attrs["outcome_names"] = json.dumps(OUTCOME_NAMES, sort_keys=True)
            raw.attrs["reference_action_provenance"] = "legacy_replay_sampled_puppet_command"
            raw.attrs["policy_action_provenance"] = "legacy_replay_sampled_puppet_command"
            raw.attrs["executed_action_provenance"] = "arbiter_puppet_command_sampled_at_dataset_fps"
            raw.attrs["human_action_provenance"] = "master_position_during_effective_teach_mode"
            raw.flush()
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--loop-transition",
        type=_range,
        action="append",
        default=[],
        help="raw frame range START:STOP, STOP exclusive; may be repeated",
    )
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--policy-id", default="replay-baseline")
    parser.add_argument(
        "--outcome",
        choices=tuple(OUTCOME_NAMES),
        default="unknown",
        help="unknown becomes success when the episode contains Human/Teach intervention",
    )
    args = parser.parse_args()
    migrate(
        args.source,
        args.output,
        loop_transitions=args.loop_transition,
        episode_id=args.episode_id,
        policy_id=args.policy_id,
        outcome=args.outcome,
    )


if __name__ == "__main__":
    main()
