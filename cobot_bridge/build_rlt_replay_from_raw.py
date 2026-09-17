"""Build RLT decisions + replay bundles directly from raw h5 archives.

For episodes that were collected but never went through the online RLT
runtime (so no source decision files exist), this tool re-derives the
observation-dependent features (z_rl, normalized state, VLA reference
actions) with the SAME Stage-1 runtime (VLA + encoder) as the existing
replay root, and takes executed actions from the authoritative 30 Hz raw
record. It never opens a bridge and cannot publish robot commands.

Data contract (verified byte-exact against the do02 refhumanilgt root,
max diff 0.0000 on actions, ~0.007 on state):
  state   = normalize(puppet_position[7] @ start)         VLA state stats
  actions = normalize(executed_action[start:start+10])    VLA action stats
  ref     = VLA re-inference @ start, contiguous 10-tick, model-normalized
  z_rl    = Stage-1 encoder over VLA prefix (960-dim, rlinf_prefix_v1)
  rewards = sparse terminal: 1.0 (success) / 0.0 (failure) on last transition
Decisions are placed every 10 raw frames (contiguous_30hz_v1 timebase).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .rlt_policy_server import RLTPolicyRuntime, _decode_jpeg
from .rlt_stage2_data import (
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
)

LOG = logging.getLogger("cobot_bridge.build_rlt_replay_from_raw")

# Decision-grid stride is derived per run from the stage1 checkpoint
# (dimensions.action_steps == n_action_steps_rl; the do02-era contract was a
# hardcoded 10-tick window, which broke against the chunk30 stage1).


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help=(
            "Contiguous Stage-2 action/replay horizon. Must be no larger than "
            "the frozen Stage-1 reference horizon. For ref30/actor10 use 10."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pattern", default="*.raw.h5")
    parser.add_argument("--limit", type=int)
    # Optional explicit stem list (without .raw.h5); overrides --pattern.
    parser.add_argument("--names", default="")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _outcome(raw: h5py.File, path: Path) -> str:
    value = str(raw.attrs.get("outcome", "")).lower()
    if value in ("success", "failure"):
        return value
    sidecar = path.with_name(path.name + ".result")
    if sidecar.is_file():
        value = sidecar.read_text().strip().lower()
    if value not in ("success", "failure"):
        raise ValueError(f"{path.name}: missing success/failure label")
    return value


def _jpeg(dataset: h5py.Dataset, index: int) -> np.ndarray:
    return _decode_jpeg(bytes(np.asarray(dataset[index], dtype=np.uint8)))


def build_one(
    raw_path: Path,
    output_root: Path,
    runtime: RLTPolicyRuntime,
) -> dict[str, Any]:
    stem = raw_path.name.removesuffix(".raw.h5")
    decision_path = output_root / f"{stem}.decisions.h5"
    replay_path = output_root / f"{stem}.npz"
    if decision_path.exists() or replay_path.exists():
        return {"state": "SKIP", "episode": stem}

    dimensions = RLTDimensions(**runtime.dimensions)
    action_steps = int(dimensions.action_steps)
    writer: RLTDecisionWriter | None = None
    started = time.time()
    try:
        with h5py.File(raw_path, "r") as raw:
            if not bool(raw.attrs.get("completed", False)):
                raise ValueError(f"{raw_path.name}: raw archive is incomplete")
            cameras = tuple(json.loads(str(raw.attrs["cameras"])))
            if cameras != runtime.cameras:
                raise ValueError(
                    f"camera mismatch: raw={cameras} runtime={runtime.cameras}"
                )
            task = str(raw.attrs["task"])
            outcome = _outcome(raw, raw_path)
            side = str(raw.attrs.get("side", "right"))

            n_frames = int(raw.attrs.get("num_frames", -1))
            if n_frames <= 0:
                n_frames = int(raw["frames/sequence"].shape[0])
            starts = np.arange(0, max(0, n_frames - action_steps), action_steps,
                               dtype=np.int64)

            puppet = raw["frames/puppet_position"][:].astype(np.float32)
            executed = raw["frames/executed_action"][:].astype(np.float32)
            executed_valid = raw["frames/executed_action_valid"][:].astype(bool)
            human = raw["frames/is_human_intervention"][:].astype(bool)
            sequence = raw["frames/sequence"][:].astype(np.int64)
            timestamps = raw["frames/pc_timestamp_ns"][:].astype(np.int64)
            image_datasets = {name: raw[f"images/{name}/jpeg"] for name in cameras}

            if executed.shape[0] != n_frames or puppet.shape[0] != n_frames:
                raise ValueError(f"{raw_path.name}: frame length mismatch")

            any_human = bool(human.any())
            workflow_mode = 2 if any_human else 1
            # Mode 1 replay cannot contain intervention transitions
            # (build_replay_bundle enforces this); episodes with human
            # windows are written as Mode 2, which allows them.
            writer = RLTDecisionWriter(
                decision_path,
                dimensions=dimensions,
                stage1_checkpoint=str(runtime.stage1_path),
                task=task,
                workflow_mode=workflow_mode,
                side=side,
                collection_phase="actor",
            )
            writer.file.attrs.update(
                {
                    "built_from_raw": "10tick_contiguous_30hz_v1",
                    "behavior_chunk_source": "raw_executed_contiguous_30hz",
                    "raw_archive": str(raw_path),
                }
            )
            runtime.begin_episode(workflow_mode=workflow_mode, allow_actor=False,
                                  actor_exploration_std=0.0)
            written = 0
            try:
                for number, start in enumerate(starts, 1):
                    window = slice(int(start), int(start) + action_steps)
                    if not executed_valid[window].all():
                        continue
                    inference = runtime.predict(
                        state=np.asarray(puppet[int(start)], dtype=np.float32),
                        images={
                            name: _jpeg(dataset, int(start))
                            for name, dataset in image_datasets.items()
                        },
                        task=task,
                        robot_type="cobot_magic_piper",
                    )
                    behavior = runtime.normalize_actions(
                        np.asarray(executed[window], dtype=np.float32)
                    )
                    is_human_window = bool(human[window].any())
                    writer.append(
                        z_rl=inference["z_rl"],
                        state=inference["state"],
                        reference_actions=inference["reference_actions"],
                        policy_actions=behavior,
                        executed_actions=behavior,
                        action_source="human" if is_human_window else "actor",
                        is_intervention=is_human_window,
                        bridge_sequence=int(sequence[int(start)]),
                        time_ns=int(timestamps[int(start)]),
                    )
                    written += 1
                    if number == 1 or number % 25 == 0:
                        LOG.info("%s decision=%d/%d", stem, number, len(starts))
            finally:
                # begin_episode raises if an episode is still active, so a
                # mid-episode failure must always release the runtime state.
                runtime.end_episode()
            if writer.count < 2:
                raise ValueError(
                    f"{raw_path.name}: fewer than two valid ten-tick windows"
                )
            writer.close(outcome=outcome)
            writer = None
            metadata = build_replay_bundle(decision_path, replay_path)
            return {
                "state": "DONE",
                "episode": stem,
                "outcome": outcome,
                "workflow_mode": workflow_mode,
                "decisions_written": written,
                "transitions": int(metadata["transition_count"]),
                "elapsed_seconds": round(time.time() - started, 1),
            }
    except BaseException:
        if writer is not None:
            writer.abort()
        decision_path.unlink(missing_ok=True)
        replay_path.unlink(missing_ok=True)
        raise


def main() -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parser().parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.names:
        stems = [name.strip() for name in args.names.split(",") if name.strip()]
        paths = sorted(raw_root / f"{name}.raw.h5" for name in stems)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"raw archives not found: {missing}")
    else:
        paths = sorted(raw_root.glob(args.pattern))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no raw archives matched in {raw_root}")

    stage1 = args.stage1_checkpoint.expanduser().resolve()
    runtime = RLTPolicyRuntime(
        stage1_checkpoint=stage1,
        stage2_checkpoint=stage1.parent / ".no_actor.pt",
        device=args.device,
        cameras=("front", "right"),
        action_steps=args.action_steps,
    )

    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        try:
            LOG.info("episode=%d/%d start %s", index, len(paths), path.name)
            row = build_one(path, output_root, runtime)
            results.append(row)
            LOG.info("episode=%d/%d %s %s", index, len(paths), path.name, row.get("state"))
        except Exception as error:
            LOG.exception("episode=%d/%d FAIL %s", index, len(paths), path.name)
            failed.append({"raw": str(path), "error": str(error)})
            if args.fail_fast:
                raise
    summary = {
        "stage1_checkpoint": str(stage1),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "total": len(paths),
        "done": sum(item["state"] == "DONE" for item in results),
        "skip": sum(item["state"] == "SKIP" for item in results),
        "failed": failed,
        "transitions": sum(int(item.get("transitions", 0)) for item in results),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "results": results,
    }
    (output_root / "build_from_raw_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "results"},
        indent=2,
    ))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
