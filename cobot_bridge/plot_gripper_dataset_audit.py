"""Plot measured gripper state and the exact targets used by the abs-joint run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def plot(training_root: Path, output: Path, episode: int = 0) -> dict:
    view = json.loads((training_root / "meta/training_view_contract.json").read_text())
    repair = json.loads(Path(view["gripper_repair_report"]).read_text())
    original_root = Path(repair["contract"]["source"])
    original_dim = repair["contract"]["modified_columns"]["action"]
    train_dim = view["action_indices"].index(original_dim)
    state_dim = view["state_indices"].index(original_dim)
    fps = json.loads((training_root / "meta/info.json").read_text())["fps"]
    records, selected = [], None
    for record in repair["episode_report"]:
        ep = record["episode_index"]
        before = pq.read_table(original_root / f"data/chunk-000/episode_{ep:06d}.parquet",
                               columns=["observation.state", "action"])
        trained = pq.read_table(training_root / f"data/chunk-000/file-{ep:03d}.parquet",
                                columns=["timestamp", "observation.state", "action"])
        state = np.asarray(trained["observation.state"].to_pylist())[:, state_dim]
        new = np.asarray(trained["action"].to_pylist())[:, train_dim]
        old = np.asarray(before["action"].to_pylist())[:, original_dim]
        assert np.array_equal(state, np.asarray(before["observation.state"].to_pylist())[:, original_dim])
        mask = np.zeros(len(new), dtype=bool)
        for start, stop in record["corrected_ranges"]:
            mask[start:stop] = True
        assert int(mask.sum()) == record["corrected_frames"]
        assert np.all(new[mask] == 0) and np.array_equal(new[~mask], old[~mask])
        records.append({"episode": ep, "frames": len(new), "corrected_frames": int(mask.sum()),
                        "state_median_mm": float(np.median(state[mask]) * 1000),
                        "old_action_median_mm": float(np.median(old[mask]) * 1000),
                        "new_action_median_mm": float(np.median(new[mask]) * 1000)})
        if ep == episode:
            selected = (np.asarray(trained["timestamp"].to_pylist()), state * 1000,
                        old * 1000, new * 1000, record["corrected_ranges"])
    if selected is None:
        raise ValueError(f"episode {episode} not found")
    t, state, old, new, intervals = selected
    start, stop = intervals[0]
    med_state = float(np.median(state[start:stop]))
    med_old = float(np.median(old[start:stop]))
    med_new = float(np.median(new[start:stop]))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax = fig.add_subplot(grid[0, :])
    zoom = fig.add_subplot(grid[1, 0])
    population = fig.add_subplot(grid[1, 1])
    for target in (ax, zoom):
        target.plot(t, state, color="#273747", linewidth=2.4, label="Measured STATE (unchanged)")
        target.plot(t, old, color="#e59223", linewidth=1.8, linestyle="--", label="Original ACTION = state[t+3]")
        target.plot(t, new, color="#087cac", linewidth=2.3, label="Repaired ACTION (training target)")
        for a, b in intervals:
            target.axvspan(t[a], t[b - 1] + 1 / fps, color="#e2f1e9", alpha=.7, zorder=-1)
        target.set_ylim(-5, 87)
        target.set_ylabel("Gripper aperture / target (mm)")
        target.set_xlabel("Episode time (s)")
        target.grid(alpha=.2)
    ax.set_title(f"Episode {episode}: the object blocks the fingers, but the action must still command closure", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=1)
    middle = .5 * (t[start] + t[stop - 1])
    ax.annotate(f"Holding the object\nSTATE = {med_state:.1f} mm\nOld target = {med_old:.1f} mm\nNew target = {med_new:.1f} mm (fully closed)",
                xy=(middle, med_new), xytext=(middle, 26), ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": "#97b8a5"},
                arrowprops={"arrowstyle": "->", "color": "#087cac"})
    ax.text(.98, .02, "0 mm = fully closed target; ~80 mm = open", transform=ax.transAxes, ha="right", color="#58616b", fontsize=10)
    zoom.set_xlim(max(0, t[start] - 1), min(t[-1], t[start] + 2.2))
    zoom.set_title("Close-up: grasp starts, then the new target stays at zero", loc="left", fontsize=11)
    zoom.axhline(0, color="#087cac", alpha=.3, linewidth=1)
    ids = np.array([r["episode"] for r in records])
    population.scatter(ids, [r["state_median_mm"] for r in records], s=14, color="#273747", alpha=.7, label="Measured STATE")
    population.scatter(ids, [r["old_action_median_mm"] for r in records], s=15, color="#e59223", marker="x", label="Original ACTION")
    population.plot(ids, [r["new_action_median_mm"] for r in records], color="#087cac", linewidth=2.5, label="Repaired ACTION = 0")
    population.set_ylim(-5, 65)
    population.set_xlabel("Episode index")
    population.set_ylabel("Median during repaired hold intervals (mm)")
    population.set_title(f"All {len(records)} episodes: sustained-close target verified", loc="left", fontsize=11)
    population.grid(alpha=.2)
    population.legend(loc="center right", fontsize=9)
    total = sum(r["frames"] for r in records)
    changed = sum(r["corrected_frames"] for r in records)
    fig.suptitle(f"Right-gripper dataset audit | {len(records)} episodes | {changed:,} repaired frames / {total:,} total\n"
                 "Shows the saved data used for the 100k SmolVLA abs-joint run, not model predictions", fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"training_dataset": str(training_root), "episodes": len(records),
               "frames": total, "corrected_frames": changed,
               "all_repaired_hold_targets_are_zero": True, "all_measured_states_unchanged": True,
               "all_targets_outside_repaired_intervals_unchanged": True,
               "example_episode": episode, "example_first_hold_frames": [start, stop],
               "example_first_hold_seconds": [float(t[start]), float(t[stop - 1] + 1 / fps)],
               "example_hold_state_mm": med_state, "example_hold_old_target_mm": med_old,
               "example_hold_new_target_mm": med_new,
               "interpretation": "The saved labels now demand continued closure; actual grip strength and learned model output require rollout verification."}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/cobot_magic_5482_grippercmd_right_joint_v3")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "web_runs/gripper_dataset_audit/gripper_state_action.png")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(plot(args.training_root, args.output, args.episode), indent=2))
