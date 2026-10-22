"""Infer missing close targets from calibrated contact holds, keeping measured state intact.

This does not recover original commands. It corrects sustained grasp plateaus
where measured aperture has been used as the action target. Videos are shared
with the source through a symlink; parquet and metadata are independent copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def infer_contact_command(
    feedback_target: np.ndarray,
    *,
    contact_upper: float,
    close_target: float = 0.0,
    hold_frames: int = 15,
    contact_lower: float = 0.02,
    plateau_span: float = 0.0015,
    entry_margin: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct only long contact holds reached by closing from a wider aperture.

    Initial ambiguous holds, short transits, partial openings above the calibrated
    contact band, and release motion are left as their original state-derived targets.
    No additional time shift is introduced.
    """
    g = np.asarray(feedback_target, dtype=np.float32)
    if g.ndim != 1 or not np.isfinite(g).all():
        raise ValueError("gripper targets must be a finite one-dimensional array")
    fixed = g.copy()
    corrected = np.zeros(len(g), dtype=bool)
    for start, end in ranges((g >= contact_lower) & (g <= contact_upper)):
        if start == 0 or end - start < hold_frames:
            continue
        # A wider preceding position and downward entry distinguish a grasp
        # from an opening ramp that pauses at a similar measured width.
        previous = g[max(0, start - 4 * hold_frames):start]
        if g[start - 1] <= g[start] or previous.max() < contact_upper + entry_margin:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(g[start:end], hold_frames)
        stable = np.flatnonzero(np.ptp(windows, axis=1) <= plateau_span)
        if not len(stable):
            continue
        baseline = float(np.median(windows[stable[0]]))
        # Stop at the first visible opening rise after the stable hold.
        after = start + int(stable[0]) + hold_frames
        released = np.flatnonzero(g[after:end] > baseline + plateau_span)
        stop = after + int(released[0]) if len(released) else end
        corrected[start:stop] = True
    fixed[corrected] = close_target
    return fixed, corrected


def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as raw:
        state = np.asarray(raw["frames/puppet_position"], dtype=np.float32)[:, -1]
        command = np.asarray(raw["frames/puppet_command"], dtype=np.float32)[:, -1]
        valid = np.asarray(raw["frames/puppet_command_valid"], dtype=bool)
    valid &= np.isfinite(state) & np.isfinite(command)
    age = np.zeros(len(state), dtype=np.int64)
    for i in range(1, len(state)):
        if valid[i] and valid[i - 1] and abs(command[i] - command[i - 1]) < 0.002:
            age[i] = age[i - 1] + 1
    return state, command, valid, age


def calibrate(raw_collection: Path) -> tuple[dict, dict]:
    files = sorted(raw_collection.glob("*.h5"))
    if len(files) < 8:
        raise ValueError("at least 8 independent local recordings are needed")
    split = int(len(files) * 0.75)
    gripping, close_commands = [], []
    for path in files[:split]:
        state, command, valid, age = load_raw(path)
        close = valid & (age >= 15) & (np.abs(command) <= 0.002)
        contact = close & (state >= 0.02)
        gripping.extend(state[contact].tolist())
        close_commands.extend(command[close].tolist())
    if len(gripping) < 100:
        raise ValueError("too few measured contact holds in calibration data")
    contact_q = np.quantile(gripping, [0.001, 0.01, 0.5, 0.99, 0.999])
    config = {
        "contact_upper": float(contact_q[-1] + 0.001),
        "close_target": float(max(0.0, np.median(close_commands))),
        "hold_frames": 15,
        "contact_lower": 0.02,
        "plateau_span": 0.0015,
        "entry_margin": 0.003,
    }
    sums = dict(before=0.0, after=0.0, frames=0, corrected=0, true_close=0,
                steady_before=0.0, steady_after=0.0, steady_frames=0,
                steady_corrected=0, steady_true_close=0)
    for path in files[split:]:
        state, command, valid, age = load_raw(path)
        target = state[np.minimum(np.arange(len(state)) + 3, len(state) - 1)]
        fixed, mask = infer_contact_command(target, **config)
        sums["before"] += float(np.abs(target[valid] - command[valid]).sum())
        sums["after"] += float(np.abs(fixed[valid] - command[valid]).sum())
        sums["frames"] += int(valid.sum())
        sums["corrected"] += int((mask & valid).sum())
        sums["true_close"] += int((mask & valid & (np.abs(command) <= 0.002)).sum())
        steady = valid & (age >= 15)
        sums["steady_before"] += float(np.abs(target[steady] - command[steady]).sum())
        sums["steady_after"] += float(np.abs(fixed[steady] - command[steady]).sum())
        sums["steady_frames"] += int(steady.sum())
        sums["steady_corrected"] += int((mask & steady).sum())
        sums["steady_true_close"] += int((mask & steady & (np.abs(command) <= 0.002)).sum())
    report = {
        "raw_collection": str(raw_collection.resolve()),
        "calibration_episodes": [str(p.resolve()) for p in files[:split]],
        "held_out_episodes": [str(p.resolve()) for p in files[split:]],
        "contact_state_quantiles": contact_q.tolist(),
        "close_command_median_unclipped": float(np.median(close_commands)),
        "config": config,
        "held_out": {
            "frames": sums["frames"],
            "mae_state_t3": sums["before"] / sums["frames"],
            "mae_repaired": sums["after"] / sums["frames"],
            "steady_frames": sums["steady_frames"],
            "steady_mae_state_t3": sums["steady_before"] / sums["steady_frames"],
            "steady_mae_repaired": sums["steady_after"] / sums["steady_frames"],
            "corrected_frames": sums["corrected"],
            "actual_close_fraction_of_corrected": sums["true_close"] / max(1, sums["corrected"]),
            "steady_corrected_frames": sums["steady_corrected"],
            "steady_actual_close_fraction_of_corrected": sums["steady_true_close"] / max(1, sums["steady_corrected"]),
        },
    }
    return config, report


def statistics(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    result = {"count": [len(v)], "mean": v.mean(axis=0).tolist(),
              "std": v.std(axis=0).tolist(), "min": v.min(axis=0).tolist(),
              "max": v.max(axis=0).tolist()}
    for q in (1, 10, 50, 90, 99):
        result[f"q{q:02d}"] = np.quantile(v, q / 100, axis=0).tolist()
    return result


def dimension_names(info: dict, key: str) -> list[str]:
    names = info["features"][key]["names"]
    return names["motors"] if isinstance(names, dict) else names


def repair(source: Path, output: Path, config: dict, calibration: dict) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or output.exists():
        raise ValueError("output must be a new independent directory outside source")
    info = json.loads((source / "meta/info.json").read_text())
    indices = {key: dimension_names(info, key).index("right_gripper")
               for key in ("action", "action_joint")}
    indices["action_eef"] = dimension_names(info, "action_eef").index("gripper")
    paths = sorted((source / "data").rglob("*.parquet"))
    if len(paths) != info["total_episodes"]:
        raise ValueError("this converter requires one parquet file per episode")
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    (output / "videos").symlink_to(source / "videos", target_is_directory=True)
    ep_stats = {row["episode_index"]: row for row in
                (json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines())}
    chunks = {key: [] for key in indices}
    records = []
    for path in paths:
        table = pq.read_table(path)
        episode = int(table["episode_index"][0].as_py())
        original = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        g = original[:, indices["action"]]
        fixed, mask = infer_contact_command(g, **config)
        for key, index in indices.items():
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if not np.array_equal(values[:, index], g):
                raise ValueError(f"{path.name}: inconsistent gripper mirrors in {key}")
            values[:, index] = fixed
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field,
                                     pa.array(values.tolist(), type=field.type))
            chunks[key].append(values)
            ep_stats[episode]["stats"][key] = statistics(values)
        dest = output / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest, compression="snappy")
        records.append({"episode_index": episode, "frames": len(g),
                        "corrected_frames": int(mask.sum()),
                        "corrected_ranges": [[int(a), int(b)] for a, b in ranges(mask)],
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "repaired_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    global_stats = json.loads((source / "meta/stats.json").read_text())
    for key, arrays in chunks.items():
        global_stats[key] = statistics(np.concatenate(arrays))
    (output / "meta/stats.json").write_text(json.dumps(global_stats, indent=2) + "\n")
    (output / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(ep_stats[i]) + "\n" for i in sorted(ep_stats)))
    contract = {
        "type": "inferred_contact_close_command_v1", "source": str(source),
        "raw_calibration": calibration, "modified_columns": indices,
        "original_actions": "state[t+3]; actual historical commands were not recorded",
        "new_actions": "arm and left gripper unchanged; right contact holds use inferred close target",
        "time_alignment": "existing t+3 action timeline retained, no additional shift",
        "observations": "all measured states and all other parquet columns unchanged",
        "videos": "symlink to original videos, which must remain available",
        "limitations": "not exact command recovery; no inference for ambiguous initial holds; transition timing remains approximate",
    }
    info["representations"]["joint"]["note"] = contract["new_actions"] + "; arm uses state[t+3]"
    info["representations"]["eef"]["gripper_semantics"] = "inferred contact-close command; transitions remain state-derived"
    info["gripper_action_repair"] = "meta/gripper_action_repair.json"
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    dataset_contract = json.loads((output / "meta/dataset_contract.json").read_text())
    dataset_contract["dataset"] = str(output)
    dataset_contract["columns"]["action"] = info["representations"]["joint"]["note"]
    dataset_contract["columns"]["action_eef"] = "original absolute EEF arm pose + repaired right gripper target"
    dataset_contract["gripper_action_repair"] = contract
    (output / "meta/dataset_contract.json").write_text(json.dumps(dataset_contract, indent=2) + "\n")
    # This file is explicitly unused. Its arm statistics remain valid because
    # only absolute gripper targets changed; repair its gripper entry as well.
    relative_path = output / "meta/relative_eef_action_stats.json.unused"
    if relative_path.exists():
        relative = json.loads(relative_path.read_text())
        relative["contract"]["gripper"] = "inferred_contact_close_target_not_original_executed_command"
        for key in list(relative["action"]):
            if key != "count" and key in global_stats["action_eef"]:
                relative["action"][key][-1] = global_stats["action_eef"][key][-1]
        relative_path.write_text(json.dumps(relative, indent=2) + "\n")
    report = {"contract": contract, "episodes": len(records),
              "frames": sum(r["frames"] for r in records),
              "episodes_corrected": sum(r["corrected_frames"] > 0 for r in records),
              "corrected_frames": sum(r["corrected_frames"] for r in records),
              "uncorrected_episode_ids": [r["episode_index"] for r in records if not r["corrected_frames"]],
              "before_gripper_stats": json.loads((source / "meta/stats.json").read_text())["action_eef"],
              "after_gripper_stats": global_stats["action_eef"], "episode_report": records}
    (output / "meta/gripper_action_repair.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README_GRIPPER_REPAIR.md").write_text(
        "# 夹爪 action 修正版\n\n"
        f"源数据：`{source}`\n\n"
        f"{len(records)} 条 episode、{report['frames']} 帧，修正 {report['corrected_frames']} 帧右夹爪目标。\n\n"
        "本机金属工件 IL 记录显示，接触工件后夹爪反馈约为 0.055，但闭合控制目标接近 0。"
        "本版本只修正从较大开口闭合进入的持续夹持段，保留实测 state、机械臂动作、左夹爪、"
        "图像和原有 state[t+3] 时间关系。未识别为夹持段的开合过渡沿用原数据值。\n\n"
        "修正的目标是根据本机记录推断的持续闭合目标，不是找回下载数据丢失的原始 action。"
        "初始状态不明、短暂过渡和部分张开不做推断；夹持段边界的指令时刻仍是近似。\n\n"
        "`action`、`action_joint`、`action_eef` 同步修正，逐 episode 和全局 action 统计重新计算。"
        "已有 checkpoint 的 processor/统计和权重没有修改；使用新数据训练时应明确选用新统计。\n\n"
        "视频使用源数据 videos 的符号链接；迁移到其他机器时需复制链接目标。"
        "完整参数、独立留出验证、各集修正区间及 parquet 校验码见 `meta/gripper_action_repair.json`。\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new dataset directory; omitted = calibration only")
    parser.add_argument("--verify-only", action="store_true", help="check an existing repaired output")
    args = parser.parse_args()
    if args.verify_only:
        if args.output is None:
            parser.error("--verify-only requires --output")
        print(json.dumps(verify_repair(args.source, args.output), indent=2))
        return
    config, calibration = calibrate(args.raw_collection)
    print(json.dumps({"config": config, "held_out": calibration["held_out"]}, indent=2), flush=True)
    if args.output:
        report = repair(args.source, args.output, config, calibration)
        verification = verify_repair(args.source, args.output)
        print(json.dumps({k: report[k] for k in
                          ("episodes", "frames", "episodes_corrected", "corrected_frames", "uncorrected_episode_ids")}, indent=2))
        print(json.dumps(verification, indent=2))


def verify_repair(source: Path, output: Path) -> dict:
    """Independently read every saved parquet and check the permitted changes."""
    source, output = source.resolve(), output.resolve()
    report_path = output / "meta/gripper_action_repair.json"
    report = json.loads(report_path.read_text())
    config = report["contract"]["raw_calibration"]["config"]
    indices = report["contract"]["modified_columns"]
    episode_reports = {r["episode_index"]: r for r in report["episode_report"]}
    ep_stats = {r["episode_index"]: r["stats"] for r in
                (json.loads(line) for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines())}
    global_stats = json.loads((output / "meta/stats.json").read_text())
    all_values = {key: [] for key in indices}
    frames = changed = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        before = pq.read_table(path)
        after = pq.read_table(output / path.relative_to(source))
        assert before.schema.equals(after.schema, check_metadata=True), path
        episode = int(before["episode_index"][0].as_py())
        record = episode_reports[episode]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["source_sha256"], path
        assert hashlib.sha256((output / path.relative_to(source)).read_bytes()).hexdigest() == record["repaired_sha256"], path
        g = np.asarray(before["action"].to_pylist(), dtype=np.float32)[:, indices["action"]]
        expected, mask = infer_contact_command(g, **config)
        frames += len(g)
        changed += int(mask.sum())
        for key in before.column_names:
            if key not in indices:
                assert before[key].equals(after[key]), (path, key)
                continue
            old = np.asarray(before[key].to_pylist(), dtype=np.float32)
            new = np.asarray(after[key].to_pylist(), dtype=np.float32)
            idx = indices[key]
            assert np.array_equal(np.delete(old, idx, axis=1), np.delete(new, idx, axis=1)), (path, key)
            assert np.array_equal(new[:, idx], expected), (path, key)
            for name, value in statistics(new).items():
                assert np.allclose(value, ep_stats[episode][key][name], atol=1e-10, rtol=1e-10), (path, key, name)
            all_values[key].append(new)
    for key, arrays in all_values.items():
        for name, value in statistics(np.concatenate(arrays)).items():
            assert np.allclose(value, global_stats[key][name], atol=1e-10, rtol=1e-10), (key, name)
    source_stats = json.loads((source / "meta/stats.json").read_text())
    for key in source_stats.keys() - indices.keys():
        assert source_stats[key] == global_stats[key], key
    assert frames == report["frames"] and changed == report["corrected_frames"]
    assert (output / "videos").resolve() == (source / "videos").resolve()
    verification = {
        "all_episodes_checked": len(episode_reports), "all_frames_checked": frames,
        "corrected_frames": changed, "source_parquet_sha256_unchanged": True,
        "source_and_output_schema_identical": True, "all_observations_and_other_columns_identical": True,
        "arm_targets_and_left_gripper_identical": True, "right_gripper_action_mirrors_identical": True,
        "global_and_episode_action_stats_match_saved_data": True, "state_stats_unchanged": True,
        "videos_shared_with_source": True,
    }
    report["verification"] = verification
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(source, output, report)
    return verification


def plot_comparison(source: Path, output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration = report["contract"]["raw_calibration"]
    local_path = Path(calibration["held_out_episodes"][0])
    measured, command, valid, _ = load_raw(local_path)
    old_target = measured[np.minimum(np.arange(len(measured)) + 3, len(measured) - 1)]
    inferred, _ = infer_contact_command(old_target, **calibration["config"])
    before = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    after = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    t = np.arange(len(measured)) / 30
    axes[0].plot(t, measured * 1000, label="Measured aperture", color="gray")
    axes[0].plot(t, np.where(valid, command, np.nan) * 1000, label="Actual recorded command", color="tab:blue")
    axes[0].plot(t, inferred * 1000, label="Inferred command", color="tab:red", linestyle="--")
    axes[0].set_title("Local held-out metal-workpiece recording: measured width is not the close command")
    idx = report["contract"]["modified_columns"]["action"]
    original = np.asarray(before["action"].to_pylist())[:, idx]
    repaired = np.asarray(after["action"].to_pylist())[:, idx]
    t = np.arange(len(original)) / 30
    axes[1].plot(t, original * 1000, label="Original action = state[t+3]", color="gray")
    axes[1].plot(t, repaired * 1000, label="Repaired right-gripper action", color="tab:red")
    axes[1].set_title("Downloaded dataset, episode 0: retain close command throughout contact hold")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Aperture / target (mm)")
        ax.grid(alpha=.25)
        ax.legend(loc="upper right")
    fig.savefig(output / "gripper_action_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
