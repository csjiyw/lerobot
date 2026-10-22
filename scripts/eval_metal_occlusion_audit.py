#!/usr/bin/env python3
"""Four-way occlusion audit for metal-workpiece SmolVLA checkpoints.

Modes: normal | gray (both cameras zeroed) | state_zero (normalized state zeroed)
| both. Per sampler-group metrics reveal whether the policy still performs when
inputs are removed: gray ~= normal => vision ignored (state shortcut);
state_zero ~= normal => state ignored (vision-driven, desired).

Offline only; never contacts the robot.
"""
from __future__ import annotations

import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.relative_eef import chunk_to_relative, relative_chunk_to_absolute

MODES = ("normal", "gray", "state_zero", "both")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/eval")
    ap.add_argument("--episodes", default="all", help="'all' or comma list of episode indices")
    ap.add_argument("--max-samples-per-episode", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-groups-json", type=Path, default=None,
                    help="override episode->group map (for datasets without meta/env_groups.json)")
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = str(device)
    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}})
    policy.eval()
    chunk = int(cfg.chunk_size)
    groups_path = args.env_groups_json or (args.dataset_root / "meta/env_groups.json")
    env_groups = json.loads(Path(groups_path).read_text()) if Path(groups_path).is_file() else {}

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root,
                        delta_timestamps={"action": [i / meta.fps for i in range(chunk)]},
                        video_backend="pyav")
    if args.episodes.strip().lower() == "all":
        episodes = [int(e) for e in meta.episodes["episode_index"]]
    else:
        episodes = [int(e) for e in args.episodes.split(",")]

    rng = random.Random(20261014)
    rows = defaultdict(lambda: defaultdict(list))  # mode -> group -> per-sample list
    for ep in episodes:
        em = meta.episodes[ep]
        start, end = int(em["dataset_from_index"]), int(em["dataset_to_index"])
        n = end - start
        grid = list(range(start, end - chunk + 1, max(1, n // max(1, args.max_samples_per_episode))))
        group = env_groups.get(str(ep), "unknown")
        for gi, idx in enumerate(grid):
            item = ds[idx]
            base_obs = {k: item[k] for k in ("observation.state", "observation.images.camera_f",
                                             "observation.images.camera_r")}
            task = item.get("task", "")
            state_np = item["observation.state"].numpy().astype(np.float64)
            targets = item["action"].numpy().astype(np.float64)[:chunk]
            target_rel = chunk_to_relative(state_np, targets)
            for mode in MODES:
                obs = dict(base_obs)
                if mode in ("gray", "both"):
                    obs["observation.images.camera_f"] = torch.zeros_like(obs["observation.images.camera_f"])
                    obs["observation.images.camera_r"] = torch.zeros_like(obs["observation.images.camera_r"])
                obs["task"] = task
                torch.manual_seed(4242 + gi)
                with torch.inference_mode():
                    processed = pre(obs)
                    if mode in ("state_zero", "both") and "observation.state" in processed:
                        processed["observation.state"] = torch.zeros_like(processed["observation.state"])
                    pred = post(policy.predict_action_chunk(processed))
                pred_rel = pred.detach().cpu().numpy()[0, :chunk].astype(np.float64)
                d = pred_rel - target_rel
                pos = float(np.sqrt(np.mean(d[:, :3] ** 2)))
                rot = float(np.sqrt(np.mean(d[:, 3:6] ** 2)))
                grip = float(np.mean(np.abs(d[:, 6])))
                rows[mode][group].append((pos, rot, grip))
                rows[mode]["__all__"].append((pos, rot, grip))
                phase = ("early" if gi < len(grid) / 3 else "mid" if gi < 2 * len(grid) / 3 else "late")
                rows[mode][f"{group}#{phase}"].append((pos, rot, grip))

    def agg(mode, g):
        v = rows[mode].get(g)
        if not v:
            return None
        a = np.array(v)
        return {"n": len(v), "pos_rmse": float(a[:, 0].mean()),
                "rot_rmse": float(a[:, 1].mean()), "grip_mae": float(a[:, 2].mean())}

    groups = sorted(g for g in rows["normal"] if "#" not in g)
    report = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
              "modes": {m: {g: agg(m, g) for g in sorted(rows[m])} for m in MODES}}
    ratios = {}
    for g in groups:
        base = agg("normal", g)
        if not base or base["pos_rmse"] <= 0:
            continue
        ratios[g] = {"gray_over_normal": round(agg("gray", g)["pos_rmse"] / base["pos_rmse"], 3),
                     "state_zero_over_normal": round(agg("state_zero", g)["pos_rmse"] / base["pos_rmse"], 3)}
    report["dependence_ratios"] = ratios
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "occlusion_audit.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [f"# Occlusion audit {args.checkpoint.name}", "",
             "| group | n | normal pos | gray pos | state_zero pos | both pos | gray/norm | zero/norm |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        a = {m: agg(m, g) for m in MODES}
        r = ratios.get(g, {})
        lines.append(f"| {g} | {a['normal']['n']} | {a['normal']['pos_rmse']:.4f} | {a['gray']['pos_rmse']:.4f} "
                     f"| {a['state_zero']['pos_rmse']:.4f} | {a['both']['pos_rmse']:.4f} "
                     f"| {r.get('gray_over_normal','-')} | {r.get('state_zero_over_normal','-')} |")
    (args.out / "occlusion_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
