"""T1/T2 offline evaluation for the RLT-Latent route.

T1 (direction correction on recorded errors): for every intervention row in
the replay buffer, compare three policies' decoded 10-step executed chunks
(in absolute EEF space) against the human-corrected executed action:
  A) base VLA (r_lat = 0)
  B) old iv50 action-residual pipeline (compose_actor_action, no RTC)
  C) latent actor (trained r_lat through the frozen VLA)
Metrics: mean direction angle (deg) vs the human correction, mean endpoint
distance (mm), and the fraction of rows where the latent route is closer
than each baseline.  No robot is involved.

T2 (inter-chunk seam / fluctuation): replay a decision chain on selected
episodes (offline open-loop, state from the raw archive at the executed
frame).  For every consecutive pair of decisions, the seam is the mm
difference between chunk_i's tail plan (steps 10..29) and chunk_{i+1}'s head
plan (steps 0..9), both converted to absolute EEF with the shared anchor
convention.  Four arms:
  base_vla        : r_lat = 0, no RTC
  old_iv50        : action residual (old checkpoint), no RTC
  latent_no_rtc   : latent actor, no RTC
  latent_rtc      : latent actor + guided RTC (prev chunk leftover fed in)
Outputs JSON per subcommand plus a small printed table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rlt_latent import LatentRuntime
from .rlt_stage2_data import compose_actor_action
from .rlt_stage2_train_latent import RawReader, RowFrameMap, _noise_seed_for_row
from .rlt_stage2_train import discover_replay_bundles, load_replay_dataset

BOX_CENTER = np.asarray([0.35, 0.37, 0.095], dtype=np.float64)  # x=(0.31+0.39)/2 etc.
EVAL_SEED = 20260901


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _abs_chunk(runtime: LatentRuntime, anchor_eef: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """relative normalized chunk (C,7) -> absolute EEF (C,7) meters."""
    rel = torch.as_tensor(relative, dtype=torch.float32)
    if rel.ndim == 2:
        rel = rel.unsqueeze(0)
    return np.asarray(
        runtime.relative_chunk_to_absolute(
            anchor_eef, runtime.unnormalize_chunk(rel)
        ),
        dtype=np.float64,
    )


@dataclass
class RowSpec:
    row: int
    episode_index: int
    bundle_name: str
    is_intervention: bool


def _row_specs(arrays: dict[str, np.ndarray], row_map: RowFrameMap) -> list[RowSpec]:
    n = int(arrays["z_rl"].shape[0])
    specs: list[RowSpec] = []
    for row in range(n):
        raw_path = str(row_map.raw_path_by_row[row])
        name = Path(raw_path).stem
        specs.append(
            RowSpec(
                row=row,
                episode_index=int(arrays["replay_episode_index"][row].item()),
                bundle_name=name,
                is_intervention=bool(arrays["is_intervention"][row].item()),
            )
        )
    return specs


def _load_old_iv50_actor(
    iv50_path: Path, arrays: dict[str, np.ndarray], device: str
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Load the old direct/bounded actor + its action scale from the iv50 ckpt.

    The saved ``actor_action_scale`` is used verbatim; falls back to the
    replay-derived scale if it is absent.
    """
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP
    from .rlt_stage2_train import _derive_actor_action_scale

    ckpt = torch.load(iv50_path, map_location="cpu", weights_only=False)
    config = SmolVLARLTConfig(**ckpt["rlt_config"])
    actor = RLActorMLP(config).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    scale = ckpt.get("actor_action_scale")
    if scale is None:
        scale = _derive_actor_action_scale(arrays)
    return actor, np.asarray(scale, dtype=np.float64), ckpt


# ─────────────────────────────────────────────────────────────────────────────
# T1
# ─────────────────────────────────────────────────────────────────────────────
def run_t1(
    *,
    stage1_checkpoint: Path,
    latent_ckpt: Path,
    iv50_ckpt: Path,
    replay_root: Path,
    device: str,
    max_rows: int | None,
    output: Path,
) -> None:
    replay_paths = discover_replay_bundles(replay_root)
    arrays, meta = load_replay_dataset(replay_paths, stage1_checkpoint=stage1_checkpoint)
    row_map = RowFrameMap(replay_paths, meta)
    reader = RawReader(max_open=8)
    specs = _row_specs(arrays, row_map)

    iv_rows = [s.row for s in specs if s.is_intervention]
    if max_rows is not None:
        rng = np.random.default_rng(EVAL_SEED)
        iv_rows = list(rng.choice(np.asarray(iv_rows), size=min(max_rows, len(iv_rows)), replace=False).tolist())
    print(f"T1: {len(iv_rows)} intervention rows")

    # Two runtimes: base_rt = stage1 VLA (no stage2 weights) for the base arm;
    # runtime = stage1 + stage2 (policy weights when present + actor + projector).
    base_rt = LatentRuntime(stage1_checkpoint=stage1_checkpoint, device=device)
    runtime = LatentRuntime(stage1_checkpoint=stage1_checkpoint, device=device)
    runtime.load_stage2(latent_ckpt)
    old_actor, old_scale, old_ckpt = _load_old_iv50_actor(iv50_ckpt, arrays, device)
    print(f"old iv50 checkpoint semantics: {old_ckpt.get('actor_output_semantics')}, "
          f"total_updates={old_ckpt.get('total_updates')}")

    per_row = []
    with torch.no_grad():
        for idx, row in enumerate(iv_rows):
            spec = specs[row]
            raw_path = str(row_map.raw_path_by_row[row])
            frame = int(row_map.frame_by_row[row])
            task = str(row_map.task_by_row[row])
            images, puppet = reader.read_row(raw_path, frame, runtime.cameras)
            anchor_eef = runtime.eef_state(puppet)  # absolute EEF at the decision frame

            z = torch.as_tensor(arrays["z_rl"][row], dtype=torch.float32, device=device).unsqueeze(0)
            s = torch.as_tensor(arrays["state"][row], dtype=torch.float32, device=device).unsqueeze(0)
            ref = torch.as_tensor(arrays["ref_actions"][row], dtype=torch.float32, device=device).unsqueeze(0)

            # A) base VLA (stage1 weights only)
            out_base = base_rt.decode(anchor_eef, images, task, r_lat=None,
                                      seed=EVAL_SEED + row)
            base_rel = np.asarray(out_base["_reference"].cpu().numpy(), dtype=np.float64)[:10]
            # B) old iv50 (action residual on the SAME base decode)
            mean_raw, _ = old_actor(z, s, ref, training=False)
            old_exec = compose_actor_action(
                mean_raw.cpu(), ref.cpu(),
                torch.as_tensor(old_scale, dtype=torch.float32),
                output_semantics=old_ckpt.get("actor_output_semantics", "direct"),
            ).numpy()[:10]
            # C) latent
            r_lat, _ = runtime.actor(z, s, ref, training=False)
            out_lat = runtime.decode(anchor_eef, images, task, r_lat=r_lat,
                                     seed=EVAL_SEED + row)
            lat_rel = np.asarray(out_lat["_reference"].cpu().numpy(), dtype=np.float64)[:10]

            human_abs = _abs_chunk(runtime, anchor_eef,
                                   arrays["actions"][row].astype(np.float64))[:10]
            base_abs = _abs_chunk(runtime, anchor_eef, base_rel)
            old_abs = _abs_chunk(runtime, anchor_eef, old_exec.astype(np.float64))
            lat_abs = _abs_chunk(runtime, anchor_eef, lat_rel)

            def metrics(chunk: np.ndarray) -> dict[str, float]:
                # per-step displacement direction vs human, steps 0..9
                anchor = anchor_eef
                angles, dists = [], []
                for t in range(10):
                    vh = human_abs[t] - anchor
                    vc = chunk[t] - anchor
                    angles.append(_angle_deg(vh, vc))
                    dists.append(float(np.linalg.norm(chunk[t] - human_abs[t])))
                return {"angle_deg": float(np.mean(angles)),
                        "endpoint_mm": float(np.mean(dists) * 1000.0)}

            mb, mo, ml = metrics(base_abs), metrics(old_abs), metrics(lat_abs)
            per_row.append({
                "row": row, "bundle": spec.bundle_name,
                "base": mb, "old_iv50": mo, "latent": ml,
                "latent_vs_base": mb["endpoint_mm"] - ml["endpoint_mm"],
                "latent_vs_old": mo["endpoint_mm"] - ml["endpoint_mm"],
                "angle_base": mb["angle_deg"], "angle_old": mo["angle_deg"],
                "angle_latent": ml["angle_deg"],
            })
            if (idx + 1) % 50 == 0:
                print(f"  T1 {idx+1}/{len(iv_rows)} rows done")

    def agg(key, subkey):
        return float(np.mean([r[key][subkey] for r in per_row]))

    summary = {
        "n_rows": len(per_row),
        "endpoint_mm": {
            "base_vla": agg("base", "endpoint_mm"),
            "old_iv50": agg("old_iv50", "endpoint_mm"),
            "latent": agg("latent", "endpoint_mm"),
        },
        "angle_deg": {
            "base_vla": agg("base", "angle_deg"),
            "old_iv50": agg("old_iv50", "angle_deg"),
            "latent": agg("latent", "angle_deg"),
        },
        "latent_closer_than_base_frac": float(np.mean([r["latent_vs_base"] > 0 for r in per_row])),
        "latent_closer_than_old_frac": float(np.mean([r["latent_vs_old"] > 0 for r in per_row])),
    }
    print("T1 summary:", json.dumps(summary, indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "per_row": per_row}
    output.write_text(json.dumps(out, indent=2))
    print(f"T1 written: {output}")
    reader.close()


# ─────────────────────────────────────────────────────────────────────────────
# T2
# ─────────────────────────────────────────────────────────────────────────────
def _decision_chain(
    base_rt: LatentRuntime,
    anchors: list[np.ndarray],
    images_task: list[tuple[dict, str]],
    z_s_ref: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    runtime: LatentRuntime,
    old_actor,
    old_scale: np.ndarray,
    old_ckpt: dict[str, Any],
) -> dict[str, list[dict]]:
    """Decode every decision for all arms, then measure the seam between
    consecutive chunks in absolute EEF (chunk i tail plan vs chunk i+1 head).

    The RTC arm feeds the previous chunk's unexecuted tail (steps 10..29,
    20 steps, normalized) as ``prev_chunk_left_over``, matching the
    ``ActionQueue.get_leftover`` convention of the RTC stack.
    """
    arms = ["base_vla", "old_iv50", "latent_no_rtc", "latent_rtc"]
    n = len(anchors)
    full: dict[str, list[np.ndarray]] = {a: [] for a in arms}
    for i in range(n):
        z, s, ref = z_s_ref[i]
        anchor = anchors[i]
        images, task = images_task[i]
        out_base = base_rt.decode(anchor, images, task, r_lat=None, seed=EVAL_SEED + i)
        base_full = np.asarray(out_base["reference_actions_full"], dtype=np.float64)[:30, :7]
        rlat, _ = runtime.actor(z, s, ref, training=False)
        out_nrtc = runtime.decode(anchor, images, task, r_lat=rlat, seed=EVAL_SEED + i)
        nrtc_full = np.asarray(out_nrtc["reference_actions_full"], dtype=np.float64)[:30, :7]
        leftover = None
        if i > 0:
            leftover = torch.as_tensor(
                full["latent_rtc"][i - 1][10:30],
                dtype=torch.float32,
                device=runtime.device,
            ).unsqueeze(0)
        out_rtc = runtime.decode(
            anchor, images, task, r_lat=rlat,
            prev_chunk_left_over=leftover, inference_delay=0, seed=EVAL_SEED + i,
        )
        rtc_full = np.asarray(out_rtc["reference_actions_full"], dtype=np.float64)[:30, :7]
        full["base_vla"].append(base_full)
        full["old_iv50"].append(base_full)
        full["latent_no_rtc"].append(nrtc_full)
        full["latent_rtc"].append(rtc_full)
        if (i + 1) % 10 == 0:
            print(f"    T2 chain {i + 1}/{n}")

    seams: dict[str, list[dict]] = {a: [] for a in arms}
    for i in range(n - 1):
        for arm in arms:
            # Overlap window: old chunk's forecast for the NEXT window
            # (steps 10..19) vs the new chunk's plan for that window (0..9).
            tail = full[arm][i][10:20]
            head = full[arm][i + 1][:10]
            if arm == "old_iv50":
                z1, s1, ref1 = z_s_ref[i + 1]
                mean_raw, _ = old_actor(z1, s1, ref1, training=False)
                head = compose_actor_action(
                    mean_raw.cpu(),
                    ref1.cpu(),
                    torch.as_tensor(old_scale, dtype=torch.float32),
                    output_semantics=old_ckpt.get(
                        "actor_output_semantics", "direct_bounded_normalized_action_contiguous_30hz_v3"
                    ),
                ).numpy()[:10]
            tail_abs = runtime.relative_chunk_to_absolute(
                anchors[i], runtime.unnormalize_chunk(torch.as_tensor(tail))
            )
            head_abs = runtime.relative_chunk_to_absolute(
                anchors[i + 1], runtime.unnormalize_chunk(torch.as_tensor(head))
            )
            seam_mm = float(np.abs(np.asarray(tail_abs) - np.asarray(head_abs)).mean() * 1000.0)
            seams[arm].append({"chunk": i, "seam_mm": seam_mm})
    return seams


def run_t2(
    *,
    stage1_checkpoint: Path,
    latent_ckpt: Path,
    iv50_ckpt: Path,
    replay_root: Path,
    device: str,
    episodes: list[str],
    max_decisions_per_episode: int,
    output: Path,
) -> None:
    replay_paths = discover_replay_bundles(replay_root)
    arrays, meta = load_replay_dataset(replay_paths, stage1_checkpoint=stage1_checkpoint)
    row_map = RowFrameMap(replay_paths, meta)
    reader = RawReader(max_open=8)
    specs = _row_specs(arrays, row_map)

    base_rt = LatentRuntime(stage1_checkpoint=stage1_checkpoint, device=device)
    runtime = LatentRuntime(stage1_checkpoint=stage1_checkpoint, device=device)
    runtime.enable_rtc(execution_horizon=10, max_guidance_weight=10.0)
    runtime.load_stage2(latent_ckpt)
    old_actor, old_scale, old_ckpt = _load_old_iv50_actor(iv50_ckpt, arrays, device)

    all_seams: dict[str, list[dict]] = {"base_vla": [], "old_iv50": [],
                                        "latent_no_rtc": [], "latent_rtc": []}
    per_episode: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for ep_name in episodes:
            rows = sorted([s.row for s in specs if s.bundle_name.startswith(ep_name)])
            if not rows:
                print(f"T2: no rows found for episode prefix {ep_name!r}")
                continue
            rows = rows[:max_decisions_per_episode]
            frames = [int(row_map.frame_by_row[r]) for r in rows]
            anchors, images_tasks, z_s_ref = [], [], []
            for r in rows:
                raw_path = str(row_map.raw_path_by_row[r])
                frame = int(row_map.frame_by_row[r])
                task = str(row_map.task_by_row[r])
                images, puppet = reader.read_row(raw_path, frame, runtime.cameras)
                anchor = runtime.eef_state(puppet)
                anchors.append(anchor)
                images_tasks.append((images, task))
                z = torch.as_tensor(arrays["z_rl"][r], dtype=torch.float32, device=device).unsqueeze(0)
                s = torch.as_tensor(arrays["state"][r], dtype=torch.float32, device=device).unsqueeze(0)
                ref = torch.as_tensor(arrays["ref_actions"][r], dtype=torch.float32, device=device).unsqueeze(0)
                z_s_ref.append((z, s, ref))
            print(f"T2: {ep_name}: {len(rows)} decisions")
            seams = _decision_chain(
                base_rt, anchors, images_tasks, z_s_ref, runtime,
                old_actor, old_scale, old_ckpt,
            )
            for arm, vals in seams.items():
                all_seams[arm].extend([dict(v, episode=ep_name) for v in vals])
            per_episode[ep_name] = {
                arm: float(np.mean([v["seam_mm"] for v in vals]))
                for arm, vals in seams.items() if vals
            }
    summary = {
        arm: {
            "mean_seam_mm": float(np.mean([v["seam_mm"] for v in vals])),
            "max_seam_mm": float(np.max([v["seam_mm"] for v in vals])),
            "n_seams": len(vals),
        }
        for arm, vals in all_seams.items() if vals
    }
    per_episode_mm = {ep: {arm: round(v, 3) for arm, v in d.items()} for ep, d in per_episode.items()}
    print("T2 summary:", json.dumps(summary, indent=2))
    print("T2 per-episode mean seam (mm):")
    for ep, d in per_episode_mm.items():
        print(f"  {ep}: {d}")
    output.parent.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "per_episode": per_episode, "seams": all_seams}
    output.write_text(json.dumps(out, indent=2))
    print(f"T2 written: {output}")
    reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--stage1-checkpoint", type=Path, required=True)
        p.add_argument("--replay-root", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--iv50-ckpt", type=Path, required=True)
        p.add_argument("--latent-ckpt", type=Path, required=True)

    p1 = sub.add_parser("t1")
    _common(p1)
    p1.add_argument("--max-rows", type=int, default=None)
    p1.add_argument("--output", type=Path, required=True)

    p2 = sub.add_parser("t2")
    _common(p2)
    p2.add_argument("--episodes", type=str, required=True,
                    help="comma-separated bundle-name prefixes")
    p2.add_argument("--max-decisions", type=int, default=12)
    p2.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if args.cmd == "t1":
        run_t1(stage1_checkpoint=args.stage1_checkpoint,
               latent_ckpt=args.latent_ckpt, iv50_ckpt=args.iv50_ckpt,
               replay_root=args.replay_root, device=args.device,
               max_rows=args.max_rows, output=args.output)
    else:
        run_t2(stage1_checkpoint=args.stage1_checkpoint,
               latent_ckpt=args.latent_ckpt, iv50_ckpt=args.iv50_ckpt,
               replay_root=args.replay_root, device=args.device,
               episodes=[e.strip() for e in args.episodes.split(",") if e.strip()],
               max_decisions_per_episode=args.max_decisions,
               output=args.output)


if __name__ == "__main__":
    main()
