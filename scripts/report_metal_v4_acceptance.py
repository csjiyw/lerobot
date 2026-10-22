#!/usr/bin/env python3
"""Merge occlusion-audit JSONs into a single metal-v4 acceptance report with a verdict."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset_audit" / "v4"
FINAL = ROOT / "audit_v4_final"

# (dir under audit_v4_final or audit_baseline60k, model label, dataset label)
SOURCES = [
    ("audit_baseline60k", "60k-baseline(all407)", "train-mix(all407)"),
    ("baseline60k_env3", "60k-baseline", "holdout env3"),
    ("baseline60k_env2", "60k-baseline", "holdout env2"),
    ("baseline60k_oldenv", "60k-baseline*", "holdout oldenv*"),
    ("baseline60k_trainmix", "60k-baseline", "train-mix v4"),
    ("phase1_early10k_trainmix", "v4 phase1@10k", "train-mix v4"),
    ("phase1_env3", "v4 phase1@60k", "holdout env3"),
    ("phase1_env2", "v4 phase1@60k", "holdout env2"),
    ("phase1_oldenv", "v4 phase1@60k", "holdout oldenv"),
    ("phase1_trainmix", "v4 phase1@60k", "train-mix v4"),
    ("phase2_env3", "v4 phase2@80k", "holdout env3"),
    ("phase2_env2", "v4 phase2@80k", "holdout env2"),
    ("phase2_oldenv", "v4 phase2@80k", "holdout oldenv"),
    ("phase2_trainmix", "v4 phase2@80k", "train-mix v4"),
]


def load(rel: str) -> dict | None:
    for base in (FINAL, ROOT):
        p = base / rel / "occlusion_audit.json"
        if p.is_file():
            return json.loads(p.read_text())
    return None


def main() -> None:
    rows = []
    for rel, model, ds in SOURCES:
        rep = load(rel)
        if rep is None:
            continue
        modes = rep["modes"]
        for g in sorted(modes["normal"]):
            if g in ("unknown",):
                g = "__all__"
            get = lambda m: modes[m].get(g)  # noqa: E731
            if not all(get(m) for m in ("normal", "gray", "state_zero", "both")):
                continue
            rows.append({
                "model": model, "dataset": ds, "group": g, "n": get("normal")["n"],
                "normal_pos": get("normal")["pos_rmse"], "gray_pos": get("gray")["pos_rmse"],
                "zero_pos": get("state_zero")["pos_rmse"], "both_pos": get("both")["pos_rmse"],
            })
    for r in rows:
        r["gray_ratio"] = round(r["gray_pos"] / r["normal_pos"], 3) if r["normal_pos"] > 0 else None
        r["zero_ratio"] = round(r["zero_pos"] / r["normal_pos"], 3) if r["normal_pos"] > 0 else None

    lines = ["# metal v4 验收报告（自动生成）", "",
             "| model | dataset | group | n | normal | gray | zero | both | gray/n | zero/n |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['group']} | {r['n']} | {r['normal_pos']:.4f} "
                     f"| {r['gray_pos']:.4f} | {r['zero_pos']:.4f} | {r['both_pos']:.4f} "
                     f"| {r['gray_ratio']} | {r['zero_ratio']} |")
    lines.append("\n\\* eval_oldenv 与 all407 训练集可能部分重叠，基线该行仅参考。")

    # verdict logic
    def pick(model, ds, group):
        c = [r for r in rows if r["model"] == model and r["dataset"].startswith(ds) and r["group"] == group]
        return c[0] if c else None

    verdict = "INCOMPLETE — 评测尚未跑全，见上表现有行"
    p2_cur = pick("v4 phase2@80k", "train-mix", "current")
    p2_env3 = pick("v4 phase2@80k", "holdout env3", "__all__")
    b_cur = pick("60k-baseline", "train-mix v4", "current")
    b_env3 = pick("60k-baseline", "holdout env3", "__all__")
    if p2_cur and p2_env3:
        vis_ok = b_cur and p2_cur["gray_ratio"] > max(1.4, b_cur["gray_ratio"] * 1.2)
        gen_ok = (b_env3 is None) or (p2_env3["normal_pos"] <= b_env3["normal_pos"] * 1.25)
        if vis_ok and gen_ok:
            verdict = ("DEPLOY-READY — 视觉依赖显著高于 60k 基线且 holdout 泛化达标；"
                       "按 DEPLOY_V4_README 做真机 A/B。")
        elif gen_ok:
            verdict = "VISION-WEAK — 泛化可以但 gray/norm 未明显超过基线：建议加严 camera_dropout/state_dropout 复训或加大 phase2 解冻层数。"
        else:
            verdict = "GENERALIZATION-WEAK — 先跑 current-env FT 配方（train_metal_v4_finetune.sh）再评估真机。"
    lines.append(f"\n## VERDICT: {verdict}\n")
    out = FINAL / "ACCEPTANCE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines[-3:]))
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
