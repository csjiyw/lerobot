#!/usr/bin/env python3
"""Attach audited scene labels to Cobot Magic raw HDF5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


SCENES = {
    "large_white_table": "White large table",
    "small_table_yellow_base": "Small table + yellow base",
    "camera_misaligned_no_task_view": "No valid task view",
}

TASK_SETUPS = {
    "large_white_table": {
        "id": "large_table_four_hole_peg_insertion",
        "label": "白色大桌面 · 蓝色四孔积木与长条木制插柱板",
        "objective": "抓取蓝色方形四孔积木，将圆孔对准长条木板上的对应插柱并插入。",
    },
    "small_table_yellow_base": {
        "id": "small_table_cross_shape_insertion",
        "label": "小桌＋黄色底座 · 红色十字积木与木制形状盒",
        "objective": "抓取红色十字积木，搬运到木制形状盒上方，对准十字孔并插入。",
    },
    "camera_misaligned_no_task_view": {
        "id": "right_arm_teach_intervention_test",
        "label": "早期右臂 Teach／人工介入管线测试 · 相机未覆盖操作区",
        "objective": "验证右臂 Teach、跟随和人工介入记录流程；该录制没有可辨识的完整操作任务。",
    },
}

COLLECTION_SCENES = {
    "cobot_magic_il": "large_white_table",
    "cobot_magic_il_1.0": "small_table_yellow_base",
    "cobot_magic_rl_pnp_cross": "small_table_yellow_base",
    "cobot_magic_rlt": "small_table_yellow_base",
    "cobot_magic_right_intervention": "camera_misaligned_no_task_view",
    "cobot_magic_right_intervention_rlt": "camera_misaligned_no_task_view",
    "cobot_magic_right_intervention_rlt_success": "camera_misaligned_no_task_view",
    "cobot_magic_right_test": "camera_misaligned_no_task_view",
}

MIXED_COLLECTION = "cobot_magic_rl"
MIXED_SMALL_TABLE_EPISODE = "episode_20260814_144110_843908.raw.h5"
TARGET_COLLECTION = "cobot_magic_rl_pnp_cross"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.raw_root.resolve()

    assignments: list[tuple[Path, str]] = []
    for collection, scene in COLLECTION_SCENES.items():
        assignments.extend((path, scene) for path in sorted((root / collection).glob("*.h5")))
    mixed = root / MIXED_COLLECTION
    assignments.extend(
        (path, "small_table_yellow_base" if path.name == MIXED_SMALL_TABLE_EPISODE else "large_white_table")
        for path in sorted(mixed.glob("*.h5"))
    )

    for path, scene in assignments:
        print(f"TAG {path.relative_to(root)} -> {scene}")
        if args.apply:
            with h5py.File(path, "r+") as raw:
                raw.attrs["scene_layout"] = scene
                raw.attrs["scene_layout_label"] = SCENES[scene]
                raw.attrs["scene_label_source"] = "manual_visual_audit_2026-08-20"
                setup = TASK_SETUPS[scene]
                raw.attrs["task_setup"] = setup["id"]
                raw.attrs["task_setup_label"] = setup["label"]
                raw.attrs["task_objective"] = setup["objective"]
                raw.attrs["task_annotation_source"] = "manual_visual_audit_2026-08-20"

    source = mixed / MIXED_SMALL_TABLE_EPISODE
    target = root / TARGET_COLLECTION / MIXED_SMALL_TABLE_EPISODE
    print(f"MOVE {source.relative_to(root)} -> {target.relative_to(root)}")
    if args.apply:
        if source.exists() and target.exists():
            raise FileExistsError(target)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        elif not target.exists():
            raise FileNotFoundError(source)


if __name__ == "__main__":
    main()
