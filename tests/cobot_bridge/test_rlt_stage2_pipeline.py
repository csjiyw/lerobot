import pytest
from pathlib import Path
from types import SimpleNamespace

from cobot_bridge.rlt_stage2_pipeline import (
    build_archive_command,
    build_train_command,
    should_train_stage2,
)


def test_warmup_phase_is_removed():
    with pytest.raises(ValueError, match="invalid rollout_phase"):
        should_train_stage2("warmup")


def test_every_actor_episode_requests_training():
    assert should_train_stage2("actor") is True


def test_async_pipeline_trains_actor_and_does_not_trim_rollout():
    args = SimpleNamespace(
        raw=Path("episode.raw.h5"),
        archive_repo_id="local/episode",
        archive_root=Path("archive"),
        fps=30,
        stage1_checkpoint=Path("stage1.pt"),
        replay_root=Path("replay"),
        stage2_output_dir=Path("stage2"),
        trainer_device="cuda:1",
        actor_exploration_std=0.002,
        actor_validation_cache=Path("actor_bc_features.npz"),
        il_anchor_weight=30.0,
        q_weight=0.01,
        bc_weight=5.0,
        intervention_bc_weight=1.0,
        actor_output_semantics="absolute",
        ref_anchor_weight=0.0,
        ref_anchor_mask="none",
        action_stats=None,
    )
    archive = build_archive_command(args)
    train = build_train_command(args, Path("progress.json"))

    assert "--trim-stationary-holds" not in archive
    assert archive[archive.index("--action-source") + 1] == "executed_action"
    assert train[train.index("--training-phase") + 1] == "actor"
    assert train[train.index("--critic-target-mode") + 1] == "mc"
    assert train[train.index("--il-anchor-weight") + 1] == "30.0"
    assert train[train.index("--progress-file") + 1] == "progress.json"
    assert "warmup" not in " ".join(train).lower()
