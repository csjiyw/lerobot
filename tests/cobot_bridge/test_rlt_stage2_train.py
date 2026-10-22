import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from cobot_bridge.rlt_stage2_data import (
    ACTION_CHUNK_TIMEBASE,
    ACTOR_OUTPUT_SEMANTICS,
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
)
from cobot_bridge.rlt_stage2_train import (
    _actor_bc_target_and_mask,
    _assert_reference_reachable,
    _derive_actor_action_scale,
    _masked_action_mse,
    _reachable_bc_target,
    _sample_balanced_outcome_indices,
    _sample_balanced_episode_indices,
    discover_replay_bundles,
    train_stage2,
    load_replay_dataset,
)
from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP, RLTokenEncoder


def make_stage1(path):
    config = SmolVLARLTConfig(
        input_features={},
        output_features={},
        vlm_hidden_dim=8,
        rlt_hidden_dim=8,
        rlt_prefix_seq_len=16,
        rlt_encoder_layers=1,
        rlt_decoder_layers=1,
        rlt_num_heads=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        modality_projection_dim=8,
        actor_output_mode="residual",
        actor_residual_scale=0.25,
        state_dim=3,
        action_dim=3,
        chunk_size=4,
        n_action_steps_rl=2,
        action_stride=1,
        rl_batch_size=4,
        utd_ratio=2,
        critic_per_actor=2,
    )
    encoder = RLTokenEncoder(config)
    torch.save(
        {
            "stage": "rlt_stage1_joint",
            "rlt_architecture": "rlinf_prefix_v1",
            "rlt_config": asdict(config),
            "encoder_state_dict": encoder.state_dict(),
            "smolvla_path": "/tmp/base-policy",
        },
        path,
    )
    return config


def make_replay(root, stage1, config, episode, outcome):
    dimensions = RLTDimensions(
        config.rlt_hidden_dim,
        config.state_dim,
        config.n_action_steps_rl,
        config.action_dim,
    )
    decision_path = root / f"episode_{episode}.h5"
    writer = RLTDecisionWriter(
        decision_path,
        dimensions=dimensions,
        stage1_checkpoint=str(stage1),
        task="place cross",
        workflow_mode=2,
    )
    for index in range(4):
        ref = np.full((2, 3), 0.1 * (episode + index), dtype=np.float32)
        human = index == 1
        writer.append(
            z_rl=np.full(8, 0.01 * (episode + index), dtype=np.float32),
            state=np.full(3, 0.02 * (episode + index), dtype=np.float32),
            reference_actions=ref,
            policy_actions=ref + 0.03,
            executed_actions=ref + (0.2 if human else 0.03),
            action_source="human" if human else "actor",
            is_intervention=human,
        )
    writer.close(outcome=outcome)
    replay = root / f"episode_{episode}.npz"
    build_replay_bundle(decision_path, replay)
    return replay


def make_actor_init(path, stage1, config):
    actor = RLActorMLP(config)
    torch.save(
        {
            "stage": "rlt_actor_bc",
            "stage1_checkpoint": str(stage1.resolve()),
            "actor_output_semantics": ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
            "rlt_config": asdict(config),
            "actor_state_dict": actor.state_dict(),
            "actor_action_scale": [2.0] * config.action_dim,
        },
        path,
    )


def test_stage2_trainer_publishes_and_resumes_checkpoint(tmp_path):
    stage1 = tmp_path / "stage1.pt"
    config = make_stage1(stage1)
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    make_replay(replay_root, stage1, config, 1, "success")
    output = tmp_path / "stage2"
    progress_file = tmp_path / "trainer_progress.json"
    actor_init = tmp_path / "actor_bc.pt"
    make_actor_init(actor_init, stage1, config)

    first = train_stage2(
        stage1_checkpoint=stage1,
        actor_init_checkpoint=actor_init,
        replay_paths=discover_replay_bundles(replay_root),
        output_dir=output,
        device="cpu",
        updates=4,
        batch_size=4,
        seed=3,
        log_freq=10,
        progress_file=progress_file,
        training_phase="actor",
        actor_exploration_std=0.0,
        balanced_outcomes=False,
    )
    assert first["total_updates"] == 4
    assert first["actor_updates_this_run"] == 1
    checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "rlt_stage2"
    assert checkpoint["actor_init_checkpoint"] == str(actor_init.resolve())
    assert checkpoint["actor_output_semantics"] == ACTOR_OUTPUT_SEMANTICS
    assert checkpoint["action_chunk_timebase"] == ACTION_CHUNK_TIMEBASE
    assert len(checkpoint["actor_action_scale"]) == 3
    assert checkpoint["grad_clip_norm"] == 1.0
    assert checkpoint["ref_anchor_weight"] == 0.5
    assert checkpoint["training_phase"] == "actor"
    assert checkpoint["ready_for_online"] is True
    assert "warmup_updates" not in checkpoint
    assert checkpoint["q_weight"] == 1.0
    assert checkpoint["bc_weight"] == 5.0
    assert checkpoint["actor_validation"]["p95_abs_error_to_reference"] >= 0.0
    assert checkpoint["actor_warnings"] == []
    assert checkpoint["buffer_transitions"] == 3
    assert len(checkpoint["actor_state_dict"]) > 0
    assert checkpoint["trained_replay_names"] == ["episode_1.npz"]
    progress = json.loads(progress_file.read_text())
    assert progress["state"] == "completed"
    assert progress["current_update"] == 4
    assert progress["total_updates"] == 4
    assert progress["global_update"] == 4
    assert progress["percent"] == 100.0
    assert progress["eta_seconds"] == 0.0
    assert progress["updates_per_second"] > 0
    assert progress["checkpoint"] == first["checkpoint"]

    # Simulate a v3 checkpoint and relocation from an audit directory into the
    # canonical dashboard replay directory.  Historical episodes must not be
    # counted as new merely because their absolute paths changed.
    checkpoint.pop("trained_replay_names")
    checkpoint["trained_replay_paths"] = [str((replay_root / "episode_1.npz").resolve())]
    torch.save(checkpoint, output / "latest.pt")
    relocated_root = tmp_path / "canonical_replay"
    replay_root.rename(relocated_root)
    make_replay(relocated_root, stage1, config, 2, "failure")
    second = train_stage2(
        stage1_checkpoint=stage1,
        replay_paths=discover_replay_bundles(relocated_root),
        output_dir=output,
        device="cpu",
        updates=None,
        batch_size=4,
        seed=3,
        log_freq=10,
        training_phase="actor",
        actor_exploration_std=0.0,
    )
    assert second["new_transitions"] == 3
    assert second["new_episodes"] == 1
    assert second["updates_this_run"] == 32
    assert second["total_updates"] == 36
    assert second["training_phase"] == "actor"
    assert second["ready_for_online"] is True
    assert json.loads((output / "status.json").read_text())["total_updates"] == 36


def test_balanced_sampler_uses_equal_success_and_failure_transitions() -> None:
    labels = np.asarray([[1], [1], [0], [0], [0]], dtype=np.float32)
    indices = _sample_balanced_outcome_indices(
        np.random.default_rng(3), labels, batch_size=8
    )
    assert labels[indices].sum() == 4
    assert len(indices) == 8


def test_replay_dataset_adds_episode_reset_monte_carlo_returns(tmp_path):
    """RECAP/MC critic targets must never bootstrap across episode boundaries."""

    stage1 = tmp_path / "stage1.pt"
    config = make_stage1(stage1)
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    success = make_replay(replay_root, stage1, config, 1, "success")
    failure = make_replay(replay_root, stage1, config, 2, "failure")

    arrays, _ = load_replay_dataset([success, failure], stage1_checkpoint=stage1)
    # ``make_replay`` creates three transitions per episode.  The binary
    # terminal reward is 1 for success / 0 for failure at gamma=.96.
    assert np.allclose(arrays["mc_returns"][:3, 0], [0.96**2, 0.96, 1.0])
    assert np.allclose(arrays["mc_returns"][3:, 0], 0.0)


def test_episode_balanced_sampler_does_not_overweight_long_episode() -> None:
    outcomes = np.asarray([[1]] * 100 + [[1]] + [[0]] * 4, dtype=np.float32)
    episodes = np.asarray([[0]] * 100 + [[1]] + [[2]] * 4, dtype=np.int64)
    indices = _sample_balanced_episode_indices(
        np.random.default_rng(7), outcomes, episodes, batch_size=20_000
    )
    selected_episodes = episodes[indices].reshape(-1)

    assert (outcomes[indices].sum()) == 10_000
    success_episodes = selected_episodes[:0]  # keep dtype for static analyzers
    success_episodes = episodes[indices][outcomes[indices].reshape(-1).astype(bool)]
    ratio = float((success_episodes == 0).sum()) / float(success_episodes.size)
    assert 0.47 < ratio < 0.53


def test_actor_bc_uses_reference_online_and_human_or_il_ground_truth():
    actions = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    reference = actions + 100.0
    batch = {
        "actions": actions,
        "ref_actions": reference,
        "outcome_success": torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        "is_intervention": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        "is_demo": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
    }

    target, supervised = _actor_bc_target_and_mask(batch)

    expected = reference.clone()
    expected[1:3] = actions[1:3]
    assert torch.equal(target, expected)
    assert supervised.tolist() == [True, True, True, True]
    predicted = expected.clone()
    assert _masked_action_mse(predicted, target, supervised).item() == 0.0


def test_actor_bc_action_weights_reduce_gripper_contribution():
    predicted = torch.zeros(1, 1, 3)
    target = torch.ones_like(predicted)
    supervised = torch.tensor([True])

    uniform = _masked_action_mse(predicted, target, supervised)
    weighted = _masked_action_mse(
        predicted,
        target,
        supervised,
        torch.tensor([1.0, 1.0, 0.5]),
    )

    assert uniform.item() == 1.0
    assert weighted.item() == 1.0

    target[..., :2] = 0.0
    weighted_gripper_only = _masked_action_mse(
        predicted,
        target,
        supervised,
        torch.tensor([1.0, 1.0, 0.5]),
    )
    assert torch.isclose(weighted_gripper_only, torch.tensor(0.2))


def test_actor_bc_empty_mask_has_differentiable_zero_loss():
    predicted = torch.ones(2, 2, 3, requires_grad=True)
    loss = _masked_action_mse(
        predicted, torch.zeros_like(predicted), torch.tensor([False, False])
    )

    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_residual_bc_target_is_projected_into_reachable_box():
    reference = torch.zeros(1, 2, 3)
    target = torch.tensor([[[2.0, -2.0, 0.25], [0.5, -0.5, 0.0]]])
    scale = torch.tensor([4.0, 2.0, 1.0])

    reachable = _reachable_bc_target(
        target,
        reference,
        scale,
        output_semantics=ACTOR_OUTPUT_SEMANTICS,
        residual_scale=0.25,
    )

    assert torch.equal(
        reachable,
        torch.tensor([[[1.0, -0.5, 0.25], [0.5, -0.5, 0.0]]]),
    )


def _reachability_arrays(peak: float):
    dim = 3
    ref = np.full((2, 4, dim), peak, dtype=np.float32)
    return {
        "ref_actions": ref,
        "next_ref_actions": np.zeros_like(ref),
        "actions": np.zeros_like(ref),
    }


def test_reference_reachability_rejects_a_truncating_scale():
    arrays = _reachability_arrays(peak=4.0)
    narrow = np.array([2.6, 5.0, 5.0], dtype=np.float32)

    with pytest.raises(ValueError, match="truncates the replay reference") as error:
        _assert_reference_reachable(arrays, narrow)

    assert "dim 0: |action| up to 4.000 > scale 2.600" in str(error.value)
    assert "--actor-action-scale" in str(error.value)
    _assert_reference_reachable(arrays, narrow, allow_clipped=True)


def test_reference_reachability_accepts_the_derived_scale():
    arrays = _reachability_arrays(peak=4.0)
    _assert_reference_reachable(arrays, _derive_actor_action_scale(arrays))


def test_heldout_il_regression_is_diagnostic_only(tmp_path, monkeypatch):
    stage1 = tmp_path / "stage1.pt"
    config = make_stage1(stage1)
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    make_replay(replay_root, stage1, config, 1, "success")
    output = tmp_path / "stage2"
    actor_init = tmp_path / "actor_bc.pt"
    make_actor_init(actor_init, stage1, config)
    payload = torch.load(actor_init, map_location="cpu", weights_only=False)
    payload["best_val_loss"] = 0.01
    torch.save(payload, actor_init)
    cache = tmp_path / "actor_bc_features.npz"
    cache.write_bytes(b"validation is monkeypatched")
    monkeypatch.setattr(
        "cobot_bridge.rlt_stage2_train._heldout_actor_gt_bc_loss",
        lambda *_args, **_kwargs: 0.02,
    )

    result = train_stage2(
        stage1_checkpoint=stage1,
        actor_init_checkpoint=actor_init,
        replay_paths=discover_replay_bundles(replay_root),
        output_dir=output,
        device="cpu",
        updates=2,
        batch_size=3,
        seed=3,
        log_freq=10,
        training_phase="actor",
        actor_exploration_std=0.0,
        balanced_outcomes=False,
        actor_validation_cache=cache,
        max_heldout_bc_ratio=1.25,
    )

    assert result["total_updates"] == 2
    assert result["heldout_validation"]["ratio_to_actor_bc"] == 2.0
    assert result["heldout_validation"]["diagnostic_only"] is True
    assert result["actor_warnings"] == []
    assert result["ready_for_online"] is True
    assert (output / "latest.pt").is_file()
