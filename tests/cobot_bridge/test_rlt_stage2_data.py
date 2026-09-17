import json

import h5py
import numpy as np
import pytest
import torch

from cobot_bridge.rlt_stage2_data import (
    RLTDecisionWriter,
    RLTDimensions,
    build_replay_bundle,
    calibrated_actor_exploration_std,
    compose_actor_action,
    load_replay_bundle,
    inspect_warmup_replays,
    repair_mode2_human_action_chunks,
    squash_actor_action,
    contiguous_action_chunk,
)


def test_piper_exploration_grows_physically_from_j1_to_j6() -> None:
    action_scale = torch.tensor(
        [2.0670104, 1.8038399, 2.0153348, 3.8637834, 5.0442343, 3.5502837, 1.446517]
    )
    action_physical_std = torch.tensor(
        [0.2694869, 0.8065177, 0.5445433, 0.1276350, 0.3232516, 0.1923895, 0.0338265]
    )

    raw_std = calibrated_actor_exploration_std(
        0.002,
        action_scale,
        action_physical_std,
        residual_scale=0.25,
    )
    physical_arm_std = raw_std[:6] * action_scale[:6] * action_physical_std[:6] * 0.25
    expected_ratios = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    torch.testing.assert_close(
        physical_arm_std / physical_arm_std[-1], expected_ratios
    )
    assert torch.all(physical_arm_std[1:] > physical_arm_std[:-1])
    assert raw_std[-1] == pytest.approx(0.002)


def test_contiguous_action_chunk_preserves_30hz_order() -> None:
    actions = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)

    selected = contiguous_action_chunk(actions, action_steps=4, action_dim=2)

    torch.testing.assert_close(selected, actions[:, :4, :2])


def decision(index, dims, *, source="reference"):
    base = np.float32(index * 0.1)
    ref = np.full((dims.action_steps, dims.action_dim), base + 0.2, dtype=np.float32)
    policy = ref + 0.05
    executed = policy if source == "actor" else ref
    if source == "human":
        executed = np.full_like(ref, base + 0.7)
    return {
        "z_rl": np.full(dims.z_dim, base, dtype=np.float32),
        "state": np.full(dims.state_dim, base + 0.1, dtype=np.float32),
        "reference_actions": ref,
        "policy_actions": policy,
        "executed_actions": executed,
        "action_source": source,
        "is_intervention": source == "human",
        "bridge_sequence": index,
    }


def test_mode2_decisions_build_terminal_and_intervention_rewards(tmp_path):
    dims = RLTDimensions(z_dim=8, state_dim=3, action_steps=2, action_dim=3)
    decisions = tmp_path / "episode.decisions.h5"
    writer = RLTDecisionWriter(
        decisions,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="place cross",
        workflow_mode=2,
    )
    writer.append(**decision(0, dims, source="actor"))
    writer.append(**decision(1, dims, source="human"))
    writer.append(**decision(2, dims, source="actor"))
    writer.close(outcome="success")

    output = tmp_path / "replay" / "episode.npz"
    metadata = build_replay_bundle(decisions, output)
    arrays, loaded = load_replay_bundle(output)

    assert metadata == loaded
    assert metadata["transition_count"] == 2
    assert metadata["intervention_transitions"] == 1
    np.testing.assert_allclose(arrays["rewards"], [[0.0], [1.0]])
    np.testing.assert_allclose(arrays["dones"], [[0.0], [1.0]])
    np.testing.assert_allclose(arrays["is_intervention"], [[0.0], [1.0]])
    np.testing.assert_allclose(arrays["is_demo"], [[0.0], [0.0]])
    np.testing.assert_allclose(arrays["actions"][1], np.full((2, 3), 0.8))
    sidecar = json.loads((output.with_suffix(".npz.json")).read_text())
    assert sidecar["outcome"] == "success"


def test_intervention_is_a_bc_label_and_does_not_create_reward(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    decisions = tmp_path / "failure.h5"
    writer = RLTDecisionWriter(
        decisions,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=2,
    )
    writer.append(**decision(0, dims, source="human"))
    writer.append(**decision(1, dims, source="human"))
    writer.append(**decision(2, dims, source="actor"))
    writer.close(outcome="failure")
    output = tmp_path / "failure.npz"

    build_replay_bundle(decisions, output)
    arrays, _ = load_replay_bundle(output)

    np.testing.assert_allclose(arrays["rewards"], [[0.0], [0.0]])


def test_recap_potential_shaping_uses_adjacent_decision_values(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    decisions = tmp_path / "recap.decisions.h5"
    writer = RLTDecisionWriter(
        decisions,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
    )
    for index in range(3):
        writer.append(**decision(index, dims, source="reference"))
    writer.close(outcome="failure")

    output = tmp_path / "recap.npz"
    metadata = build_replay_bundle(
        decisions,
        output,
        trim_stationary=False,
        recap_values=np.asarray([-0.8, -0.5, -0.1], dtype=np.float32),
        recap_discount=0.96,
    )
    arrays, _ = load_replay_bundle(output)

    # r0=.96*(-.5)-(-.8)=.32; r1=.96*(-.1)-(-.5)-1=-.596.
    np.testing.assert_allclose(arrays["rewards"], [[0.32], [-0.596]], atol=1e-6)
    assert metadata["reward_source"] == "recap_potential_v1"
    assert metadata["terminal_failure_reward"] == -1.0


def test_mode2_human_chunks_are_rebuilt_from_future_raw_commands(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=3, action_dim=2)
    decision_path = tmp_path / "mode2.decisions.h5"
    writer = RLTDecisionWriter(
        decision_path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=2,
        side="right",
    )
    values = decision(0, dims, source="human")
    values["time_ns"] = 1_000_000_000
    writer.append(**values)
    writer.close(outcome="success")

    raw_path = tmp_path / "mode2.raw.h5"
    timestamps = 1_000_000_000 + np.arange(5, dtype=np.int64) * 100_000_000
    executed = np.stack(
        [np.asarray([index, index + 0.5], dtype=np.float32) for index in range(5)]
    )
    with h5py.File(raw_path, "w") as raw:
        raw.attrs.update({"completed": True, "side": "right"})
        frames = raw.create_group("frames")
        frames.create_dataset("pc_timestamp_ns", data=timestamps)
        frames.create_dataset("sequence", data=np.arange(5, dtype=np.int64) * 10)
        frames.create_dataset("is_human_intervention", data=np.ones(5, dtype=np.uint8))
        frames.create_dataset("executed_action_valid", data=np.ones(5, dtype=np.uint8))
        frames.create_dataset("executed_action", data=executed)

    summary = repair_mode2_human_action_chunks(
        decision_path,
        raw_path,
        normalize_action_chunk=lambda chunk: chunk * 2.0,
        action_fps=10.0,
    )

    with h5py.File(decision_path, "r") as repaired:
        np.testing.assert_allclose(repaired["executed_actions"][0], executed[:3] * 2.0)
        assert repaired.attrs["human_chunk_source"] == "raw_executed_action_future"
    assert summary["repaired_decisions"] == 1
    assert summary["constant_human_chunks"] == 0


def test_mode2_human_chunk_prefers_observation_sequence_over_late_append_time(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    decision_path = tmp_path / "sequence.decisions.h5"
    writer = RLTDecisionWriter(
        decision_path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=2,
        side="right",
    )
    values = decision(0, dims, source="human")
    values["bridge_sequence"] = 20
    values["time_ns"] = 1_300_000_000  # Appended two frames after observation.
    writer.append(**values)
    writer.close(outcome="success")

    raw_path = tmp_path / "sequence.raw.h5"
    with h5py.File(raw_path, "w") as raw:
        raw.attrs.update({"completed": True, "side": "right"})
        frames = raw.create_group("frames")
        frames.create_dataset(
            "pc_timestamp_ns",
            data=1_000_000_000 + np.arange(5, dtype=np.int64) * 100_000_000,
        )
        frames.create_dataset("sequence", data=np.arange(5, dtype=np.int64) * 10)
        frames.create_dataset("is_human_intervention", data=np.ones(5, dtype=np.uint8))
        frames.create_dataset("executed_action_valid", data=np.ones(5, dtype=np.uint8))
        frames.create_dataset(
            "executed_action",
            data=np.stack([np.full(2, index, dtype=np.float32) for index in range(5)]),
        )

    summary = repair_mode2_human_action_chunks(
        decision_path,
        raw_path,
        normalize_action_chunk=lambda chunk: chunk,
        action_fps=10.0,
    )

    with h5py.File(decision_path, "r") as repaired:
        np.testing.assert_allclose(repaired["executed_actions"][0], [[2, 2], [3, 3]])
    assert summary["alignment_source"] == "bridge_sequence"
    assert summary["max_sequence_error"] == 0


def test_mode2_human_chunk_tail_holds_last_teach_command(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=3, action_dim=2)
    decision_path = tmp_path / "tail.decisions.h5"
    writer = RLTDecisionWriter(
        decision_path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=2,
        side="right",
    )
    values = decision(0, dims, source="human")
    values["time_ns"] = 1_100_000_000
    values["bridge_sequence"] = 1
    writer.append(**values)
    writer.close(outcome="success")

    raw_path = tmp_path / "tail.raw.h5"
    with h5py.File(raw_path, "w") as raw:
        raw.attrs.update({"completed": True, "side": "right"})
        frames = raw.create_group("frames")
        frames.create_dataset(
            "pc_timestamp_ns",
            data=1_000_000_000 + np.arange(4, dtype=np.int64) * 100_000_000,
        )
        frames.create_dataset("sequence", data=np.arange(4, dtype=np.int64))
        frames.create_dataset(
            "is_human_intervention", data=np.asarray([1, 1, 1, 0], dtype=np.uint8)
        )
        frames.create_dataset("executed_action_valid", data=np.ones(4, dtype=np.uint8))
        frames.create_dataset(
            "executed_action",
            data=np.asarray([[0, 0], [1, 1], [2, 2], [9, 9]], dtype=np.float32),
        )

    summary = repair_mode2_human_action_chunks(
        decision_path,
        raw_path,
        normalize_action_chunk=lambda chunk: chunk,
        action_fps=10.0,
    )

    with h5py.File(decision_path, "r") as repaired:
        np.testing.assert_allclose(
            repaired["executed_actions"][0], [[1, 1], [2, 2], [2, 2]]
        )
    assert summary["padded_action_samples"] == 1


def test_mode1_rejects_human_decisions(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    writer = RLTDecisionWriter(
        tmp_path / "mode1.h5",
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
    )
    with pytest.raises(ValueError, match="Mode 1"):
        writer.append(**decision(0, dims, source="human"))
    writer.abort()


def test_warmup_count_is_scoped_to_exact_stage1_and_reference_only(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    stage1 = (tmp_path / "stage1.pt").resolve()
    stage1.write_bytes(b"stage1")
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    for episode in range(2):
        path = tmp_path / f"warmup_{episode}.h5"
        writer = RLTDecisionWriter(
            path,
            dimensions=dims,
            stage1_checkpoint=str(stage1),
            task="task",
            workflow_mode=1,
            collection_phase="warmup",
        )
        for index in range(3):
            writer.append(**decision(index + episode, dims, source="reference"))
        writer.close(outcome="success" if episode == 0 else "failure")
        build_replay_bundle(path, replay_root / f"warmup_{episode}.npz")

    actor_path = tmp_path / "actor.h5"
    actor = RLTDecisionWriter(
        actor_path,
        dimensions=dims,
        stage1_checkpoint=str(stage1),
        task="task",
        workflow_mode=1,
        collection_phase="actor",
    )
    for index in range(3):
        actor.append(**decision(index, dims, source="actor"))
    actor.close(outcome="success")
    build_replay_bundle(actor_path, replay_root / "actor.npz")

    progress = inspect_warmup_replays(replay_root, stage1_checkpoint=stage1, target=5)

    assert progress["count"] == 2
    assert progress["target"] == 5
    assert progress["complete"] is False
    assert len(progress["rejected"]) == 1


def test_warmup_writer_requires_mode1_and_reference_decisions(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    with pytest.raises(ValueError, match="requires Mode 1"):
        RLTDecisionWriter(
            tmp_path / "bad-mode.h5",
            dimensions=dims,
            stage1_checkpoint=str(tmp_path / "stage1.pt"),
            task="task",
            workflow_mode=2,
            collection_phase="warmup",
        )

    path = tmp_path / "bad-source.h5"
    writer = RLTDecisionWriter(
        path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
        collection_phase="warmup",
    )
    writer.append(**decision(0, dims, source="actor"))
    writer.append(**decision(1, dims, source="actor"))
    writer.close(outcome="success")
    with pytest.raises(ValueError, match="only Mode 1 reference"):
        build_replay_bundle(path, tmp_path / "bad-source.npz")


def test_reference_diagnostic_writer_is_mode1_only_and_never_builds_replay(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    with pytest.raises(ValueError, match="reference collection requires Mode 1"):
        RLTDecisionWriter(
            tmp_path / "bad-reference-mode.h5",
            dimensions=dims,
            stage1_checkpoint=str(tmp_path / "stage1.pt"),
            task="task",
            workflow_mode=2,
            collection_phase="reference",
        )

    path = tmp_path / "reference.h5"
    writer = RLTDecisionWriter(
        path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
        collection_phase="reference",
    )
    writer.append(**decision(0, dims, source="reference"))
    writer.append(**decision(1, dims, source="reference"))
    writer.close(outcome="success")
    with pytest.raises(ValueError, match="cannot be converted to Stage 2 replay"):
        build_replay_bundle(path, tmp_path / "reference.npz")


def test_builder_rejects_unfinalized_or_too_short_episode(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    path = tmp_path / "short.h5"
    writer = RLTDecisionWriter(
        path,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
    )
    writer.append(**decision(0, dims))
    writer.close(outcome="success")
    with pytest.raises(ValueError, match="at least two"):
        build_replay_bundle(path, tmp_path / "short.npz")

    unfinished = tmp_path / "unfinished.h5"
    writer = RLTDecisionWriter(
        unfinished,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
    )
    writer.append(**decision(0, dims))
    writer.file.flush()
    writer.file.close()
    with pytest.raises(ValueError, match="not finalized"):
        build_replay_bundle(unfinished, tmp_path / "unfinished.npz")


def test_decision_writer_rejects_wrong_shapes_and_non_finite_values(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    writer = RLTDecisionWriter(
        tmp_path / "bad.h5",
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=2,
    )
    values = decision(0, dims)
    values["z_rl"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="z_rl must have shape"):
        writer.append(**values)
    values = decision(0, dims)
    values["state"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        writer.append(**values)
    writer.abort()


def test_replay_trims_stationary_prefix_and_suffix(tmp_path):
    dims = RLTDimensions(z_dim=4, state_dim=2, action_steps=2, action_dim=2)
    decisions = tmp_path / "holds.h5"
    writer = RLTDecisionWriter(
        decisions,
        dimensions=dims,
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="task",
        workflow_mode=1,
    )
    states = [0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.4]
    for index, state_value in enumerate(states):
        values = decision(index, dims)
        values["state"] = np.full(dims.state_dim, state_value, dtype=np.float32)
        writer.append(**values)
    writer.close(outcome="success")
    output = tmp_path / "holds.npz"

    metadata = build_replay_bundle(decisions, output)
    arrays, _ = load_replay_bundle(output)

    assert metadata["trimmed_prefix_decisions"] == 1
    assert metadata["trimmed_suffix_decisions"] == 2
    assert metadata["transition_count"] == 4
    assert arrays["state"].shape[0] == 4


def test_actor_action_squash_preserves_units_and_bounds() -> None:
    raw = torch.tensor([[[-100.0, 0.5, 100.0]]])
    scale = torch.tensor([2.0, 3.0, 4.0])
    bounded = squash_actor_action(raw, scale)

    assert bounded.shape == raw.shape
    assert bounded[0, 0, 0] >= -2.0
    assert bounded[0, 0, 2] <= 4.0
    assert torch.allclose(bounded[0, 0, 1], torch.tensor(0.5), atol=0.005)


def test_offline_async_and_inference_share_one_actor_squash() -> None:
    import cobot_bridge.rlt_policy_server as inference
    import cobot_bridge.rlt_stage2_train as trainer
    from cobot_bridge import rlt_stage2_data as contract

    assert trainer.compose_actor_action is contract.compose_actor_action
    assert inference.compose_actor_action is contract.compose_actor_action


def test_zero_residual_reproduces_reference_exactly() -> None:
    reference = torch.tensor([[[0.2, -0.4, 1.1]]])
    composed = compose_actor_action(
        torch.zeros_like(reference), reference, torch.tensor([2.0, 2.0, 2.0])
    )

    torch.testing.assert_close(composed, reference, rtol=0, atol=0)
