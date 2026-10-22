import json

import h5py
import numpy as np
import pytest
import torch

from cobot_bridge.audit_rlt_replay import audit_replay
from cobot_bridge.build_rlt_replay_from_decisions import build_all
from cobot_bridge.rlt_replay_provenance import PROVENANCE_FIELDS, actor_training_mask
from cobot_bridge.rlt_stage2_data import RLTDecisionWriter, RLTDimensions, build_replay_bundle, load_replay_bundle
from cobot_bridge.rlt_stage2_train import _actor_bc_target_and_mask, _masked_action_mse, _valid_critic_mse, load_replay_dataset


def make_bundle(tmp_path, *, sources=("actor", "human", "human"), times=(0, 2.5, 2.66), frames=(-1, -1, -1), attrs=None):
    path = tmp_path / "episode.decisions.h5"
    writer = RLTDecisionWriter(path, dimensions=RLTDimensions(4, 3, 30, 3), stage1_checkpoint=str(tmp_path / "stage1.pt"), task="align", workflow_mode=2)
    if attrs:
        writer.file.attrs.update(attrs)
    for i, source in enumerate(sources):
        writer.append(z_rl=np.full(4, i), state=np.full(3, i), reference_actions=np.zeros((30, 3)), policy_actions=np.zeros((30, 3)), executed_actions=np.full((30, 3), i + 1), action_source=source, is_intervention=source == "human", bridge_sequence=i * 30, time_ns=int(1e9 + times[i] * 1e9), episode_frame_index=frames[i])
    writer.close(outcome="success")
    replay = tmp_path / "episode.npz"
    build_replay_bundle(path, replay, trim_stationary=False)
    return replay


def test_actor_and_human_survive_gate_with_omitted_prefix(tmp_path):
    replay = make_bundle(tmp_path)
    arrays, _ = load_replay_bundle(replay)
    assert actor_training_mask(arrays, 780).tolist() == [True, True]
    assert arrays["episode_frame_index"].reshape(-1).tolist() == [-1, -1]


def test_reference_uses_real_frame_not_row_number(tmp_path):
    replay = make_bundle(tmp_path, sources=("reference",) * 3, frames=(610, 810, 830))
    arrays, _ = load_replay_bundle(replay)
    assert actor_training_mask(arrays, 780).tolist() == [False, True]


def test_unknown_frame_reference_gate_fails_closed(tmp_path):
    replay = make_bundle(tmp_path, sources=("reference",) * 3)
    arrays, _ = load_replay_bundle(replay)
    assert not actor_training_mask(arrays, 1).any()
    assert actor_training_mask(arrays, None).all()


def test_reconstruction_does_not_claim_actor_and_detects_skipped_window(tmp_path):
    replay = make_bundle(tmp_path, sources=("actor",) * 3, frames=(0, 30, 90), times=(0, 1, 3), attrs={"built_from_raw": "10tick_contiguous_30hz_v1"})
    arrays, meta = load_replay_bundle(replay)
    assert arrays["action_source"].reshape(-1).tolist() == [3, 3]
    assert arrays["td_valid"].reshape(-1).tolist() == [1, 0]
    assert meta["transition_action_source_counts"]["actor"] == 0


def test_human_future_chunks_keep_bc_but_not_false_td(tmp_path):
    replay = make_bundle(tmp_path, attrs={"rollout_speed_scale": 0.5, "sent_action_hz": 30.0, "human_chunk_source": "raw_executed_action_future"})
    arrays, meta = load_replay_bundle(replay)
    assert arrays["td_valid"].reshape(-1).tolist() == [1, 0]
    assert arrays["rewards"].reshape(-1).tolist() == [0, 1]
    assert meta["outcome"] == "success"
    batch = {k: torch.as_tensor(v) for k, v in arrays.items()}
    target, supervised = _actor_bc_target_and_mask(batch)
    assert supervised.all()
    torch.testing.assert_close(target[1], batch["actions"][1])
    torch.testing.assert_close(target[0], batch["ref_actions"][0])


def test_partial_human_chunk_only_clones_owned_timesteps(tmp_path):
    replay = make_bundle(tmp_path)
    arrays, _ = load_replay_bundle(replay)
    arrays["action_human_mask"][1] = 0
    arrays["action_human_mask"][1, 10:20] = 1
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}

    target, supervised = _actor_bc_target_and_mask(batch)

    assert supervised.all()
    torch.testing.assert_close(target[1, :10], batch["ref_actions"][1, :10])
    torch.testing.assert_close(target[1, 10:20], batch["actions"][1, 10:20])
    torch.testing.assert_close(target[1, 20:], batch["ref_actions"][1, 20:])


def test_mc_discount_uses_duration_and_requested_gamma(tmp_path):
    replay = make_bundle(tmp_path, attrs={"rollout_speed_scale": 0.5})
    arrays, _ = load_replay_dataset([replay], stage1_checkpoint=tmp_path / "stage1.pt", discount=0.9)
    np.testing.assert_allclose(arrays["mc_returns"].reshape(-1), [0.9 ** 2.5, 1.0])


def test_raw_sequence_recovers_omitted_prefix_and_observation_clock(tmp_path):
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        f.create_dataset("frames/sequence", data=np.arange(-800, 101))
        f.create_dataset("frames/pc_timestamp_ns", data=(1e9 + np.arange(901) * 1e9 / 30).astype(np.int64))
    replay = make_bundle(tmp_path, sources=("reference",) * 3, attrs={"raw_episode_path": str(raw)})
    arrays, meta = load_replay_bundle(replay)
    np.testing.assert_array_equal(arrays["episode_frame_index"].reshape(-1), [800, 830])
    np.testing.assert_allclose(arrays["transition_duration_s"], 1.0)
    assert meta["observation_clock"] == "raw_sequence"
    assert actor_training_mask(arrays, 780).all()


def test_legacy_backfill_is_read_only(tmp_path):
    replay = make_bundle(tmp_path)
    with np.load(replay) as f:
        payload = {k: f[k] for k in f.files if k not in PROVENANCE_FIELDS}
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **payload)
    before = legacy.read_bytes()
    arrays, _ = load_replay_bundle(legacy)
    assert actor_training_mask(arrays, 780).all()
    assert legacy.read_bytes() == before


def test_recap_shaping_retains_original_discount_contract(tmp_path):
    replay = make_bundle(tmp_path, attrs={"rollout_speed_scale": 0.5})
    shaped = tmp_path / "shaped.npz"
    build_replay_bundle(tmp_path / "episode.decisions.h5", shaped, trim_stationary=False, recap_values=np.array([0.1, 0.2, 0.3]))
    arrays, _ = load_replay_bundle(shaped)
    np.testing.assert_array_equal(arrays["discount_steps"], 1.0)


def test_sample_only_weighted_mse_has_elementwise_denominator():
    prediction = torch.ones(2, 30, 7, requires_grad=True)
    loss = _masked_action_mse(prediction, torch.zeros_like(prediction), torch.tensor([True, True]), sample_weights=torch.tensor([1.0, 5.0]))
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_invalid_critic_rows_have_no_gradient():
    prediction = torch.tensor([[2.0], [100.0]], requires_grad=True)
    loss = _valid_critic_mse(prediction, torch.zeros_like(prediction), torch.tensor([[1], [0]]))
    loss.backward()
    assert loss.item() == 4.0
    assert prediction.grad.reshape(-1).tolist() == [4.0, 0.0]


def test_all_invalid_critic_batch_is_finite_and_differentiable():
    prediction = torch.ones(2, 1, requires_grad=True)
    loss = _valid_critic_mse(prediction, prediction.detach(), torch.zeros(2, 1))
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad.eq(0).all()


def test_audit_copy_preserves_tensors_and_success_not_autonomy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    replay = make_bundle(source, attrs={"rollout_speed_scale": 0.5, "human_chunk_source": "raw_executed_action_future"})
    before = replay.read_bytes()
    report = audit_replay(source, output_root=tmp_path / "copy")
    assert report["totals"]["human_assisted_success_episodes"] == 1
    assert report["totals"]["actor_without_human_success_episodes"] == 0
    original, _ = load_replay_bundle(replay)
    copied, _ = load_replay_bundle(tmp_path / "copy" / replay.name)
    for key in original:
        np.testing.assert_array_equal(original[key], copied[key])
    assert before == replay.read_bytes()
    with pytest.raises(FileExistsError):
        audit_replay(source, output_root=tmp_path / "copy")


def test_build_all_from_finalized_decisions_requires_fresh_output(tmp_path):
    source = tmp_path / "decisions"
    source.mkdir()
    make_bundle(source)

    manifest = build_all(source, tmp_path / "bundles")

    assert manifest["episodes"] == 1
    assert manifest["transitions"] == 2
    assert (tmp_path / "bundles" / "episode.npz").is_file()
    with pytest.raises(FileExistsError):
        build_all(source, tmp_path / "bundles")
