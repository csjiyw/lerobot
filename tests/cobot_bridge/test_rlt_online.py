import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cobot_bridge.rlt_online import (
    DecisionTakeoverGate,
    _actor_control_allowed,
    _is_mode2_intervention_boundary,
    _raise_for_rejected_action,
    _requested_outcome,
    normalize_action_chunk,
    unnormalize_action_chunk,
)
from cobot_bridge.remote_robot import ControlNotReady, RESYNC_REQUIRED, RLT_SYNC, TEACH


class FakeNormalizer:
    def _normalize_action(self, values, inverse=False):
        assert inverse is False
        return (values - 1.0) / 2.0


class FakePostprocessor:
    def __call__(self, values):
        return values * 2.0 + 1.0


class FakeObservation:
    def __init__(self, *, teach_active=False, effective_mode=RLT_SYNC):
        self.teach_active = teach_active
        self.effective_mode = effective_mode


def test_human_chunk_uses_same_action_normalization_round_trip():
    raw = np.arange(21, dtype=np.float32).reshape(3, 7) * 0.01
    normalized = normalize_action_chunk(FakeNormalizer(), raw, torch.device("cpu"))
    restored = unnormalize_action_chunk(FakePostprocessor(), normalized.unsqueeze(0))
    np.testing.assert_allclose(restored, raw, atol=1e-6)


def test_action_chunk_rejects_non_seven_dimensional_output():
    with pytest.raises(RuntimeError, match="invalid unnormalized"):
        unnormalize_action_chunk(FakePostprocessor(), torch.zeros(1, 3, 6))


def test_requested_outcome_survives_immediate_safe_transition(tmp_path):
    result = tmp_path / "episode.result"
    assert _requested_outcome(result) is None
    result.write_text("failure")
    assert _requested_outcome(result) == "failure"


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (FakeObservation(teach_active=True, effective_mode=TEACH), None),
        (FakeObservation(effective_mode=RESYNC_REQUIRED), None),
        (FakeObservation(), "teach_active"),
        (FakeObservation(), "resync_required"),
    ],
)
def test_mode2_accepts_teach_as_normal_intervention_boundary(observation, reason):
    assert _is_mode2_intervention_boundary(2, observation, rejection_reason=reason)


def test_mode1_never_accepts_teach_as_intervention_boundary():
    observation = FakeObservation(teach_active=True, effective_mode=TEACH)
    assert not _is_mode2_intervention_boundary(
        1, observation, rejection_reason="teach_active"
    )


def test_bridge_control_transition_rejection_is_typed():
    with pytest.raises(ControlNotReady, match="teach_active"):
        _raise_for_rejected_action("action chunk", {"accepted": False, "reason": "teach_active"})


def test_non_control_rejection_remains_fatal():
    with pytest.raises(RuntimeError, match="invalid_action"):
        _raise_for_rejected_action("action chunk", {"accepted": False, "reason": "invalid_action"})


def test_decision_gate_waits_for_complete_reference_chunks():
    gate = DecisionTakeoverGate.__new__(DecisionTakeoverGate)
    gate._needed = 2
    gate._seen = 0
    gate.latched = False
    gate.last_position = None

    assert gate.update(None) is False
    assert gate.update(None) is False
    assert gate.update(None) is True
    assert gate.update(None) is True


def test_decision_gate_opens_after_eight_reference_chunks():
    gate = DecisionTakeoverGate.__new__(DecisionTakeoverGate)
    gate._needed = 8
    gate._seen = 0
    gate.latched = False
    gate.last_position = None

    assert all(gate.update(None) is False for _ in range(8))
    assert gate.update(None) is True


def test_actor_decision_budget_stops_after_configured_exposure():
    assert _actor_control_allowed(True, 0, 1)
    assert not _actor_control_allowed(True, 1, 1)
    assert _actor_control_allowed(True, 100, 0)
    assert not _actor_control_allowed(False, 0, 3)


def test_aborted_episode_is_quarantined_without_orphans(tmp_path):
    from argparse import Namespace

    from cobot_bridge.rlt_online import _quarantine_aborted_episode

    batch = tmp_path / "masterpiece_online_v7"
    batch.mkdir()
    stem = "rlt_actor_mode2_20260918_181500_123456"
    raw = batch / f"{stem}.raw.h5"
    args = Namespace(
        raw_output=raw,
        decision_output=batch / f"{stem}.decisions.h5",
        replay_output=tmp_path / "replay" / f"{stem}.npz",
        result_file=batch / f"{stem}.raw.h5.result",
        recorder_ready_file=batch / f"{stem}.raw.h5.ready",
    )
    leftovers = [
        raw,
        raw.with_name(raw.name + ".partial"),
        args.recorder_ready_file,
        Path(f"{raw}.phases.jsonl"),
        args.decision_output,
        args.replay_output,
        args.result_file,
    ]
    for path in leftovers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    _quarantine_aborted_episode(args)

    assert list(batch.rglob("*.h5")) == [], "the bin must stay invisible to raw scans"
    assert list(batch.glob("*.phases.jsonl")) == []
    bin_dir = batch / "_discarded" / stem
    moved = sorted(path.name for path in bin_dir.iterdir())
    assert moved == [
        "abort_manifest.json",
        f"{stem}.decisions.h5.aborted",
        f"{stem}.npz.aborted",
        f"{stem}.raw.h5.aborted",
        f"{stem}.raw.h5.partial.aborted",
        f"{stem}.raw.h5.phases.jsonl.aborted",
        f"{stem}.raw.h5.ready.aborted",
        f"{stem}.raw.h5.result.aborted",
    ]
    assert json.loads((bin_dir / "abort_manifest.json").read_text())["episode"] == stem
