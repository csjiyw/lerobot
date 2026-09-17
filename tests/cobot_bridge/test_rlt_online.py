import numpy as np
import pytest
import torch

from cobot_bridge.rlt_online import (
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
