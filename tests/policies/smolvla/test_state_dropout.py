import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import _drop_state_features


def test_state_dropout_drops_complete_samples() -> None:
    state_features = torch.ones(1024, 1, 64)

    torch.manual_seed(0)
    result, fraction = _drop_state_features(state_features, probability=0.5, training=True)

    rows_are_kept_or_dropped = torch.logical_or(
        torch.all(result == 1, dim=(1, 2)), torch.all(result == 0, dim=(1, 2))
    )
    assert torch.all(rows_are_kept_or_dropped)
    assert 0.4 < fraction.item() < 0.6


def test_state_dropout_is_disabled_during_inference() -> None:
    state_features = torch.randn(4, 1, 64)

    result, fraction = _drop_state_features(state_features, probability=1.0, training=False)

    torch.testing.assert_close(result, state_features)
    assert fraction.item() == 0.0


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_state_dropout_probability_is_validated(probability: float) -> None:
    with pytest.raises(ValueError, match="state_dropout"):
        SmolVLAConfig(state_dropout=probability)
