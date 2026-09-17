import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from cobot_bridge.policy_server import _processor_uses_relative_eef_actions, _restore_relative_eef_chunk
from lerobot.policies.smolvla.processor_smolvla import SmolVLARelativeEEFChunkProcessor
from lerobot.policies.smolvla.relative_eef import chunk_to_relative, relative_chunk_to_absolute
from lerobot.processor import TransitionKey


def test_groot_relative_eef_chunk_round_trip():
    state = np.asarray([0.3, 0.1, 0.2, 0.4, -0.2, 0.3, 0.08], dtype=np.float32)
    targets = np.asarray(
        [
            [0.31, 0.12, 0.19, 0.5, -0.1, 0.2, 0.05],
            [0.34, 0.08, 0.17, -0.2, 0.6, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    relative = chunk_to_relative(state, targets)
    restored = relative_chunk_to_absolute(state, relative)
    np.testing.assert_allclose(restored, targets, atol=1e-6)


def test_policy_server_restores_relative_eef_chunk():
    class RelativeStep:
        _registry_name = "smolvla_relative_eef_chunk"

    class Preprocessor:
        steps = [RelativeStep()]

    state = np.asarray([0.32, -0.08, 0.24, 0.1, -0.2, 0.3, 0.05], dtype=np.float32)
    targets = np.asarray(
        [
            [0.33, -0.07, 0.23, 0.11, -0.18, 0.29, 0.04],
            [0.35, -0.06, 0.25, 0.09, -0.17, 0.31, 0.02],
        ],
        dtype=np.float32,
    )
    relative = chunk_to_relative(state, targets)
    restored = _restore_relative_eef_chunk(Preprocessor(), state, relative)
    np.testing.assert_allclose(restored, targets, atol=1e-6)


def test_policy_server_only_restores_when_loaded_processor_is_relative():
    class AbsoluteStep:
        _registry_name = "normalizer_processor"

    class Preprocessor:
        steps = [AbsoluteStep()]

    state = np.zeros(7, dtype=np.float32)
    chunk = np.ones((2, 7), dtype=np.float32)
    assert not _processor_uses_relative_eef_actions(Preprocessor())
    np.testing.assert_array_equal(_restore_relative_eef_chunk(Preprocessor(), state, chunk), chunk)


def test_processor_uses_one_state_for_the_complete_chunk():
    state = torch.tensor([[0.3, 0.1, 0.2, 0.4, -0.2, 0.3, 0.08]])
    actions = torch.tensor(
        [[[0.31, 0.12, 0.19, 0.5, -0.1, 0.2, 0.05], [0.34, 0.08, 0.17, -0.2, 0.6, 0.4, 0.0]]]
    )
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": state},
        TransitionKey.ACTION: actions,
    }
    converted = SmolVLARelativeEEFChunkProcessor()(transition)[TransitionKey.ACTION]
    expected = chunk_to_relative(state[0].numpy(), actions[0].numpy())
    np.testing.assert_allclose(converted[0].numpy(), expected, atol=1e-6)


def test_pretrained_relative_eef_factory_keeps_anchor_relative_action_pipeline(tmp_path):
    """The fine-tuning factory must not reload the base absolute-action pipeline."""
    relative_stats_path = tmp_path / "relative_stats.json"
    relative_stats_path.write_text(
        '{"action":{"mean":[0.01,-0.02,0.03,0.04,-0.05,0.06,0.07],'
        '"std":[0.1,0.1,0.1,0.2,0.2,0.2,0.03]}}'
    )
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    config = SmolVLAConfig(
        input_features={"observation.state": features["observation.state"]},
        output_features={"action": features["action"]},
        use_relative_eef_actions=True,
        relative_eef_action_stats_path=str(relative_stats_path),
        device="cpu",
    )
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        # Deliberately unmistakable absolute-action statistics. The pipeline
        # must replace these with the JSON relative action statistics.
        "action": {"mean": torch.full((7,), 99.0), "std": torch.full((7,), 77.0)},
    }
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path="base-model-path-is-not-loaded-for-relative-eef",
        dataset_stats=dataset_stats,
    )
    assert any(step._registry_name == "smolvla_relative_eef_chunk" for step in preprocessor.steps)
    normalizer = next(step for step in preprocessor.steps if step._registry_name == "normalizer_processor")
    np.testing.assert_allclose(
        np.asarray(normalizer.stats["action"]["mean"]),
        [0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.07],
    )

    state = torch.tensor([[0.3, 0.1, 0.2, 0.4, -0.2, 0.3, 0.08]])
    targets = torch.tensor([[[0.31, 0.12, 0.19, 0.5, -0.1, 0.2, 0.05]]])
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": state},
        TransitionKey.ACTION: targets,
    }
    relative_step = next(step for step in preprocessor.steps if step._registry_name == "smolvla_relative_eef_chunk")
    converted = relative_step(transition)[TransitionKey.ACTION]
    np.testing.assert_allclose(converted[0].numpy(), chunk_to_relative(state[0].numpy(), targets[0].numpy()), atol=1e-6)
