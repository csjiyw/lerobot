from pathlib import PurePosixPath

import pytest

from cobot_bridge.remote_policy_server import _validated_policy_id


def test_policy_id_is_confined_to_remote_checkpoint_root():
    assert _validated_policy_id("run/checkpoints/020000/pretrained_model") == PurePosixPath(
        "run/checkpoints/020000/pretrained_model"
    )
    with pytest.raises(ValueError):
        _validated_policy_id("../../outside")
    with pytest.raises(ValueError):
        _validated_policy_id("/absolute")
