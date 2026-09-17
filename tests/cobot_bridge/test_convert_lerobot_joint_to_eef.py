import json

import numpy as np

from cobot_bridge.convert_lerobot_joint_to_eef import _update_episode_metadata


def test_update_episode_metadata_rewrites_v21_jsonl_stats(tmp_path):
    metadata = tmp_path / "meta"
    metadata.mkdir()
    episodes_stats = metadata / "episodes_stats.jsonl"
    episodes_stats.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "stats": {
                            "observation.state": {
                                "mean": [99.0, 99.0],
                                "std": [99.0, 99.0],
                            },
                            "untouched": {"mean": [episode_index]},
                        },
                    }
                )
                for episode_index in (0, 1)
            ]
        )
        + "\n"
    )
    episode_values = {
        0: {"observation.state": [np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]},
        1: {"observation.state": [np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]},
    }

    _update_episode_metadata(tmp_path, episode_values, ["observation.state"])

    rows = [json.loads(line) for line in episodes_stats.read_text().splitlines()]
    assert rows[0]["stats"]["observation.state"]["mean"] == [2.0, 3.0]
    assert rows[1]["stats"]["observation.state"]["mean"] == [6.0, 7.0]
    assert rows[0]["stats"]["observation.state"]["count"] == [2]
    assert rows[0]["stats"]["untouched"] == {"mean": [0]}
