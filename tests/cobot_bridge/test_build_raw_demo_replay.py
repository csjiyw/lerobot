import numpy as np

from cobot_bridge.build_raw_demo_replay import complete_demo_rows


def test_complete_demo_rows_keeps_only_full_valid_chunks() -> None:
    learning = np.zeros(50, dtype=bool)
    learning[5:45] = True
    valid = np.ones(50, dtype=bool)
    valid[17] = False

    rows, chunks = complete_demo_rows(
        learning,
        valid,
        action_steps=4,
        action_stride=1,
        decision_stride_frames=10,
    )

    assert rows.tolist() == [5, 19, 29, 39]
    assert chunks.tolist() == [
        [5, 6, 7, 8],
        [19, 20, 21, 22],
        [29, 30, 31, 32],
        [39, 40, 41, 42],
    ]
