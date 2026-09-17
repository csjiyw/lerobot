from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from cobot_bridge.reconstruct_rlt_replay import (
    build_parser,
    decision_indices,
    first_log_timestamp_ns,
    match_policy_log_offline,
    nearest_indices,
    parse_chunk_completion_times,
)
from cobot_bridge.reencode_rlt_online_replay import (
    contiguous_indices_at_times,
    parse_online_decision_log,
)


def test_initial_reconstruction_selection_excludes_actor_rollouts(tmp_path) -> None:
    # Keep this naming contract explicit: Actor-generated rollouts belong to
    # subsequent online iterations, not to the initial offline RL update.
    for name in (
        "episode_20260812_174120_110575.raw.h5",
        "rlt_warmup_mode1_20260812_162154_941720.raw.h5",
        "rlt_actor_mode1_20260812_203727_684391.raw.h5",
    ):
        (tmp_path / name).touch()
    selected = sorted(tmp_path.glob("episode_*.raw.h5"))
    assert [path.name for path in selected] == [
        "episode_20260812_174120_110575.raw.h5",
    ]


def test_parser_can_force_persistent_logs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--raw-root", "raw",
            "--output-root", "out",
            "--stage1-checkpoint", "stage1.pt",
            "--job-log-root", "logs",
            "--prefer-job-logs",
        ]
    )
    assert args.prefer_job_logs is True


def test_parse_chunk_completion_times() -> None:
    text = """
2026-08-12 18:56:36,293 [INFO] received action chunk size=30 consuming=30
noise
2026-08-12 18:56:37,408 [INFO] received action chunk size=30 consuming=30
"""
    values = parse_chunk_completion_times(text, timezone="Asia/Shanghai")
    expected = datetime(2026, 8, 12, 18, 56, 36, 293000, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert values[0] == int(expected.replace(microsecond=0).timestamp()) * 1_000_000_000 + 293_000_000
    assert np.diff(values).tolist() == [1_115_000_000]


def test_nearest_indices_chooses_closest_sample() -> None:
    timestamps = np.asarray([0, 10, 20, 30], dtype=np.int64)
    assert nearest_indices(timestamps, np.asarray([4, 6, 25, 31])).tolist() == [0, 1, 2, 3]


def test_decision_indices_uses_log_timing_and_keeps_complete_tail() -> None:
    timestamps = np.arange(0, 4_000_000_001, 33_333_333, dtype=np.int64)
    completions = np.asarray([200_000_000, 1_310_000_000, 2_420_000_000, 3_530_000_000])
    observations, actions, latency_ms = decision_indices(
        timestamps,
        completions,
        source_fps=30,
        action_steps=10,
        action_stride=1,
    )
    assert len(observations) == 4
    assert actions.shape == (4, 10)
    assert 100 <= latency_ms <= 120
    assert np.all(np.diff(actions, axis=1) == 1)


def test_nearest_indices_can_preserve_duplicate_samples() -> None:
    timestamps = np.asarray([0, 100, 200], dtype=np.int64)
    assert nearest_indices(timestamps, np.asarray([1, 2, 199])).tolist() == [0, 0, 2]


def test_match_policy_log_offline_uses_exact_record_and_mode1_log(tmp_path) -> None:
    raw = tmp_path / "episode_20260812_185635_026388.raw.h5"
    raw.touch()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "record_record.log").write_text(
        "2026-08-12 18:56:35,079 [INFO] raw recording started: " + str(raw) + ".partial\n"
    )
    policy = logs / "policy_policy_execute.log"
    policy.write_text(
        "2026-08-12 18:56:36,148 [INFO] policy_server={'ready': True}\n"
        "2026-08-12 18:56:36,149 [INFO] mode_result={'effective_mode': 'POLICY_PUPPET'}\n"
        "2026-08-12 18:56:36,293 [INFO] received action chunk size=30 consuming=30\n"
    )
    assert match_policy_log_offline(
        raw, logs, timezone="Asia/Shanghai", max_start_delta_s=3.0
    ) == policy.resolve()


def test_first_log_timestamp_ns_is_exact() -> None:
    value = first_log_timestamp_ns(
        "2026-08-12 18:56:36,293 [INFO] message", timezone="Asia/Shanghai"
    )
    expected = datetime(2026, 8, 12, 18, 56, 36, 293000, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert value == int(expected.replace(microsecond=0).timestamp()) * 1_000_000_000 + 293_000_000


def test_parse_rlt_online_log_and_contiguous_30hz_rows() -> None:
    entries = parse_online_decision_log(
        "2026-08-13 17:24:10,690 [INFO] decision=1 source=actor actor_ready=True\n"
        "2026-08-13 17:24:11,133 [INFO] decision=2 source=human actor_ready=True\n",
        timezone="Asia/Shanghai",
    )
    assert [(item.number, item.source) for item in entries] == [(1, "actor"), (2, "human")]

    timestamps = np.arange(0, 1_000_000_000, 33_333_333, dtype=np.int64)
    rows = contiguous_indices_at_times(
        timestamps,
        np.asarray([0, 333_333_330], dtype=np.int64),
        action_steps=4,
        action_fps=30,
    )
    assert rows.tolist() == [[0, 1, 2, 3], [10, 11, 12, 13]]
