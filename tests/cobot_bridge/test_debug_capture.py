from pathlib import Path
import json

import h5py
import numpy as np
import pytest

from cobot_bridge.debug_capture import publish_completed_debug_record


def _record(path: Path, *, completed: bool, frames: int) -> None:
    with h5py.File(path, "w") as raw:
        raw.attrs["completed"] = completed
        raw.attrs["num_frames"] = frames


def test_completed_debug_capture_atomically_replaces_only_latest(tmp_path):
    root = tmp_path / "debug_rollouts"
    root.mkdir()
    latest = root / "latest.raw.h5"
    latest.write_bytes(b"previous-complete-record")
    candidate = root / ".inflight.raw.h5.partial"
    _record(candidate, completed=True, frames=15)

    published = publish_completed_debug_record(
        candidate,
        root,
        metadata={"sampled_fps": 15, "motion_kind": "policy_execute", "job_id": "debug-1"},
    )

    assert published == latest
    assert not candidate.exists()
    with h5py.File(latest, "r") as raw:
        assert raw.attrs["completed"]
        assert raw.attrs["debug_record"]
        assert raw.attrs["debug_sampled_fps"] == 15
    assert '"frames": 15' in (root / "latest.json").read_text()


def test_incomplete_capture_does_not_replace_previous_latest(tmp_path):
    root = tmp_path / "debug_rollouts"
    root.mkdir()
    latest = root / "latest.raw.h5"
    latest.write_bytes(b"previous-complete-record")
    candidate = root / ".inflight.raw.h5.partial"
    _record(candidate, completed=False, frames=15)

    with pytest.raises(RuntimeError, match="did not complete"):
        publish_completed_debug_record(
            candidate,
            root,
            metadata={"sampled_fps": 15, "motion_kind": "policy_execute", "job_id": "debug-2"},
        )

    assert latest.read_bytes() == b"previous-complete-record"
    assert candidate.exists()


def test_debug_capture_attaches_exact_policy_decisions_by_bridge_sequence(tmp_path):
    root = tmp_path / "debug_rollouts"
    root.mkdir()
    candidate = root / ".inflight.raw.h5.partial"
    with h5py.File(candidate, "w") as raw:
        raw.attrs["completed"] = True
        raw.attrs["num_frames"] = 3
        frames = raw.create_group("frames")
        frames.create_dataset("sequence", data=np.asarray([10, 20, 30], dtype=np.int64))
    trace = root / "policy.jsonl"
    decisions = []
    for sequence in (18, 31):
        chunk = np.full((2, 7), float(sequence), dtype=np.float32).tolist()
        decisions.append(
            {
                "format": "cobot_magic_policy_decisions_v1",
                "time_ns": sequence * 100,
                "bridge_time_ns": sequence * 10,
                "bridge_sequence": sequence,
                "model_state": [0.0] * 7,
                "model_native_action_chunk": chunk,
                "model_absolute_action_chunk": chunk,
                "rollout_action_chunk": chunk,
                "model_output_semantics": "relative_eef",
                "action_space": "eef_target",
                "consumed_actions": 2,
                "latency_ms": 123.4,
                "rtc_prefix_steps": 1,
            }
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in decisions))

    published = publish_completed_debug_record(
        candidate,
        root,
        metadata={
            "sampled_fps": 15,
            "motion_kind": "policy_execute",
            "job_id": "debug-trace",
            "policy_decision_trace": str(trace),
        },
    )

    with h5py.File(published, "r") as raw:
        assert raw.attrs["policy_decision_trace_status"] == "attached"
        assert raw.attrs["policy_decision_count"] == 2
        decisions = raw["policy_decisions"]
        assert decisions.attrs["model_output_semantics"] == "relative_eef"
        assert decisions.attrs["action_space"] == "eef_target"
        assert decisions["model_native_action_chunk"].shape == (2, 2, 7)
        assert decisions["raw_frame_index"][:].tolist() == [1, 2]
        assert decisions["raw_sequence_error"][:].tolist() == [2, -1]
