"""Intervention-phase attribution from dashboard sidecar marks."""

import json

import numpy as np

from cobot_bridge.audit_rlt_replay import audit_replay
from cobot_bridge.rlt_stage2_data import RLTDecisionWriter, RLTDimensions, build_replay_bundle, load_replay_bundle


def make_episode(tmp_path, *, sources, times, phase_marks=None, raw_name="clip.raw.h5", sidecar_raw=None):
    path = tmp_path / "episode.decisions.h5"
    writer = RLTDecisionWriter(
        path,
        dimensions=RLTDimensions(4, 3, 30, 3),
        stage1_checkpoint=str(tmp_path / "stage1.pt"),
        task="align",
        workflow_mode=2,
    )
    writer.file.attrs.update({
        "rollout_speed_scale": 0.5,
        "sent_action_hz": 30.0,
        "raw_episode_path": str(tmp_path / raw_name),
    })
    for i, source in enumerate(sources):
        writer.append(
            z_rl=np.full(4, i), state=np.full(3, i), reference_actions=np.zeros((30, 3)),
            policy_actions=np.zeros((30, 3)), executed_actions=np.full((30, 3), i + 1),
            action_source=source, is_intervention=source == "human",
            bridge_sequence=i * 30, time_ns=int(1e9 + times[i] * 1e9), episode_frame_index=-1,
        )
    writer.close(outcome="success")
    if sidecar_raw is not None:
        (tmp_path / (raw_name + ".phases.jsonl")).write_text(sidecar_raw)
    elif phase_marks is not None:
        sidecar = tmp_path / (raw_name + ".phases.jsonl")
        sidecar.write_text("".join(
            json.dumps({"t_ns": int(1e9 + t * 1e9), "phase": phase}) + "\n" for t, phase in phase_marks
        ))
    replay = tmp_path / "episode.npz"
    build_replay_bundle(path, replay, trim_stationary=False)
    return replay


def test_phase_marks_attribute_human_rows(tmp_path):
    replay = make_episode(
        tmp_path,
        sources=("human", "human", "human", "actor"),
        times=(0.0, 2.5, 3.5, 5.0),
        phase_marks=[(1.0, "calibrate"), (3.0, "rescue")],
    )
    arrays, meta = load_replay_bundle(replay)
    # Transition rows cover the first three observations; the actor row keeps 0.
    assert arrays["intervention_phase"].reshape(-1).tolist() == [0, 1, 2]
    counts = meta["human_phase_counts"]
    assert counts == {"unknown": 1, "calibrate": 1, "rescue": 1, "retreat": 0}


def test_legacy_episode_without_sidecar_stays_unknown(tmp_path):
    replay = make_episode(tmp_path, sources=("human", "human", "actor"), times=(0.0, 2.5, 4.0))
    arrays, meta = load_replay_bundle(replay)
    assert arrays["intervention_phase"].reshape(-1).tolist() == [0, 0]
    assert meta["human_phase_counts"]["unknown"] == 2


def test_malformed_mark_lines_are_ignored(tmp_path):
    replay = make_episode(
        tmp_path,
        sources=("human", "human", "actor"),
        times=(2.0, 4.0, 5.0),
        sidecar_raw='not json\n{"phase": "retreat"}\n' + json.dumps({"t_ns": int(3e9), "phase": "retreat"}) + "\n",
    )
    # Row transition times are 3e9/5e9 ns; both follow the valid mark, and the
    # two malformed lines must have been skipped, not applied or fatal.
    arrays, _ = load_replay_bundle(replay)
    assert arrays["intervention_phase"].reshape(-1).tolist() == [3, 3]


def test_audit_totals_include_phases(tmp_path):
    make_episode(
        tmp_path,
        sources=("human", "human", "human"),
        times=(0.0, 2.5, 3.5),
        phase_marks=[(1.0, "calibrate"), (3.0, "rescue")],
    )
    report = audit_replay(tmp_path)
    totals = report["totals"]
    assert totals["phase_calibrate_human"] + totals["phase_rescue_human"] + totals["phase_unknown_human"] == 2


def test_post_episode_labels_apply_after_bundle_was_built(tmp_path):
    replay = make_episode(
        tmp_path, sources=("human", "human", "human", "actor"), times=(0.0, 2.5, 3.5, 5.0)
    )
    arrays, _ = load_replay_bundle(replay)
    assert arrays["intervention_phase"].reshape(-1).tolist() == [0, 0, 0]
    # The dashboard teach-label endpoint writes interval rows later in time;
    # attribution happens at load, so the baked copy must not shadow them.
    (tmp_path / "clip.raw.h5.phases.jsonl").write_text(json.dumps(
        {"t0_ns": int(2.4e9), "t1_ns": int(5.0e9), "phase": "rescue", "src": "teach_labels"}
    ) + "\n")
    arrays, _ = load_replay_bundle(replay)
    assert arrays["intervention_phase"].reshape(-1).tolist() == [0, 2, 2]
