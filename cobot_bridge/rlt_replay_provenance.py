"""Read-only recovery of replay provenance; never infer Actor from 'not human'.

Legacy online decisions have a variable clock (and sometimes omit the VLA
prefix). Row number times action horizon is not an episode frame index.
Unaligned human future chunks remain useful BC examples, not TD transitions.
"""

from pathlib import Path
import json

import h5py
import numpy as np


REFERENCE, ACTOR, HUMAN, UNKNOWN = 0, 1, 2, 3
PHASE_UNKNOWN, PHASE_CALIBRATE, PHASE_RESCUE, PHASE_RETREAT = 0, 1, 2, 3
PHASE_CODES = {"unknown": PHASE_UNKNOWN, "calibrate": PHASE_CALIBRATE, "rescue": PHASE_RESCUE, "retreat": PHASE_RETREAT}
PROVENANCE_VERSION = "source_and_clock_v1"
PROVENANCE_FIELDS = (
    "action_source", "episode_frame_index", "transition_duration_s",
    "discount_steps", "td_valid", "intervention_phase",
)


def _nearest(values, targets):
    right = np.clip(np.searchsorted(values, targets), 0, len(values) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(values[left] - targets) <= abs(values[right] - targets), left, right)


def human_phases(times, human, raw_name):
    """Attribute intervention phases from the episode's ``.phases.jsonl`` sidecar.

    Two record kinds are supported, both keyed on the same wall clock as the
    raw ``pc_timestamp_ns`` stream: point marks ``{"t_ns","phase"}`` written
    live from the dashboard, and interval marks ``{"t0_ns","t1_ns","phase"}``
    written by post-episode teach-segment labelling.  Intervals take priority
    over point marks.  Rows without any evidence keep PHASE_UNKNOWN; legacy
    episodes are never assigned a fabricated default.
    """
    phases = np.zeros(len(times), dtype=np.int64)
    if not raw_name:
        return phases
    sidecar = Path(str(raw_name)).with_name(Path(str(raw_name)).name + ".phases.jsonl")
    if not sidecar.is_file():
        return phases
    points = []
    intervals = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            code = PHASE_CODES.get(str(item.get("phase")), PHASE_UNKNOWN)
            if code == PHASE_UNKNOWN:
                continue
            if "t0_ns" in item and "t1_ns" in item:
                intervals.append((int(item["t0_ns"]), int(item["t1_ns"]), code))
            elif "t_ns" in item:
                points.append((int(item["t_ns"]), code))
        except (ValueError, TypeError, KeyError):
            continue
    for t0, t1, code in intervals:  # later file entries win on overlap
        phases[(times >= t0) & (times < t1)] = code
    if points:
        points.sort()
        mark_times = np.fromiter((m[0] for m in points), dtype=np.int64)
        codes = np.fromiter((m[1] for m in points), dtype=np.int64)
        idx = np.searchsorted(mark_times, times, side="right") - 1
        point_phases = np.where(idx >= 0, codes[np.clip(idx, 0, len(codes) - 1)], PHASE_UNKNOWN)
        unclaimed = phases == PHASE_UNKNOWN
        phases[unclaimed] = point_phases[unclaimed]
    phases[~human.astype(bool)] = PHASE_UNKNOWN
    return phases


def decision_provenance(source, start, end):
    """Return per-transition arrays plus an audit for decisions [start, end)."""
    count = end - start - 1
    ids = np.asarray(source["action_source"][start:end], dtype=np.int64)
    human = np.asarray(source["is_intervention"][start:end], dtype=bool)
    if not np.array_equal(ids == HUMAN, human):
        raise ValueError("decision human source and intervention mask disagree")
    reconstructed = bool(source.attrs.get("built_from_raw") or source.attrs.get("reconstruction"))
    # Old reconstruction scripts called *every* autonomous action 'actor'.
    # New writers explicitly certify that they retain unknown provenance.
    if reconstructed and not source.attrs.get("action_source_verified", False):
        ids[~human] = UNKNOWN
    frames = np.full(end - start, -1, dtype=np.int64)
    if "episode_frame_index" in source:
        frames[:] = source["episode_frame_index"][start:end]
    times = np.asarray(source["time_ns"][start:end], dtype=np.int64)
    clock = "decision_timestamp"
    raw_name = source.attrs.get("raw_episode_path") or source.attrs.get("raw_archive")
    if raw_name and Path(str(raw_name)).is_file():
        with h5py.File(str(raw_name), "r") as raw:
            raw_times = np.asarray(raw["frames/pc_timestamp_ns"], dtype=np.int64)
            seq = np.asarray(raw["frames/sequence"], dtype=np.int64)
            targets = np.asarray(source["bridge_sequence"][start:end], dtype=np.int64)
            if len(seq) and np.all(np.diff(seq) >= 0) and np.all(targets >= 0):
                indices = _nearest(seq, targets)
                aligned = np.abs(seq[indices] - targets) <= 2
                frames[aligned] = indices[aligned]
                times[aligned] = raw_times[indices[aligned]]
                clock = "raw_sequence" if aligned.all() else "partially_aligned_raw_sequence"
    # No row*chunk fallback: use the recorded origin, or keep frame unknown.
    origin = source.attrs.get("raw_recorder_first_frame_time_ns")
    if origin is not None:
        missing = frames < 0
        frames[missing] = np.maximum(0, np.rint((times[missing] - int(origin)) * 30 / 1e9)).astype(np.int64)
    duration = np.diff(times) / 1e9
    steps = np.ones(count, dtype=np.float32)
    valid = np.ones(count, dtype=bool)
    online = "rollout_speed_scale" in source.attrs
    known_clock = online or reconstructed
    horizon = int(source.attrs.get("execution_frame_span", source.attrs["action_steps"]))
    if known_clock:
        valid &= duration > 0
        # discount is defined per second (= 30 raw frames), not per inference.
        steps = np.maximum(duration, 1e-6).astype(np.float32)
    if reconstructed:
        # A skipped invalid raw window must never be bridged by a false TD edge.
        valid &= (frames[:-1] >= 0) & (np.diff(frames) == horizon)
    if online:
        hz = float(source.attrs.get("sent_action_hz", source.attrs.get("source_fps", 30.0)))
        if reconstructed and "execution_frame_span" in source.attrs:
            # The reconstruction already stores the physical span after
            # retiming (e.g. 60 raw frames for model30 at speed 0.5).  Applying
            # the speed factor a second time would incorrectly expect 4 s.
            expected = horizon / hz
        else:
            speed = float(source.attrs["rollout_speed_scale"])
            expected = int(source.attrs["action_steps"]) / (hz * speed)
        # Autonomy may include inference/communication latency, but cannot end
        # before execution, or bridge an arbitrarily long omitted interval.
        valid &= (duration >= expected - 1 / hz) & (duration <= expected + 1.0)
    if source.attrs.get("human_chunk_source") == "raw_executed_action_future":
        # Existing files do not record per-row padding/Teach release endpoints.
        # Keep all human BC targets; do not pretend these are verified TD edges.
        valid[human[:-1]] = False
    phases = human_phases(times, human, raw_name)
    arrays = {
        "action_source": ids[:-1, None].astype(np.float32),
        "episode_frame_index": frames[:-1, None].astype(np.float32),
        "transition_duration_s": duration[:, None].astype(np.float32),
        "discount_steps": steps[:, None],
        "td_valid": valid[:, None].astype(np.float32),
        "intervention_phase": phases[:-1, None].astype(np.float32),
    }
    audit = {
        "provenance_version": PROVENANCE_VERSION,
        "observation_clock": clock,
        "timing_checked": known_clock,
        "td_valid_transitions": int(valid.sum()),
        "bc_only_transitions": int((~valid).sum()),
        "unknown_frame_transitions": int((frames[:-1] < 0).sum()),
        "transition_action_source_counts": {
            name: int((ids[:-1] == value).sum())
            for name, value in (("reference", REFERENCE), ("actor", ACTOR), ("human", HUMAN), ("unknown", UNKNOWN))
        },
        "human_phase_counts": {
            name: int(((ids[:-1] == HUMAN) & (phases[:-1] == value)).sum())
            for name, value in PHASE_CODES.items()
        },
    }
    return arrays, audit


def enrich_replay_provenance(arrays, metadata, replay_path):
    """Backfill old bundles in memory, without changing recorded tensors/files."""
    count = len(arrays["z_rl"])
    complete = all(name in arrays for name in PROVENANCE_FIELDS)
    # Prefer adjacent decision files after relocation, then the recorded path.
    candidates = [Path(replay_path).with_suffix(".decisions.h5")]
    if metadata.get("decision_path"):
        candidates.append(Path(metadata["decision_path"]))
    decision = next((p for p in candidates if p.is_file()), None)
    if decision is not None:
        with h5py.File(decision, "r") as source:
            start = int(metadata.get("selected_decision_start", 0))
            if start + count + 1 > len(source["z_rl"]):
                raise ValueError(f"decision/replay length mismatch: {decision}")
            # Avoid attaching provenance from a stale file with the same name.
            if not np.allclose(source["state"][start:start + count], arrays["state"], rtol=1e-5, atol=1e-6):
                raise ValueError(f"decision/replay state mismatch: {decision}")
            recovered, audit = decision_provenance(source, start, start + count + 1)
        if complete:
            # Post-episode teach labels may have arrived after the bundle was
            # built; re-attribute phases without trusting the baked copy.
            arrays["intervention_phase"] = recovered["intervention_phase"]
        else:
            arrays.update(recovered)
        metadata.update(audit)
        if metadata.get("reward_source") == "recap_potential_v1":
            arrays["discount_steps"][:] = 1.0
        return
    if complete:
        return
    arrays.update({
        "action_source": np.where(arrays["is_intervention"].astype(bool), HUMAN, UNKNOWN).astype(np.float32),
        "episode_frame_index": np.full((count, 1), -1, dtype=np.float32),
        "transition_duration_s": np.full((count, 1), -1, dtype=np.float32),
        "discount_steps": np.ones((count, 1), dtype=np.float32),
        "td_valid": np.ones((count, 1), dtype=np.float32),
        "intervention_phase": np.zeros((count, 1), dtype=np.float32),
    })
    metadata["provenance_warning"] = "No decision evidence; legacy TD clock retained; autonomous actor gate fails closed."
    # An explicitly recorded IL stride is trustworthy; a default chunk is not.
    if metadata.get("collection_phase") == "il_demo" and metadata.get("decision_stride_frames"):
        stride = int(metadata["decision_stride_frames"])
        arrays["episode_frame_index"][:, 0] = np.arange(count) * stride
    if metadata.get("reward_source") == "recap_potential_v1":
        # Existing shaping used a fixed gamma. Do not change its discount
        # without recomputing the potential rewards from the teacher trace.
        arrays["discount_steps"][:] = 1.0


def actor_training_mask(arrays, actor_start_step):
    """Explicit Actor/human rows bypass a deployment-only fixed-frame gate."""
    count = len(arrays["z_rl"])
    if actor_start_step is None:
        return np.ones(count, dtype=bool)
    source = arrays["action_source"].reshape(-1)
    human = arrays["is_intervention"].reshape(-1).astype(bool)
    frames = arrays["episode_frame_index"].reshape(-1)
    return (source == ACTOR) | human | ((frames >= 0) & (frames >= actor_start_step))
