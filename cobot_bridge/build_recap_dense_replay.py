#!/usr/bin/env python3
"""Create immutable 10-tick RLT replays with fixed RECAP potential rewards.

This is deliberately offline-only: it reads completed decision logs plus their
raw camera archives, evaluates the already validated frozen RECAP value head,
and writes a *new* replay directory.  It never starts a policy server or
connects to the robot.

The output reward is

    r_t = gamma * V_recap(o_next) - V_recap(o_t) + terminal(success/failure),

where one decision is one contiguous Stage-2 action chunk.  For the RLinf
replication contract that chunk must be 10 consecutive 30 Hz ticks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from .rlt_stage2_data import build_replay_bundle


class ValueHead(nn.Module):
    """Checkpoint-compatible RECAP categorical value head."""

    def __init__(self, feature_dim: int, hidden: int, num_bins: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_bins),
        )
        self.register_buffer("atoms", torch.linspace(-1.0, 0.0, num_bins), persistent=False)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(self.net(features), dim=-1) * self.atoms).sum(dim=-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recap-cache", type=Path)
    parser.add_argument("--recap-checkpoint", type=Path)
    parser.add_argument(
        "--value-trace",
        type=Path,
        help=(
            "Per-raw-episode RECAP values from recap_online_value.py. "
            "This is the online path and is mutually exclusive with --recap-cache."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recap-discount", type=float, default=0.96)
    parser.add_argument(
        "--max-observation-time-error-ms",
        type=float,
        default=200.0,
        help="maximum raw/decision observation timestamp mismatch; sequence is diagnostic only",
    )
    parser.add_argument("--pattern", default="*.decisions.h5")
    parser.add_argument("--limit", type=int)
    return parser


def _predict_values(cache: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
    data = np.load(cache / "data.npz", allow_pickle=False)
    features = np.load(cache / "features.npz", allow_pickle=False)
    meta = json.loads((cache / "meta.json").read_text())
    if not np.array_equal(features["row"], np.arange(len(features["row"]))):
        raise ValueError("RECAP feature rows are not contiguous")
    if len(data["source"]) != len(features["features"]):
        raise ValueError("RECAP data/features row count mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ValueHead(
        int(payload["feature_dim"]), int(payload["hidden"]), int(payload["num_bins"])
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    values = np.empty(len(features["features"]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), 8192):
            x = torch.from_numpy(features["features"][start : start + 8192]).to(device)
            values[start : start + len(x)] = model.value(x.float()).cpu().numpy()
    by_raw: dict[str, np.ndarray] = {}
    for episode in meta["episode_table"]:
        if int(episode["source"]) != 1:
            continue
        rows = np.flatnonzero(data["ep"] == int(episode["ep"]))
        trace = values[rows]
        expected = int(episode["n_frames"])
        if trace.shape != (expected,):
            raise ValueError(f"RECAP trace mismatch for {episode['name']}: {trace.shape} != {(expected,)}")
        by_raw[str(episode["name"])] = trace
    return by_raw, {"checkpoint": str(checkpoint.resolve()), "cache": str(cache.resolve())}


def _raw_path(source: h5py.File, decision: Path, raw_root: Path) -> Path:
    for key in ("raw_episode_path", "feature_reencode_raw"):
        value = source.attrs.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return path.resolve()
    stem = decision.name.removesuffix(".decisions.h5")
    candidate = raw_root / f"{stem}.raw.h5"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve raw archive for {decision}; add raw_episode_path metadata")


def _nearest_rows(raw_sequence: np.ndarray, decision_sequence: np.ndarray, maximum: int) -> np.ndarray:
    right = np.searchsorted(raw_sequence, decision_sequence, side="left")
    right = np.clip(right, 0, len(raw_sequence) - 1)
    left = np.maximum(right - 1, 0)
    rows = np.where(
        np.abs(raw_sequence[left] - decision_sequence) <= np.abs(raw_sequence[right] - decision_sequence),
        left,
        right,
    )
    error = np.abs(raw_sequence[rows] - decision_sequence)
    if int(error.max()) > maximum:
        raise ValueError(f"decision/raw sequence error {int(error.max())} > {maximum}")
    return rows.astype(np.int64)


def _nearest_timestamp_rows(
    raw_time_ns: np.ndarray,
    decision_time_ns: np.ndarray,
    maximum_error_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each policy input observation to the nearest persisted raw frame.

    The recorder and policy consume the bridge independently.  Their internal
    sequence counters are not a common clock, while their PC timestamps are.
    """

    raw = np.asarray(raw_time_ns, dtype=np.int64)
    decisions = np.asarray(decision_time_ns, dtype=np.int64)
    if raw.ndim != 1 or decisions.ndim != 1 or not len(raw) or np.any(np.diff(raw) < 0):
        raise ValueError("raw timestamps must be a non-empty monotonic vector")
    if not np.isfinite(maximum_error_ms) or maximum_error_ms <= 0:
        raise ValueError("maximum observation time error must be positive")
    right = np.searchsorted(raw, decisions, side="left")
    right = np.clip(right, 0, len(raw) - 1)
    left = np.maximum(right - 1, 0)
    rows = np.where(
        np.abs(raw[left] - decisions) <= np.abs(raw[right] - decisions), left, right
    ).astype(np.int64)
    errors_ns = np.abs(raw[rows] - decisions)
    maximum_ns = int(round(float(maximum_error_ms) * 1e6))
    if int(errors_ns.max()) > maximum_ns:
        raise ValueError(
            "decision/raw observation timestamp error "
            f"{errors_ns.max() / 1e6:.3f} ms > {maximum_error_ms:.3f} ms"
        )
    return rows, errors_ns


def _online_trace(path: Path, raw_sequence: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load a newly scored raw episode without rebuilding the global feature cache."""

    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float32).reshape(-1)
        trace_sequence = np.asarray(payload["sequence"], dtype=np.int64).reshape(-1)
    if values.shape != raw_sequence.shape or not np.isfinite(values).all():
        raise ValueError("online RECAP trace values do not match raw frame count")
    if not np.array_equal(trace_sequence, raw_sequence):
        raise ValueError("online RECAP trace sequence does not match the raw archive")
    return values, {"trace": str(path.resolve()), "source": "online_frozen_recap_v1"}


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if not 0.0 < args.recap_discount <= 1.0:
        raise ValueError("--recap-discount must be in (0, 1]")
    if args.value_trace is not None and (args.recap_cache is not None or args.recap_checkpoint is not None):
        raise ValueError("--value-trace is mutually exclusive with --recap-cache/--recap-checkpoint")
    if args.value_trace is None and (args.recap_cache is None or args.recap_checkpoint is None):
        raise ValueError("provide --value-trace or both --recap-cache and --recap-checkpoint")
    raw_values: dict[str, np.ndarray] | None = None
    value_meta: dict
    if args.value_trace is None:
        raw_values, value_meta = _predict_values(
            args.recap_cache.expanduser().resolve(),
            args.recap_checkpoint.expanduser().resolve(),
            device,
        )
    else:
        value_meta = {"trace": str(args.value_trace.expanduser().resolve())}
    outputs = args.output_root.expanduser().resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    decisions = sorted(args.decision_root.expanduser().resolve().glob(args.pattern))
    if args.limit is not None:
        decisions = decisions[: args.limit]
    if not decisions:
        raise FileNotFoundError("no decision logs matched")
    result = []
    for decision in decisions:
        out = outputs / decision.name.replace(".decisions.h5", ".npz")
        if out.exists():
            raise FileExistsError(out)
        with h5py.File(decision, "r") as source:
            if not bool(source.attrs.get("completed", False)):
                raise ValueError(f"unfinished decision log: {decision}")
            if int(source.attrs["action_steps"]) != 10:
                raise ValueError(
                    f"{decision.name} has action_steps={source.attrs['action_steps']}; "
                    "RLinf reproduction requires a 10-tick Stage-2 decision log"
                )
            raw = _raw_path(source, decision, args.raw_root.expanduser().resolve())
            with h5py.File(raw, "r") as archive:
                raw_sequence = np.asarray(archive["frames/sequence"], dtype=np.int64)
                rows, timestamp_errors_ns = _nearest_timestamp_rows(
                    np.asarray(archive["frames/pc_timestamp_ns"], dtype=np.int64),
                    np.asarray(source["time_ns"], dtype=np.int64),
                    args.max_observation_time_error_ms,
                )
            if args.value_trace is not None:
                online_values, online_meta = _online_trace(
                    args.value_trace.expanduser().resolve(), raw_sequence
                )
                trace = online_values[rows]
                value_meta = online_meta
            else:
                assert raw_values is not None
                if raw.name not in raw_values:
                    raise KeyError(f"RECAP cache has no trace for raw episode {raw.name}")
                trace = raw_values[raw.name][rows]
        metadata = build_replay_bundle(
            decision,
            out,
            recap_values=trace,
            recap_discount=args.recap_discount,
        )
        sidecar = out.with_suffix(out.suffix + ".json")
        payload = json.loads(sidecar.read_text())
        payload.update(
            {
                "recap_value": value_meta,
                "raw_episode_path": str(raw),
                "observation_alignment": "pc_timestamp_nearest_v1",
                "max_observation_time_error_ms": float(timestamp_errors_ns.max() / 1e6),
                "median_observation_time_error_ms": float(np.median(timestamp_errors_ns) / 1e6),
            }
        )
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result.append({"decision": str(decision), "replay": str(out), "transitions": metadata["transition_count"]})
        print(json.dumps(result[-1]), flush=True)
    print(json.dumps({"episodes": len(result), "output_root": str(outputs), "items": result}, indent=2))


if __name__ == "__main__":
    main()
