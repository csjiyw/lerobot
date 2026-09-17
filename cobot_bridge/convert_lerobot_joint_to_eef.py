"""Convert joint-space LeRobot datasets to absolute and incremental EEF data.

The source datasets are never modified.  For every compatible source dataset
the converter can create:

* ``<source>_eef``: state and target/action vectors are
  ``[x, y, z, rx, ry, rz, gripper_command]`` in the Piper base frame.
* ``<source>_eef_delta``: state remains absolute EEF state, while action-like
  vectors are ``[dx, dy, dz, drx, dry, drz, gripper_command]`` relative to the
  current EEF state.  The gripper value intentionally remains a command, not a
  temporal difference.

RLT fields are converted with their validity masks.  When both
``rlt.reference_action`` and ``rlt.executed_action`` exist, the output also
contains ``rlt.eef_residual_action``.  Its first six dimensions are
``executed_eef - reference_eef`` and its last dimension is the executed
gripper command.  This is the correction signal used by the incremental EEF
actor; it is not confused with the temporal action delta above.

Videos are hard-linked when possible, so the conversion does not duplicate
the large MP4 payload.  Metadata and parquet files are copied and rewritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats

from .piper_eef_metrics import DEFAULT_PIPER_URDF, PiperForwardKinematics
from .piper_eef_transform import eef_pose_delta


LOG = logging.getLogger("cobot_bridge.convert_lerobot_joint_to_eef")

JOINT_NAMES = [f"joint{index}" for index in range(7)]
EEF_ABSOLUTE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
EEF_DELTA_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
EEF_RESIDUAL_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]

RLT_ACTION_FIELDS = (
    "action",
    "rlt.reference_action",
    "rlt.policy_action",
    "rlt.human_action",
    "rlt.executed_action",
)
RLT_VALIDITY_FIELDS = {
    "rlt.reference_action": "rlt.reference_action_valid",
    "rlt.policy_action": "rlt.policy_action_valid",
    "rlt.human_action": "rlt.human_action_valid",
    "rlt.executed_action": "rlt.executed_action_valid",
}
VECTOR_FIELDS = ("observation.state", "observation.master_state", *RLT_ACTION_FIELDS)
RESIDUAL_FIELD = "rlt.eef_residual_action"
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _copy_source_file(source: str, destination: str) -> None:
    """Copy metadata/data, hard-linking video payloads on the same filesystem."""

    if "videos" in Path(source).parts:
        try:
            os.link(source, destination)
            return
        except OSError:
            # Cross-filesystem copies are still valid; only the storage
            # optimization is unavailable.
            pass
    shutil.copy2(source, destination)


def _copy_dataset_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=_copy_source_file)


def _feature_is_joint_vector(info: dict[str, Any], name: str) -> bool:
    feature = info.get("features", {}).get(name)
    if not isinstance(feature, dict):
        return False
    shape = feature.get("shape")
    dtype = feature.get("dtype")
    names = feature.get("names")
    return (
        dtype == "float32"
        and list(shape or []) == [7]
        and list(names or []) == JOINT_NAMES
    )


def _column_array(table: pa.Table, name: str, *, dtype: Any = np.float32) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=dtype)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _validity_mask(table: pa.Table, field: str) -> np.ndarray:
    validity_field = RLT_VALIDITY_FIELDS.get(field)
    if validity_field is None or validity_field not in table.column_names:
        return np.ones(table.num_rows, dtype=bool)
    values = _column_array(table, validity_field, dtype=np.float64).reshape(-1)
    return values > 0.5


def _rotation_vectors(kinematics: PiperForwardKinematics, rotations: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to axis-angle vectors."""

    try:
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(rotations).as_rotvec()
    except ImportError:
        # Pinocchio is already required by the EEF conversion.  This fallback
        # keeps the script usable in a minimal environment without scipy.
        return np.stack([kinematics.pin.log3(rotation) for rotation in rotations], axis=0)


def _joint_to_eef(
    joints: np.ndarray,
    valid: np.ndarray,
    kinematics: PiperForwardKinematics,
) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if joints.ndim != 2 or joints.shape[1] != 7:
        raise ValueError(f"joint vectors must have shape (N, 7), got {joints.shape}")
    if valid.shape != (len(joints),):
        raise ValueError(f"validity mask must have shape ({len(joints)},), got {valid.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("joint vectors contain non-finite values")

    result = np.zeros((len(joints), 7), dtype=np.float32)
    if not valid.any():
        return result
    fk = kinematics.forward(joints[valid, :6])["gripper_center"]
    positions, rotations = fk
    result[valid, :3] = np.asarray(positions, dtype=np.float32)
    result[valid, 3:6] = np.asarray(
        _rotation_vectors(kinematics, np.asarray(rotations, dtype=np.float64)),
        dtype=np.float32,
    )
    # Gripper is a command/state scalar and is deliberately not transformed.
    result[valid, 6] = joints[valid, 6].astype(np.float32)
    return result


def _eef_delta(eef_action: np.ndarray, eef_state: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(eef_action, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    for index in np.flatnonzero(mask):
        result[index] = eef_pose_delta(eef_state[index], eef_action[index])
    return result


def _eef_residual(
    reference: np.ndarray,
    executed: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(reference, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    result[mask, :6] = executed[mask, :6] - reference[mask, :6]
    result[mask, 6] = executed[mask, 6]
    return result


def _replace_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    values = np.asarray(values, dtype=np.float32)
    array = pa.array(values.tolist(), type=table.schema.field(name).type)
    return table.set_column(table.schema.get_field_index(name), table.schema.field(name), array)


def _append_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    values = np.asarray(values, dtype=np.float32)
    return table.append_column(name, pa.array(values.tolist(), type=pa.list_(pa.float32())))


def _stats(values: np.ndarray) -> dict[str, Any]:
    result = get_feature_stats(np.asarray(values, dtype=np.float32), axis=0, keepdims=False)
    return {key: _jsonable(result[key]) for key in STAT_KEYS if key in result}


def _replace_or_append_column(table: pa.Table, name: str, values: list[Any], field_type: pa.DataType) -> pa.Table:
    array = pa.array(values, type=field_type)
    if name in table.column_names:
        index = table.schema.get_field_index(name)
        return table.set_column(index, table.schema.field(index), array)
    return table.append_column(name, array)


def _update_episode_metadata(
    output_root: Path,
    episode_values: dict[int, dict[str, list[np.ndarray]]],
    transformed_fields: Iterable[str],
) -> None:
    fields = tuple(transformed_fields)
    legacy_path = output_root / "meta" / "episodes_stats.jsonl"
    if legacy_path.is_file():
        output_lines: list[str] = []
        for line in legacy_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            episode_id = int(item["episode_index"])
            item_stats = item.setdefault("stats", {})
            for field in fields:
                batches = episode_values[episode_id].get(field, [])
                if not batches:
                    raise ValueError(
                        f"missing transformed data for episode {episode_id}, field {field}"
                    )
                item_stats[field] = _stats(np.concatenate(batches, axis=0))
            output_lines.append(json.dumps(_jsonable(item), separators=(",", ":")))

        temporary_path = legacy_path.with_suffix(f"{legacy_path.suffix}.tmp-{os.getpid()}")
        temporary_path.write_text("\n".join(output_lines) + "\n")
        os.replace(temporary_path, legacy_path)

    for episode_path in sorted((output_root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(episode_path)
        episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        for field in fields:
            for stat_key in STAT_KEYS:
                column_name = f"stats/{field}/{stat_key}"
                values: list[Any] = []
                for episode_id in episode_ids:
                    batches = episode_values[int(episode_id)].get(field, [])
                    if not batches:
                        raise ValueError(
                            f"missing transformed data for episode {episode_id}, field {field}"
                        )
                    values.append(_stats(np.concatenate(batches, axis=0))[stat_key])
                if column_name in table.column_names:
                    field_type = table.schema.field(column_name).type
                elif stat_key == "count":
                    field_type = pa.list_(pa.int64())
                else:
                    field_type = pa.list_(pa.float64())
                table = _replace_or_append_column(table, column_name, values, field_type)
        pq.write_table(table, episode_path, compression="snappy", use_dictionary=True)


def _transform_table(
    table: pa.Table,
    info: dict[str, Any],
    mode: str,
    kinematics: PiperForwardKinematics,
) -> tuple[pa.Table, dict[str, np.ndarray], list[str]]:
    if "observation.state" not in table.column_names:
        raise ValueError("dataset is missing observation.state")
    state_valid = np.ones(table.num_rows, dtype=bool)
    eef_state = _joint_to_eef(
        _column_array(table, "observation.state"), state_valid, kinematics
    )

    output = table
    converted: dict[str, np.ndarray] = {}
    transformed_fields: list[str] = []
    absolute: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}

    for field in VECTOR_FIELDS:
        if field not in table.column_names or not _feature_is_joint_vector(info, field):
            continue
        valid = _validity_mask(table, field)
        value = _joint_to_eef(_column_array(table, field), valid, kinematics)
        absolute[field] = value
        valid_masks[field] = valid
        if mode == "delta" and field in RLT_ACTION_FIELDS:
            value = _eef_delta(value, eef_state, valid)
        converted[field] = value
        output = _replace_vector_column(output, field, value)
        transformed_fields.append(field)

    # State is always absolute, even in the incremental-action dataset.
    if "observation.state" not in transformed_fields:
        raise ValueError("observation.state is not a supported 7D joint vector")
    if "rlt.reference_action" in absolute and "rlt.executed_action" in absolute:
        residual_valid = valid_masks["rlt.reference_action"] & valid_masks["rlt.executed_action"]
        residual = _eef_residual(
            absolute["rlt.reference_action"],
            absolute["rlt.executed_action"],
            residual_valid,
        )
        output = _append_vector_column(output, RESIDUAL_FIELD, residual)
        converted[RESIDUAL_FIELD] = residual
        transformed_fields.append(RESIDUAL_FIELD)

    return output, converted, transformed_fields


def _update_info(info: dict[str, Any], mode: str, transformed_fields: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(info)
    action_names = EEF_DELTA_NAMES if mode == "delta" else EEF_ABSOLUTE_NAMES
    for field in transformed_fields:
        if field == RESIDUAL_FIELD:
            result["features"][field] = {
                "dtype": "float32",
                "shape": [7],
                "names": EEF_RESIDUAL_NAMES,
            }
            continue
        if field not in result.get("features", {}):
            continue
        names = action_names if field in RLT_ACTION_FIELDS else EEF_ABSOLUTE_NAMES
        result["features"][field]["shape"] = [7]
        result["features"][field]["names"] = names
    return result


def _write_global_stats(
    output_root: Path,
    transformed_values: dict[str, list[np.ndarray]],
) -> None:
    stats_path = output_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for field, batches in transformed_values.items():
        stats[field] = _stats(np.concatenate(batches, axis=0))
    stats_path.write_text(json.dumps(_jsonable(stats), indent=2) + "\n")


def _convert_dataset(source: Path, destination: Path, mode: str, urdf_path: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("dataset is empty")
    if not _feature_is_joint_vector(info, "observation.state"):
        raise ValueError("dataset is not a joint-space 7D dataset")

    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"destination already exists or has an unfinished conversion: {destination}")
    LOG.info("copying %s -> temporary %s (videos are hard-linked when possible)", source.name, temporary.name)
    _copy_dataset_tree(source, temporary)

    kinematics = PiperForwardKinematics(urdf_path=urdf_path, frames=("gripper_center",))
    transformed_values: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_values: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    all_transformed_fields: set[str] = set()
    data_files = sorted((source / "data").glob("**/*.parquet"))
    if not data_files:
        raise ValueError("dataset has no data parquet files")

    try:
        for index, source_data in enumerate(data_files, start=1):
            table = pq.read_table(source_data)
            output_table, converted, transformed_fields = _transform_table(
                table, info, mode, kinematics
            )
            all_transformed_fields.update(transformed_fields)
            episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            for field, values in converted.items():
                transformed_values[field].append(values)
                for episode_id in np.unique(episode_ids):
                    episode_values[int(episode_id)][field].append(
                        values[episode_ids == episode_id]
                    )
            destination_data = temporary / source_data.relative_to(source)
            pq.write_table(output_table, destination_data, compression="snappy", use_dictionary=True)
            LOG.info(
                "%s: parquet %d/%d rows=%d fields=%s",
                source.name,
                index,
                len(data_files),
                table.num_rows,
                ",".join(transformed_fields),
            )

        output_info = _update_info(info, mode, sorted(all_transformed_fields))
        output_info["eef_conversion"] = {
            "mode": "absolute" if mode == "absolute" else "incremental_action",
            "frame": "gripper_center",
            "orientation": "axis_angle_rotation_vector_rad",
            "delta_rotation": "spatial_relative_rotation_log_R_target_R_state_inverse",
            "state_vector": "[x,y,z,rx,ry,rz,gripper]",
            "action_vector": (
                "[x,y,z,rx,ry,rz,gripper_command]"
                if mode == "absolute"
                else "[dx,dy,dz,drx,dry,drz,gripper_command]"
            ),
            "gripper_semantics": "absolute_command_or_state; never differenced",
        }
        (temporary / "meta" / "info.json").write_text(
            json.dumps(_jsonable(output_info), indent=2) + "\n"
        )
        _write_global_stats(temporary, transformed_values)
        _update_episode_metadata(temporary, episode_values, sorted(all_transformed_fields))

        manifest = {
            "source_dataset": str(source.resolve()),
            "output_dataset": str(destination.resolve()),
            "mode": mode,
            "urdf": str(urdf_path.resolve()),
            "eef_frame": "gripper_center",
            "orientation": "axis_angle_rotation_vector_rad",
            "source_total_episodes": info.get("total_episodes"),
            "source_total_frames": info.get("total_frames"),
            "output_total_episodes": output_info.get("total_episodes"),
            "output_total_frames": output_info.get("total_frames"),
            "transformed_fields": sorted(all_transformed_fields),
            "residual_definition": (
                "executed EEF absolute - reference EEF absolute for first 6 dimensions; "
                "executed gripper command in dimension 7"
            ),
            "video_storage": "hard-linked when possible",
        }
        (temporary / "meta" / "eef_conversion.json").write_text(
            json.dumps(_jsonable(manifest), indent=2) + "\n"
        )
        os.replace(temporary, destination)
        LOG.info("completed %s (%s): episodes=%s frames=%s", destination.name, mode, output_info.get("total_episodes"), output_info.get("total_frames"))
        return manifest
    except Exception:
        LOG.exception("conversion failed for %s", source)
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _discover_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "meta" / "info.json").is_file():
            continue
        info = json.loads((path / "meta" / "info.json").read_text())
        if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
            LOG.info("skip empty dataset %s", path.name)
            continue
        if not _feature_is_joint_vector(info, "observation.state"):
            LOG.info("skip non-joint/EEF dataset %s", path.name)
            continue
        sources.append(path)
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("../datasets"))
    parser.add_argument("--source", type=Path, nargs="*", help="specific dataset directories; default: all compatible datasets")
    parser.add_argument("--mode", choices=("absolute", "delta", "both"), default="both")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_PIPER_URDF)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    root = args.datasets_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not args.urdf.is_file():
        raise FileNotFoundError(args.urdf)

    if args.source:
        sources = [(root / item if not item.is_absolute() else item).resolve() for item in args.source]
    else:
        sources = _discover_sources(root)
    sources = [path for path in sources if path.name not in {"rlt_archive"}]
    if args.max_datasets is not None:
        if args.max_datasets <= 0:
            raise ValueError("--max-datasets must be positive")
        sources = sources[: args.max_datasets]
    if not sources:
        raise ValueError("no compatible joint-space datasets found")

    suffixes = ("_eef", "_eef_delta") if args.mode == "both" else (("_eef",) if args.mode == "absolute" else ("_eef_delta",))
    plan = [(source, root / f"{source.name}{suffix}", "absolute" if suffix == "_eef" else "delta") for source in sources for suffix in suffixes]
    for source, destination, mode in plan:
        print(f"{mode:8s} {source.name} -> {destination.name}")
    if args.dry_run:
        return

    converted = 0
    skipped = 0
    for source, destination, mode in plan:
        if destination.exists():
            LOG.warning("skip existing output %s; source data was not overwritten", destination)
            skipped += 1
            continue
        _convert_dataset(source, destination, mode, args.urdf.expanduser().resolve())
        converted += 1
    LOG.info("conversion summary: converted=%d skipped_existing=%d planned=%d", converted, skipped, len(plan))


if __name__ == "__main__":
    main()
