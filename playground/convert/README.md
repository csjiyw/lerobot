# LeRobot Dataset Converter

Unified base converter for multiple robot types with robot-specific JSON configurations.

## Features

- **Single base converter** for all robot types
- **Robot-specific JSON configs** in `configs/` directory
- **Memory optimized**: 
  - Lazy image loading (one frame at a time from HDF5)
  - Periodic image writer queue draining (every 500 frames)
  - Configurable batch size
  - HDF5 file handle caching
  - Preallocated numpy arrays
  - Incremental merge for large datasets
- **No normalization code** - raw data preserved
- **Early validation** - H5 structure validated before processing

## Optimizations

| # | Optimization | Benefit |
|---|-------------|---------|
| 1 | HDF5 file handle cache | Reduces open/close overhead |
| 2 | Preallocated numpy arrays | Avoids multiple array concatenations |
| 3 | Reduced image writer threads (2 vs 3) | Lower memory footprint |
| 4 | Early H5 validation | Fails fast on invalid files |
| 5 | Incremental merge | Handles 10+ episodes without OOM |
| 6 | Per-episode progress bar | Better visibility for long episodes |

## Supported Robots

| Robot | Config File | State Dim | Action Dim | Notes |
|-------|-------------|-----------|------------|-------|
| `cobotmagic` | `cobotmagic.json` | 20 | 20 | arm(12) + effector(2) + angular(3) + velocity(3) |
| `a2` | `a2.json` | varies | varies | arm(14) + effector(2) + head |
| `gr2` | `gr2.json` | varies | varies | arm + effector + head + leg + waist |
| `fr3` | `fr3.json` | 28 | 28 | left/right arm(7) + gripper(1) + eef_6d(6) |
| `ginie1` | `ginie1.json` | varies | varies | Uses default loaders |
| `lejukuafu` | `lejukuafu.json` | varies | varies | arm + effector + head + leg |
| `qinglongros1` | `qinglongros1.json` | varies | varies | arm + effector |
| `xinghaitu_r1` | `xinghaitu_r1.json` | varies | varies | arm + effector |
| `astribot_s1` | `astribot_s1.json` | varies | varies | arm + effector + head + torso + chassis |

## Usage

### Basic Conversion

```bash
python base_converter.py \
    --robot cobotmagic \
    --input ./raw_data \
    --output ./lerobot_output \
    --workers 4 \
    --fps 30
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--robot` | Robot type (required) | - |
| `--input` | Input directory with H5 files | - |
| `--output` | Output directory | - |
| `--repo_id` | HuggingFace repo ID | output dir name |
| `--task` | Task description | manipulation_task |
| `--fps` | Dataset FPS | 30 |
| `--workers` | Parallel worker processes | 4 |
| `--batch_size` | Episodes per batch | 1 |
| `--show_progress` | Show per-episode progress bar | False |
| `--skip_merge` | Skip merge step | False |

### Examples

**CobotMagic:**
```bash
python base_converter.py --robot cobotmagic \
    --input ./cobot_data \
    --output ./cobot_lerobot \
    --workers 4 --batch_size 1
```

**FR3 (Franka):**
```bash
python base_converter.py --robot fr3 \
    --input ./fr3_data \
    --output ./fr3_lerobot \
    --workers 4
```

**A2 (ZhiYuan):**
```bash
python base_converter.py --robot a2 \
    --input ./a2_data \
    --output ./a2_lerobot \
    --workers 4
```

**GR2:**
```bash
python base_converter.py --robot gr2 \
    --input ./gr2_data \
    --output ./gr2_lerobot \
    --workers 4
```

**Astribot S1:**
```bash
python base_converter.py --robot astribot_s1 \
    --input ./s1_data \
    --output ./s1_lerobot \
    --workers 4
```

## Memory Optimization Tips

1. **Reduce batch_size** for large episodes:
   ```bash
   --batch_size 1  # Process one episode at a time
   ```

2. **Reduce workers** if OOM:
   ```bash
   --workers 2  # Fewer parallel processes
   ```

3. **Periodic queue drain** - Already built-in at every 500 frames

## Adding New Robot Types

Create a new JSON config in `configs/your_robot.json`:

```json
{
    "robot_type": "YourRobot",
    "target_image_size": [640, 480],
    "state_joints": {
        "order": ["arm", "effector"],
        "paths": {
            "arm": "joints/state/arm/position",
            "effector": "joints/state/effector/position"
        }
    },
    "action_joints": {
        "order": ["arm", "effector"],
        "paths": {
            "arm": "joints/action/arm/position",
            "effector": "joints/action/effector/position"
        }
    }
}
```

### Advanced: End-Effector 6D Pose (for FR3-like robots)

```json
{
    "robot_type": "YourRobot",
    "target_image_size": [640, 480],
    "state_joints": {
        "order": ["arm", "effector"],
        "paths": {
            "arm": "joints/state/arm/position",
            "effector": "joints/state/effector/position"
        },
        "end_effector": {
            "position_path": "joints/state/end/position",
            "orientation_path": "joints/state/end/orientation",
            "convert_to_6d": true
        }
    },
    "action_joints": {
        "order": ["arm", "effector"],
        "paths": {
            "arm": "joints/action/arm/position",
            "effector": "joints/action/effector/position"
        },
        "use_next_state_eef": true,
        "end_effector": {
            "position_path": "joints/state/end/position",
            "orientation_path": "joints/state/end/orientation",
            "convert_to_6d": true
        }
    }
}
```

### Advanced: Optional Joints & Camera Filtering (for Astribot S1-like robots)

```json
{
    "robot_type": "YourRobot",
    "target_image_size": [640, 480],
    "state_joints": {
        "order": ["arm", "effector", "head", "torso", "chassis"],
        "paths": {
            "arm": "joints/state/arm/position",
            "effector": "joints/state/effector/position",
            "head": "joints/state/head/position",
            "torso": "joints/state/torso/position",
            "chassis": "joints/state/chassis/position"
        },
        "optional": ["torso", "chassis"]
    },
    "action_joints": {
        "order": ["arm", "effector", "head", "torso", "chassis"],
        "paths": {
            "arm": "joints/action/arm/position",
            "effector": "joints/action/effector/position",
            "head": "joints/action/head/position",
            "torso": "joints/action/torso/position",
            "chassis": "joints/action/chassis/position"
        },
        "optional": ["torso", "chassis"],
        "chassis_fallback_to_state": true
    },
    "camera_filter": {
        "include_patterns": ["head", "hand_", "torso"],
        "description": "Only include head, hand_*, and torso cameras"
    }
}
```

**Features:**
- `optional`: List of joints that may be missing (will be skipped if not found)
- `chassis_fallback_to_state`: If action/chassis is missing, use state/chassis with t+1 offset
- `camera_filter.include_patterns`: Filter cameras by exact match or prefix

## File Structure

```
convert/
├── base_converter.py      # Main converter script
├── configs/               # Robot-specific JSON configs
│   ├── cobotmagic.json
│   ├── a2.json
│   ├── gr2.json
│   ├── fr3.json
│   ├── ginie1.json
│   ├── lejukuafu.json
│   ├── qinglongros1.json
│   ├── xinghaitu_r1.json
│   └── astribot_s1.json
└── README.md              # This file
```

## Output Structure

```
output_dir/
├── info.json              # Dataset metadata
├── stats.json             # Dataset statistics
├── episodes/              # Episode metadata
├── data/                  # Parquet files with state/action
└── videos/                # Encoded video files
```
