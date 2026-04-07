# LeRobot Dataset Converter

Unified base converter for multiple robot types with robot-specific JSON configurations.

## Features

- **Single base converter** for all robot types
- **Robot-specific JSON configs** in `configs/` directory
- **Flexible image input handling**:
  - Supports both 1D raw encoded bytes (JPEG/PNG) and 3D HWC numpy arrays
  - Automatic format detection (`detect_img_ext`) based on array dimensions or byte headers
  - JPEG decoding with BGR→RGB conversion and resize via `imgbyte2rgb_hwc`
- **Memory optimized**:
  - Lazy image loading (one frame at a time from HDF5)
  - Periodic image writer queue draining (every 500 frames)
  - Configurable batch size
  - HDF5 file handle caching
  - Preallocated numpy arrays
  - Incremental merge for large datasets
- **Dynamic image format detection** - `dataset.img_format` set from first frame
- **Environment variable control** - `LEROBOT_W` and `LEROBOT_H` for target image size
- **No normalization code** - raw data preserved
- **Early validation** - H5 structure validated before processing
- **Action t+1 offset support** - actions use next state when configured

## Optimizations

| # | Optimization | Benefit |
|---|-------------|---------|
| 1 | HDF5 file handle cache | Reduces open/close overhead |
| 2 | Preallocated numpy arrays | Avoids multiple array concatenations |
| 3 | Reduced image writer threads (2 vs 3) | Lower memory footprint |
| 4 | Early H5 validation | Fails fast on invalid files |
| 5 | Incremental merge | Handles 10+ episodes without OOM |
| 6 | Per-episode progress bar | Better visibility for long episodes |
| 7 | Dual image input support | Handles both raw bytes and HWC arrays |
| 8 | Dynamic image format detection | Auto-detects JPEG/PNG/Array format |

## Supported Robots

| Robot | Config File | State Dim | Action Dim | Notes |
|-------|-------------|-----------|------------|-------|
| `cobotmagic` | `cobotmagic.json` | 20 | 20 | arm(12) + effector(2) + angular(3) + velocity(3) |
| `a2` | `a2.json` | varies | varies | arm(14) + effector(2) + head |
| `gr2` | `gr2.json` | varies | varies | arm + effector + head + leg + waist |
| `fr3` | `fr3.json` | 30 | 30 | left/right arm(14) + gripper(2) + end_pos(6) + end_ori(8) |
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

## Image Handling

The converter supports two image input formats from HDF5:

### 1D Raw Encoded Bytes (JPEG/PNG)
- Stored as 1D numpy array in HDF5
- Automatically decoded via `cv2.imdecode`
- BGR→RGB conversion applied
- Resized to target dimensions (`target_image_size` in config)

### 3D HWC Arrays
- Stored as 3D numpy array (H, W, C) in RGB format
- Written directly without conversion
- Format detected from array dimensions

### Format Detection
```python
def detect_img_ext(arr) -> str:
    if arr.ndim == 3:
        return "png"  # 3D array = raw image
    # Check byte headers for 1D data
    if x.startswith(b"\x89PNG\r\n\x1a\n"): return "png"
    if x.startswith(b"\xff\xd8\xff"): return "jpg"
```

### Target Image Size
Set via environment variables or config:
```bash
export LEROBOT_W=640
export LEROBOT_H=480
```
Or in robot config JSON:
```json
"target_image_size": [640, 480]
```

## Adding New Robot Types

Create a new JSON config in `configs/your_robot.json`:

```json
{
    "robot_type": "YourRobot",
    "target_image_size": [640, 480],
    "hardcoded_names": {
        "arm": ["arm_joint_1", "arm_joint_2"],
        "effector": ["gripper"]
    },
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

### Config Fields

| Field | Description |
|-------|-------------|
| `robot_type` | Robot identifier for LeRobotDataset |
| `target_image_size` | `[width, height]` for image resizing |
| `hardcoded_names` | Optional: manually define joint names (bypasses H5 joint_info) |
| `state_joints.order` | Order of state joints |
| `state_joints.paths` | H5 paths for each state joint |
| `action_joints.order` | Order of action joints |
| `action_joints.paths` | H5 paths for each action joint |
| `action_joints.use_next_state_eef` | Optional: use t+1 state as action for end effector |

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
    "hardcoded_names": {
        "arm": ["arm_joint_1", "arm_joint_2"],
        "effector": ["gripper"]
    },
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
- `hardcoded_names`: Manually define joint names, bypassing H5 `joint_info` lookup
- `camera_filter.include_patterns`: Filter cameras by exact match or prefix
- `use_next_state_eef`: Use t+1 state as action for end effector (reduces action count by 1)

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
