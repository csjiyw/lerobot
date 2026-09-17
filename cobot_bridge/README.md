# Cobot Magic PC/AGX bridge

This package is the non-ROS side of the Cobot Magic RLT integration.

```text
PC: LeRobot / IL / RLT / dataset
              |
          ZeroMQ TCP
              |
AGX: bridge -> canonical ROS control API -> Piper
```

The AGX bridge does not decode cameras and does not save data. It keeps only
the latest ROS samples in memory. The PC receives the original compressed JPEG
frames and owns datasets, replay buffers, checkpoints and policy inference.

## IL/RL data collection dashboard

Start the local dashboard on the PC:

```bash
cd /media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot
bash cobot_bridge/start_data_dashboard.sh
```

Open <http://127.0.0.1:8088>. The IL workflow starts a raw recorder and ends
the episode after Teach release. The RL workflow starts the recorder first,
then a selected LeRobot policy; Teach release stops that policy but the recorder
remains open until the operator saves Success or Failure. Real motion requires
both a UI checkbox and a browser confirmation. **EMERGENCY STOP** stops motion
jobs and latches policy/alignment output off. Physical Teach has the highest
priority and continues to follow while held. Automatic control remains blocked
after Teach release until the operator clicks **EXIT STOP**; clearing the latch
keeps the controller in SAFE with policy output off.

RL recording is finalized explicitly with the **Save Success** or **Save
Failure** button. Either button stores the terminal result and stops the active
rollout; Teach transitions remain frame-level intervention labels and do not by
themselves close an RL episode unless **Exit Teach mode -> auto-save** is
selected. The same option is available for IL collection. New IL and RL
recordings also carry an explicit episode-level collection type through raw and
converted data.

An active IL recorder can also be completed explicitly with **Save Success**
or **Save Failure**. This writes the selected terminal result and finalizes the
raw HDF5 without requiring Teach release or using the generic Jobs stop
control.

The RL form also has **Auto-start selected policy with recorder**. It is only
available when both a local checkpoint and normalization dataset exist. The
operator must still enable the Policy Rollout real-motion checkbox and accept
one browser confirmation. If policy startup fails, the newly started recorder
is stopped instead of being left running alone.

The System tab is a read-only live device monitor. It displays cached frames
from the front, left and right cameras plus freshness, age and 7D joint state
for master-left, master-right, puppet-left and puppet-right. A device is marked
OK only when its bridge sample is present and fresh; the top strip summarizes
all seven devices.

The IL Collect tab also exposes a guarded **Move to Ready Pose** operation.
Its fixed non-singular target is
`[0.1441, 1.1020, -1.1085, -0.2355, 0.7648, 0.2386, 0.08]`, applied to
master-left, master-right, puppet-left and puppet-right through direct joint
alignment.  This avoids using the all-zero Piper pose as a Cartesian-IK
starting point. Teach and EMERGENCY SAFE can interrupt the motion, and
completion always disables output and returns to SAFE.

The dashboard binds to localhost by default. Set `COBOT_WEB_HOST=0.0.0.0` only
when LAN access is intentionally required. `COBOT_WEB_TOKEN` enables an API
token check for remote use.

## Safety and ownership

The bridge never publishes `/master/*_cmd` or `/puppet/*_cmd`. It publishes
only `/policy/*_cmd` and `/control/*`; the existing ROS arbiter remains the
single physical command owner.

The bridge starts read-only. Control forwarding requires a separate explicit
opt-in:

```bash
COBOT_BRIDGE_ALLOW_CONTROL=true \
  bash /home/agilex/lerobot/cobot_bridge/start_robot_bridge.sh
```

This does not bypass the ROS arbiter. The arbiter's process-level dry-run,
policy-output, Teach-priority, emergency-latch, epoch and resync gates still
apply.

## Start on AGX

```bash
cd /home/agilex/lerobot
bash cobot_bridge/start_robot_bridge.sh
```

The default endpoint is `tcp://192.168.250.2:5555` and the process uses ROS
domain 0 with `ROS_LOCALHOST_ONLY=1`.

## Test from the PC

```bash
conda activate lerobot0_4_0
cd /media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot

python -m cobot_bridge.smoke_test \
  --endpoint tcp://192.168.250.2:5555 \
  --cycles 20
```

`--exercise-control` is accepted only when the ROS arbiter reports
process-level dry-run. It verifies mode, epoch, resync, stale-epoch rejection
and dry-run action rejection.

## Record raw demonstrations on the PC

The live path records one append-only HDF5 file. JPEG images remain compressed,
and no LeRobot conversion or video encoding happens while operating the robot.
Episode recording is controlled by the PC and is independent of Teach mode.

```bash
python -m cobot_bridge.record_raw \
  --output /path/to/episode_000000.raw.h5 \
  --stop-file /tmp/cobot_stop_episode_000000 \
  --task "pick up the object" \
  --side right \
  --cameras front right \
  --fps 30
```

Creating the stop file flushes and closes the raw episode without starting an
expensive conversion in the live recorder:

```bash
touch /tmp/cobot_stop_episode_000000
```

Convert the finalized file offline:

```bash
python -m cobot_bridge.convert_raw_to_lerobot \
  --raw /path/to/episode_000000.raw.h5 \
  --repo-id local/cobot_magic_right_demo \
  --root /path/to/cobot_magic_right_demo \
  --action-source master_state \
  --fps 20 \
  --use-videos
```

Replay is generated at 200 Hz by timestamp interpolation. It is dry-run unless
the explicit execution flag is supplied:

```bash
python -m cobot_bridge.replay_raw \
  --raw /path/to/episode_000000.raw.h5 \
  --side right \
  --waypoint-hz 20 \
  --speed-scale 0.5
```

For RLT-style synchronized replay, mode 2 routes each identical 200 Hz action
to both the selected master and puppet:

```bash
python -m cobot_bridge.replay_raw \
  --raw /path/to/episode_000000.raw.h5 \
  --side right \
  --mode sync \
  --waypoint-hz 20 \
  --speed-scale 0.5 \
  --execute
```

`--speed-scale 0.5` is the default and replays the 20 Hz recorded waypoints on
a 10 Hz timeline, doubling the episode duration. The AGX scheduler still
interpolates and publishes at 200 Hz. Robot-side Mode 2 alignment keeps its
independent configured speed and is not scaled by this replay option. Use
`--speed-scale 1.0` for the original recorded speed.

Leading and trailing stationary holds are removed automatically. By default,
the detector uses a `1e-4 rad` per-waypoint threshold and keeps 0.5 seconds on
each side of the actual motion. Disable this with
`--no-trim-stationary-holds`.

For a supervised human-in-the-loop test, loop the clipped trajectory:

```bash
python -m cobot_bridge.replay_raw \
  --raw /path/to/episode_000000.raw.h5 \
  --side right \
  --mode sync \
  --speed-scale 0.5 \
  --loop \
  --execute
```

Between normal cycles, a smooth transition returns both arms from the final
target to the first target. Pressing Teach immediately clears the active
chunk. The runner waits without timing out while the operator teaches. Teach
release ends the episode: the runner does **not** resync or resume policy; it
requests output off and `SAFE`, then exits. Ctrl-C has the same safe shutdown.

## Record a policy-to-human intervention episode

Start raw recording before policy motion. The first Teach activation marks the
human intervention; its release frame closes the raw episode and starts the
offline LeRobot conversion in a separate thread:

```bash
python -m cobot_bridge.record_raw \
  --output /path/to/episode_000001.raw.h5 \
  --task "human correction during replay" \
  --side right \
  --fps 30 \
  --stop-on-teach-release \
  --convert-repo-id local/cobot_magic_right_intervention_001 \
  --convert-root /path/to/cobot_magic_right_intervention_001 \
  --convert-action-source master_state
```

Raw and converted frames contain explicit `control_source` labels
(`0=other`, `1=policy`, `2=human`, `3=alignment`), `is_policy`,
`is_human_intervention`, `intervention_id`, `teach_event` (`+1` press, `-1`
release), effective mode and epoch. The finalized raw file remains the source
of truth; conversion starts only after its atomic rename.

`collect_lerobot.py` remains for small synthetic tests, but real robot episodes
must use `record_raw.py` followed by offline conversion.

## Run a trained LeRobot policy

The runner computes actions but sends nothing by default:

```bash
python -m cobot_bridge.run_lerobot_policy \
  --policy-path /path/to/policy \
  --task "pick up the object"
```

The checkpoint restores its own feature schema, normalization state, and action
denormalization state. Inference does not require the training dataset.

`--send-actions` is the explicit PC-side action opt-in. Real motion additionally
requires an AGX bridge started with control forwarding and a ROS arbiter started
with live output.
