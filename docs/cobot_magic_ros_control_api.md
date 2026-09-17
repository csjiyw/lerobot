# Cobot Magic ROS Control API

This document is the interface contract between the AGX robot-control node and
an upstream policy, IL or RLT process. The upstream process may run directly on
AGX or inside a host-network container. No ZMQ bridge is part of this module.

## Safety model

The AGX control node is the only publisher allowed on the physical command
paths:

```text
/master/left_cmd
/master/right_cmd
/puppet/left_cmd
/puppet/right_cmd
```

Policy/RLT processes publish only `/policy/*_cmd`. They must never publish the
master or puppet command topics directly.

The node starts in `SAFE`, with logical output disabled. It also starts in
process-level dry-run unless launched with `--live-output`. Both gates must be
open before a command can reach Piper ROS.

`SAFE` means “publish no new position command.” It does not disable the motors;
the arm controller may continue holding its last target.

## Modes

| ID | Name | Master output | Puppet output |
|---:|---|---|---|
| 0 | `SAFE` | none | none |
| 1 | `POLICY_PUPPET` | none | policy action |
| 2 | `RLT_SYNC` | policy action | identical policy action |
| 3 | `TEACH` | none; human moves master | measured master state |
| 4 | `RESYNC_REQUIRED` | none | none |
| 5 | `ALIGN_SYNC` | independent smooth trajectory | independent smooth trajectory |

Modes 3, 4 and 5 are effective internal states. An upstream process may request
only modes 0, 1 and 2. Requesting mode 2 first enters `ALIGN_SYNC`; mode 2 becomes
effective only after both arms independently reach the first policy target.
The physical teach status overrides all requested modes.

Teach activation is immediate. Teach release is debounced, clears the old
action cache and enters `RESYNC_REQUIRED`.

## Canonical state topics

| Topic | Type | Source |
|---|---|---|
| `/master/left_state` | `sensor_msgs/msg/JointState` | measured left rear/master feedback |
| `/master/right_state` | `sensor_msgs/msg/JointState` | measured right rear/master feedback |
| `/puppet/left_state` | `sensor_msgs/msg/JointState` | measured left front/puppet feedback |
| `/puppet/right_state` | `sensor_msgs/msg/JointState` | measured right front/puppet feedback |

The default physical feedback mapping is:

```text
/puppet/joint_rear_left  -> /master/left_state
/puppet/joint_rear_right -> /master/right_state
/puppet/joint_left       -> /puppet/left_state
/puppet/joint_right      -> /puppet/right_state
```

Do not treat the legacy `/master/joint_*` Piper topics as measured master arm
state; Piper ROS uses those topics for command-state feedback.

## Control topics

| Topic | Type | Direction | Meaning |
|---|---|---|---|
| `/control/request_mode` | `std_msgs/msg/UInt8` | upstream -> AGX | request mode 0, 1 or 2 |
| `/control/output_enabled` | `std_msgs/msg/Bool` | upstream -> AGX | policy/alignment output gate; does not block physical Teach |
| `/control/resync` | `std_msgs/msg/Empty` | upstream -> AGX | acknowledge cleared action queue after teach |
| `/control/emergency_safe` | `std_msgs/msg/Empty` | upstream -> AGX | latch all policy/alignment output off |
| `/control/clear_emergency` | `std_msgs/msg/Empty` | upstream -> AGX | clear the emergency latch; remains SAFE with policy output off |
| `/control/effective_mode` | `std_msgs/msg/UInt8` | AGX -> upstream | current effective mode |
| `/control/epoch` | `std_msgs/msg/UInt64` | AGX -> upstream | current action generation |
| `/control/teach_active` | `std_msgs/msg/Bool` | AGX -> upstream | physical teach status |
| `/control/owner` | `std_msgs/msg/String` | AGX -> upstream | `SAFE`, `POLICY`, `ALIGN`, `RLT` or `TEACH` |
| `/control/state` | `std_msgs/msg/String` | AGX -> upstream | complete state as compact JSON |
| `/control/action_result` | `std_msgs/msg/String` | AGX -> upstream | request/action result as JSON |

The complete state is published at 4 Hz and includes `emergency_latched`.
Teach detection and command routing run at 200 Hz. Ownership priority is
physical Teach, then the emergency latch, then policy/replay. Consequently,
Teach continues to drive its puppet while the emergency latch is set; after
Teach release, automatic control remains blocked until `/control/clear_emergency`.

## Policy action topics

| Topic | Type |
|---|---|
| `/policy/left_cmd` | `sensor_msgs/msg/JointState` |
| `/policy/right_cmd` | `sensor_msgs/msg/JointState` |

`position` must contain exactly seven finite values: joints 1-6 in radians and
the gripper in the existing Piper ROS convention.

Every action must include the current epoch in `header.frame_id`:

```text
epoch=<control_epoch>;seq=<optional_sequence>
```

Example:

```yaml
header:
  frame_id: "epoch=12;seq=408"
name: [joint0, joint1, joint2, joint3, joint4, joint5, joint6]
position: [0.0, 0.2, -0.3, 0.0, 0.1, 0.0, 0.04]
```

Actions with a missing or old epoch are rejected. The result is published on
`/control/action_result`, for example:

```json
{"accepted":false,"command":"SET_ACTION","reason":"teach_active","side":"left"}
```

Other rejection reasons include `missing_epoch`, `stale_epoch`,
`resync_required`, `output_disabled`, `dry_run`, `master_status_stale` and
`owner_safe`. During individual alignment, additional policy targets are
rejected with `align_sync_in_progress`.

## Mode 2 individual alignment

Mode 2 has an internal `ALIGN_SYNC` phase. For each active side, the first
policy command is treated as the common alignment target `q_first`. The
arbiter captures the current measured positions independently and generates:

```text
q_master(t) = smoothstep(q_master_now -> q_first)
q_puppet(t) = smoothstep(q_puppet_now -> q_first)
```

All active master and puppet arms use one common duration, chosen from the
largest required displacement. Their command values are therefore normally
different during alignment and become identical only at `q_first`.

The arbiter remains the sole physical command publisher. It waits until both
measured arms are within tolerance and stable before changing effective mode
from 5 (`ALIGN_SYNC`) to 2 (`RLT_SYNC`). A teach-button press immediately
interrupts alignment and clears the first target. Stale feedback, alignment
timeout, or failure to receive a fresh post-alignment action sends the arbiter
to `SAFE` with a new epoch.

## Required RLT sequence

Initial startup:

1. Subscribe to `/control/state`, `/control/epoch` and all observation topics.
2. Request mode 2 (`RLT_SYNC`).
3. Enable logical output.
4. Read the resulting epoch.
5. Wait for effective mode 5 (`ALIGN_SYNC`).
6. Publish the recorded first target once on every active `/policy/*_cmd`, using
   that epoch. Do not publish physical `/master/*_cmd` or `/puppet/*_cmd`.
7. Wait for effective mode 2 (`RLT_SYNC`) and owner `RLT`.
8. Discard any old action chunk and stream the shared RLT actions to
   `/policy/*_cmd` with the same current epoch.

Teach intervention:

1. A physical teach button makes `/control/teach_active=true` immediately.
2. AGX invalidates the action epoch and stops policy output.
3. The master command path is silent; puppet follows measured master state.
4. Upstream stops sending and clears its action chunk.
5. After button release, wait for effective mode 4 (`RESYNC_REQUIRED`).
6. Publish one `/control/resync` message.
7. Read the new epoch and wait for mode 5 (`ALIGN_SYNC`).
8. Recompute policy/RLT action from a fresh observation and submit its first
   target for individual alignment.
9. Wait for effective mode 2, then resume the new shared action chunk.

Mode 1 does not enter `ALIGN_SYNC`: request mode 1, enable output, read the new
epoch, then stream fresh actions; only the puppet receives them. Its teach
intervention/release handshake is otherwise the same. If master and puppet
differ by more than 15 degrees when teach begins, the puppet is smoothly
aligned before direct 200 Hz following.

## ROS CLI dry-run example

Start the stack in its default non-publishing mode:

```bash
bash /home/agilex/lerobot/scripts/start_cobot_magic_ros_control.sh
```

Request Mode 2 and logically enable output. The process remains dry-run, so no
arm command is published:

```bash
ros2 topic pub --once /control/request_mode std_msgs/msg/UInt8 '{data: 2}'
ros2 topic pub --once /control/output_enabled std_msgs/msg/Bool '{data: true}'
ros2 topic echo /control/state
```

Live output requires the explicit process-level opt-in:

```bash
COBOT_LIVE_OUTPUT=true bash /home/agilex/lerobot/scripts/start_cobot_magic_ros_control.sh
```

## Command motion limiting

Edge V4 and RLT online apply the policy velocity/acceleration limits at the PC
policy waypoint rate, normally 30 Hz. A distant overlap therefore gets more
policy ticks instead of being forced into a fixed three-frame blend. The PC
sends consecutive limited waypoints as a chunk; the AGX bridge only linearly
interpolates that chunk from its 200 Hz timer. The normal launch passes
`--no-command-smoothing` to the ROS arbiter for policy output, so a second
200 Hz limiter does not add lag or alter the planned trajectory.

The bridge and ROS arbiter still retain ownership, epoch, Teach, emergency,
stale-command, and output-off safety gates. Teach follower output remains a
direct 200 Hz feedback path. Planner defaults are arm velocity
`[0.7, 0.7, 0.7, 1.0, 1.0, 1.0]`, gripper velocity `0.12`, arm acceleration
`1.5`, and gripper acceleration `0.4` in the command units used by the Piper
joint topics.

Teach slow alignment is one-shot per side and per physical button press. If
the initial master/puppet arm error exceeds 15 degrees, the puppet first moves
to the captured master pose at the configured slow alignment speed. This
one-time alignment trajectory is the only Teach speed shaping. Once it
completes, all later master motion uses direct, unsmoothed 200 Hz following and
cannot restart slow alignment. Releasing and pressing Teach again resets the
one-shot gate.

The Piper ROS driver does not interpret `JointState.velocity[0:6]` as
per-joint velocity. It treats element 6 as a global speed percentage, so the
outgoing velocity field stays zero. For non-Teach commands, bounded position
increments implement the actual limits; `last_applied_velocity` is exposed in
the controller observation for diagnostics.

Do not run `cobot_magic_handover.py` or another publisher of the canonical
command topics at the same time.
