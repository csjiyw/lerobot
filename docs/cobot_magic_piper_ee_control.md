# Piper end-effector control with the Piper SDK

## Decision

The runtime uses the official Piper SDK endpoint controller. It supports two
explicit EEF contracts:

* ``eef_delta``: the policy emits a state-relative delta and the runner sends
  one target before observing again.
* ``eef_target``: the policy emits a complete chunk of absolute EEF targets in
  ``[x, y, z, rx, ry, rz, gripper]``. The runner queues and sends the complete
  chunk sequentially, which is the Groot-style rollout contract.

The ROS arbiter remains the only process allowed to forward either target to
the Piper SDK.

## Frame boundary

The AGX Piper ``end_pose`` feedback and ``PosCmd`` controller use the
``link6`` pose. The existing converted EEF datasets and checkpoints use the
fixed ``gripper_center`` TCP in the official Piper URDF. The rollout path now
converts explicitly at the policy boundary:

```text
AGX SDK/ROS feedback (link6)
        -> fixed URDF SE(3) transform
policy observation (gripper_center)
        -> VLA/RLT inference
model absolute target or delta (gripper_center)
        -> fixed URDF inverse SE(3) transform
AGX SDK/ROS command target (link6)
```

This is a rigid pose transform, not ``x += dx; y += dy; z += dz`` with a
constant world-frame offset. The TCP translation rotates with the current
``link6`` orientation. The gripper scalar is passed through unchanged. The
default model frame is ``gripper_center`` for backward compatibility;
``--model-eef-frame link6`` is available for a checkpoint trained directly on
SDK-frame data.

The SDK path is intentionally the first integration because it avoids adding a
second inverse-kinematics implementation to the rollout experiment. It uses
the robot driver's existing ``PosCmd`` callback and ``EndPoseCtrl`` conversion
to the Piper's internal controller. Pink remains available as an offline
diagnostic/fallback path; it is not mixed into the SDK rollout path.

```text
30 Hz VLA EEF target chunk or delta + gripper
                 |
       link6 -> gripper_center for the model
                 |
        gripper_center -> link6 for PosCmd
                 |
       ZeroMQ bridge -> ROS policy EEF topic
                 |
       existing Cobot Magic arbiter
       - Teach and emergency priority
       - epoch/action ownership
       - stale-command and output safety gates
                 |
       200 Hz repeated PosCmd publication
                  |
       Piper ROS driver -> Piper SDK EndPoseCtrl
```

Pink does not bypass the arbiter and does not publish robot commands itself.

## Implemented SDK path and computation-only components

- The policy bridge uses ``/policy/<side>_eef_cmd`` with ``piper_msgs/PosCmd``.
- ``PosCmd`` carries meters/radians/meters at the public interface. The SDK
  callback converts the Euler fields to its integer millimeter/millidegree
  units.
- EEF mode currently supports Mode 1 only. Mode 2 continues to use the
  tested joint-space individual alignment until EEF alignment has its own
  feedback/tolerance tests.
- ``eef_delta`` rollout consumes only the first predicted action and observes
  again. Each converted delta is relative to its own 30 Hz frame.
- ``eef_target`` rollout consumes the complete model chunk. Its dataset action
  rows are already absolute EEF targets, so no measured pose is added again.

- `cobot_bridge/piper_pink_ik.py` loads the official Piper URDF, removes the two
  finger joints from the IK model, verifies `joint1` through `joint6` ordering,
  and returns one bounded joint target per differential-IK tick.
- `cobot_bridge/validate_piper_pink_ik.py` reconstructs a recorded joint
  trajectory through FK then IK. It does not initialize ROS, ZMQ, CAN, or any
  command publisher.
- `tests/cobot_bridge/test_piper_pink_ik.py` checks official joint limits,
  nearby-pose convergence, and optional hard velocity/acceleration bounds.

The AGX runtime must use the robot's official URDF:

```text
/home/agilex/piper_ros/src/piper_description/urdf/piper_description.urdf
```

The selected training frame is currently ``gripper_center``. The conversion
script keeps writing that metadata, while the runtime bridge treats incoming
feedback as ``link6`` and applies the URDF transform before inference. The
inverse transform is applied immediately before the SDK command. Therefore
existing ``gripper_center`` datasets/checkpoints do not need retraining merely
because the SDK feedback is expressed at ``link6``.

The transform implementation is
``cobot_bridge/piper_eef_transform.py``. It resolves the local official URDF
or the synchronized model URDF at
``/home/agilex/smolvla_runtime/cobot_bridge/assets/piper_model.urdf``. The
AGX driver URDF is also a fallback path, but it does not define the legacy
``gripper_center`` frame on some installations; in that case the synchronized
model URDF must be used. The driver path is
``/home/agilex/piper_ros/src/piper_description/urdf/piper_description.urdf``;
``COBOT_PIPER_URDF`` can override the path. A wrong URDF/TCP still produces a
systematic insertion offset, so the path is recorded and checked offline.

## Current limits and diagnostics

Defaults in the computation adapter are:

- IK rate: 200 Hz;
- joint velocity: 0.45 rad/s;
- Pink hard acceleration limit: disabled;
- soft low-acceleration task: enabled;
- singularity warning: minimum Jacobian singular value below `1e-3` or
  condition number above `1000`.

The Pink hard acceleration limit is disabled by default because the policy
waypoint planner applies the hard command limit before the 200 Hz bridge
interpolation. Stacking another hard limiter in the ROS arbiter caused
avoidable lag and can make a QP infeasible near a joint limit. The Pink
low-acceleration task remains active as a soft continuity objective.

Near-singularity is diagnostic in the current code. It must not silently alter
or publish a command. Runtime integration should reject a non-finite result,
hold or enter SAFE on repeated QP failures, and report the singularity metrics
to the dashboard.

## Offline validation

Validation source:

```text
raw_datasets/cobot_magic_rl/episode_20260814_144110_843908.raw.h5
636 frames at 30 Hz, puppet joints 1-6
```

The validator used an exact cumulative 30 Hz to 200 Hz clock, alternating six
and seven IK ticks per recorded frame. It reconstructed each measured joint
frame into an endpoint target with FK, then tracked that target with Pink:

| IK joint velocity limit | Position error P95 | Position error max | Joint error P95 | Solver failures |
|---:|---:|---:|---:|---:|
| 0.45 rad/s | 14.75 mm | 18.88 mm | 0.0641 rad | 0 |
| 0.60 rad/s | 0.23 mm | 3.83 mm | 0.00448 rad | 0 |

The recorded trajectory itself reaches roughly 0.53 rad/s. Therefore the
0.45-rad/s result is primarily rate-limit lag, not an IK convergence failure.
The 0.60-rad/s result demonstrates feasibility offline; it is not authorization
to raise the physical limit. Any limit change requires a separately reviewed,
small-motion test.

The trajectory generated 71 near-singularity warnings, concentrated around the
near-zero initial posture. This supports retaining posture regularization and
explicit diagnostics. An endpoint representation does not remove robot
singularities; it makes handling them an explicit controller responsibility.

## Reproduce the offline check

Use the LeRobot environment with Pink 4.2.0, Pinocchio 4.0.0, qpsolvers 4.10.0,
and DAQP 0.8.5:

```bash
conda activate lerobot0_4_0
python -m cobot_bridge.validate_piper_pink_ik \
  --urdf /path/to/piper_description.urdf \
  --raw /path/to/episode.raw.h5 \
  --max-joint-velocity 0.45
```

The AGX `robot_bridge` environment did not contain Pink, Pinocchio, or
qpsolvers when this check was written. Installing them and copying code to AGX
are intentionally separate deployment steps.

## Safe integration sequence

1. **Live shadow only:** verify the four raw Piper EEF feedback topics and the
   bridge's `master_eef`/`puppet_eef` fields; publish no command.
2. **Recorded target shadow:** replay converted EEF deltas offline and check
   rotvec/Euler round trips, gripper units, workspace bounds, and target jumps.
3. **Arbiter dry-run:** send an EEF target through the policy topic with the
   control process still in dry-run. Verify epoch, Teach preemption, SAFE
   transitions, stale-feedback handling, and that no PosCmd reaches the live
   Piper node.
4. **Explicitly authorized small motion:** test one arm in open space with a
   millimetre-scale Cartesian displacement using the SDK path.
5. Only after those checks, enable the web rollout's `EEF delta · Piper SDK
   PosCmd` selector. Pink remains a separate offline diagnostic path.

Training and inference must use the same Cartesian convention, including base
frame, TCP, translation units, orientation representation, action horizon, and
30 Hz timestamps. IK is an execution transform; it cannot correct a mismatched
learning schema.
