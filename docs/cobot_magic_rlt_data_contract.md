# Cobot Magic RLT data contract

Raw episodes use `cobot_magic_raw_v2`. Collection runs at the dataset rate
(normally 20 or 30 Hz). The AGX may interpolate commands at 200 Hz, but the
dataset only stores the command sampled at each dataset frame.

Every new episode has an explicit `collection_type` HDF5 attribute (`IL`, `RL`,
or `REPLAY`). Conversion copies its numeric value to `rlt.collection_type`
(`IL=0`, `RL=1`, `REPLAY=2`; legacy unmarked episodes use `-1`).

## Action fields

- `reference_action`: the frozen VLA action before the RLT actor changes it.
- `policy_action`: the action produced by the active policy/RLT actor.
- `human_action`: the master-arm target while effective control mode is Teach.
- `executed_action`: the arbiter puppet command sampled at the dataset frame.
- `puppet_position`: measured puppet feedback; it is state, not an action.

Every action has a matching `_valid` field. Invalid vectors are stored as zero;
consumers must use validity flags. In the replay-baseline migration,
`reference_action == policy_action` because no RLT actor was present.

## Phase and learning eligibility

`phase` values are `other`, `alignment`, `policy`, `loop_transition`, `human`,
and `terminal`. `learning_mask` is one only for policy and human frames.
Alignment, loop-return transitions, SAFE/RESYNC, and the terminal frame are
retained for sequence context but excluded from normal policy/RL sampling.

`cycle_id` identifies a replay/policy cycle. `intervention_event` and
`intervention_id` are derived from the effective Teach mode, not solely from
the delayed Teach topic. A rising event is `+1`; release is `-1`.

## Episode result

The final frame has `done=1`. `success` uses `-1=unknown`, `0=failure`,
`1=success`, and `2=aborted`. Reward is only meaningful where
`reward_valid=1`. By task definition, any episode containing a Human/Teach
intervention is labeled successful and receives terminal reward 1. An explicit
failure or aborted outcome still overrides this automatic rule.

RL dashboard recording does not stop on Teach release. The operator ends the
episode with **Save Success** or **Save Failure**. That result is written before
the recorder closes, sets the terminal reward/success fields, and stops the
active rollout. This permits a no-intervention policy rollout to be explicitly
saved as either success or failure.

The LeRobot conversion stores the auxiliary fields under `rlt.*` so standard
policy feature inference does not mistake all four action records for model
outputs. The standard `action` column is selected at conversion time and falls
back to a measured hold target on frames where the selected action is invalid;
training must still honor `rlt.learning_mask` and the action validity flags.

## Persistent Stage 2 rollout service

Online RLT uses three cooperating processes:

- GPU 0 runs `cobot_bridge.rlt_policy_server` and keeps the frozen SmolVLA,
  Stage 1 encoder and rollout actor resident across episodes.
- `cobot_bridge.rlt_online` is a collection/control client. It records the
  server-returned normalized `z_rl`, state, reference, policy and selected
  executed chunk, while sending only the unnormalized selected chunk to AGX.
- GPU 1 runs the detached post-episode actor/critic trainer. It publishes
  `latest.pt` atomically.

The server checks `latest.pt` only on `BEGIN_EPISODE`. A checkpoint published
during an episode is reported as pending and is adopted at the next episode
boundary. This guarantees that one episode never mixes two actor generations.
Starting or stopping the server does not connect to robot control; stopping is
blocked while an Online RLT episode is active.

Stage 2 starts with a separate five-episode warm-up phase. A valid warm-up
episode must belong to the exact selected Stage 1 checkpoint, declare
`collection_phase=warmup`, use Mode 1, and contain only `reference` decisions.
The dashboard reports this verified count directly as `x/5`; it is not a
countdown and does not include legacy, actor, intervention, or another Stage 1
run's replay bundles. Warm-up episodes 1–4 are archived without optimizer
updates. Episode 5/5 triggers the first GPU 1 actor/critic bootstrap. Actor
Mode 1/2 rollout remains hard-locked until an atomic Stage 2 checkpoint with
`total_updates > 0` is published for the same Stage 1 checkpoint.
