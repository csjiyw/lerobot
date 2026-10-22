# Offline z / residual-actor pipeline

Stage-1 z encoder + bounded residual actor for the cobot workpieces, trained
offline from the recorded episodes. Checkpoints and datasets are NOT committed
(`lerobot/outputs/` is gitignored).

## Layout expected

```
<workspace>/datasets/...          # LeRobot datasets (not in git)
<workspace>/lerobot/              # this repo
<workspace>/lerobot/outputs/...   # checkpoints and .npz bundles (gitignored)
```

Scripts resolve `REPO_ROOT = <repo>/`, `OUTPUTS = <repo>/outputs`,
`DATASETS = <workspace>/datasets` from their own location, so they can be run
from any cwd.

## Scripts

| script | purpose |
|---|---|
| `train_z_stage1.py` | stage-1 z (frozen VLA prefix -> attention pool -> 128-d z, future-EEF t+15/t+30 + gripper heads) |
| `train_z_pairs.py` | experimental z with temporal smoothness (term measured ~0; not needed) |
| `build_actor_data.py` | per-decision dataset: z, state, 60k ref chunk, executed action, policy/human label |
| `build_actor_data_old.py` | historical zero-correction anchors (policy rows, target = ref) |
| `merge_actor_data.py` | merge current-env + anchor arrays (old episodes offset by 1000) |
| `train_residual_actor.py` | `action = ref + cap*tanh(MLP(z,state,ref))`, BC, optional input standardization |
| `eval_composed_offline.py` | per-episode / truncation offline evaluation |
| `probe_z.py` | linear probe: z -> needed correction (R^2 acceptance check) |
| `run_offline_pipeline.sh` | one-click: z -> data -> anchors -> merge -> actor -> eval |

## Validated configuration

- z: 128 dims, EEF + gripper heads, per-sample LayerNorm;
- actor: cap = 1.0 x action std, human weight 1, historical anchors,
  per-dim input standardization (mu/sd saved in `config.json`);
- held-out episode: human-row error -54%, policy-row correction ~0.008;
- recommended dataset flags: `STEPS_Z=3000 STEPS_A=2000 CUR_STRIDE=2 OLD_STRIDE=200`.

## Deploy

```bash
python -m cobot_bridge.policy_server --bind tcp://127.0.0.1:5560 --cameras front right \
  --policy-path <60k checkpoint> \
  --residual-actor outputs/actor_v5 --z-encoder outputs/z_stage1_v2_128
```

or via the dashboard API with `residual_actor` / `z_encoder` fields
(`/api/policy-server/start`).
