#!/usr/bin/env bash
# Wait for the metal_v4 pipeline to finish, then run occlusion + holdout audits on GPU0.
set -uo pipefail
cd /media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot
PY=/home/fuxin/anaconda3/envs/lerobot0_4_0/bin/python
BASE=outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model
P2=outputs/cobot_magic_smolvla_metal_v4_phase2_unfreeze3_20k_bs32_gpu_256_photoaug_gw/checkpoints/020000/pretrained_model
OLD=outputs/cobot_magic_smolvla_workpiece_all407_anchor_relative_eef_chunk30_resume20k_to60k_bs32_gpu1_256_photoaug_cs8/checkpoints/060000/pretrained_model
OUT=../dataset_audit/v4/audit_v4_final
G=../dataset_audit/v4
TR=../datasets/cobot_magic_metal_v4
TR_EP="393,396,399,402,405,408,412,416,420,423,0,120,250,320,380,24,50,100,430,445,460,480,497"
deadline=$(( $(date +%s) + 8*3600 ))
miss=0
while (( $(date +%s) < deadline )); do
  if [[ -f "$P2/config.json" ]]; then break; fi
  if grep -q "Traceback" outputs/metal_v4_phase2_20k.log 2>/dev/null; then break; fi
  if grep -q "\[pipeline\] phase2 done" outputs/metal_v4_pipeline.log 2>/dev/null; then sleep 30; break; fi
  if pgrep -f "lerobot.scripts.lerobot_train" >/dev/null; then
    miss=0
  else
    miss=$((miss+1))
    # tolerate the phase1->phase2 relaunch gap (~60-90s); 3 misses at 40s spacing = real exit
    if (( miss >= 3 )); then sleep 30; break; fi
  fi
  sleep 40
done
echo "[watch] start audits $(date)"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
audit() { # ckpt ds repo groups out eparg...
  local ckpt="$1" ds="$2" repo="$3" groups="$4" out="$5"; shift 5
  $PY scripts/eval_metal_occlusion_audit.py --checkpoint "$ckpt" --dataset-root "$ds" --repo-id "$repo" \
    --max-samples-per-episode 6 --env-groups-json "$groups" --device cuda --out "$out" "$@" || echo "[watch] FAILED $out"
}
for ep in env3 env2 oldenv; do
  DS=../datasets/cobot_magic_metal_v4_eval_${ep}
  GR=$G/cobot_magic_metal_v4_eval_${ep}_groups.json
  audit "$OLD" "$DS" local/eval_$ep "$GR" $OUT/baseline60k_$ep --episodes all
  audit "$BASE" "$DS" local/eval_$ep "$GR" $OUT/phase1_$ep --episodes all
  [[ -f "$P2/config.json" ]] && audit "$P2" "$DS" local/eval_$ep "$GR" $OUT/phase2_$ep --episodes all
done
audit "$OLD" "$TR" local/train "$TR/meta/env_groups.json" $OUT/baseline60k_trainmix --episodes "$TR_EP"
audit "$BASE" "$TR" local/train "$TR/meta/env_groups.json" $OUT/phase1_trainmix --episodes "$TR_EP"
[[ -f "$P2/config.json" ]] && audit "$P2" "$TR" local/train "$TR/meta/env_groups.json" $OUT/phase2_trainmix --episodes "$TR_EP"
echo "[watch] audits done $(date)"
