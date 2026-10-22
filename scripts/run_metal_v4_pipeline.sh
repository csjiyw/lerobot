#!/usr/bin/env bash
# Chain: phase1 (60k, vision frozen) -> phase2 (20k, last 3 SigLIP blocks unfrozen, LR 2e-5).
# Each phase is a fresh optimizer run; phase2 loads phase1 weights via pretrained_path.
set -euo pipefail
cd /media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/lerobot
P1=outputs/cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw
P2=outputs/cobot_magic_smolvla_metal_v4_phase2_unfreeze3_20k_bs32_gpu_256_photoaug_gw
echo "[pipeline] phase1 start $(date)"
COBOT_RUN_NAME=cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw \
COBOT_TRAIN_STEPS=60000 COBOT_LR=1e-4 COBOT_WARMUP=500 COBOT_DECAY_STEPS=60000 COBOT_DECAY_LR=2.5e-6 \
COBOT_UNFREEZE_VISION=0 COBOT_BATCH_SIZE=32 COBOT_CUDA_DEVICE=1 COBOT_SAVE_FREQ=5000 \
bash scripts/train_metal_v4.sh > outputs/metal_v4_phase1_60k.log 2>&1
echo "[pipeline] phase1 done $(date)"
echo "[pipeline] phase2 start $(date)"
COBOT_BASE_MODEL="${PWD}/${P1}/checkpoints/060000/pretrained_model" \
COBOT_RUN_NAME=cobot_magic_smolvla_metal_v4_phase2_unfreeze3_20k_bs32_gpu_256_photoaug_gw \
COBOT_TRAIN_STEPS=20000 COBOT_LR=2e-5 COBOT_WARMUP=200 COBOT_DECAY_STEPS=20000 COBOT_DECAY_LR=5e-7 \
COBOT_UNFREEZE_VISION=3 COBOT_BATCH_SIZE=32 COBOT_CUDA_DEVICE=1 COBOT_SAVE_FREQ=5000 \
bash scripts/train_metal_v4.sh > outputs/metal_v4_phase2_20k.log 2>&1
echo "[pipeline] phase2 done $(date)"
