#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_chap5_generation.sh
#
# Optional env vars:
#   SFT_MODEL_PATH
#   CLASSIFIER_CKPTS   # comma-separated classifier checkpoints for filter modes != full
#   AUG_STRATEGY       # fixed|balance|ratio
#   FIXED_ADD          # default 100
#   TARGET_NUM         # default 500
#   RATIO              # default 1.0
#   BUDGET_MODE        # static|entropy_dynamic, default static
#   BUDGET_TOTAL       # default -1 (use sum of base budget)
#   ENTROPY_CKPT       # classifier ckpt used for entropy budgeting
#   ENTROPY_MIX_IMB    # default 0.3
#   ENTROPY_TEMP       # default 1.0
#   ENTROPY_MAX_SPC    # max samples per class when estimating entropy, default 200
#   USE_MULTISCALE     # 0|1, default 1
#   CONF_MIN           # default 0.50
#   ENT_MAX            # default 0.80
#   DISAG_MAX          # default 0.20
#   SEED               # default 42

# Required env vars:
DATA_NAME=DOTA
PERSONAL_DATA_PATH=/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1
OUTPUT_ROOT="./outputs/Generated_Results/${DATA_NAME}"
SFT_MODEL_PATH=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/DOTA_030321/ar-ckpt-giter005K-ep3-iter365-last.pth
CLASSIFIER_CKPTS=resnet18:/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/models/Mulit_Classifer/FGSC/resnet18_bestmodel.pth,mobilenet_v2:/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/models/Mulit_Classifer/DIOR/mobilenet_v2_bestmodel.pth,efficientnet_b0:/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/models/Mulit_Classifer/DIOR/efficientnet_b0_bestmodel.pth

AUG_STRATEGY="${AUG_STRATEGY:-balance}"
FIXED_ADD="${FIXED_ADD:-100}"
TARGET_NUM="${TARGET_NUM:-1000}"
RATIO="${RATIO:-1.0}"
BUDGET_MODE="${BUDGET_MODE:-static}"
BUDGET_TOTAL="${BUDGET_TOTAL:--1}"
ENTROPY_CKPT="${ENTROPY_CKPT:-}"
ENTROPY_MIX_IMB="${ENTROPY_MIX_IMB:-0.3}"
ENTROPY_TEMP="${ENTROPY_TEMP:-1.0}"
ENTROPY_MAX_SPC="${ENTROPY_MAX_SPC:-200}"
USE_MULTISCALE="${USE_MULTISCALE:-0}"
CONF_MIN="${CONF_MIN:-0.10}"
ENT_MAX="${ENT_MAX:-9.66}"
DISAG_MAX="${DISAG_MAX:-0.20}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --data_name "${DATA_NAME}"
  --personal_data_path "${PERSONAL_DATA_PATH}"
  --augment_strategy "${AUG_STRATEGY}"
  --fixed_add "${FIXED_ADD}"
  --target_num "${TARGET_NUM}"
  --ratio "${RATIO}"
  --budget_mode "${BUDGET_MODE}"
  --budget_total "${BUDGET_TOTAL}"
  --entropy_mix_imbalance "${ENTROPY_MIX_IMB}"
  --entropy_temperature "${ENTROPY_TEMP}"
  --entropy_max_samples_per_class "${ENTROPY_MAX_SPC}"
  --use_multiscale_candidates "${USE_MULTISCALE}"
  --seed "${SEED}"
  --confidence_min "${CONF_MIN}"
  --entropy_max "${ENT_MAX}"
  --disagreement_max "${DISAG_MAX}"
  --use_reference_image 1
)

if [[ -n "${SFT_MODEL_PATH:-}" ]]; then
  COMMON_ARGS+=(--sft_model_path "${SFT_MODEL_PATH}")
fi

if [[ -n "${ENTROPY_CKPT}" ]]; then
  COMMON_ARGS+=(--entropy_ckpt "${ENTROPY_CKPT}")
fi

echo "[RUN] mode=full"
python3 generate_data.py \
  "${COMMON_ARGS[@]}" \
  --filter_mode full \
  --save_dir "${OUTPUT_ROOT}/var_full"


# if [[ -z "${CLASSIFIER_CKPTS:-}" ]]; then
#   echo "[WARN] CLASSIFIER_CKPTS is empty. Skip confidence/entropy/joint modes."
#   exit 0
# fi

# echo "[RUN] mode=topk_confidence"
# python3 generate_data.py \
#   "${COMMON_ARGS[@]}" \
#   --filter_mode topk_confidence \
#   --classifier_ckpts "${CLASSIFIER_CKPTS}" \
#   --save_dir "${OUTPUT_ROOT}/var_topk_confidence"

# echo "[RUN] mode=topk_entropy"
# python3 generate_data.py \
#   "${COMMON_ARGS[@]}" \
#   --filter_mode topk_entropy \
#   --classifier_ckpts "${CLASSIFIER_CKPTS}" \
#   --save_dir "${OUTPUT_ROOT}/var_topk_entropy"

# echo "[RUN] mode=joint"
# python3 generate_data.py \
#   "${COMMON_ARGS[@]}" \
#   --filter_mode joint \
#   --classifier_ckpts "${CLASSIFIER_CKPTS}" \
#   --save_dir "${OUTPUT_ROOT}/var_joint"

echo "[DONE] generation experiments completed under ${OUTPUT_ROOT}"