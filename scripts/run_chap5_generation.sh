#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_chap5_generation.sh
#
# Required env vars:
#   TRAIN_PATH
#   TEST_PATH
#   PERSONAL_DATA_PATH
#   OUTPUT_ROOT
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

if [[ -z "${TRAIN_PATH:-}" || -z "${TEST_PATH:-}" || -z "${PERSONAL_DATA_PATH:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "ERROR: TRAIN_PATH, TEST_PATH, PERSONAL_DATA_PATH, OUTPUT_ROOT are required."
  exit 1
fi

AUG_STRATEGY="${AUG_STRATEGY:-fixed}"
FIXED_ADD="${FIXED_ADD:-100}"
TARGET_NUM="${TARGET_NUM:-500}"
RATIO="${RATIO:-1.0}"
BUDGET_MODE="${BUDGET_MODE:-static}"
BUDGET_TOTAL="${BUDGET_TOTAL:--1}"
ENTROPY_CKPT="${ENTROPY_CKPT:-}"
ENTROPY_MIX_IMB="${ENTROPY_MIX_IMB:-0.3}"
ENTROPY_TEMP="${ENTROPY_TEMP:-1.0}"
ENTROPY_MAX_SPC="${ENTROPY_MAX_SPC:-200}"
USE_MULTISCALE="${USE_MULTISCALE:-1}"
CONF_MIN="${CONF_MIN:-0.50}"
ENT_MAX="${ENT_MAX:-0.80}"
DISAG_MAX="${DISAG_MAX:-0.20}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --train_path "${TRAIN_PATH}"
  --test_path "${TEST_PATH}"
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

if [[ -z "${CLASSIFIER_CKPTS:-}" ]]; then
  echo "[WARN] CLASSIFIER_CKPTS is empty. Skip confidence/entropy/joint modes."
  exit 0
fi

echo "[RUN] mode=confidence"
python3 generate_data.py \
  "${COMMON_ARGS[@]}" \
  --filter_mode confidence \
  --classifier_ckpts "${CLASSIFIER_CKPTS}" \
  --save_dir "${OUTPUT_ROOT}/var_confidence"

echo "[RUN] mode=entropy"
python3 generate_data.py \
  "${COMMON_ARGS[@]}" \
  --filter_mode entropy \
  --classifier_ckpts "${CLASSIFIER_CKPTS}" \
  --save_dir "${OUTPUT_ROOT}/var_entropy"

echo "[RUN] mode=joint"
python3 generate_data.py \
  "${COMMON_ARGS[@]}" \
  --filter_mode joint \
  --classifier_ckpts "${CLASSIFIER_CKPTS}" \
  --save_dir "${OUTPUT_ROOT}/var_joint"

echo "[DONE] generation experiments completed under ${OUTPUT_ROOT}"
