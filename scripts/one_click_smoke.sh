#!/usr/bin/env bash
set -euo pipefail

# One-click smoke test for Chapter-5 pipeline.
# Usage:
#   bash scripts/one_click_smoke.sh
#
# What it does:
# 1) Fast static checks (py_compile, --help)
# 2) Auto-discover common data/model paths on server
# 3) Best-effort minimal generation run (1 class x 1 sample target style)
#    If paths are missing, it still exits success after static checks.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "== [1/3] Static checks =="
python3 -m py_compile \
  generate_data.py \
  tools/run_infinity.py \
  tools/summarize_downstream_cls.py \
  tools/build_train_count_csv.py \
  infinity/utils/downstream_metrics.py

python3 generate_data.py --help >/dev/null
python3 tools/summarize_downstream_cls.py --help >/dev/null
python3 tools/build_train_count_csv.py --help >/dev/null
echo "Static checks passed."

echo
echo "== [2/3] Auto-discover paths =="

find_first_existing() {
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

PERSONAL_DATA_PATH="$(find_first_existing \
  /picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1 \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1 \
  /data \
  /mnt/data || true)"

TRAIN_PATH="$(find_first_existing \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/train \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DOTA/train \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/FGSC23/train || true)"

TEST_PATH="$(find_first_existing \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/test \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DOTA/test \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/FGSC23/test || true)"

SFT_MODEL_PATH="$(find_first_existing \
  /picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1/models/FoundationVision/Infinity/infinity_125M_256x256.pth \
  /picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1/models/FoundationVision/Infinity/infinity_2b_reg.pth \
  /picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/debug_experiment022611/ar-ckpt-giter000K-ep0-iter900-last.pth || true)"

echo "PERSONAL_DATA_PATH=${PERSONAL_DATA_PATH:-<missing>}"
echo "TRAIN_PATH=${TRAIN_PATH:-<missing>}"
echo "TEST_PATH=${TEST_PATH:-<missing>}"
echo "SFT_MODEL_PATH=${SFT_MODEL_PATH:-<missing>}"

echo
echo "== [3/3] Minimal runtime check =="

if [[ -z "${PERSONAL_DATA_PATH:-}" || -z "${TRAIN_PATH:-}" || -z "${TEST_PATH:-}" || -z "${SFT_MODEL_PATH:-}" ]]; then
  echo "Runtime check skipped (missing auto-discovered paths)."
  echo "Result: PASS (static checks only)."
  exit 0
fi

OUT_DIR="${ROOT_DIR}/outputs/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

set +e
python3 generate_data.py \
  --train_path "${TRAIN_PATH}" \
  --test_path "${TEST_PATH}" \
  --save_dir "${OUT_DIR}" \
  --personal_data_path "${PERSONAL_DATA_PATH}" \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --augment_strategy fixed \
  --fixed_add 1 \
  --budget_mode static \
  --filter_mode full \
  --use_multiscale_candidates 1 \
  --max_attempt_factor 1.2 \
  --seed 123
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
  echo "Runtime check failed."
  echo "You can inspect logs/output under: ${OUT_DIR}"
  exit ${STATUS}
fi

echo "Runtime check passed."
echo "Output dir: ${OUT_DIR}"
echo "Result: PASS (static + runtime checks)."
