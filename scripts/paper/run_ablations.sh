#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/shamisa_a0.yaml}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data_base_path}"
DRY_RUN=0
CASES=(A0 A1 A2 A3 A4 B1 B2 B3 B4 B5 B6 B7 B8 C1 D1a D1b D2 D3 D4 E1 F1)

usage() {
  cat <<USAGE
Usage: scripts/paper/run_ablations.sh [options]

Options:
  --dry-run              Print commands without executing them.
  --python-bin PATH      Python executable (default: ${PYTHON_BIN}).
  --config PATH          Base config file (default: ${CONFIG_PATH}).
  --data-root PATH       Dataset root containing the training and evaluation sets.
  --cases LIST           Space-separated case IDs (default: all paper ablations).
  -h, --help             Show this message.
USAGE
}

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --cases)
      shift
      CASES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CASES+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"

STAMP=$(date +"%Y%m%d_%H%M%S")
SUFFIX=$($PYTHON_BIN - <<'PY'
import random
import string
print(''.join(random.choices(string.ascii_letters + string.digits, k=8)))
PY
)

base_cmd=(
  "$PYTHON_BIN"
  main.py
  "--config=${CONFIG_PATH}"
  "--data_base_path=${DATA_ROOT}"
  "--logging.use_wandb=False"
  "--logging.wandb.online=False"
  "--logging.wandb.imgs=False"
)

run_case() {
  local case_id="$1"
  shift
  local experiment_name="${case_id}_${STAMP}_${SUFFIX}"
  run_cmd "${base_cmd[@]}" "--experiment_name=${experiment_name}" "$@"
}

for case_id in "${CASES[@]}"; do
  case "$case_id" in
    A0)
      run_case "A0_REFERENCE"
      ;;
    A1)
      run_case "A1_BINARY_RELATIONS" \
        --model.relations.soft_afgrl=False \
        --model.relations.branches.distortion_ref_dist.soft_entries=False \
        --model.relations.branches.distortion_dist_dist.soft_entries=False \
        --model.relations.branches.distortion_ref_ref.soft_entries=False \
        --model.relations.branches.feature_knn.soft_entries=False \
        --model.relations.branches.feature_transport.soft_entries=False
      ;;
    A2)
      run_case "A2_NO_RELATION_REG" \
        --model.relations.regularizer.active=False
      ;;
    A3)
      run_case "A3_FIXED_BRANCH_WEIGHTS" \
        --model.relations.weighting.active=False
      ;;
    A4)
      run_case "A4_NO_STOP_GRAD" \
        --model.relations.weighting.stop_grad=False
      ;;
    B1)
      run_case "B1_NO_TRANSPORT" \
        --model.relations.branches.feature_transport.active=False
      ;;
    B2)
      run_case "B2_NO_KNN" \
        --model.relations.branches.feature_knn.active=False
      ;;
    B3)
      run_case "B3_NO_REF_DIST" \
        --model.relations.branches.distortion_ref_dist.active=False
      ;;
    B4)
      run_case "B4_NO_DIST_DIST" \
        --model.relations.branches.distortion_dist_dist.active=False
      ;;
    B5)
      run_case "B5_NO_REF_REF" \
        --model.relations.branches.distortion_ref_ref.active=False
      ;;
    B6)
      run_case "B6_KNN_ONLY" \
        --model.relations.branches.distortion_structural.active=False \
        --model.relations.branches.distortion_ref_dist.active=False \
        --model.relations.branches.distortion_ref_ref.active=False \
        --model.relations.branches.distortion_dist_dist.active=False \
        --model.relations.branches.feature_transport.active=False
      ;;
    B7)
      run_case "B7_METADATA_ONLY" \
        --model.relations.branches.feature_knn.active=False \
        --model.relations.branches.feature_transport.active=False
      ;;
    B8)
      run_case "B8_STRUCTURE_ONLY" \
        --model.relations.branches.distortion_structural.active=False \
        --model.relations.branches.distortion_ref_dist.active=False \
        --model.relations.branches.distortion_ref_ref.active=False \
        --model.relations.branches.distortion_dist_dist.active=False
      ;;
    C1)
      run_case "C1_TRANSPORT_TOPK_EACH" \
        --model.relations.branches.feature_transport.transport.g_sparse=topk_each \
        --model.relations.branches.feature_transport.transport.g_sparse_k=8
      ;;
    D1a)
      run_case "D1A_MAX_DIST_1" \
        --training.data.max_distortions=1 \
        --training.data.cache_mode=na
      ;;
    D1b)
      run_case "D1B_MAX_DIST_7" \
        --training.data.max_distortions=7 \
        --training.data.cache_mode=na
      ;;
    D2)
      run_case "D2_DISCRETE_SEVERITY" \
        --training.data.severity_discrete=True \
        --training.data.num_levels=5 \
        --training.data.cache_mode=na
      ;;
    D3)
      run_case "D3_UNIFORM_SEVERITY" \
        --training.data.severity_dist=uniform \
        --training.data.cache_mode=na
      ;;
    D4)
      run_case "D4_FIXED_ORDER" \
        --training.data.fixed_order=True \
        --training.data.cache_mode=na
      ;;
    E1)
      run_case "E1_NO_TRANSPORT_ALIGNMENT" \
        --model.relations.branches.feature_transport.transport.pq_alignment_coeff=0.0
      ;;
    F1)
      run_case "F1_ORIGINAL_CURVE" \
        --model.relations.distortion_curve=original
      ;;
    *)
      echo "Unknown ablation case: ${case_id}" >&2
      exit 1
      ;;
  esac
done
