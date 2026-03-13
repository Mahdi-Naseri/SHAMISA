#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${REPO_ROOT}/.smoke/readme"
mkdir -p "$OUT_DIR"

run_and_mark() {
  local name="$1"
  shift
  local log_path="${OUT_DIR}/${name}.log"
  local ok_path="${OUT_DIR}/${name}.ok"

  echo "[SMOKE] ${name}"
  "$@" >"$log_path" 2>&1
  touch "$ok_path"
}

cd "$REPO_ROOT"

run_and_mark "prepare_splits_help" "$PYTHON_BIN" scripts/data/prepare_splits.py --help
run_and_mark "main_help" "$PYTHON_BIN" main.py --help
run_and_mark "test_help" "$PYTHON_BIN" test.py --help
run_and_mark "reproduce_help" bash scripts/paper/reproduce.sh --help
run_and_mark "reproduce_list" bash scripts/paper/reproduce.sh --list
run_and_mark "reproduce_prepare_splits_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data prepare-splits
run_and_mark "reproduce_train_a0_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data train-a0
run_and_mark "reproduce_eval_main_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data eval-main
run_and_mark "reproduce_eval_cross_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data eval-cross
run_and_mark "reproduce_eval_fr_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data eval-fr
run_and_mark "reproduce_ablations_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data ablations
run_and_mark "reproduce_tsne_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data tsne
run_and_mark "reproduce_umap_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data umap
run_and_mark "reproduce_gmad_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data --waterloo-root /tmp/waterloo gmad
run_and_mark "reproduce_dynamics_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data dynamics
run_and_mark "reproduce_plot_dynamics_dry_run" bash scripts/paper/reproduce.sh --dry-run plot-dynamics
run_and_mark "reproduce_all_dry_run" bash scripts/paper/reproduce.sh --dry-run --data-root /tmp/shamisa_data --waterloo-root /tmp/waterloo all
run_and_mark "ablations_help" bash scripts/paper/run_ablations.sh --help
run_and_mark "ablations_dry_run" bash scripts/paper/run_ablations.sh --dry-run --data-root /tmp/shamisa_data --cases A0 B1 F1
run_and_mark "unit_tests" "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v

echo "README smoke checks passed. Logs are in ${OUT_DIR}."
