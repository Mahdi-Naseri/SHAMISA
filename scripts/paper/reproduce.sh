#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/shamisa_a0.yaml}"
TRAIN_EXPERIMENT="${TRAIN_EXPERIMENT:-shamisa_a0_reference}"
ANALYSIS_EXPERIMENT="${ANALYSIS_EXPERIMENT:-shamisa_tsne}"
COMPARISON_EXPERIMENT="${COMPARISON_EXPERIMENT:-baseline_reference}"
DYNAMICS_EXPERIMENT="${DYNAMICS_EXPERIMENT:-shamisa_dynamics_a0}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data_base_path}"
WATERLOO_ROOT="${WATERLOO_ROOT:-${DATA_ROOT}/WaterlooExploration}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN=0
ACTION=""

usage() {
  cat <<USAGE
Usage: scripts/paper/reproduce.sh [options] <action>

Actions:
  prepare-splits  Create the split files expected by the dataset loaders.
  train-a0        Pre-train the SHAMISA A0 encoder.
  eval-main       Evaluate the six NR-IQA benchmarks.
  eval-cross      Evaluate cross-dataset transfer on synthetic benchmarks.
  eval-fr         Evaluate the FR extension on synthetic datasets.
  ablations       Launch the paper ablation runs.
  tsne            Generate the KADID-10K t-SNE figures.
  umap            Generate the manifold plots.
  gmad            Generate the Waterloo gMAD artifacts.
  dynamics        Run the metrics-over-time training pass.
  plot-dynamics   Render the metrics-over-time figures.
  all             Run the public paper workflow in order.

Options:
  --dry-run                Print commands without executing them.
  --python-bin PATH        Python executable (default: ${PYTHON_BIN}).
  --config PATH            Base config file (default: ${CONFIG_PATH}).
  --train-experiment NAME  Experiment name for train/eval-main/eval-fr (default: ${TRAIN_EXPERIMENT}).
  --analysis-experiment NAME
                           Experiment name for analysis actions (default: ${ANALYSIS_EXPERIMENT}).
  --comparison-experiment NAME
                           Comparison experiment for UMAP and gMAD (default: ${COMPARISON_EXPERIMENT}).
  --dynamics-experiment NAME
                           Experiment name for metrics-over-time (default: ${DYNAMICS_EXPERIMENT}).
  --data-root PATH         Dataset root containing LIVE/CSIQ/TID2013/KADID10K/FLIVE/SPAQ.
  --waterloo-root PATH     Waterloo Exploration root for gMAD.
  --device VALUE           Device string for analysis tools (default: ${DEVICE}).
  --list                   List available actions.
  -h, --help               Show this message.
USAGE
}

list_actions() {
  cat <<LIST
prepare-splits
train-a0
eval-main
eval-cross
eval-fr
ablations
tsne
umap
gmad
dynamics
plot-dynamics
all
LIST
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

run_action() {
  case "$1" in
    prepare-splits)
      run_cmd "$PYTHON_BIN" scripts/data/prepare_splits.py --data-root "$DATA_ROOT"
      ;;
    train-a0)
      run_cmd "$PYTHON_BIN" main.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT"
      ;;
    eval-main)
      run_cmd "$PYTHON_BIN" test.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT" \
        --eval_type scratch \
        --model.method vicreg \
        --test.fast_mode false \
        --test.datasets live csiq tid2013 kadid10k flive spaq
      ;;
    eval-cross)
      run_cmd "$PYTHON_BIN" test.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT" \
        --eval_type scratch \
        --model.method vicreg \
        --test.fast_mode false \
        --test.cross_eval.enabled true \
        --test.cross_eval.train_datasets live csiq tid2013 kadid10k \
        --test.cross_eval.test_datasets live csiq tid2013 kadid10k
      ;;
    eval-fr)
      run_cmd "$PYTHON_BIN" test.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT" \
        --eval_type scratch \
        --model.method vicreg \
        --fr_iqa \
        --fr_sanity_checks \
        --fr_sanity_samples 64 \
        --plcc_logistic 1 \
        --test.fast_mode false \
        --test.datasets live csiq tid2013 kadid10k
      ;;
    ablations)
      run_cmd bash scripts/paper/run_ablations.sh \
        --python-bin "$PYTHON_BIN" \
        --config "$CONFIG_PATH" \
        --data-root "$DATA_ROOT"
      ;;
    tsne)
      run_cmd "$PYTHON_BIN" tools/tsne_embeddings.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$ANALYSIS_EXPERIMENT" \
        --eval_type scratch \
        --checkpoint best \
        --dataset_name kadid10k \
        --seed 123 \
        --color_by distortion_group \
        --also_plot_severity \
        --crop_policy mean \
        --make_bundle_zip true \
        --palette_mode shared
      ;;
    umap)
      run_cmd "$PYTHON_BIN" tools/umap_manifold.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT" \
        --eval_type scratch \
        --checkpoint best \
        --device "$DEVICE"
      ;;
    gmad)
      run_cmd "$PYTHON_BIN" tools/gmad_score_waterloo.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$TRAIN_EXPERIMENT" \
        --eval_type scratch \
        --dataset_root "$WATERLOO_ROOT" \
        --subset distorted \
        --output_scores "results/gmad/${TRAIN_EXPERIMENT}_waterloo_scores.npz" \
        --regressor_train_dataset kadid10k \
        --device "$DEVICE"
      run_cmd "$PYTHON_BIN" tools/gmad_score_waterloo.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$COMPARISON_EXPERIMENT" \
        --eval_type scratch \
        --dataset_root "$WATERLOO_ROOT" \
        --subset distorted \
        --output_scores "results/gmad/${COMPARISON_EXPERIMENT}_waterloo_scores.npz" \
        --regressor_train_dataset kadid10k \
        --device "$DEVICE"
      run_cmd "$PYTHON_BIN" tools/gmad_pairs.py \
        --scores_a "results/gmad/${TRAIN_EXPERIMENT}_waterloo_scores.npz" \
        --scores_b "results/gmad/${COMPARISON_EXPERIMENT}_waterloo_scores.npz" \
        --name_a shamisa \
        --name_b comparison \
        --num_bins 10 \
        --top_k 1 \
        --output_dir "results/gmad/${TRAIN_EXPERIMENT}_vs_${COMPARISON_EXPERIMENT}" \
        --copy_images \
        --image_root "$WATERLOO_ROOT"
      run_cmd "$PYTHON_BIN" tools/gmad_render_panels.py \
        --summary "results/gmad/${TRAIN_EXPERIMENT}_vs_${COMPARISON_EXPERIMENT}/gmad_summary.json" \
        --images_dir "results/gmad/${TRAIN_EXPERIMENT}_vs_${COMPARISON_EXPERIMENT}/images" \
        --output_dir "results/gmad/${TRAIN_EXPERIMENT}_vs_${COMPARISON_EXPERIMENT}/panels" \
        --panel_style paper \
        --disable_wandb
      ;;
    dynamics)
      run_cmd "$PYTHON_BIN" main.py \
        --config "$CONFIG_PATH" \
        --data_base_path "$DATA_ROOT" \
        --experiment_name "$DYNAMICS_EXPERIMENT" \
        --enable_metrics_over_time true \
        --metrics_eval_interval_steps 5000 \
        --metrics_eval_dtype float32 \
        --metrics_eval_save_every 1 \
        --training.epochs 2 \
        --logging.wandb.imgs false
      ;;
    plot-dynamics)
      run_cmd "$PYTHON_BIN" tools/plot_training_metrics.py \
        --metrics_file "experiments/${DYNAMICS_EXPERIMENT}/metrics_over_time.pt" \
        --output_dir "experiments/${DYNAMICS_EXPERIMENT}/figs" \
        --make_multipanel_main_metric true
      ;;
    all)
      run_action prepare-splits
      run_action train-a0
      run_action eval-main
      run_action eval-cross
      run_action eval-fr
      run_action ablations
      run_action tsne
      run_action umap
      run_action gmad
      run_action dynamics
      run_action plot-dynamics
      ;;
    *)
      echo "Unknown action: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
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
    --train-experiment)
      TRAIN_EXPERIMENT="$2"
      shift 2
      ;;
    --analysis-experiment)
      ANALYSIS_EXPERIMENT="$2"
      shift 2
      ;;
    --comparison-experiment)
      COMPARISON_EXPERIMENT="$2"
      shift 2
      ;;
    --dynamics-experiment)
      DYNAMICS_EXPERIMENT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --waterloo-root)
      WATERLOO_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --list)
      list_actions
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$ACTION" ]]; then
        echo "Only one action can be specified." >&2
        usage >&2
        exit 1
      fi
      ACTION="$1"
      shift
      ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"
run_action "$ACTION"
