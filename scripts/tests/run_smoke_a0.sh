#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DEFAULT=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT="$REPO_ROOT_DEFAULT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="smoke_a0"
OUTPUT_JSON=""
MAX_STEPS=32
BATCH_SIZE=2
CONFIG_PATH="configs/shamisa_a0.yaml"
DATA_ROOT="${REPO_ROOT_DEFAULT}/data_base_path"
DEVICE=""

usage() {
  cat <<USAGE
Usage: scripts/tests/run_smoke_a0.sh [options]

Options:
  --run-name NAME       Logical run label (default: ${RUN_NAME})
  --output-json PATH    Output JSON path (default: .smoke/smoke_a0/<run-name>.json)
  --max-steps N         Training max steps (default: ${MAX_STEPS})
  --batch-size N        Training batch size (default: ${BATCH_SIZE})
  --python-bin PATH     Python executable (default: ${PYTHON_BIN})
  --config PATH         Config file (default: ${CONFIG_PATH})
  --repo-root PATH      Repository root to run against (default: ${REPO_ROOT_DEFAULT})
  --data-root PATH      Dataset root containing KADIS700 and the evaluation datasets
  --device VALUE        Device for train/test (default: auto -> 0 or -1)
  -h, --help            Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
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

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repository root not found: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi
if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="${REPO_ROOT}/${DATA_ROOT}"
fi
if [[ -z "$OUTPUT_JSON" ]]; then
  OUTPUT_JSON="${REPO_ROOT}/.smoke/smoke_a0/${RUN_NAME}.json"
elif [[ "$OUTPUT_JSON" != /* ]]; then
  OUTPUT_JSON="${REPO_ROOT}/${OUTPUT_JSON}"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found on PATH: ${PYTHON_BIN}" >&2
  exit 1
fi

for dataset_name in KADIS700 LIVE; do
  if [[ ! -e "${DATA_ROOT}/${dataset_name}" ]]; then
    echo "Missing dataset entry under data root: ${DATA_ROOT}/${dataset_name}" >&2
    exit 1
  fi
done

if [[ -z "$DEVICE" ]]; then
  DEVICE=$(
    "$PYTHON_BIN" - <<'PY'
try:
    import torch
    print("0" if torch.cuda.is_available() else "-1")
except Exception:
    print("-1")
PY
  )
fi

OUT_DIR=$(dirname "$OUTPUT_JSON")
mkdir -p "$OUT_DIR"
OUTPUT_JSON=$(cd "$OUT_DIR" && pwd)/$(basename "$OUTPUT_JSON")

RUN_TS=$(date +"%Y%m%d_%H%M%S")
RAND_SUFFIX=$(
  "$PYTHON_BIN" - <<'PY'
import random
import string
print(''.join(random.choices(string.ascii_letters + string.digits, k=8)))
PY
)
EXPERIMENT_NAME="SMOKE_A0_${RUN_NAME}_${RUN_TS}_${RAND_SUFFIX}"
TRAIN_LOG="${OUT_DIR}/${RUN_NAME}_train.log"
TEST_LOG="${OUT_DIR}/${RUN_NAME}_test.log"
SPLIT_LOG="${OUT_DIR}/${RUN_NAME}_splits.log"

cd "$REPO_ROOT"

export PYTORCH_SHARING_STRATEGY=file_system
export PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" scripts/data/prepare_splits.py \
  --data-root "$DATA_ROOT" \
  --datasets live \
  --num-splits 1 \
  >"$SPLIT_LOG" 2>&1

"$PYTHON_BIN" main.py \
  --config "$CONFIG_PATH" \
  --experiment_name "$EXPERIMENT_NAME" \
  --data_base_path "$DATA_ROOT" \
  --device "$DEVICE" \
  --seed 27 \
  --logging.use_wandb false \
  --logging.wandb.online false \
  --model.encoder.embedding_dim 64 \
  --model.projector.out_dim 128 \
  --model.projector.hidden_dim 256 \
  --model.relations.branches.feature_transport.transport.n_prototypes 8 \
  --training.epochs 1 \
  --training.max_steps "$MAX_STEPS" \
  --training.batch_size "$BATCH_SIZE" \
  --training.num_workers 0 \
  --training.pin_memory false \
  --training.persistent_workers false \
  --training.data.shuffle false \
  --training.data.patch_size 96 \
  --training.data.max_distortions 2 \
  --training.data.num_levels 3 \
  --training.data.n_refs 1 \
  --training.data.n_dist_comps 1 \
  --training.data.n_dist_comp_levels 2 \
  --training.data.extended_int_distortions false \
  --training.data.cache_mode na \
  --training.test false \
  --validation.skip_phase true \
  --validation.datasets live \
  --validation.num_splits 1 \
  --validation.cross_eval.enabled false \
  --test.datasets live \
  --test.num_splits 1 \
  --test.cross_eval.enabled false \
  2>&1 | tee "$TRAIN_LOG"

BEST_CKPT=$(ls -1t "${REPO_ROOT}/experiments/${EXPERIMENT_NAME}/pretrain"/best*.pth | head -n1)
if [[ -z "$BEST_CKPT" ]]; then
  echo "Best checkpoint not found for ${EXPERIMENT_NAME}" >&2
  exit 1
fi
CKPT_SHA=$(sha256sum "$BEST_CKPT" | awk '{print $1}')
export DATA_ROOT_FOR_SMOKE="$DATA_ROOT"
export SMOKE_DEVICE="$DEVICE"
export SMOKE_RUN_NAME="$RUN_NAME"
export SMOKE_EXPERIMENT_NAME="$EXPERIMENT_NAME"
export SMOKE_CONFIG_PATH="$CONFIG_PATH"
export SMOKE_MAX_STEPS="$MAX_STEPS"
export SMOKE_BATCH_SIZE="$BATCH_SIZE"
export SMOKE_TRAIN_LOG="$TRAIN_LOG"
export SMOKE_TEST_LOG="$TEST_LOG"
export SMOKE_BEST_CKPT="$BEST_CKPT"
export SMOKE_CKPT_SHA="$CKPT_SHA"
export SMOKE_OUTPUT_JSON="$OUTPUT_JSON"
export SMOKE_REPO_ROOT="$REPO_ROOT"

"$PYTHON_BIN" - <<'PY' 2>&1 | tee "$TEST_LOG"
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Subset

from models.associations import finalize_association_config
from models.simclr import SimCLR
from models.vicreg import Vicreg
from test import _load_pretrained_weights, get_features_scores, prepare_dataset
from utils.utils import parse_config, replace_none_string

repo_root = Path(os.environ["SMOKE_REPO_ROOT"])
data_root = Path(os.environ["DATA_ROOT_FOR_SMOKE"])
experiment_name = os.environ["SMOKE_EXPERIMENT_NAME"]
checkpoint_path = Path(os.environ["SMOKE_BEST_CKPT"])
device_arg = os.environ["SMOKE_DEVICE"]

resolved_config = repo_root / "experiments" / experiment_name / "config_resolved.yaml"
args = replace_none_string(parse_config(str(resolved_config)))
args = finalize_association_config(args)
args.data_base_path = data_root

device = torch.device("cpu" if device_arg == "-1" else f"cuda:{device_arg}")

projector_out_dim = getattr(args.model.projector, "out_dim", None)
projector_hidden_dim = getattr(args.model.projector, "hidden_dim", None)

if args.model.method == "simclr":
    model = SimCLR(
        encoder_params=args.model.encoder,
        temperature=args.model.temperature,
        projector_out_dim=projector_out_dim,
        projector_hidden_dim=projector_hidden_dim,
    )
elif args.model.method == "vicreg":
    model = Vicreg(
        args,
        projector_out_dim=projector_out_dim,
        projector_hidden_dim=projector_hidden_dim,
    )
else:
    raise ValueError(f"Unsupported method: {args.model.method}")

_load_pretrained_weights(model, checkpoint_path)
model = model.to(device)
model.eval()

dataset, _, _ = prepare_dataset(
    dataset_key="live",
    data_base_path=data_root,
    crop_size=96,
    default_num_splits=1,
    fr_iqa=False,
)
test_indices = dataset.get_split_indices(split=0, phase="test")
subset_indices = test_indices[: min(len(test_indices), 8)]
subset = Subset(dataset, subset_indices)
loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)

features, _, scores = get_features_scores(
    model,
    loader,
    device,
    eval_type="scratch",
    dtype=torch.float32,
)

feature_dim = features.shape[1]
features = features.reshape(-1, 5, feature_dim).mean(axis=1)
scores = scores.reshape(-1, 5)[:, 0]

split_point = max(1, len(scores) // 2)
train_features = features[:split_point]
train_scores = scores[:split_point]
test_features = features[split_point:]
test_scores = scores[split_point:]

if len(test_scores) < 2:
    raise RuntimeError("Smoke subset is too small to compute correlation metrics.")

regressor = Ridge(alpha=0.1)
regressor.fit(train_features, train_scores)
predictions = regressor.predict(test_features)

srcc = float(stats.spearmanr(predictions, test_scores)[0])
plcc = float(stats.pearsonr(predictions, test_scores)[0])

print(json.dumps(
    {
        "subset_size": int(len(subset_indices)),
        "global_avg_srocc": srcc,
        "global_avg_plcc": plcc,
        "val_avg_srocc": srcc,
        "val_avg_plcc": plcc,
    },
    indent=2,
))
PY

GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
export SMOKE_GIT_COMMIT="$GIT_COMMIT"

"$PYTHON_BIN" - <<'PY'
import json
import math
import os
import re
from pathlib import Path


def relativize(path_str: str, repo_root: Path) -> str:
    path = Path(path_str)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return path_str


def to_float(value: str):
    try:
        return float(value)
    except Exception:
        return math.nan

train_log = Path(os.environ["SMOKE_TRAIN_LOG"]).read_text()
test_log = Path(os.environ["SMOKE_TEST_LOG"]).read_text()

cfg_match = re.search(r"^CONFIG_SHA1=([0-9a-f]+)$", train_log, re.MULTILINE)
config_sha1 = cfg_match.group(1) if cfg_match else None

metrics_payload = None
try:
    metrics_payload = json.loads(test_log)
except Exception:
    json_match = re.search(r"\{[\s\S]*\}", test_log)
    if json_match is not None:
        try:
            metrics_payload = json.loads(json_match.group(0))
        except Exception:
            metrics_payload = None

if isinstance(metrics_payload, dict):
    global_srocc = to_float(metrics_payload.get("global_avg_srocc"))
    global_plcc = to_float(metrics_payload.get("global_avg_plcc"))
    val_srocc = to_float(metrics_payload.get("val_avg_srocc"))
    val_plcc = to_float(metrics_payload.get("val_avg_plcc"))
else:
    global_matches = re.findall(r"^Global avg\s+([+-]?\d*\.?\d+|nan)\s+([+-]?\d*\.?\d+|nan)", test_log, re.MULTILINE)
    val_matches = re.findall(r"^Val avg \(from test\)\s+([+-]?\d*\.?\d+|nan)\s+([+-]?\d*\.?\d+|nan)", test_log, re.MULTILINE)

    global_srocc = to_float(global_matches[0][0]) if global_matches else math.nan
    global_plcc = to_float(global_matches[0][1]) if global_matches else math.nan
    val_srocc = to_float(val_matches[0][0]) if val_matches else math.nan
    val_plcc = to_float(val_matches[0][1]) if val_matches else math.nan

payload = {
    "run_name": os.environ["SMOKE_RUN_NAME"],
    "experiment_name": os.environ["SMOKE_EXPERIMENT_NAME"],
    "git_commit": os.environ["SMOKE_GIT_COMMIT"],
    "config_path": relativize(os.environ["SMOKE_CONFIG_PATH"], Path(os.environ["SMOKE_REPO_ROOT"])),
    "config_sha1": config_sha1,
    "max_steps": int(os.environ["SMOKE_MAX_STEPS"]),
    "batch_size": int(os.environ["SMOKE_BATCH_SIZE"]),
    "train_log": relativize(os.environ["SMOKE_TRAIN_LOG"], Path(os.environ["SMOKE_REPO_ROOT"])),
    "test_log": relativize(os.environ["SMOKE_TEST_LOG"], Path(os.environ["SMOKE_REPO_ROOT"])),
    "best_checkpoint": relativize(os.environ["SMOKE_BEST_CKPT"], Path(os.environ["SMOKE_REPO_ROOT"])),
    "best_checkpoint_sha256": os.environ["SMOKE_CKPT_SHA"],
    "global_avg_srocc": global_srocc,
    "global_avg_plcc": global_plcc,
    "val_avg_srocc": val_srocc,
    "val_avg_plcc": val_plcc,
}

out = Path(os.environ["SMOKE_OUTPUT_JSON"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2))
print(f"Wrote smoke summary: {out}")
PY

echo "Smoke run completed: ${OUTPUT_JSON}"
