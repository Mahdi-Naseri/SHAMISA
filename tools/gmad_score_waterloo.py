"""
Score Waterloo Exploration images with a trained NR-IQA model and ridge regressor.

Usage:
    python tools/gmad_score_waterloo.py \
        --config configs/shamisa_a0.yaml \
        --experiment_name shamisa_best \
        --dataset_root /path/to/WaterlooExploration \
        --subset distorted \
        --output_scores /path/to/out/shamisa_waterloo_scores.npz
"""

import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.torch_amp_compat import patch_cuda_amp_custom_autocast
from utils.utils import (
    PROJECT_ROOT,
    parse_command_line_args,
    parse_config,
    merge_configs,
    replace_none_string,
)
from data.dataset_waterloo_exploration import WaterlooExplorationDataset
from models.simclr import SimCLR
from models.vicreg import Vicreg
from test import (
    prepare_dataset,
    evaluate_ridge_metrics,
    _load_pretrained_weights,
)

patch_cuda_amp_custom_autocast(device_type="cuda")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_identifier(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars)
    return sanitized.strip("_") or "result"


def _parse_wandb_tags(tags: Optional[str]):
    if not tags:
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _resolve_wandb_mode(arg_mode: Optional[str]) -> str:
    if arg_mode:
        return arg_mode
    env_mode = os.getenv("WANDB_MODE")
    if env_mode:
        return env_mode
    if os.getenv("WANDB_OFFLINE"):
        return "offline"
    return "online"


def _init_wandb(args, metadata: dict, model_name: str):
    if not args.enable_wandb:
        return None
    if args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but --enable_wandb was set.") from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT") or "shamisa"
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    run_name = args.wandb_run_name or f"gmad_scores_{model_name}"
    mode = _resolve_wandb_mode(args.wandb_mode)
    tags = _parse_wandb_tags(args.wandb_tags)

    init_kwargs = {
        "project": project,
        "entity": entity,
        "name": run_name,
        "group": args.wandb_group,
        "tags": tags,
        "config": metadata,
        "mode": mode,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def _log_wandb_scores(run, output_path: Path, scores: np.ndarray, model_tag: str, hist_counts, hist_edges, args):
    if run is None:
        return

    import wandb

    metrics = {
        f"scores/{model_tag}_mean": float(scores.mean()),
        f"scores/{model_tag}_std": float(scores.std()),
        f"scores/{model_tag}_min": float(scores.min()),
        f"scores/{model_tag}_max": float(scores.max()),
    }
    run.log(metrics)
    run.log({f"scores/{model_tag}_hist": wandb.Histogram(scores)})

    artifact_name = args.wandb_artifact_name or f"gmad_scores_{model_tag}"
    artifact = wandb.Artifact(artifact_name, type="gmad_scores")
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)
    run.finish()


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if device_str.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_str}")
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_checkpoint_path(
    experiment_name: str, checkpoint_path: Optional[str]
) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    experiment_dir = PROJECT_ROOT / "experiments" / experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment does not exist: {experiment_dir}")

    pretrain_dir = experiment_dir / "pretrain"
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"Pretrain directory does not exist: {pretrain_dir}")

    checkpoints = [ckpt for ckpt in pretrain_dir.glob("*.pth") if "best" in ckpt.name]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint containing 'best' found in {pretrain_dir}"
        )
    return checkpoints[0]


def _load_model(args, eval_type: str, checkpoint_path: Optional[str]) -> Tuple[torch.nn.Module, Optional[Path]]:
    if eval_type != "scratch":
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    if args.model.method == "simclr":
        model = SimCLR(
            encoder_params=args.model.encoder, temperature=args.model.temperature
        )
    elif args.model.method == "vicreg":
        model = Vicreg(args)
    else:
        raise ValueError(f"Model method {args.model.method} not supported")

    ckpt_path = _resolve_checkpoint_path(args.experiment_name, checkpoint_path)
    _load_pretrained_weights(model, ckpt_path)
    return model, ckpt_path


def _build_regressor_cache_path(
    cache_dir: Optional[Path],
    experiment_name: str,
    eval_type: str,
    train_dataset: str,
    checkpoint_path: Optional[Path],
) -> Optional[Path]:
    if cache_dir is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    stem_parts = [experiment_name, eval_type, train_dataset]
    if checkpoint_path is not None:
        stem_parts.append(checkpoint_path.stem)
    cache_key = _sanitize_identifier("_".join(stem_parts))
    return cache_dir / f"ridge_{cache_key}.pkl"


def _load_cached_regressor(cache_path: Path, train_dataset: str):
    if not cache_path.exists():
        return None, None
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "regressor" in payload:
        meta = payload.get("meta", {})
        if meta.get("train_dataset") not in (None, train_dataset):
            return None, None
        return payload["regressor"], meta.get("alpha")
    return payload, None


def _train_regressor(
    model: torch.nn.Module,
    args,
    device: torch.device,
    train_dataset: str,
    batch_size: int,
    num_workers: int,
    eval_type: str,
) -> Tuple[object, float]:
    dataset, num_splits, _ = prepare_dataset(
        train_dataset,
        data_base_path=args.data_base_path,
        crop_size=args.test.crop_size,
        default_num_splits=args.test.num_splits,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    from test import get_features_scores

    reps, _, scores = get_features_scores(
        model, dataloader, device, eval_type, dtype=torch.float64
    )

    try:
        _, _, regressor, best_alpha, _ = evaluate_ridge_metrics(
            dataset=dataset,
            features=reps,
            scores=scores,
            num_splits=num_splits,
            phase="test",
            alpha=args.test.alpha,
            grid_search=args.test.grid_search,
            args=args,
        )
    except Exception as exc:
        raise RuntimeError(f"Regressor training failed: {exc}") from exc

    if regressor is None:
        raise RuntimeError("Regressor training failed: no regressor returned.")
    return regressor, best_alpha


def _score_waterloo(
    model: torch.nn.Module,
    regressor,
    dataloader: DataLoader,
    device: torch.device,
    eval_type: str,
) -> Tuple[list, np.ndarray]:
    model.eval()
    scores = np.empty((len(dataloader.dataset),), dtype=np.float32)
    paths = []
    offset = 0

    amp_context = (
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )

    for batch in dataloader:
        img_orig = batch["img"].to(device, non_blocking=True)
        img_ds = batch["img_ds"].to(device, non_blocking=True)
        batch_paths = list(batch["path"])

        img_orig = img_orig.flatten(0, 1)
        img_ds = img_ds.flatten(0, 1)

        with torch.no_grad():
            if amp_context is None:
                f_orig, _ = model(img_orig)
                f_ds, _ = model(img_ds)
                feats = torch.cat((f_orig, f_ds), dim=1)
            else:
                with amp_context:
                    f_orig, _ = model(img_orig)
                    f_ds, _ = model(img_ds)
                    feats = torch.cat((f_orig, f_ds), dim=1)

        feats_np = feats.detach().float().cpu().numpy()
        preds = regressor.predict(feats_np)
        preds = np.mean(preds.reshape(-1, 5), axis=1).astype(np.float32)

        batch_size = len(batch_paths)
        scores[offset : offset + batch_size] = preds
        paths.extend(batch_paths)
        offset += batch_size

    if offset != len(dataloader.dataset):
        raise RuntimeError(
            f"Scoring mismatch: expected {len(dataloader.dataset)} images, got {offset}."
        )
    return paths, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Waterloo Exploration images.")
    parser.add_argument("--config", type=str, required=True, help="Path to a config file")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="scratch",
        choices=["scratch"],
        help="Evaluation mode. This release supports only scratch checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to the best checkpoint in the experiment.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Waterloo Exploration root directory",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="distorted",
        choices=["pristine", "distorted"],
        help="Select subset of Waterloo images to score (default: distorted)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference (default: config test.batch_size)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of dataloader workers (default: config test.num_workers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. cuda, cuda:0, cpu (default: auto)",
    )
    parser.add_argument(
        "--output_scores",
        type=str,
        required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--regressor_train_dataset",
        type=str,
        default="kadid10k",
        help="Dataset to train ridge regressor (default: kadid10k)",
    )
    parser.add_argument(
        "--regressor_cache_dir",
        type=str,
        default=None,
        help="Optional directory to cache trained ridge regressor",
    )
    parser.add_argument(
        "--limit_images",
        type=int,
        default=None,
        help="Limit number of Waterloo images for debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: config seed)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name for metadata (default: experiment_name)",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=True,
        help="Log score summary to Weights & Biases",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project (default: WANDB_PROJECT or 'shamisa')",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode (online/offline/disabled)",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID (for resuming/logging across scripts)",
    )
    parser.add_argument(
        "--wandb_artifact_name",
        type=str,
        default=None,
        help="Optional W&B artifact name",
    )

    args, unknown = parser.parse_known_args()
    if args.disable_wandb:
        args.enable_wandb = False

    config = parse_config(args.config)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + unknown
        config_args = parse_command_line_args(config)
    finally:
        sys.argv = original_argv

    config_args = merge_configs(config, config_args)
    config_args = replace_none_string(config_args)
    config_args.experiment_name = args.experiment_name
    config_args.eval_type = args.eval_type
    config_args.data_base_path = Path(config_args.data_base_path)

    batch_size = args.batch_size or config_args.test.batch_size
    num_workers = args.num_workers or config_args.test.num_workers
    seed = args.seed if args.seed is not None else int(config_args.seed)
    _set_seed(seed)

    device = _resolve_device(args.device)
    model, resolved_ckpt = _load_model(
        config_args, args.eval_type, args.checkpoint_path
    )
    model.to(device)
    model.eval()

    train_dataset_key = args.regressor_train_dataset.lower()

    cache_dir = None
    if args.regressor_cache_dir:
        cache_dir = Path(args.regressor_cache_dir).expanduser()
    elif args.eval_type == "scratch":
        cache_dir = PROJECT_ROOT / "experiments" / args.experiment_name / "regressors"

    cache_path = _build_regressor_cache_path(
        cache_dir,
        args.experiment_name,
        args.eval_type,
        train_dataset_key,
        resolved_ckpt,
    )

    regressor = None
    best_alpha = None
    if cache_path is not None:
        regressor, best_alpha = _load_cached_regressor(cache_path, train_dataset_key)
        if regressor is not None:
            print(f"Loaded cached regressor from {cache_path}")

    if regressor is None:
        regressor, best_alpha = _train_regressor(
            model=model,
            args=config_args,
            device=device,
            train_dataset=train_dataset_key,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_type=args.eval_type,
        )
        if cache_path is not None:
            payload = {
                "regressor": regressor,
                "meta": {
                    "train_dataset": train_dataset_key,
                    "alpha": best_alpha,
                    "eval_type": args.eval_type,
                    "experiment_name": args.experiment_name,
                    "checkpoint_path": str(resolved_ckpt) if resolved_ckpt else None,
                    "timestamp": datetime.now().isoformat(),
                },
            }
            with open(cache_path, "wb") as handle:
                pickle.dump(payload, handle)
            print(f"Saved regressor cache to {cache_path}")

    dataset = WaterlooExplorationDataset(
        root_dir=args.dataset_root,
        return_relpath=True,
        crop_size=config_args.test.crop_size,
        subset=args.subset,
    )
    if args.limit_images is not None and args.limit_images > 0:
        dataset.images = dataset.images[: args.limit_images]
    print(f"Waterloo root (requested): {Path(args.dataset_root).resolve()}")
    print(f"Waterloo root (scan): {dataset.data_root.resolve()}")
    print(f"Waterloo subset: {dataset.subset_resolved} (requested: {args.subset})")
    print(f"Waterloo image count: {len(dataset)}")
    if len(dataset) == 4744:
        print(
            "WARNING: Waterloo image count is 4744, which matches pristine-only. "
            "Distorted pool should be much larger."
        )
    sample_count = min(5, len(dataset.images))
    if sample_count:
        sample_paths = [
            str(p.relative_to(dataset.root_dir)) for p in dataset.images[:sample_count]
        ]
        print(f"Waterloo sample paths: {sample_paths}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    paths, scores = _score_waterloo(
        model=model,
        regressor=regressor,
        dataloader=dataloader,
        device=device,
        eval_type=args.eval_type,
    )

    scores = np.asarray(scores, dtype=np.float32)
    model_name = args.model_name or args.experiment_name
    output_path = Path(args.output_scores).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(resolved_ckpt) if resolved_ckpt else None,
        "regressor_train_dataset": train_dataset_key,
        "regressor_alpha": float(best_alpha) if best_alpha is not None else None,
        "experiment_name": args.experiment_name,
        "eval_type": args.eval_type,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "subset": args.subset,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    np.savez_compressed(
        output_path,
        paths=np.array(paths, dtype=object),
        scores=scores,
        model_name=model_name,
        metadata_json=json.dumps(metadata),
    )

    print(
        f"Saved scores to {output_path} (N={len(scores)}) "
        f"mean={scores.mean():.4f} std={scores.std():.4f} "
        f"min={scores.min():.4f} max={scores.max():.4f}"
    )
    hist_counts, hist_edges = np.histogram(scores, bins=10)
    print(f"Histogram counts: {hist_counts.tolist()}")
    print(f"Histogram edges: {[float(x) for x in hist_edges]}")

    wandb_run = _init_wandb(args, metadata, model_name)
    if wandb_run is not None:
        model_tag = _sanitize_identifier(model_name)
        _log_wandb_scores(
            wandb_run,
            output_path,
            scores,
            model_tag,
            hist_counts,
            hist_edges,
            args,
        )


if __name__ == "__main__":
    main()
