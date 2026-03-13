import argparse
import inspect
import json
import os
import pickle
import random
import time
import urllib.request
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dotmap import DotMap
from einops import rearrange
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset

import wandb
from wandb.wandb_run import Run

from data import (
    LIVEDataset,
    CSIQDataset,
    TID2013Dataset,
    KADID10KDataset,
    FLIVEDataset,
    SPAQDataset,
)
from models.simclr import SimCLR
from models.vicreg import Vicreg
from utils.torch_amp_compat import patch_cuda_amp_custom_autocast
from utils.utils import (
    PROJECT_ROOT,
    parse_command_line_args,
    merge_configs,
    parse_config,
    replace_none_string,
    prepare_wandb_config,
)

patch_cuda_amp_custom_autocast(device_type="cuda")

KADID_COARSE_MAP = {
    "gaussian_blur": "blur",
    "lens_blur": "blur",
    "motion_blur": "blur",
    "color_diffusion": "color",
    "color_shift": "color",
    "color_quantization": "color",
    "color_saturation_1": "color",
    "color_saturation_2": "color",
    "jpeg2000": "compression",
    "jpeg": "compression",
    "white_noise": "noise",
    "white_noise_color_component": "noise",
    "impulse_noise": "noise",
    "multiplicative_noise": "noise",
    "denoise": "noise",
    "brighten": "brightness",
    "darken": "brightness",
    "mean_shift": "brightness",
    "jitter": "spatial",
    "non_eccentricity_patch": "spatial",
    "pixelate": "spatial",
    "quantization": "spatial",
    "color_block": "spatial",
    "high_sharpen": "sharpness_contrast",
    "contrast_change": "sharpness_contrast",
}


def prepare_dataset(
    dataset_key: str,
    data_base_path: Path,
    crop_size: int,
    default_num_splits: int,
    fr_iqa: bool = False,
) -> Tuple[Dataset, int, str]:
    dataset_key = dataset_key.lower()
    if dataset_key == "live":
        dataset = LIVEDataset(
            data_base_path / "LIVE",
            phase="all",
            crop_size=crop_size,
            fr_mode=fr_iqa,
        )
        dataset_num_splits = default_num_splits
        dataset_name = "LIVE"
    elif dataset_key == "csiq":
        dataset = CSIQDataset(
            data_base_path / "CSIQ",
            phase="all",
            crop_size=crop_size,
            fr_mode=fr_iqa,
        )
        dataset_num_splits = default_num_splits
        dataset_name = "CSIQ"
    elif dataset_key == "tid2013":
        dataset = TID2013Dataset(
            data_base_path / "TID2013",
            phase="all",
            crop_size=crop_size,
            fr_mode=fr_iqa,
        )
        dataset_num_splits = default_num_splits
        dataset_name = "TID2013"
    elif dataset_key == "kadid10k":
        dataset = KADID10KDataset(
            data_base_path / "KADID10K",
            phase="all",
            crop_size=crop_size,
            fr_mode=fr_iqa,
        )
        dataset_num_splits = default_num_splits
        dataset_name = "KADID-10K"
    elif dataset_key == "flive":
        if fr_iqa:
            raise ValueError(
                "FR-IQA is only supported for LIVE, CSIQ, TID2013, and KADID-10K."
            )
        dataset = FLIVEDataset(
            data_base_path / "FLIVE", phase="all", crop_size=crop_size
        )
        dataset_num_splits = 1
        dataset_name = "FLIVE"
    elif dataset_key == "spaq":
        if fr_iqa:
            raise ValueError(
                "FR-IQA is only supported for LIVE, CSIQ, TID2013, and KADID-10K."
            )
        dataset = SPAQDataset(
            data_base_path / "SPAQ", phase="all", crop_size=crop_size
        )
        dataset_num_splits = default_num_splits
        dataset_name = "SPAQ"
    else:
        raise ValueError(f"Dataset {dataset_key} not supported")
    return dataset, dataset_num_splits, dataset_name


def _safe_torch_load(checkpoint_path: Path) -> Dict[str, Any]:
    load_kwargs = {"map_location": "cpu"}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
    except (TypeError, ValueError):
        pass

    try:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    return checkpoint


def _load_pretrained_weights(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = _safe_torch_load(checkpoint_path)

    state_dict = checkpoint
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            candidate = state_dict.get(key) if isinstance(state_dict, dict) else None
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Unsupported checkpoint structure in {checkpoint_path}: expected a state_dict mapping."
        )

    normalized_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized_state_dict[key] = value

    if not any(k.startswith("encoder.") for k in normalized_state_dict):
        remapped_state_dict = {}
        for key, value in normalized_state_dict.items():
            if key.startswith("model.") or key.startswith("projector."):
                remapped_state_dict[f"encoder.{key}"] = value
            else:
                remapped_state_dict[key] = value
        normalized_state_dict = remapped_state_dict

    model.load_state_dict(normalized_state_dict, strict=True)


def _no_internet(url: str = "http://www.google.com", timeout: int = 3) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return False
    except Exception:
        return True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_to_id(path_value: Any) -> Optional[str]:
    if path_value is None:
        return None
    try:
        return Path(path_value).stem
    except TypeError:
        return str(path_value)


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items = [item.strip().lower() for item in value.split(",")]
    return [item for item in items if item]


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _resolve_distortion_group(
    dist_type: Optional[str],
    dist_group: Optional[str],
    dataset_key: str,
) -> str:
    if isinstance(dist_group, str) and dist_group:
        return dist_group
    if dataset_key == "kadid10k" and isinstance(dist_type, str):
        return KADID_COARSE_MAP.get(dist_type, "unknown")
    if isinstance(dist_type, str):
        return dist_type
    return "unknown"


def _build_subset_dir_name(
    include_types: List[str],
    include_groups: List[str],
) -> Optional[str]:
    if not include_types and not include_groups:
        return None

    types_set = set(include_types)
    groups_set = set(include_groups)

    if include_types and not include_groups:
        if types_set == {"gaussian_blur", "lens_blur", "motion_blur"}:
            return "subset_blur_types"
        if types_set == {"jpeg", "jpeg2000"}:
            return "subset_compression_types"
    if include_groups and not include_types:
        if groups_set == {"blur"}:
            return "subset_blur_group"
        if groups_set == {"compression"}:
            return "subset_compression_group"

    parts = []
    if include_types:
        parts.append("types_" + "-".join(include_types))
    if include_groups:
        parts.append("groups_" + "-".join(include_groups))
    name = "subset_" + "_".join(parts)
    return name.replace("/", "-")


def _compute_alpha_values(
    meta: Dict[str, List[Any]],
    alpha_by: str,
    base_alpha: float,
) -> Optional[np.ndarray]:
    if alpha_by == "none":
        return None
    if alpha_by == "severity":
        severity = np.array(
            [
                float(_coerce_int(val)) if _coerce_int(val) is not None else np.nan
                for val in meta["severity"]
            ]
        )
        min_alpha = 0.25
        max_alpha = 0.95
        alpha_values = np.full(severity.shape, base_alpha, dtype=float)
        valid = ~np.isnan(severity)
        if np.any(valid):
            clipped = np.clip(severity[valid], 1.0, 5.0)
            scaled = (clipped - 1.0) / 4.0
            alpha_values[valid] = min_alpha + scaled * (max_alpha - min_alpha)
        return np.clip(alpha_values, min_alpha, max_alpha)
    return None


def _make_bundle_zip(output_root: Path) -> Path:
    bundle_path = output_root / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_root):
            for filename in files:
                file_path = Path(root) / filename
                if file_path == bundle_path:
                    continue
                rel_path = file_path.relative_to(output_root)
                arcname = Path(output_root.name) / rel_path
                zf.write(file_path, arcname.as_posix())
    return bundle_path


def _slice_attr(values: Optional[np.ndarray], indices: np.ndarray) -> List[Any]:
    if values is None or len(values) == 0:
        return [None] * len(indices)
    return [values[idx] for idx in indices]


def _resolve_checkpoint_path(
    experiment_name: str, checkpoint_value: str
) -> Path:
    if checkpoint_value == "best":
        checkpoint_base_path = PROJECT_ROOT / "experiments" / experiment_name
        checkpoint_path = checkpoint_base_path / "pretrain"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint folder {checkpoint_path} does not exist"
            )
        candidates = sorted(
            [
                ckpt_path
                for ckpt_path in checkpoint_path.glob("*.pth")
                if "best" in ckpt_path.name
            ]
        )
        if not candidates:
            raise FileNotFoundError(
                f"No best checkpoint found under {checkpoint_path}"
            )
        return candidates[0]

    candidate = Path(checkpoint_value)
    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        repo_candidate = PROJECT_ROOT / checkpoint_value
        if repo_candidate.exists():
            return repo_candidate
        exp_candidate = (
            PROJECT_ROOT / "experiments" / experiment_name / "pretrain" / checkpoint_value
        )
        if exp_candidate.exists():
            return exp_candidate

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_value}")


def _init_wandb(args: DotMap, cli_args: argparse.Namespace) -> Optional[Run]:
    use_wandb = bool(args.logging.use_wandb) or cli_args.enable_wandb
    if not use_wandb:
        return None

    wandb_config = prepare_wandb_config(args)
    project = cli_args.wandb_project or args.logging.wandb.project
    run_name = cli_args.wandb_run_name or args.experiment_name
    mode = "online" if args.logging.wandb.online else "offline"

    logger = wandb.init(
        project=project,
        name=run_name,
        config=wandb_config,
        mode=mode,
        resume="never",
    )
    return logger


def _extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    dataset: Dataset,
    device: torch.device,
    eval_type: str,
    dataset_key: str,
    index_map: Optional[List[int]] = None,
    crop_policy: str = "all",
) -> Tuple[np.ndarray, Dict[str, List[Any]]]:
    model.eval()

    dist_types_all = (
        np.array(dataset.distortion_types)
        if hasattr(dataset, "distortion_types")
        else None
    )
    dist_groups_all = (
        np.array(dataset.distortion_groups)
        if hasattr(dataset, "distortion_groups")
        else None
    )
    if dist_groups_all is None and dist_types_all is not None and dataset_key == "kadid10k":
        dist_groups_all = np.array(
            [
                KADID_COARSE_MAP.get(dist_type, "unknown")
                for dist_type in dist_types_all
            ],
            dtype=object,
        )
    dist_levels_all = (
        np.array(dataset.distortion_levels)
        if hasattr(dataset, "distortion_levels")
        else None
    )
    images_all = np.array(dataset.images) if hasattr(dataset, "images") else None
    ref_images_all = (
        np.array(dataset.ref_images)
        if hasattr(dataset, "ref_images")
        else None
    )

    feat_dim = model.encoder.feat_dim * 2
    features_list: List[np.ndarray] = []
    image_idx = 0

    meta: Dict[str, List[Any]] = {
        "distortion_type": [],
        "distortion_group": [],
        "severity": [],
        "ref_id": [],
        "image_id": [],
        "image_index": [],
        "crop_index": [],
        "split": [],
        "score": [],
        "score_type": [],
        "mos": [],
        "dmos": [],
        "dataset_name": [],
    }

    dataset_split = getattr(dataset, "phase", "all")
    mos_type = getattr(dataset, "mos_type", "mos")

    amp_context = (
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and torch.cuda.is_available()
        else nullcontext()
    )

    for batch in dataloader:
        img = batch["img"].to(device)
        img_ds = batch["img_ds"].to(device)
        mos = batch["mos"]

        if crop_policy == "first":
            img = img[:, :1, ...]
            img_ds = img_ds[:, :1, ...]

        num_crops_batch = int(img.shape[1])

        img = rearrange(img, "b n c h w -> (b n) c h w")
        img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
        mos_repeat = mos.repeat_interleave(num_crops_batch)

        with amp_context, torch.no_grad():
            if eval_type == "scratch":
                f_orig, _ = model(img)
                f_ds, _ = model(img_ds)
                f = torch.cat((f_orig, f_ds), dim=1)
            else:
                raise ValueError(f"Eval type {eval_type} not supported")

        f = f.float()
        img_batch_size = mos.shape[0]

        if crop_policy == "all":
            feature_block = f.detach().cpu().numpy()
            features_list.append(feature_block)
            crop_indices = np.tile(np.arange(num_crops_batch), img_batch_size).tolist()
            score_values = mos_repeat.detach().cpu().numpy().tolist()
            repeat_factor = num_crops_batch
            output_size = img_batch_size * repeat_factor
        else:
            f_view = f.view(img_batch_size, num_crops_batch, -1)
            if crop_policy == "mean":
                f_out = f_view.mean(dim=1)
                crop_indices = [-1] * img_batch_size
            elif crop_policy == "first":
                f_out = f_view[:, 0, :]
                crop_indices = [0] * img_batch_size
            else:
                raise ValueError(f"Unknown crop_policy {crop_policy}")
            feature_block = f_out.detach().cpu().numpy()
            features_list.append(feature_block)
            score_values = mos.detach().cpu().numpy().tolist()
            repeat_factor = 1
            output_size = img_batch_size

        if index_map is not None:
            batch_indices = np.array(
                index_map[image_idx : image_idx + img_batch_size]
            )
        else:
            batch_indices = np.arange(image_idx, image_idx + img_batch_size)

        dist_types = _slice_attr(dist_types_all, batch_indices)
        dist_groups = _slice_attr(dist_groups_all, batch_indices)
        dist_levels = _slice_attr(dist_levels_all, batch_indices)
        image_paths = _slice_attr(images_all, batch_indices)
        ref_paths = _slice_attr(ref_images_all, batch_indices)

        resolved_groups = [
            _resolve_distortion_group(dt, dg, dataset_key)
            for dt, dg in zip(dist_types, dist_groups)
        ]
        severity_vals = [_coerce_int(val) for val in dist_levels]
        image_ids = []
        for idx, path in zip(batch_indices, image_paths):
            resolved = _path_to_id(path)
            if not resolved:
                resolved = f"image_{idx}"
            image_ids.append(resolved)
        ref_ids = [_path_to_id(val) for val in ref_paths]
        image_indices = batch_indices.tolist()

        if crop_policy == "all":
            meta["distortion_type"].extend(
                np.repeat(np.array(dist_types, dtype=object), repeat_factor).tolist()
            )
            meta["distortion_group"].extend(
                np.repeat(np.array(resolved_groups, dtype=object), repeat_factor).tolist()
            )
            meta["severity"].extend(
                np.repeat(np.array(severity_vals, dtype=object), repeat_factor).tolist()
            )
            meta["ref_id"].extend(
                np.repeat(np.array(ref_ids, dtype=object), repeat_factor).tolist()
            )
            meta["image_id"].extend(
                np.repeat(np.array(image_ids, dtype=object), repeat_factor).tolist()
            )
            meta["image_index"].extend(
                np.repeat(np.array(image_indices, dtype=int), repeat_factor).tolist()
            )
            meta["crop_index"].extend(crop_indices)
        else:
            meta["distortion_type"].extend(dist_types)
            meta["distortion_group"].extend(resolved_groups)
            meta["severity"].extend(severity_vals)
            meta["ref_id"].extend(ref_ids)
            meta["image_id"].extend(image_ids)
            meta["image_index"].extend(image_indices)
            meta["crop_index"].extend(crop_indices)

        meta["split"].extend([dataset_split] * output_size)
        meta["score"].extend(score_values)
        meta["score_type"].extend([mos_type] * output_size)
        if mos_type == "dmos":
            meta["dmos"].extend(score_values)
            meta["mos"].extend([None] * output_size)
        else:
            meta["mos"].extend(score_values)
            meta["dmos"].extend([None] * output_size)
        meta["dataset_name"].extend([dataset_key] * output_size)

        image_idx += img_batch_size

    if not features_list:
        return np.empty((0, feat_dim), dtype=np.float32), meta

    features = np.concatenate(features_list, axis=0).astype(np.float32, copy=False)
    return features, meta


def _infer_num_crops(dataset: Dataset) -> int:
    sample = dataset[0]
    img = sample["img"]
    if hasattr(img, "shape"):
        return int(img.shape[0])
    return 1


def _prefilter_indices(
    dataset: Dataset,
    dataset_key: str,
    num_crops: int,
    crop_policy: str,
    seed: int,
    subsample_frac: float,
    max_samples: Optional[int],
    include_types: List[str],
    include_groups: List[str],
    focus_distortion: Optional[str],
    min_severity: Optional[int],
    max_severity: Optional[int],
) -> np.ndarray:
    num_images = len(dataset)
    mask = np.ones(num_images, dtype=bool)

    dist_types = (
        np.array(dataset.distortion_types, dtype=object)
        if hasattr(dataset, "distortion_types")
        else None
    )
    dist_types_lower = None
    if dist_types is not None:
        dist_types_lower = np.array(
            [
                val.lower() if isinstance(val, str) else ""
                for val in dist_types
            ],
            dtype=object,
        )

    dist_groups = (
        np.array(dataset.distortion_groups, dtype=object)
        if hasattr(dataset, "distortion_groups")
        else None
    )
    if dist_groups is None and dist_types is not None and dataset_key == "kadid10k":
        dist_groups = np.array(
            [
                _resolve_distortion_group(dist_type, None, dataset_key)
                for dist_type in dist_types
            ],
            dtype=object,
        )
    dist_groups_lower = None
    if dist_groups is not None:
        dist_groups_lower = np.array(
            [
                val.lower() if isinstance(val, str) else ""
                for val in dist_groups
            ],
            dtype=object,
        )

    if include_types:
        if dist_types_lower is None:
            mask &= False
        else:
            mask &= np.isin(dist_types_lower, np.array(include_types, dtype=object))

    if include_groups:
        if dist_groups_lower is None:
            mask &= False
        else:
            mask &= np.isin(dist_groups_lower, np.array(include_groups, dtype=object))

    if focus_distortion:
        if dist_types_lower is None:
            mask &= False
        else:
            focus = focus_distortion.lower()
            mask &= dist_types_lower == focus

    if min_severity is not None or max_severity is not None:
        if hasattr(dataset, "distortion_levels"):
            dist_levels = np.array(dataset.distortion_levels)
            levels = np.array([_coerce_int(val) for val in dist_levels], dtype=float)
            severity_mask = ~np.isnan(levels)
            if min_severity is not None:
                severity_mask &= levels >= float(min_severity)
            if max_severity is not None:
                severity_mask &= levels <= float(max_severity)
            mask &= severity_mask
        else:
            mask &= False

    indices = np.where(mask)[0]
    rng = np.random.default_rng(seed)

    samples_per_image = num_crops if crop_policy == "all" else 1
    if max_samples is not None and samples_per_image > 0:
        max_images = max(1, max_samples // samples_per_image)
        if len(indices) > max_images:
            indices = rng.choice(indices, size=max_images, replace=False)

    if subsample_frac < 1.0 and len(indices) > 0:
        target = max(1, int(len(indices) * subsample_frac))
        if target < len(indices):
            indices = rng.choice(indices, size=target, replace=False)

    return np.array(indices, copy=False)


def _apply_filters(
    meta: Dict[str, List[Any]],
    focus_distortion: Optional[str],
    min_severity: Optional[int],
    max_severity: Optional[int],
) -> np.ndarray:
    num_samples = len(meta["score"])
    mask = np.ones(num_samples, dtype=bool)

    if focus_distortion:
        focus = focus_distortion.lower()
        types = np.array(meta["distortion_type"], dtype=object)
        type_mask = np.array(
            [
                isinstance(t, str) and t.lower() == focus
                for t in types
            ],
            dtype=bool,
        )
        mask &= type_mask

    if min_severity is not None or max_severity is not None:
        severity = np.array(
            [float(_coerce_int(val)) if val is not None else np.nan for val in meta["severity"]]
        )
        mask &= ~np.isnan(severity)
        if min_severity is not None:
            mask &= severity >= float(min_severity)
        if max_severity is not None:
            mask &= severity <= float(max_severity)

    return np.where(mask)[0]


def _apply_subsample(
    indices: np.ndarray,
    seed: int,
    subsample_frac: float,
    max_samples: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.array(indices, copy=True)

    if len(selected) == 0:
        return selected

    if max_samples is not None and len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)

    if subsample_frac < 1.0:
        target = max(1, int(len(selected) * subsample_frac))
        if target < len(selected):
            selected = rng.choice(selected, size=target, replace=False)

    return selected


def _subset_meta(meta: Dict[str, List[Any]], indices: np.ndarray) -> Dict[str, List[Any]]:
    return {key: [values[idx] for idx in indices] for key, values in meta.items()}


def _build_color_labels(
    meta: Dict[str, List[Any]],
    color_by: str,
    dataset_key: str,
    distortion_granularity: str,
) -> Tuple[np.ndarray, bool]:
    if color_by == "severity":
        severity = np.array(
            [float(_coerce_int(val)) if val is not None else np.nan for val in meta["severity"]]
        )
        return severity, True

    dist_types = np.array(meta["distortion_type"], dtype=object)
    dist_groups = np.array(meta["distortion_group"], dtype=object)

    if color_by == "distortion_group":
        labels = [
            _resolve_distortion_group(dist_type, dist_group, dataset_key)
            for dist_type, dist_group in zip(dist_types, dist_groups)
        ]
        return np.array(labels, dtype=object), False

    labels = []
    for dist_type, dist_group in zip(dist_types, dist_groups):
        if isinstance(dist_type, str) and dist_type:
            labels.append(dist_type)
        elif isinstance(dist_group, str) and dist_group:
            labels.append(dist_group)
        else:
            labels.append("unknown")
    return np.array(labels, dtype=object), False


def _run_tsne(
    features: np.ndarray,
    seed: int,
    pca_dim: int,
    tsne_perplexity: float,
    tsne_learning_rate: float,
    tsne_n_iter: int,
    tsne_init: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = features
    pca_info: Dict[str, Any] = {"used": False}

    if data.shape[0] < 2:
        raise ValueError("Need at least 2 samples to run t-SNE.")

    if pca_dim and pca_dim > 0 and data.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        data = pca.fit_transform(data)
        pca_info = {
            "used": True,
            "n_components": pca_dim,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }

    if tsne_perplexity >= data.shape[0]:
        raise ValueError(
            f"Perplexity {tsne_perplexity} is too large for {data.shape[0]} samples"
        )

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        n_iter=tsne_n_iter,
        init=tsne_init,
        random_state=seed,
    )
    coords = tsne.fit_transform(data).astype(np.float32)
    return coords, pca_info


def _save_array(path: Path, array: np.ndarray, save_format: str) -> None:
    if save_format == "pt":
        torch.save(torch.from_numpy(array), path)
    else:
        np.save(path, array)


def _load_array(path: Path, save_format: str) -> np.ndarray:
    if save_format == "pt":
        return torch.load(path, map_location="cpu").numpy()
    return np.load(path)


def _resolve_array_paths(
    output_root: Path, save_format: str
) -> Tuple[Path, Path, str]:
    features_path = output_root / f"features_H.{save_format}"
    tsne_path = output_root / f"Y_tsne.{save_format}"

    if features_path.exists() and tsne_path.exists():
        return features_path, tsne_path, save_format

    alt_format = "npy" if save_format == "pt" else "pt"
    alt_features = output_root / f"features_H.{alt_format}"
    alt_tsne = output_root / f"Y_tsne.{alt_format}"
    if alt_features.exists() and alt_tsne.exists():
        return alt_features, alt_tsne, alt_format

    return features_path, tsne_path, save_format


DISTORTION_GROUP_COLORS = {
    "blur": "#1f77b4",
    "noise": "#d62728",
    "jpeg": "#2ca02c",
    "color_distortion": "#ff7f0e",
    "brightness_change": "#9467bd",
    "spatial_distortion": "#8c564b",
    "sharpness_contrast": "#17becf",
}


def _make_palette(num_classes: int) -> List[Tuple[float, float, float, float]]:
    min_saturation = 0.45
    max_value = 0.9

    def adjust_color(rgba: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        r, g, b, a = rgba
        h, s, v = mcolors.rgb_to_hsv([r, g, b])
        s = max(s, min_saturation)
        v = min(v, max_value)
        r_adj, g_adj, b_adj = mcolors.hsv_to_rgb([h, s, v])
        return (float(r_adj), float(g_adj), float(b_adj), float(a))

    tab10 = [plt.cm.get_cmap("tab10", 10)(i) for i in range(10)]
    if num_classes <= 10:
        return [adjust_color(color) for color in tab10[:num_classes]]

    tab20 = [plt.cm.get_cmap("tab20", 20)(i) for i in range(20)]
    # Reorder to avoid adjacent similar hues (take dark then light)
    tab20_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    tab20_reordered = [tab20[i] for i in tab20_order]
    if num_classes <= 20:
        return [adjust_color(color) for color in tab20_reordered[:num_classes]]

    hues = np.linspace(0.0, 1.0, num_classes, endpoint=False)
    colors = []
    for h in hues:
        r, g, b = mcolors.hsv_to_rgb([h, 0.75, 0.85])
        colors.append((float(r), float(g), float(b), 1.0))
    return [adjust_color(color) for color in colors]


def _resolve_label_colors(
    label_names: List[str],
    label2color_path: Path,
    fixed_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    label2hex: Dict[str, str] = {}
    if fixed_map:
        for label in label_names:
            if label in fixed_map:
                label2hex[label] = fixed_map[label]
    if label2color_path.exists():
        with label2color_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for label, value in data.items():
                if fixed_map and str(label) in fixed_map:
                    continue
                try:
                    rgba = mcolors.to_rgba(value)
                except (TypeError, ValueError):
                    continue
                label2hex[str(label)] = mcolors.to_hex(rgba, keep_alpha=False)

    palette = _make_palette(len(label_names))
    palette_hex = [mcolors.to_hex(color, keep_alpha=False) for color in palette]
    used = set(label2hex.values())

    palette_iter = iter(palette_hex)
    for label in label_names:
        if label in label2hex:
            continue
        for color in palette_iter:
            if color not in used:
                label2hex[label] = color
                used.add(color)
                break
        else:
            # Fallback: generate a new HSV color if palette is exhausted.
            hue = (len(used) * 0.61803398875) % 1.0
            r, g, b = mcolors.hsv_to_rgb([hue, 0.75, 0.85])
            hex_color = mcolors.to_hex([r, g, b], keep_alpha=False)
            label2hex[label] = hex_color
            used.add(hex_color)

    with label2color_path.open("w", encoding="utf-8") as f:
        json.dump({k: label2hex[k] for k in sorted(label2hex)}, f, indent=2)

    return {label: mcolors.to_rgba(label2hex[label]) for label in label_names}


def _infer_palette_family(output_root: Path, color_by: str) -> str:
    if color_by == "distortion_group":
        return "global"
    path_str = str(output_root).lower()
    family_map = {
        "subset_blur_types": "blur",
        "subset_color_distortion_types": "color_distortion",
        "subset_jpeg_types": "jpeg",
        "subset_noise_types": "noise",
        "subset_brightness_change_types": "brightness_change",
        "subset_spatial_distortion_types": "spatial_distortion",
        "subset_sharpness_contrast_types": "sharpness_contrast",
    }
    for key, value in family_map.items():
        if key in path_str:
            return value
    return "subtype"


def _load_label2hex(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    label2hex: Dict[str, str] = {}
    for label, value in data.items():
        try:
            rgba = mcolors.to_rgba(value)
        except (TypeError, ValueError):
            continue
        label2hex[str(label)] = mcolors.to_hex(rgba, keep_alpha=False)
    return label2hex


def _write_label2hex(path: Path, mapping: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {label: mapping[label] for label in sorted(mapping)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)


def _extend_label_mapping(
    label_names: List[str],
    base_mapping: Dict[str, str],
) -> Dict[str, str]:
    mapping = dict(base_mapping)
    palette = _make_palette(len(label_names))
    palette_hex = [mcolors.to_hex(color, keep_alpha=False) for color in palette]
    used = set(mapping.values())

    palette_iter = iter(palette_hex)
    for label in label_names:
        if label in mapping:
            continue
        for color in palette_iter:
            if color not in used:
                mapping[label] = color
                used.add(color)
                break
        else:
            hue = (len(used) * 0.61803398875) % 1.0
            r, g, b = mcolors.hsv_to_rgb([hue, 0.75, 0.85])
            hex_color = mcolors.to_hex([r, g, b], keep_alpha=False)
            mapping[label] = hex_color
            used.add(hex_color)
    return mapping


def _resolve_shared_label_colors(
    label_names: List[str],
    output_root: Path,
    palette_ref_dir: Optional[str],
    family: str,
    fixed_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    output_label2color = output_root / "label2color.json"
    ref_label2color = None
    ref_mapping: Dict[str, str] = {}
    if palette_ref_dir:
        ref_label2color = Path(palette_ref_dir) / family / "label2color.json"
        ref_mapping = _load_label2hex(ref_label2color)

    ref_labels = sorted(set(ref_mapping.keys()))
    all_labels = sorted(set(label_names) | set(ref_labels))

    mapping: Dict[str, str] = {}
    if fixed_map:
        mapping.update(fixed_map)

    if all_labels:
        palette = _make_palette(len(all_labels))
        palette_hex = [mcolors.to_hex(color, keep_alpha=False) for color in palette]
        used = set(mapping.values())
        palette_iter = iter(palette_hex)
        for label in all_labels:
            if label in mapping:
                continue
            for color in palette_iter:
                if color not in used:
                    mapping[label] = color
                    used.add(color)
                    break
            else:
                hue = (len(used) * 0.61803398875) % 1.0
                r, g, b = mcolors.hsv_to_rgb([hue, 0.75, 0.85])
                hex_color = mcolors.to_hex([r, g, b], keep_alpha=False)
                mapping[label] = hex_color
                used.add(hex_color)

    if ref_label2color is not None:
        _write_label2hex(ref_label2color, mapping)
    _write_label2hex(output_label2color, mapping)

    return {label: mcolors.to_rgba(mapping[label]) for label in label_names}


def _add_ellipse(
    ax: plt.Axes,
    coords: np.ndarray,
    color: Tuple[float, float, float, float],
    n_std: float,
) -> None:
    if coords.shape[0] < 3:
        return
    cov = np.cov(coords, rowvar=False)
    if np.any(np.isnan(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=coords.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.15,
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(ellipse)


def _plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    output_root: Path,
    color_by: str,
    marker_size: float,
    alpha: float,
    legend_loc: str,
    legend_ncol: int,
    no_axes: bool,
    draw_ellipses: bool,
    ellipse_nstd: float,
    title: Optional[str],
    dpi: int,
    is_continuous: bool,
    alpha_values: Optional[np.ndarray],
    palette_mode: str,
    palette_ref_dir: Optional[str],
) -> Tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.set_aspect("auto")

    if is_continuous:
        cmap = plt.cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=np.nanmin(labels), vmax=np.nanmax(labels))
        if alpha_values is not None:
            colors = cmap(norm(labels))
            colors[:, 3] = alpha_values
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                color=colors,
                s=marker_size,
                linewidths=0,
            )
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.85)
        else:
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels,
                cmap=cmap,
                s=marker_size,
                alpha=alpha,
                linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(color_by)
    else:
        unique_labels = sorted({str(label) for label in labels})
        fixed_map = DISTORTION_GROUP_COLORS if color_by == "distortion_group" else None
        if palette_mode == "shared":
            family = _infer_palette_family(output_root, color_by)
            label_to_color = _resolve_shared_label_colors(
                unique_labels,
                output_root,
                palette_ref_dir,
                family,
                fixed_map=fixed_map,
            )
        else:
            label2color_path = output_root / "label2color.json"
            label_to_color = _resolve_label_colors(
                unique_labels, label2color_path, fixed_map=fixed_map
            )

        for label in unique_labels:
            mask = labels == label
            label_coords = coords[mask]
            color = label_to_color[label]
            if draw_ellipses:
                _add_ellipse(ax, label_coords, color, ellipse_nstd)
            if alpha_values is None:
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    s=marker_size,
                    alpha=alpha,
                    color=color,
                    label=label,
                    linewidths=0,
                    zorder=2,
                )
            else:
                colors = np.tile(np.array(color), (label_coords.shape[0], 1))
                colors[:, 3] = alpha_values[mask]
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    s=marker_size,
                    color=colors,
                    label=label,
                    linewidths=0,
                    zorder=2,
                )

        legend_ncol_effective = legend_ncol
        if color_by == "distortion_group":
            legend_ncol_effective = 1
        elif len(unique_labels) > 5:
            legend_ncol_effective = max(legend_ncol_effective, 2)
        legend_kwargs = {
            "ncol": legend_ncol_effective,
            "frameon": False,
            "markerscale": 1.2,
        }
        if len(unique_labels) > 1:
            legend_kwargs.update(
                {"loc": "upper left", "bbox_to_anchor": (1.02, 1)}
            )
        else:
            legend_kwargs.update({"loc": legend_loc})
        ax.legend(**legend_kwargs)

    if title:
        ax.set_title(title)

    if no_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    fig.tight_layout()
    pdf_path = output_root / f"tsne_{color_by}.pdf"
    png_path = output_root / f"tsne_{color_by}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _compute_cluster_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    label_strings = np.array([str(label) for label in labels], dtype=object)
    counts = Counter(label_strings)
    metrics["class_counts"] = dict(sorted(counts.items()))
    metrics["n_samples"] = int(features.shape[0])
    metrics["feature_dim"] = int(features.shape[1])

    if len(counts) < 2:
        metrics["silhouette_score"] = None
        metrics["knn_accuracy"] = None
        metrics["note"] = "Not enough classes to compute metrics."
        return metrics

    labels_array = label_strings

    valid_counts = [count for count in counts.values() if count >= 2]
    if len(valid_counts) >= 2:
        if features.shape[0] > 5000:
            rng = np.random.default_rng(seed)
            subset = rng.choice(features.shape[0], size=5000, replace=False)
            metrics["silhouette_score"] = float(
                silhouette_score(features[subset], labels_array[subset])
            )
            metrics["silhouette_sample_size"] = 5000
        else:
            metrics["silhouette_score"] = float(
                silhouette_score(features, labels_array)
            )
    else:
        metrics["silhouette_score"] = None

    k = min(10, max(1, features.shape[0] - 1))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels_array)
    preds = knn.predict(features)
    metrics["knn_accuracy"] = float(np.mean(preds == labels_array))
    metrics["knn_k"] = int(k)
    metrics["knn_note"] = "Training-set kNN accuracy."
    return metrics


def _build_manifest(
    output_root: Path,
    features: Optional[np.ndarray],
    coords: np.ndarray,
    meta: Dict[str, List[Any]],
    label_summary: Dict[str, Dict[str, int]],
    tsne_config_path: Path,
    features_path: Path,
    tsne_path: Path,
    meta_path: Path,
) -> Dict[str, Any]:
    feature_dim = int(features.shape[1]) if features is not None else None
    num_samples = int(coords.shape[0])
    return {
        "timestamp": datetime.now().isoformat(),
        "n_samples": num_samples,
        "feature_dim": feature_dim,
        "features_path": str(features_path),
        "tsne_path": str(tsne_path),
        "meta_path": str(meta_path),
        "tsne_config_path": str(tsne_config_path),
        "label_summary": label_summary,
        "output_root": str(output_root),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE feature extraction and plotting")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint path or 'best'",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kadid10k",
        help="Dataset name",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subsample_frac", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--crop_policy",
        choices=["all", "first", "mean"],
        default="mean",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root directory",
    )
    parser.add_argument(
        "--save_format",
        choices=["pt", "npy"],
        default="pt",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip feature extraction and t-SNE, only regenerate plots",
    )
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--tsne_learning_rate", type=float, default=200)
    parser.add_argument("--tsne_n_iter", type=int, default=2000)
    parser.add_argument("--tsne_init", type=str, default="pca")
    parser.add_argument(
        "--color_by",
        choices=[
            "distortion_type",
            "distortion_group",
            "severity",
        ],
        default="distortion_type",
    )
    parser.add_argument(
        "--include_types",
        type=str,
        default=None,
        help="Comma-separated list of distortion types to include",
    )
    parser.add_argument(
        "--include_groups",
        type=str,
        default=None,
        help="Comma-separated list of distortion groups to include",
    )
    parser.add_argument(
        "--distortion_granularity",
        choices=["fine", "coarse"],
        default="fine",
    )
    parser.add_argument("--focus_distortion", type=str, default=None)
    parser.add_argument("--min_severity", type=int, default=None)
    parser.add_argument("--max_severity", type=int, default=None)
    parser.add_argument("--marker_size", type=float, default=8)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--alpha_by",
        choices=["none", "severity"],
        default="none",
    )
    parser.add_argument("--legend_loc", type=str, default="best")
    parser.add_argument("--legend_ncol", type=int, default=1)
    parser.add_argument("--no_axes", action="store_true")
    parser.add_argument("--draw_ellipses", action="store_true")
    parser.add_argument("--ellipse_nstd", type=float, default=2.0)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--palette_mode",
        choices=["auto", "shared"],
        default="shared",
        help="Use a shared palette across runs when enabled.",
    )
    parser.add_argument(
        "--palette_ref_dir",
        type=str,
        default="results/tsne_kadid_flattened/SHAMISA_A0",
        help="Reference directory for shared palettes.",
    )
    parser.add_argument(
        "--make_bundle_zip",
        type=str,
        default="false",
        help="Whether to create a bundle zip (true/false)",
    )
    parser.add_argument(
        "--also_plot_severity",
        action="store_true",
        help="Generate a severity plot using the same t-SNE coordinates",
    )
    parser.add_argument("--compute_cluster_metrics", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["scratch"],
        default="scratch",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    cli_args = _parse_args()

    if cli_args.subsample_frac <= 0 or cli_args.subsample_frac > 1:
        raise ValueError("--subsample_frac must be in (0, 1].")

    config = parse_config(cli_args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)
    args.experiment_name = cli_args.experiment_name
    args.eval_type = cli_args.eval_type
    args.data_base_path = Path(args.data_base_path)
    args = replace_none_string(args)

    _set_seed(cli_args.seed)

    include_types = _parse_csv_list(cli_args.include_types)
    include_groups = _parse_csv_list(cli_args.include_groups)
    make_bundle_zip = _parse_bool(cli_args.make_bundle_zip)
    subset_dir = _build_subset_dir_name(include_types, include_groups)

    base_output_root = (
        Path(cli_args.output_root)
        if cli_args.output_root
        else PROJECT_ROOT
        / "experiments"
        / cli_args.experiment_name
        / "tsne"
        / cli_args.dataset_name
    )
    output_root = base_output_root / subset_dir if subset_dir else base_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    logger = _init_wandb(args, cli_args)

    features: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    meta: Optional[Dict[str, List[Any]]] = None
    tsne_config: Dict[str, Any] = {}
    label_summary: Dict[str, Dict[str, int]] = {}
    checkpoint_path: Optional[Path] = None
    bundle_path: Optional[Path] = None

    if not cli_args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.eval_type == "scratch":
            if args.model.method == "simclr":
                model = SimCLR(
                    encoder_params=args.model.encoder, temperature=args.model.temperature
                )
            elif args.model.method == "vicreg":
                model = Vicreg(args)
            else:
                raise ValueError(f"Unknown method {args.model.method}")

            checkpoint_path = _resolve_checkpoint_path(
                cli_args.experiment_name, cli_args.checkpoint
            )
            _load_pretrained_weights(model, checkpoint_path)
        else:
            raise ValueError(f"Eval type {args.eval_type} not supported")

        model.to(device)
        model.eval()

        dataset_key = cli_args.dataset_name.lower()
        dataset, _, dataset_display_name = prepare_dataset(
            dataset_key,
            data_base_path=args.data_base_path,
            crop_size=args.test.crop_size,
            default_num_splits=args.test.num_splits,
        )

        num_crops = _infer_num_crops(dataset)
        prefilter_indices = _prefilter_indices(
            dataset=dataset,
            dataset_key=dataset_key,
            num_crops=num_crops,
            crop_policy=cli_args.crop_policy,
            seed=cli_args.seed,
            subsample_frac=cli_args.subsample_frac,
            max_samples=cli_args.max_samples,
            include_types=include_types,
            include_groups=include_groups,
            focus_distortion=cli_args.focus_distortion,
            min_severity=cli_args.min_severity,
            max_severity=cli_args.max_severity,
        )
        if len(prefilter_indices) == 0:
            raise RuntimeError("No samples available after prefiltering.")

        prefilter_applied = len(prefilter_indices) != len(dataset)
        dataset_for_loader = dataset
        index_map = None
        if prefilter_applied:
            dataset_for_loader = torch.utils.data.Subset(
                dataset, prefilter_indices.tolist()
            )
            index_map = prefilter_indices.tolist()

        batch_size = (
            cli_args.batch_size
            if cli_args.batch_size is not None
            else args.test.batch_size
        )
        num_workers = (
            cli_args.num_workers
            if cli_args.num_workers is not None
            else args.test.num_workers
        )

        dataloader = DataLoader(
            dataset_for_loader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        t0 = time.perf_counter()
        features, meta = _extract_features(
            model,
            dataloader,
            dataset,
            device,
            args.eval_type,
            dataset_key,
            index_map=index_map,
            crop_policy=cli_args.crop_policy,
        )
        t1 = time.perf_counter()
        print(
            f"Extracted {features.shape[0]} features in {(t1 - t0):.1f}s"
        )
    else:
        dataset_key = cli_args.dataset_name.lower()
        dataset_display_name = dataset_key

    if cli_args.plot_only:
        features_path, tsne_path, resolved_format = _resolve_array_paths(
            output_root, cli_args.save_format
        )
    else:
        resolved_format = cli_args.save_format
        features_path = output_root / f"features_H.{resolved_format}"
        tsne_path = output_root / f"Y_tsne.{resolved_format}"
    meta_path = output_root / "meta.pkl"

    if cli_args.plot_only:
        if not tsne_path.exists():
            raise FileNotFoundError(f"Missing t-SNE file: {tsne_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        coords = _load_array(tsne_path, resolved_format)
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        if cli_args.compute_cluster_metrics:
            if not features_path.exists():
                raise FileNotFoundError(f"Missing feature file: {features_path}")
            features = _load_array(features_path, resolved_format).astype(np.float32)
        if include_types or include_groups:
            config_path = output_root / "tsne_config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
                if config_data.get("include_types", []) != include_types:
                    raise RuntimeError(
                        "include_types does not match the saved t-SNE configuration."
                    )
                if config_data.get("include_groups", []) != include_groups:
                    raise RuntimeError(
                        "include_groups does not match the saved t-SNE configuration."
                    )

        filter_indices = _apply_filters(
            meta,
            cli_args.focus_distortion,
            cli_args.min_severity,
            cli_args.max_severity,
        )
        selected_indices = _apply_subsample(
            filter_indices,
            cli_args.seed,
            cli_args.subsample_frac,
            cli_args.max_samples,
        )
        coords = coords[selected_indices]
        meta = _subset_meta(meta, selected_indices)
        if features is not None:
            features = features[selected_indices]
    else:
        if features is None or meta is None:
            raise RuntimeError("Feature extraction failed.")

        if not prefilter_applied:
            filter_indices = _apply_filters(
                meta,
                cli_args.focus_distortion,
                cli_args.min_severity,
                cli_args.max_severity,
            )
            selected_indices = _apply_subsample(
                filter_indices,
                cli_args.seed,
                cli_args.subsample_frac,
                cli_args.max_samples,
            )
            features = features[selected_indices]
            meta = _subset_meta(meta, selected_indices)
        elif cli_args.max_samples is not None and features.shape[0] > cli_args.max_samples:
            selected_indices = _apply_subsample(
                np.arange(features.shape[0]),
                cli_args.seed,
                1.0,
                cli_args.max_samples,
            )
            features = features[selected_indices]
            meta = _subset_meta(meta, selected_indices)

        coords, pca_info = _run_tsne(
            features,
            cli_args.seed,
            cli_args.pca_dim,
            cli_args.tsne_perplexity,
            cli_args.tsne_learning_rate,
            cli_args.tsne_n_iter,
            cli_args.tsne_init,
        )

        _save_array(features_path, features, resolved_format)
        _save_array(tsne_path, coords, resolved_format)
        with meta_path.open("wb") as f:
            pickle.dump(meta, f)

        tsne_config = {
            "experiment_name": cli_args.experiment_name,
            "dataset_name": dataset_key,
            "dataset_display_name": dataset_display_name,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "eval_type": args.eval_type,
            "seed": cli_args.seed,
            "subsample_frac": cli_args.subsample_frac,
            "max_samples": cli_args.max_samples,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "output_root": str(output_root),
            "save_format": resolved_format,
            "num_crops": num_crops,
            "crop_policy": cli_args.crop_policy,
            "include_types": include_types,
            "include_groups": include_groups,
            "subset_dir": subset_dir,
            "prefilter_num_images": int(len(prefilter_indices)),
            "pca_dim": cli_args.pca_dim,
            "tsne_perplexity": cli_args.tsne_perplexity,
            "tsne_learning_rate": cli_args.tsne_learning_rate,
            "tsne_n_iter": cli_args.tsne_n_iter,
            "tsne_init": cli_args.tsne_init,
            "distortion_granularity": cli_args.distortion_granularity,
            "focus_distortion": cli_args.focus_distortion,
            "min_severity": cli_args.min_severity,
            "max_severity": cli_args.max_severity,
            "color_by": cli_args.color_by,
            "alpha_by": cli_args.alpha_by,
            "also_plot_severity": cli_args.also_plot_severity,
            "make_bundle_zip": make_bundle_zip,
            "plot_style": {
                "marker_size": cli_args.marker_size,
                "alpha": cli_args.alpha,
                "alpha_by": cli_args.alpha_by,
                "legend_loc": cli_args.legend_loc,
                "legend_ncol": cli_args.legend_ncol,
                "no_axes": cli_args.no_axes,
                "draw_ellipses": cli_args.draw_ellipses,
                "ellipse_nstd": cli_args.ellipse_nstd,
                "title": cli_args.title,
                "dpi": cli_args.dpi,
            },
            "feature_dim": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "timestamp": datetime.now().isoformat(),
            "pca_info": pca_info,
        }

    if coords is None or meta is None:
        raise RuntimeError("Missing t-SNE data for plotting.")
    if len(meta["score"]) != coords.shape[0]:
        raise RuntimeError("Metadata and t-SNE coordinates are misaligned.")

    alpha_values = _compute_alpha_values(meta, cli_args.alpha_by, cli_args.alpha)

    labels, is_continuous = _build_color_labels(
        meta,
        cli_args.color_by,
        dataset_key,
        cli_args.distortion_granularity,
    )

    if cli_args.color_by == "severity":
        valid_mask = ~np.isnan(labels)
        coords_plot = coords[valid_mask]
        labels_plot = labels[valid_mask]
        meta_plot = _subset_meta(meta, np.where(valid_mask)[0])
        alpha_plot = alpha_values[valid_mask] if alpha_values is not None else None
    else:
        coords_plot = coords
        labels_plot = labels
        meta_plot = meta
        alpha_plot = alpha_values

    if coords_plot.shape[0] == 0:
        raise RuntimeError("No samples available after filtering for plotting.")

    pdf_path, png_path = _plot_tsne(
        coords_plot,
        labels_plot,
        output_root,
        cli_args.color_by,
        cli_args.marker_size,
        cli_args.alpha,
        cli_args.legend_loc,
        cli_args.legend_ncol,
        cli_args.no_axes,
        cli_args.draw_ellipses,
        cli_args.ellipse_nstd,
        cli_args.title,
        cli_args.dpi,
        is_continuous,
        alpha_plot,
        cli_args.palette_mode,
        cli_args.palette_ref_dir,
    )

    extra_plot_paths: Dict[str, Tuple[Path, Path]] = {}
    if cli_args.also_plot_severity and cli_args.color_by != "severity":
        severity_labels, severity_continuous = _build_color_labels(
            meta,
            "severity",
            dataset_key,
            cli_args.distortion_granularity,
        )
        severity_mask = ~np.isnan(severity_labels)
        if np.any(severity_mask):
            severity_coords = coords[severity_mask]
            severity_labels_plot = severity_labels[severity_mask]
            severity_alpha = (
                alpha_values[severity_mask] if alpha_values is not None else None
            )
            sev_pdf, sev_png = _plot_tsne(
                severity_coords,
                severity_labels_plot,
                output_root,
                "severity",
                cli_args.marker_size,
                cli_args.alpha,
                cli_args.legend_loc,
                cli_args.legend_ncol,
                cli_args.no_axes,
                cli_args.draw_ellipses,
                cli_args.ellipse_nstd,
                cli_args.title,
                cli_args.dpi,
                severity_continuous,
                severity_alpha,
                cli_args.palette_mode,
                cli_args.palette_ref_dir,
            )
            extra_plot_paths["severity"] = (sev_pdf, sev_png)
        else:
            raise RuntimeError("No valid severity values available for plotting.")

    label_summary = {}
    for key in ("distortion_type", "distortion_group", "severity", "split"):
        values = meta_plot.get(key, [])
        if key == "severity":
            values = [
                str(_coerce_int(val))
                for val in values
                if _coerce_int(val) is not None
            ]
        label_summary[key] = dict(sorted(Counter([str(v) for v in values]).items()))

    if is_continuous:
        label_values = [
            str(_coerce_int(val)) if _coerce_int(val) is not None else str(val)
            for val in labels_plot
        ]
    else:
        label_values = [str(val) for val in labels_plot]
    label_summary["plot_labels"] = dict(sorted(Counter(label_values).items()))

    tsne_config_path = output_root / "tsne_config.json"
    if not cli_args.plot_only:
        with tsne_config_path.open("w", encoding="utf-8") as f:
            json.dump(tsne_config, f, indent=2)

    manifest = _build_manifest(
        output_root,
        features,
        coords,
        meta,
        label_summary,
        tsne_config_path,
        features_path,
        tsne_path,
        meta_path,
    )
    manifest["color_by"] = cli_args.color_by
    plots = {
        cli_args.color_by: {"pdf": str(pdf_path), "png": str(png_path)}
    }
    for plot_name, (plot_pdf, plot_png) in extra_plot_paths.items():
        plots[plot_name] = {"pdf": str(plot_pdf), "png": str(plot_png)}
    manifest["plots"] = plots
    if make_bundle_zip:
        bundle_path = output_root / "bundle.zip"
        manifest["bundle_zip"] = str(bundle_path)
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if cli_args.compute_cluster_metrics:
        if features is None:
            raise RuntimeError("Features are required for cluster metrics.")
        distortion_labels, _ = _build_color_labels(
            meta,
            "distortion_type",
            dataset_key,
            cli_args.distortion_granularity,
        )
        metrics = _compute_cluster_metrics(features, distortion_labels, cli_args.seed)
        metrics_path = output_root / "cluster_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if logger:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.summary[f"tsne_{key}"] = value

    if make_bundle_zip:
        bundle_path = _make_bundle_zip(output_root)

    if logger:
        log_payload = {f"tsne_{cli_args.color_by}": wandb.Image(str(png_path))}
        for plot_name, (_, plot_png) in extra_plot_paths.items():
            log_payload[f"tsne_{plot_name}"] = wandb.Image(str(plot_png))
        logger.log(log_payload)
        logger.summary["tsne_pdf_path"] = str(pdf_path)
        logger.summary["tsne_output_root"] = str(output_root)
        logger.summary["tsne_color_by"] = cli_args.color_by
        logger.summary["tsne_crop_policy"] = cli_args.crop_policy
        logger.summary["tsne_alpha_by"] = cli_args.alpha_by
        logger.summary["tsne_include_types"] = include_types
        logger.summary["tsne_include_groups"] = include_groups
        logger.summary["tsne_label_summary"] = label_summary
        if bundle_path is not None:
            logger.summary["tsne_bundle_zip"] = str(bundle_path)

        artifact = wandb.Artifact(
            f"tsne_{cli_args.experiment_name}_{cli_args.color_by}",
            type="tsne_results",
        )
        artifact_paths = [
            pdf_path,
            png_path,
            tsne_config_path,
            manifest_path,
            features_path,
            tsne_path,
            meta_path,
        ]
        for plot_name, (plot_pdf, plot_png) in extra_plot_paths.items():
            artifact_paths.extend([plot_pdf, plot_png])
        if bundle_path is not None:
            artifact_paths.append(bundle_path)
        for path in artifact_paths:
            if path and Path(path).exists():
                artifact.add_file(str(path))
        logger.log_artifact(artifact)


if __name__ == "__main__":
    main()
