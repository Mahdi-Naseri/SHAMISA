import argparse
import inspect
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from dotmap import DotMap
from sklearn.decomposition import PCA
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from umap import UMAP

from utils.torch_amp_compat import patch_cuda_amp_custom_autocast
from utils.utils_data import distortion_functions, distortion_range, resize_crop
from utils.utils import (
    PROJECT_ROOT,
    parse_command_line_args,
    parse_config,
    merge_configs,
    replace_none_string,
    prepare_wandb_config,
)
from models.simclr import SimCLR
from models.vicreg import Vicreg

patch_cuda_amp_custom_autocast(device_type="cuda")

BLUR_KEY = "gaublur"
NOISE_KEY = "whitenoise"
BLUR_LEVELS = distortion_range[BLUR_KEY]
NOISE_LEVELS = distortion_range[NOISE_KEY]
LEVEL_INDEX_MAX = len(BLUR_LEVELS)
TOTAL_SEVERITY_MAX = LEVEL_INDEX_MAX * 2


def _sanitize_identifier(value: str) -> str:
    value = value.replace("->", "_to_").replace(" ", "_")
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars)
    return sanitized.strip("_") or "result"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_serializable(value: Any):
    if isinstance(value, DotMap):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _load_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
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


def _resolve_checkpoint_path(
    experiment_name: str, checkpoint: str, checkpoint_path: Optional[str]
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

    checkpoints = sorted(pretrain_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {pretrain_dir}")

    if checkpoint == "best":
        matches = [ckpt for ckpt in checkpoints if "best" in ckpt.name]
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint containing 'best' found in {pretrain_dir}"
            )
    else:
        matches = [ckpt for ckpt in checkpoints if checkpoint in ckpt.name]
        if not matches:
            available = ", ".join(ckpt.name for ckpt in checkpoints)
            raise FileNotFoundError(
                f"No checkpoint matching '{checkpoint}' found in {pretrain_dir}. Available: {available}"
            )

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _deterministic_resize_crop(img: Image.Image, crop_size: int, seed: int) -> Image.Image:
    state = random.getstate()
    random.seed(seed)
    try:
        return resize_crop(img, crop_size)
    finally:
        random.setstate(state)


def _load_kadis_ref_images(root: Path) -> List[Path]:
    csv_path = PROJECT_ROOT / "data" / "synthetic_filenames.csv"
    if not csv_path.exists():
        images = list((root / "ref_imgs").glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No reference images found under {root / 'ref_imgs'}")
        pd.DataFrame(images, columns=["Filename"]).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    ref_images = []
    for entry in df["Filename"].tolist():
        path = Path(entry)
        if not path.exists():
            candidate = root / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(
                f"Reference image path not found: {path}."
            )
        ref_images.append(path)
    return ref_images


def _apply_single_distortion(
    image: torch.Tensor, key: str, level_index: int
) -> torch.Tensor:
    if level_index <= 0:
        return image.clone()
    level_value = distortion_range[key][level_index - 1]
    func = distortion_functions[key]
    output = func(image.clone(), level_value)
    output = output.to(torch.float32)
    output = torch.clip(output, 0, 1)
    return output


def _build_variants(
    image: torch.Tensor,
    normalize: transforms.Normalize,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    variants: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] = []

    blur_cache = {}
    noise_cache = {}

    for idx, blur_value in enumerate(BLUR_LEVELS, start=1):
        blur_img = _apply_single_distortion(image, BLUR_KEY, idx)
        blur_cache[idx] = blur_img
        variants.append(normalize(blur_img))
        metadata.append(
            {
                "comp_type": "blur_only",
                "blur_level_index": idx,
                "noise_level_index": 0,
                "blur_level_value": float(blur_value),
                "noise_level_value": 0.0,
            }
        )

    for idx, noise_value in enumerate(NOISE_LEVELS, start=1):
        noise_img = _apply_single_distortion(image, NOISE_KEY, idx)
        noise_cache[idx] = noise_img
        variants.append(normalize(noise_img))
        metadata.append(
            {
                "comp_type": "noise_only",
                "blur_level_index": 0,
                "noise_level_index": idx,
                "blur_level_value": 0.0,
                "noise_level_value": float(noise_value),
            }
        )

    for blur_idx, blur_value in enumerate(BLUR_LEVELS, start=1):
        blur_img = blur_cache[blur_idx]
        for noise_idx, noise_value in enumerate(NOISE_LEVELS, start=1):
            combined_img = _apply_single_distortion(blur_img, NOISE_KEY, noise_idx)
            variants.append(normalize(combined_img))
            metadata.append(
                {
                    "comp_type": "combined",
                    "blur_level_index": blur_idx,
                    "noise_level_index": noise_idx,
                    "blur_level_value": float(blur_value),
                    "noise_level_value": float(noise_value),
                }
            )

    return variants, metadata


def _extract_encoder_features(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encoder"):
        output = model.encoder(batch)
    else:
        output = model(batch)

    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _compute_axis_limits(y: np.ndarray, padding: float = 0.05) -> Dict[str, float]:
    x_min = float(np.min(y[:, 0]))
    x_max = float(np.max(y[:, 0]))
    y_min = float(np.min(y[:, 1]))
    y_max = float(np.max(y[:, 1]))

    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding

    return {
        "x_min": x_min - x_pad,
        "x_max": x_max + x_pad,
        "y_min": y_min - y_pad,
        "y_max": y_max + y_pad,
    }


def _load_axis_limits(path: Path) -> Dict[str, float]:
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            axis = payload.get("axis_limits")
            if axis:
                return axis
        npz_path = path / "umap_embeddings.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            return _compute_axis_limits(data["Y"])
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        axis = payload.get("axis_limits") or payload.get("axis")
        if axis:
            return axis
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        return _compute_axis_limits(data["Y"])

    raise FileNotFoundError(f"Could not load axis limits from {path}")


def _plot_umap(
    y: np.ndarray,
    blur_indices: np.ndarray,
    noise_indices: np.ndarray,
    total_severity: np.ndarray,
    output_root: Path,
    experiment_name: str,
    n_neighbors: int,
    min_dist: float,
    marker_size: float,
    alpha_min: float,
    alpha_max: float,
    dpi: int,
    no_axes: bool,
    fixed_axis: Optional[Dict[str, float]],
    save_pdf: bool,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    blur_color = np.array([1.0, 0.0, 0.0])
    noise_color = np.array([1.0, 1.0, 0.0])

    blur_weights = blur_indices.astype(np.float32) / float(LEVEL_INDEX_MAX)
    noise_weights = noise_indices.astype(np.float32) / float(LEVEL_INDEX_MAX)
    weight_sum = blur_weights + noise_weights
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)

    rgb = (blur_weights[:, None] * blur_color + noise_weights[:, None] * noise_color) / weight_sum[:, None]
    rgb = np.clip(rgb, 0.0, 1.0)

    severity_norm = total_severity.astype(np.float32) / float(TOTAL_SEVERITY_MAX)
    alphas = alpha_min + (alpha_max - alpha_min) * severity_norm
    alphas = np.clip(alphas, 0.0, 1.0)

    colors = np.concatenate([rgb, alphas[:, None]], axis=1)

    fig = plt.figure(figsize=(6.5, 6.0), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(y[:, 0], y[:, 1], s=marker_size, c=colors, linewidths=0)

    if fixed_axis is not None:
        ax.set_xlim(fixed_axis["x_min"], fixed_axis["x_max"])
        ax.set_ylim(fixed_axis["y_min"], fixed_axis["y_max"])

    if no_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    # Add gradient legend
    gradient_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    gradient = np.linspace(0, 1, 256)[:, None]
    gradient_rgb = (1 - gradient) * blur_color + gradient * noise_color
    gradient_ax.imshow(gradient_rgb.reshape(256, 1, 3), aspect="auto", origin="lower")
    gradient_ax.set_xticks([])
    gradient_ax.set_yticks([])
    gradient_ax.text(1.6, 0, "Blur", va="bottom", ha="left", fontsize=10)
    gradient_ax.text(1.6, 255, "Noise", va="top", ha="left", fontsize=10)

    min_dist_str = f"{min_dist}".replace(" ", "")
    base_name = f"umap_gradient_umap_{n_neighbors}neigh_{min_dist_str}mindist"
    style_png = output_root / f"{base_name}.png"

    exp_name = _sanitize_identifier(experiment_name)
    main_png = output_root / f"umap_{exp_name}_{BLUR_KEY}_{NOISE_KEY}.png"

    fig.savefig(style_png, bbox_inches="tight")
    fig.savefig(main_png, bbox_inches="tight")

    outputs = {
        "style_png": str(style_png),
        "main_png": str(main_png),
    }

    if save_pdf:
        style_pdf = output_root / f"{base_name}.pdf"
        main_pdf = output_root / f"umap_{exp_name}_{BLUR_KEY}_{NOISE_KEY}.pdf"
        fig.savefig(style_pdf, bbox_inches="tight")
        fig.savefig(main_pdf, bbox_inches="tight")
        outputs["style_pdf"] = str(style_pdf)
        outputs["main_pdf"] = str(main_pdf)

    plt.close(fig)
    return outputs


def _compute_manifold_stats(
    y: np.ndarray,
    comp_type: np.ndarray,
    blur_indices: np.ndarray,
    noise_indices: np.ndarray,
) -> Dict[str, Any]:
    stats_payload: Dict[str, Any] = {}

    blur_mask = comp_type == "blur_only"
    noise_mask = comp_type == "noise_only"
    combined_mask = comp_type == "combined"

    blur_centroid = np.mean(y[blur_mask], axis=0) if np.any(blur_mask) else np.array([np.nan, np.nan])
    noise_centroid = np.mean(y[noise_mask], axis=0) if np.any(noise_mask) else np.array([np.nan, np.nan])

    stats_payload["blur_only_centroid"] = blur_centroid.tolist()
    stats_payload["noise_only_centroid"] = noise_centroid.tolist()

    if np.any(combined_mask):
        combined = y[combined_mask]
        dist_blur = np.linalg.norm(combined - blur_centroid, axis=1)
        dist_noise = np.linalg.norm(combined - noise_centroid, axis=1)
        stats_payload["combined_fraction_closer_to_noise"] = float(np.mean(dist_noise < dist_blur))
    else:
        stats_payload["combined_fraction_closer_to_noise"] = float("nan")

    def _spearman(a, b):
        if len(a) == 0:
            return {"rho": float("nan"), "p": float("nan")}
        rho, p = stats.spearmanr(a, b)
        return {"rho": float(rho), "p": float(p)}

    stats_payload["spearman"] = {
        "x_blur": _spearman(y[:, 0], blur_indices),
        "y_blur": _spearman(y[:, 1], blur_indices),
        "x_noise": _spearman(y[:, 0], noise_indices),
        "y_noise": _spearman(y[:, 1], noise_indices),
    }

    return stats_payload


class KADISPristineDataset(Dataset):
    def __init__(
        self,
        root: Path,
        patch_size: int,
        indices: List[int],
        seed: int,
    ):
        self.root = root
        self.patch_size = patch_size
        self.indices = indices
        self.seed = seed
        self.ref_images = _load_kadis_ref_images(root)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref_idx = self.indices[idx]
        img_path = self.ref_images[ref_idx]
        img = Image.open(img_path).convert("RGB")
        img = _deterministic_resize_crop(img, self.patch_size, self.seed + ref_idx)
        img_tensor = transforms.ToTensor()(img)
        return {
            "image": img_tensor,
            "content_id": img_path.stem,
            "content_index": ref_idx,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Defaults to experiments/<experiment_name>/umap_manifold/",
    )

    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=50)
    parser.add_argument("--min_dist", type=float, default=0.99)
    parser.add_argument("--metric", type=str, default="euclidean")

    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--fixed_axis_from", type=str, default=None)

    parser.add_argument("--marker_size", type=float, default=10)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.85)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_axes", action="store_true")

    parser.add_argument(
        "--enable_wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--compute_manifold_stats", action="store_true")
    parser.add_argument(
        "--eval_type",
        type=str,
        default="scratch",
        choices=["scratch"],
    )

    umap_args, unknown = parser.parse_known_args()

    config = parse_config(umap_args.config)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + unknown
        config_args = parse_command_line_args(config)
    finally:
        sys.argv = original_argv

    args = merge_configs(config, config_args)
    args = replace_none_string(args)
    args.experiment_name = umap_args.experiment_name

    args.data_base_path = Path(args.data_base_path)

    if umap_args.output_root is None:
        output_root = PROJECT_ROOT / "experiments" / umap_args.experiment_name / "umap_manifold"
    else:
        output_root = Path(umap_args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    _set_seed(umap_args.seed)

    # Device setup
    device = torch.device(umap_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")

    # Model loading
    resolved_checkpoint = None
    if args.model.method == "simclr":
        model = SimCLR(
            encoder_params=args.model.encoder, temperature=args.model.temperature
        )
    elif args.model.method == "vicreg":
        model = Vicreg(args)
    else:
        raise ValueError(f"Unsupported method: {args.model.method}")

    resolved_checkpoint = _resolve_checkpoint_path(
        umap_args.experiment_name, umap_args.checkpoint, umap_args.checkpoint_path
    )
    _load_pretrained_weights(model, resolved_checkpoint)

    model.to(device)
    model.eval()

    logger = None
    enable_wandb = (
        umap_args.enable_wandb
        if umap_args.enable_wandb is not None
        else bool(args.logging.use_wandb)
    )
    if enable_wandb:
        import wandb

        wandb_config = prepare_wandb_config({"config": args, "umap": vars(umap_args)})
        project = umap_args.wandb_project or args.logging.wandb.project
        run_name = umap_args.wandb_run_name or umap_args.experiment_name
        mode = "online" if args.logging.wandb.online else "offline"
        logger = wandb.init(
            project=project,
            name=run_name,
            config=wandb_config,
            mode=mode,
            resume="never",
        )

    embedding_path = output_root / "umap_embeddings.npz"
    config_path = output_root / "umap_config.json"
    manifest_path = output_root / "manifest.json"
    stats_path = output_root / "manifold_stats.json"

    n_neighbors_used = umap_args.n_neighbors
    min_dist_used = umap_args.min_dist

    if umap_args.plot_only or (umap_args.resume and embedding_path.exists()):
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        data = np.load(embedding_path)
        h = data["H"]
        y = data["Y"]
        content_id = data["content_id"]
        comp_type = data["comp_type"]
        blur_level_index = data["blur_level_index"]
        noise_level_index = data["noise_level_index"]
        total_severity = data["total_severity"]
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text())
                umap_payload = payload.get("umap", {})
                n_neighbors_used = int(umap_payload.get("n_neighbors", n_neighbors_used))
                min_dist_used = float(umap_payload.get("min_dist", min_dist_used))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    else:
        patch_size = args.training.data.patch_size
        dataset_root = args.data_base_path / "KADIS700"

        ref_images = _load_kadis_ref_images(dataset_root)
        if umap_args.num_images > len(ref_images):
            raise ValueError(
                f"Requested {umap_args.num_images} images but only {len(ref_images)} available."
            )

        rng = np.random.default_rng(umap_args.seed)
        selected_indices = rng.choice(len(ref_images), size=umap_args.num_images, replace=False)
        selected_indices = selected_indices.tolist()

        dataset = KADISPristineDataset(
            root=dataset_root,
            patch_size=patch_size,
            indices=selected_indices,
            seed=umap_args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=umap_args.num_workers,
            pin_memory=False,
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        features_list: List[np.ndarray] = []
        content_ids: List[str] = []
        content_indices: List[int] = []
        comp_types: List[str] = []
        blur_level_indices: List[int] = []
        noise_level_indices: List[int] = []
        blur_level_values: List[float] = []
        noise_level_values: List[float] = []
        total_severities: List[int] = []

        autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

        for batch in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            img_raw = batch["image"].squeeze(0)
            content_id = batch["content_id"][0]
            content_index = int(batch["content_index"][0])

            variants, metadata = _build_variants(img_raw, normalize)

            for meta in metadata:
                content_ids.append(content_id)
                content_indices.append(content_index)
                comp_types.append(meta["comp_type"])
                blur_level_indices.append(meta["blur_level_index"])
                noise_level_indices.append(meta["noise_level_index"])
                blur_level_values.append(meta["blur_level_value"])
                noise_level_values.append(meta["noise_level_value"])
                total_severities.append(meta["blur_level_index"] + meta["noise_level_index"])

            for start in range(0, len(variants), umap_args.batch_size):
                batch_tensor = torch.stack(variants[start : start + umap_args.batch_size])
                batch_tensor = batch_tensor.to(device, non_blocking=True)
                with torch.no_grad(), autocast_ctx():
                    h_batch = _extract_encoder_features(model, batch_tensor)
                features_list.append(h_batch.detach().cpu().float().numpy())

        h = np.concatenate(features_list, axis=0).astype(np.float32)
        content_id = np.array(content_ids)
        content_index = np.array(content_indices)
        comp_type = np.array(comp_types)
        blur_level_index = np.array(blur_level_indices)
        noise_level_index = np.array(noise_level_indices)
        blur_level_value = np.array(blur_level_values, dtype=np.float32)
        noise_level_value = np.array(noise_level_values, dtype=np.float32)
        total_severity = np.array(total_severities)

        if umap_args.pca_dim and umap_args.pca_dim > 0:
            pca_dim = min(umap_args.pca_dim, h.shape[1])
            pca = PCA(n_components=pca_dim, random_state=umap_args.seed)
            h_for_umap = pca.fit_transform(h)
        else:
            pca_dim = 0
            h_for_umap = h

        n_neighbors_used = min(umap_args.n_neighbors, h_for_umap.shape[0] - 1)
        umap_model = UMAP(
            n_components=2,
            n_neighbors=n_neighbors_used,
            min_dist=min_dist_used,
            metric=umap_args.metric,
            random_state=umap_args.seed,
        )
        y = umap_model.fit_transform(h_for_umap).astype(np.float32)

        np.savez_compressed(
            embedding_path,
            H=h,
            Y=y,
            content_id=content_id,
            content_index=content_index,
            comp_type=comp_type,
            blur_level_index=blur_level_index,
            noise_level_index=noise_level_index,
            blur_level_value=blur_level_value,
            noise_level_value=noise_level_value,
            total_severity=total_severity,
        )

        config_payload = {
            "experiment_name": umap_args.experiment_name,
            "config_path": umap_args.config,
            "resolved_checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint else None,
            "distortions": {
                "blur_key": BLUR_KEY,
                "noise_key": NOISE_KEY,
                "blur_levels": BLUR_LEVELS,
                "noise_levels": NOISE_LEVELS,
                "level_indexing": "1..5 with 0 as none",
            },
            "seed": umap_args.seed,
            "num_images": umap_args.num_images,
            "batch_size": umap_args.batch_size,
            "pca_dim": pca_dim,
            "umap": {
                "n_neighbors": n_neighbors_used,
                "min_dist": min_dist_used,
                "metric": umap_args.metric,
            },
            "device": str(device),
            "eval_type": umap_args.eval_type,
            "config": _to_serializable(args),
        }
        config_path.write_text(json.dumps(config_payload, indent=2))

    fixed_axis = None
    if umap_args.fixed_axis_from:
        fixed_axis = _load_axis_limits(Path(umap_args.fixed_axis_from))

    axis_limits = fixed_axis or _compute_axis_limits(y)

    plot_outputs = _plot_umap(
        y=y,
        blur_indices=blur_level_index,
        noise_indices=noise_level_index,
        total_severity=total_severity,
        output_root=output_root,
        experiment_name=umap_args.experiment_name,
        n_neighbors=n_neighbors_used,
        min_dist=min_dist_used,
        marker_size=umap_args.marker_size,
        alpha_min=umap_args.alpha_min,
        alpha_max=umap_args.alpha_max,
        dpi=umap_args.dpi,
        no_axes=umap_args.no_axes,
        fixed_axis=axis_limits,
        save_pdf=umap_args.save_pdf,
    )

    manifest = {
        "experiment_name": umap_args.experiment_name,
        "num_images": int(umap_args.num_images),
        "num_variants_per_image": 35,
        "num_points": int(y.shape[0]),
        "embedding_path": str(embedding_path),
        "n_neighbors": int(n_neighbors_used),
        "min_dist": float(min_dist_used),
        "plot_paths": plot_outputs,
        "axis_limits": axis_limits,
        "fixed_axis_from": umap_args.fixed_axis_from,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if umap_args.compute_manifold_stats:
        stats_payload = _compute_manifold_stats(
            y=y,
            comp_type=comp_type,
            blur_indices=blur_level_index,
            noise_indices=noise_level_index,
        )
        stats_path.write_text(json.dumps(stats_payload, indent=2))

    if logger is not None:
        import wandb

        log_payload = {
            "umap/main": wandb.Image(plot_outputs["main_png"]),
            "umap/style": wandb.Image(plot_outputs["style_png"]),
        }
        logger.log(log_payload)
        logger.summary["umap_output_root"] = str(output_root)
        logger.summary["umap_embedding_path"] = str(embedding_path)
        logger.summary["umap_config_path"] = str(config_path)
        logger.summary["umap_manifest_path"] = str(manifest_path)
        logger.summary["umap_plot_paths"] = plot_outputs
        logger.summary["umap_n_neighbors"] = int(n_neighbors_used)
        logger.summary["umap_min_dist"] = float(min_dist_used)
        logger.summary["umap_num_points"] = int(y.shape[0])
        if stats_path.exists():
            try:
                logger.summary["umap_stats"] = json.loads(stats_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.summary["umap_stats"] = str(stats_path)

        artifact_name = f"umap_{_sanitize_identifier(umap_args.experiment_name)}_{BLUR_KEY}_{NOISE_KEY}"
        artifact = wandb.Artifact(artifact_name, type="umap_results")
        artifact_paths = [
            embedding_path,
            config_path,
            manifest_path,
            stats_path if stats_path.exists() else None,
        ]
        for path_str in plot_outputs.values():
            artifact_paths.append(Path(path_str))

        for path in artifact_paths:
            if path and Path(path).exists():
                artifact.add_file(str(path))

        logger.log_artifact(artifact)
        logger.finish()


if __name__ == "__main__":
    main()
