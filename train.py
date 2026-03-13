# from line_profiler import profile
import copy
import csv
import json
import math
import random
import re
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import wandb
from datetime import datetime
from wandb.wandb_run import Run
from PIL import Image, ImageFile
from dotmap import DotMap
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from data import KADID10KDataset
from test import (
    get_results,
    get_cross_results,
    synthetic_datasets,
    authentic_datasets,
)
from utils.visualization import visualize_tsne_umap_mos

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, None)
    if value is None:
        return bool(default)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_rank0() -> bool:
    return not _is_distributed() or torch.distributed.get_rank() == 0


def _safe_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return mean, std


def _coerce_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def _atomic_torch_save(payload: Any, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp_path, target_path)
    try:
        dir_fd = os.open(str(target_path.parent), os.O_DIRECTORY)
    except OSError:
        dir_fd = None
    if dir_fd is not None:
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _build_train_resume_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args: DotMap,
    global_step: int,
    tag: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "epoch": epoch,
        "global_step": int(global_step),
        "checkpoint_tag": tag,
        "config": args,
    }
    try:
        rng_state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        payload["rng_state"] = rng_state
    except Exception:
        # RNG state capture is best-effort and should never block checkpointing.
        pass
    return payload


def _collect_grad_probe_targets(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    root_model = model.module if hasattr(model, "module") else model
    if not hasattr(root_model, "encoder"):
        return {}
    encoder = root_model.encoder
    targets: Dict[str, torch.nn.Parameter] = {}

    projector = getattr(encoder, "projector", None)
    if isinstance(projector, torch.nn.Module):
        projector_params = [p for p in projector.parameters() if p.requires_grad]
        if projector_params:
            targets["proj_first"] = projector_params[0]
            targets["proj_last"] = projector_params[-1]

    backbone = getattr(encoder, "model", None)
    if isinstance(backbone, torch.nn.Module):
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        if backbone_params:
            targets["backbone_last"] = backbone_params[-1]

    return targets


def _grad_l2_norm(grad: Optional[torch.Tensor]) -> float:
    if grad is None:
        return float("nan")
    return float(grad.detach().float().norm().item())


def _compute_grad_probe_metrics(
    *,
    loss: torch.Tensor,
    loss_terms_dict: Dict[str, Any],
    coeff: Dict[str, float],
    probe_targets: Dict[str, torch.nn.Parameter],
    include_unweighted: bool,
) -> Dict[str, float]:
    if not probe_targets:
        return {}

    probe_names = list(probe_targets.keys())
    probe_params = [probe_targets[name] for name in probe_names]
    out: Dict[str, float] = {}

    total_grads = torch.autograd.grad(
        loss, probe_params, retain_graph=True, allow_unused=True
    )
    for name, grad in zip(probe_names, total_grads):
        out[f"grad_total_{name}"] = _grad_l2_norm(grad)

    for term_name in ("var_loss", "inv_loss", "cov_loss"):
        term = loss_terms_dict.get(term_name, None)
        if (not torch.is_tensor(term)) or (not term.requires_grad):
            continue

        weighted_term = float(coeff.get(term_name, 1.0)) * term
        if not weighted_term.requires_grad:
            continue
        weighted_grads = torch.autograd.grad(
            weighted_term, probe_params, retain_graph=True, allow_unused=True
        )
        for name, grad in zip(probe_names, weighted_grads):
            out[f"grad_w_{term_name}_{name}"] = _grad_l2_norm(grad)

        if include_unweighted:
            unweighted_grads = torch.autograd.grad(
                term, probe_params, retain_graph=True, allow_unused=True
            )
            for name, grad in zip(probe_names, unweighted_grads):
                out[f"grad_u_{term_name}_{name}"] = _grad_l2_norm(grad)

    return out


class _StepTimingRecorder:
    def __init__(
        self,
        enabled: bool,
        csv_path: Path,
        warmup_steps: int,
        benchmark_steps: int,
        max_duration_minutes: Optional[float],
        run_label: str,
        sync_cuda: bool,
        stop_after_benchmark: bool,
    ):
        self.enabled = bool(enabled)
        self.csv_path = Path(csv_path)
        self.warmup_steps = max(0, int(warmup_steps))
        self.benchmark_steps = max(0, int(benchmark_steps))
        self.max_duration_seconds = (
            None
            if max_duration_minutes is None
            else max(0.0, float(max_duration_minutes)) * 60.0
        )
        self.run_label = str(run_label) if run_label is not None else ""
        self.sync_cuda = bool(sync_cuda)
        self.stop_after_benchmark = bool(stop_after_benchmark)
        self.step_count = 0
        self.benchmark_count = 0
        self._benchmark_start_time = None
        self._csv_file = None
        self._writer = None
        self._flush_every = 100
        self._columns = [
            "run_label",
            "phase",
            "step_idx",
            "global_step",
            "epoch",
            "batch_idx",
            "loss",
            "data_wait_s",
            "batch_prep_s",
            "fwd_bwd_optim_s",
            "total_step_s",
        ]
        self._benchmark_values: Dict[str, List[float]] = {
            "data_wait_s": [],
            "batch_prep_s": [],
            "fwd_bwd_optim_s": [],
            "total_step_s": [],
        }

        if not self.enabled:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._columns)
        self._writer.writeheader()
        self._csv_file.flush()

    def should_sync_cuda(self, device: torch.device) -> bool:
        return self.enabled and self.sync_cuda and device.type == "cuda"

    def record(
        self,
        *,
        global_step: int,
        epoch: int,
        batch_idx: int,
        loss: float,
        data_wait_s: float,
        batch_prep_s: float,
        fwd_bwd_optim_s: float,
        total_step_s: float,
    ) -> None:
        if not self.enabled:
            return
        self.step_count += 1
        is_warmup = self.step_count <= self.warmup_steps
        phase = "warmup" if is_warmup else "benchmark"
        if not is_warmup:
            self.benchmark_count += 1
            if self._benchmark_start_time is None:
                self._benchmark_start_time = time.perf_counter()
            self._benchmark_values["data_wait_s"].append(float(data_wait_s))
            self._benchmark_values["batch_prep_s"].append(float(batch_prep_s))
            self._benchmark_values["fwd_bwd_optim_s"].append(float(fwd_bwd_optim_s))
            self._benchmark_values["total_step_s"].append(float(total_step_s))

        row = {
            "run_label": self.run_label,
            "phase": phase,
            "step_idx": self.step_count,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "loss": float(loss),
            "data_wait_s": float(data_wait_s),
            "batch_prep_s": float(batch_prep_s),
            "fwd_bwd_optim_s": float(fwd_bwd_optim_s),
            "total_step_s": float(total_step_s),
        }
        self._writer.writerow(row)
        if self.step_count % self._flush_every == 0:
            self._csv_file.flush()

    def reached_budget(self) -> bool:
        if not self.enabled:
            return False
        reached_steps = self.benchmark_steps > 0 and self.benchmark_count >= self.benchmark_steps
        reached_time = False
        if self.max_duration_seconds is not None and self._benchmark_start_time is not None:
            elapsed = time.perf_counter() - self._benchmark_start_time
            reached_time = elapsed >= self.max_duration_seconds
        return reached_steps or reached_time

    def should_stop_training(self) -> bool:
        return self.stop_after_benchmark and self.reached_budget()

    def summary_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for name, values in self._benchmark_values.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "metric": name,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p95": float(np.percentile(arr, 95)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            )
        return rows

    def write_summary_markdown(self, summary_path: Path) -> None:
        if not self.enabled:
            return
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.summary_rows()
        lines = [
            "# Step Timing Summary",
            "",
            f"- run_label: `{self.run_label}`",
            f"- warmup_steps: {self.warmup_steps}",
            f"- benchmark_steps_recorded: {self.benchmark_count}",
            f"- benchmark_steps_target: {self.benchmark_steps}",
            f"- benchmark_max_duration_minutes: {None if self.max_duration_seconds is None else self.max_duration_seconds / 60.0}",
            "",
        ]
        if not rows:
            lines.append("No benchmark rows were recorded.")
        else:
            lines.append(
                "| metric | mean (s) | std (s) | p50 (s) | p90 (s) | p95 (s) | min (s) | max (s) |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for row in rows:
                lines.append(
                    "| {metric} | {mean:.6f} | {std:.6f} | {p50:.6f} | {p90:.6f} | "
                    "{p95:.6f} | {min:.6f} | {max:.6f} |".format(**row)
                )
        summary_path.write_text("\n".join(lines) + "\n")

    def close(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()


def _resolve_step_timing_recorder(args: DotMap, checkpoint_dir: Path) -> _StepTimingRecorder:
    benchmark_cfg = args.get("benchmark", None) if hasattr(args, "get") else None
    if benchmark_cfg is None:
        return _StepTimingRecorder(
            enabled=False,
            csv_path=checkpoint_dir / "step_timing.csv",
            warmup_steps=0,
            benchmark_steps=0,
            max_duration_minutes=None,
            run_label="",
            sync_cuda=False,
            stop_after_benchmark=False,
        )

    enabled = bool(benchmark_cfg.get("enabled", False))
    warmup_steps = int(benchmark_cfg.get("warmup_steps", 20))
    benchmark_steps = int(benchmark_cfg.get("benchmark_steps", 0))
    max_duration_minutes = _coerce_none(benchmark_cfg.get("max_duration_minutes", None))
    run_label = str(benchmark_cfg.get("run_label", getattr(args, "experiment_name", "")))
    sync_cuda = bool(benchmark_cfg.get("sync_cuda", True))
    stop_after_benchmark = bool(benchmark_cfg.get("stop_after_benchmark", True))
    csv_path_value = _coerce_none(benchmark_cfg.get("csv_path", None))
    if csv_path_value is None:
        csv_path = checkpoint_dir / "step_timing.csv"
    else:
        csv_path = Path(csv_path_value)

    return _StepTimingRecorder(
        enabled=enabled,
        csv_path=csv_path,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        max_duration_minutes=max_duration_minutes,
        run_label=run_label,
        sync_cuda=sync_cuda,
        stop_after_benchmark=stop_after_benchmark,
    )


def _normalize_dataset_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.lower() == "all":
            return ["all"]
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _resolve_eval_dtype(
    value: Any, default: torch.dtype = torch.float32
) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if lowered in mapping:
            return mapping[lowered]
    return default


# STEP 10: Optional pristine vs. distorted example dumping for paper figures.
def _tensor_to_uint8_rgb(
    img: torch.Tensor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None
) -> np.ndarray:
    """
    img: torch tensor CxHxW on CPU or GPU, float
    mean/std: optional sequences of length 3 for de-normalization
    returns HxWx3 uint8 RGB
    """
    img = img.detach().to(device="cpu", dtype=torch.float32)
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.dim() == 3 and img.shape[0] not in (1, 3) and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if mean is not None and std is not None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        img = img * std_t + mean_t
    img = torch.clamp(img, 0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    img = np.rint(img * 255.0).astype(np.uint8)
    return img


_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_tag(value: str, max_len: int = 120) -> str:
    if not value:
        return "unknown"
    cleaned = _TAG_SAFE_RE.sub("-", str(value).strip())
    cleaned = cleaned.strip("._-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._-")
    return cleaned


def _normalize_meta_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize_meta_value(item) for item in value]
    return value


def _get_batch_item(value: Any, idx: int) -> Any:
    if torch.is_tensor(value):
        if value.dim() > 0:
            return value[idx]
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > idx:
            return value[idx]
        return value
    return value


def _extract_engine_example_metadata(
    batch: Dict[str, Any],
    example_idx: int,
    ref_view_idx: int,
    comp_idx: int,
    level_idx: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    if "imgs_names" in batch:
        ref_names = _get_batch_item(batch["imgs_names"], example_idx)
        if isinstance(ref_names, (list, tuple)):
            if 0 <= ref_view_idx < len(ref_names):
                meta["ref_name"] = ref_names[ref_view_idx]
        elif isinstance(ref_names, str):
            meta["ref_name"] = ref_names

    dist_comps = batch.get("dist_comps", None)
    if isinstance(dist_comps, dict):
        indices = dist_comps.get("indices", None)
        vals = dist_comps.get("vals", None)
        n_dist = dist_comps.get("n_dist", None)
        var_dist = dist_comps.get("var_dist", None)

        if torch.is_tensor(indices) and indices.dim() >= 4:
            meta["dist_comp_indices"] = _normalize_meta_value(
                indices[example_idx, comp_idx, level_idx]
            )
        if torch.is_tensor(vals) and vals.dim() >= 4:
            meta["dist_comp_vals"] = _normalize_meta_value(
                vals[example_idx, comp_idx, level_idx]
            )
        if torch.is_tensor(n_dist) and n_dist.dim() >= 2:
            meta["num_distortions"] = int(n_dist[example_idx, comp_idx].item())
        if torch.is_tensor(var_dist) and var_dist.dim() >= 2:
            meta["variable_dist"] = int(var_dist[example_idx, comp_idx].item())

    for key in (
        "distortion",
        "distortion_name",
        "distortion_type",
        "dist_type",
        "dist_group",
        "severity",
        "dist_level",
        "distortion_functions",
        "distortion_values",
        "distortion_params",
    ):
        if key in batch:
            value = _normalize_meta_value(_get_batch_item(batch[key], example_idx))
            meta[key] = value

    return meta


def _build_distortion_tag(meta: Dict[str, Any], comp_idx: int, level_idx: int) -> str:
    parts: List[str] = []
    for key in ("distortion_name", "distortion", "dist_type", "dist_group"):
        value = meta.get(key, None)
        if value:
            parts.append(str(value))

    dist_funcs = meta.get("distortion_functions", None)
    if dist_funcs:
        if isinstance(dist_funcs, (list, tuple)):
            funcs = [str(val) for val in dist_funcs if val not in ("", None)]
            if funcs:
                parts.append("+".join(funcs))
        else:
            parts.append(str(dist_funcs))

    if meta.get("severity", None) is not None:
        parts.append(f"sev{meta['severity']}")
    if meta.get("dist_level", None) is not None:
        parts.append(f"lvl{meta['dist_level']}")
    if meta.get("num_distortions", None) is not None:
        parts.append(f"nd{meta['num_distortions']}")
    if meta.get("variable_dist", None) is not None:
        parts.append(f"var{meta['variable_dist']}")

    if not parts:
        parts.append("unknown")
    parts.append(f"c{comp_idx}")
    parts.append(f"l{level_idx}")
    return _sanitize_tag("_".join(parts))


def _select_engine_example_tensors(
    batch: Dict[str, Any], example_idx: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int, int]]:
    ref_imgs = batch.get("ref_imgs", None)
    dist_imgs = batch.get("dist_imgs", None)
    if not torch.is_tensor(ref_imgs) or not torch.is_tensor(dist_imgs):
        return None

    ref_view_idx = 0
    comp_idx = 0
    level_idx = 0

    if ref_imgs.dim() == 5:
        ref_img = ref_imgs[example_idx, ref_view_idx]
    elif ref_imgs.dim() == 4:
        ref_img = ref_imgs[example_idx]
    else:
        return None

    if dist_imgs.dim() == 7:
        dist_img = dist_imgs[example_idx, ref_view_idx, comp_idx, level_idx]
    elif dist_imgs.dim() == 5:
        dist_img = dist_imgs[example_idx, 0]
    elif dist_imgs.dim() == 4:
        dist_img = dist_imgs[example_idx]
    else:
        return None

    return ref_img, dist_img, ref_view_idx, comp_idx, level_idx


class _EngineExamplesSaver:
    def __init__(self, args: DotMap, logger: Optional[Run], device: torch.device):
        self.is_rank0 = _is_rank0()
        self.is_distributed = _is_distributed()
        viz_cfg = args.viz
        self.max_batches = max(0, int(viz_cfg.save_engine_examples_max_batches))
        self.per_batch = max(0, int(viz_cfg.save_engine_examples_per_batch))
        self.early_stop = bool(viz_cfg.early_stop_after_examples)
        self.save_pair_concat = bool(
            viz_cfg.get("save_engine_examples_save_pair_concat", True)
        )
        self.write_jsonl = bool(viz_cfg.get("save_engine_examples_write_jsonl", True))
        self.device = device
        self.batches_saved = 0
        self.done = False
        self.experiment_name = getattr(args, "experiment_name", None)
        self.run_id = None
        if logger is not None and getattr(logger, "id", None):
            self.run_id = logger.id
        else:
            logging_cfg = args.get("logging", None)
            if logging_cfg and "wandb" in logging_cfg:
                self.run_id = logging_cfg.wandb.get("run_id", None)

        output_dir = Path(viz_cfg.save_engine_examples_output_dir)
        if not output_dir.is_absolute():
            output_dir = args.checkpoint_base_path / args.experiment_name / output_dir
        self.output_dir = output_dir
        self.jsonl_path = self.output_dir / "engine_examples.jsonl"

        if self.is_rank0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(
                "STEP 10: saving engine examples to "
                f"{self.output_dir} (max_batches={self.max_batches}, "
                f"per_batch={self.per_batch}, pair_concat={self.save_pair_concat}, "
                f"jsonl={self.write_jsonl})"
            )

    def _maybe_sync_stop(self, local_stop: bool) -> bool:
        if not self.early_stop or not self.is_distributed:
            return local_stop
        stop_tensor = torch.tensor(
            1 if local_stop else 0, device=self.device, dtype=torch.int32
        )
        torch.distributed.all_reduce(
            stop_tensor, op=torch.distributed.ReduceOp.MAX
        )
        return bool(stop_tensor.item())

    def maybe_save(
        self,
        batch: Dict[str, Any],
        epoch: int,
        batch_idx: int,
        global_step: int,
    ) -> bool:
        if self.max_batches <= 0 or self.per_batch <= 0:
            return self._maybe_sync_stop(False)

        if self.done or self.batches_saved >= self.max_batches:
            self.done = True
            return self._maybe_sync_stop(self.early_stop)

        if not self.is_rank0:
            return self._maybe_sync_stop(False)

        ref_imgs = batch.get("ref_imgs", None)
        if not torch.is_tensor(ref_imgs):
            return self._maybe_sync_stop(False)

        batch_size = int(ref_imgs.shape[0])
        num_examples = min(self.per_batch, batch_size)
        saved_examples = 0

        for example_idx in range(num_examples):
            selection = _select_engine_example_tensors(batch, example_idx)
            if selection is None:
                continue
            ref_img, dist_img, ref_view_idx, comp_idx, level_idx = selection
            meta = _extract_engine_example_metadata(
                batch, example_idx, ref_view_idx, comp_idx, level_idx
            )
            meta_tag = _build_distortion_tag(meta, comp_idx, level_idx)

            sample_id = f"step{global_step:07d}_b{batch_idx:04d}_i{example_idx:03d}"
            ref_path = self.output_dir / f"{sample_id}_ref.png"
            dist_path = self.output_dir / f"{sample_id}_dist_{meta_tag}.png"
            pair_path = (
                self.output_dir / f"{sample_id}_pair_{meta_tag}.png"
                if self.save_pair_concat
                else None
            )

            ref_uint8 = _tensor_to_uint8_rgb(ref_img, IMAGENET_MEAN, IMAGENET_STD)
            dist_uint8 = _tensor_to_uint8_rgb(dist_img, IMAGENET_MEAN, IMAGENET_STD)

            Image.fromarray(ref_uint8, mode="RGB").save(ref_path)
            Image.fromarray(dist_uint8, mode="RGB").save(dist_path)

            if self.save_pair_concat and pair_path is not None:
                pair_img = np.concatenate([ref_uint8, dist_uint8], axis=1)
                Image.fromarray(pair_img, mode="RGB").save(pair_path)

            if self.write_jsonl:
                record = {
                    "sample_id": sample_id,
                    "experiment_name": self.experiment_name,
                    "run_id": self.run_id,
                    "epoch": int(epoch),
                    "batch_idx": int(batch_idx),
                    "global_step": int(global_step),
                    "example_index": int(example_idx),
                    "ref_view_idx": int(ref_view_idx),
                    "comp_idx": int(comp_idx),
                    "level_idx": int(level_idx),
                    "ref_path": str(ref_path),
                    "dist_path": str(dist_path),
                    "pair_path": str(pair_path) if pair_path else None,
                    "distortion_tag": meta_tag,
                }
                record.update(meta)
                try:
                    with open(self.jsonl_path, "a") as jsonl_file:
                        jsonl_file.write(json.dumps(record, default=str) + "\n")
                except Exception as exc:
                    print(f"Exception while writing engine_examples.jsonl: {exc}")

            saved_examples += 1

        print(
            f"Saved {saved_examples} engine examples for batch {batch_idx} "
            f"at step {global_step}."
        )

        self.batches_saved += 1
        if self.batches_saved >= self.max_batches:
            self.done = True
            print("Done saving engine examples.")

        return self._maybe_sync_stop(self.done and self.early_stop)


def _resolve_metrics_over_time_config(
    args: DotMap, steps_per_epoch: int
) -> Dict[str, Any]:
    enabled = bool(args.get("enable_metrics_over_time", False))
    interval = args.get("metrics_eval_interval_steps", None)
    if interval is None or int(interval) <= 0:
        interval = max(1, steps_per_epoch // 10)

    datasets = _normalize_dataset_list(args.get("metrics_eval_datasets", "all"))
    if datasets == ["all"]:
        if "validation" in args and "datasets" in args.validation:
            datasets = list(args.validation.datasets)
        elif "test" in args and "datasets" in args.test:
            datasets = list(args.test.datasets)
        else:
            datasets = []

    trials = args.get("metrics_eval_trials", None)
    if trials is None:
        if "validation" in args and "num_splits" in args.validation:
            trials = args.validation.num_splits
        elif "test" in args and "num_splits" in args.test:
            trials = args.test.num_splits
        else:
            trials = 20

    save_every = args.get("metrics_eval_save_every", 1)
    try:
        save_every = int(save_every)
    except (TypeError, ValueError):
        save_every = 1
    if save_every < 0:
        save_every = 0

    return {
        "enabled": enabled,
        "interval": int(interval),
        "datasets": [str(ds).lower() for ds in datasets],
        "trials": int(trials),
        "save_raw": bool(args.get("metrics_eval_save_raw_trials", False)),
        "compute_h": bool(args.get("metrics_eval_compute_H", True)),
        "compute_z": bool(args.get("metrics_eval_compute_Z", True)),
        "eval_dtype": _resolve_eval_dtype(
            args.get("metrics_eval_dtype", None), torch.float32
        ),
        "save_every": save_every,
    }


def _map_vicreg_metrics(
    vicreg_metrics: Dict[str, float], compute_h: bool, compute_z: bool
) -> Dict[str, float]:
    metric_map = {
        "corr": "corr",
        "std": "std",
        "nstd": "nstd",
        "cov_rank": "rank",
        "nrank": "nrank",
        "inv": "inv",
    }
    suffix_map = {"rep": "H", "emb": "Z"}
    mapped = {}
    for key, value in vicreg_metrics.items():
        if "_" not in key:
            continue
        metric_name, suffix = key.rsplit("_", 1)
        if metric_name not in metric_map:
            continue
        target_suffix = suffix_map.get(suffix)
        if target_suffix is None:
            continue
        if target_suffix == "H" and not compute_h:
            continue
        if target_suffix == "Z" and not compute_z:
            continue
        mapped[f"{metric_map[metric_name]}_{target_suffix}"] = float(value)
    return mapped


def _evaluate_over_time(
    args: DotMap,
    model: torch.nn.Module,
    device: torch.device,
    datasets: List[str],
    num_splits: int,
    epoch: int,
    global_step: int,
    compute_h: bool,
    compute_z: bool,
    save_raw: bool,
    eval_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    eval_args = copy.deepcopy(args)
    eval_args.alpha_cache_signature = (
        f"{getattr(args, 'experiment_name', 'runtime')}:metrics_over_time:{global_step}"
    )
    (
        srocc_all,
        plcc_all,
        _,
        _,
        _,
        vicreg_metrics_all,
        result_metadata,
    ) = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=datasets,
        num_splits=num_splits,
        phase="val",
        alpha=args.validation.alpha if "validation" in args else args.test.alpha,
        grid_search=False,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device,
        eval_type=args.get("eval_type", "scratch"),
        eval_dtype=eval_dtype,
        logger=None,
        args=eval_args,
    )

    snapshot: Dict[str, Any] = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "time": datetime.now().isoformat(),
        "datasets": {},
    }
    log_payload: Dict[str, float] = {
        "eval/global_step": float(global_step),
        "eval/epoch": float(epoch),
    }

    for dataset_key, srocc_dataset in srocc_all.items():
        dataset_label = result_metadata.get(dataset_key, {}).get(
            "display_name", dataset_key
        )
        downstream: Dict[str, Any] = {}
        diag: Dict[str, Any] = {}

        srocc_vals = srocc_dataset.get("global", [])
        srocc_mean, srocc_std = _safe_mean_std(srocc_vals)
        downstream["SRCC/mean"] = srocc_mean
        downstream["SRCC/std"] = srocc_std
        if save_raw:
            downstream["SRCC/raw"] = [float(v) for v in srocc_vals]

        plcc_vals = plcc_all.get(dataset_key, {}).get("global", [])
        plcc_mean, plcc_std = _safe_mean_std(plcc_vals)
        downstream["PLCC/mean"] = plcc_mean
        downstream["PLCC/std"] = plcc_std
        if save_raw:
            downstream["PLCC/raw"] = [float(v) for v in plcc_vals]

        diag = _map_vicreg_metrics(
            vicreg_metrics_all.get(dataset_key, {}), compute_h, compute_z
        )

        snapshot["datasets"][dataset_label] = {
            "downstream": downstream,
            "diag": diag,
        }

        for key, value in downstream.items():
            if key.endswith("/mean") or key.endswith("/std"):
                log_payload[f"eval/{key}/{dataset_label}"] = float(value)
        for key, value in diag.items():
            log_payload[f"eval/{key}/{dataset_label}"] = float(value)

    return snapshot, log_payload


def _resolve_hp_eval_fractions(hp_eval: DotMap) -> List[float]:
    if hp_eval is None:
        return []
    fractions = hp_eval.get("fractions", [])
    if fractions:
        resolved = [float(v) for v in fractions]
    else:
        schedule = str(hp_eval.get("schedule", "") or "")
        num_points = int(hp_eval.get("num_points", 5) or 5)
        resolved = None
        if schedule.startswith("linspace_"):
            parts = schedule.split("_")
            if len(parts) >= 3:
                try:
                    start = float(parts[1]) / 100.0
                    end = float(parts[2]) / 100.0
                    resolved = np.linspace(start, end, num_points).tolist()
                except ValueError:
                    resolved = None
        if resolved is None:
            resolved = np.linspace(0.2, 1.0, num_points).tolist()
    cleaned = []
    for value in resolved:
        if value is None:
            continue
        frac = float(value)
        if frac > 1.0 and frac <= 100.0:
            frac = frac / 100.0
        if frac > 0:
            cleaned.append(frac)
    return sorted(set(cleaned))


def _resolve_hp_eval_steps(total_steps: int, fractions: List[float]) -> List[int]:
    if total_steps <= 0 or not fractions:
        return []
    steps = []
    for frac in fractions:
        step = int(round(frac * total_steps))
        step = max(1, min(total_steps, step))
        steps.append(step)
    return sorted(set(steps))


def _resolve_hp_eval_datasets(args: DotMap, hp_eval: DotMap) -> List[str]:
    hp_datasets = hp_eval.get("datasets", [])
    if isinstance(hp_datasets, str):
        hp_datasets = _normalize_dataset_list(hp_datasets)
    elif isinstance(hp_datasets, (list, tuple)):
        if hp_datasets and all(isinstance(item, str) for item in hp_datasets):
            if all(len(item) == 1 for item in hp_datasets):
                hp_datasets = _normalize_dataset_list("".join(hp_datasets))
            elif len(hp_datasets) == 1:
                hp_datasets = _normalize_dataset_list(hp_datasets[0])
    if hp_datasets:
        return list(hp_datasets)
    val_datasets = getattr(getattr(args, "validation", None), "datasets", [])
    if val_datasets:
        return list(val_datasets)
    return list(getattr(args.test, "datasets", []))


def _coerce_hp_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null", ""}:
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _extract_last_metrics(hp_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    last_metrics: Dict[str, Dict[str, float]] = {}
    for dataset, metrics in hp_state.get("metrics", {}).items():
        dataset_metrics: Dict[str, float] = {}
        for metric_name, values in metrics.items():
            if values:
                dataset_metrics[metric_name] = float(values[-1])
        if dataset_metrics:
            last_metrics[dataset] = dataset_metrics
    return last_metrics


def _update_hp_eval_state(
    hp_state: Dict[str, Any],
    eval_point: int,
    metrics_by_dataset: Dict[str, Dict[str, float]],
) -> None:
    hp_state["eval_points"].append(int(eval_point))
    eval_idx = len(hp_state["eval_points"]) - 1

    existing_datasets = set(hp_state["metrics"].keys())
    for dataset in existing_datasets - set(metrics_by_dataset.keys()):
        for _, values in hp_state["metrics"][dataset].items():
            values.append(float("nan"))

    for dataset, metrics in metrics_by_dataset.items():
        dataset_store = hp_state["metrics"].setdefault(dataset, {})
        for _, values in dataset_store.items():
            values.append(float("nan"))
        for metric_name, value in metrics.items():
            if metric_name not in dataset_store:
                dataset_store[metric_name] = [float("nan")] * eval_idx
                dataset_store[metric_name].append(float(value))
            else:
                dataset_store[metric_name][-1] = float(value)


def _log_hp_eval_metrics(
    logger: Optional[Run],
    hp_eval: DotMap,
    metrics_by_dataset: Dict[str, Dict[str, float]],
    step: int,
    final_only: bool = False,
) -> None:
    if not logger:
        return
    prefix = "hp_final" if final_only else "hp"
    payload = {}
    hp_name = hp_eval.get("hp_name", "")
    hp_value = hp_eval.get("hp_value", None)
    for dataset, metrics in metrics_by_dataset.items():
        for metric_name, value in metrics.items():
            key = f"{prefix}/{hp_name}/{hp_value}/{dataset}/{metric_name}"
            payload[key] = float(value)
    if payload:
        logger.log(payload, step=step)


def _compute_hp_eval_metrics(
    args: DotMap,
    model: torch.nn.Module,
    device: torch.device,
    datasets: List[str],
    num_splits: int,
    grid_search: bool,
    alpha: float,
    fast_mode: bool,
    fast_mode_strategy: Optional[str] = None,
    fast_mode_budget: Optional[float] = None,
    fast_mode_feature_ratio: Optional[float] = None,
    fast_mode_alpha_points: Optional[int] = None,
    fast_mode_reduce_eval_splits: Optional[bool] = None,
    fast_mode_alpha_cache_enabled: Optional[bool] = None,
    fast_mode_cache_fallback_strategy: Optional[str] = None,
    alpha_cache_signature: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    eval_args = copy.deepcopy(args)
    if hasattr(eval_args, "test"):
        eval_args.test.fast_mode = fast_mode
        eval_args.test.num_splits = num_splits
        eval_args.test.grid_search = grid_search
        eval_args.test.alpha = alpha
        if fast_mode_strategy is not None:
            eval_args.test.fast_mode_strategy = fast_mode_strategy
        if fast_mode_budget is not None:
            eval_args.test.fast_mode_budget = fast_mode_budget
        if fast_mode_feature_ratio is not None:
            eval_args.test.fast_mode_feature_ratio = fast_mode_feature_ratio
        if fast_mode_alpha_points is not None:
            eval_args.test.fast_mode_alpha_points = fast_mode_alpha_points
        if fast_mode_reduce_eval_splits is not None:
            eval_args.test.fast_mode_reduce_eval_splits = fast_mode_reduce_eval_splits
        if fast_mode_alpha_cache_enabled is not None:
            eval_args.test.fast_mode_alpha_cache_enabled = fast_mode_alpha_cache_enabled
        if fast_mode_cache_fallback_strategy is not None:
            eval_args.test.fast_mode_cache_fallback_strategy = (
                fast_mode_cache_fallback_strategy
            )
    if alpha_cache_signature is not None:
        eval_args.alpha_cache_signature = alpha_cache_signature

    was_training = model.training
    model.eval()
    with torch.no_grad():
        (
            srocc_all,
            plcc_all,
            _,
            _,
            _,
            vicreg_metrics_all,
            _,
        ) = get_results(
            model=model,
            data_base_path=args.data_base_path,
            datasets=datasets,
            num_splits=num_splits,
            phase="test",
            alpha=alpha,
            grid_search=grid_search,
            crop_size=args.test.crop_size,
            batch_size=args.test.batch_size,
            num_workers=args.test.num_workers,
            device=device,
            eval_type=args.get("eval_type", "scratch"),
            logger=None,
            num_logging_steps=None,
            args=eval_args,
        )
    if was_training:
        model.train()

    metrics_by_dataset: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        dataset_metrics: Dict[str, float] = {}
        if dataset in srocc_all:
            dataset_metrics["SRCC"] = float(np.median(srocc_all[dataset]["global"]))
        if dataset in plcc_all:
            dataset_metrics["PLCC"] = float(np.median(plcc_all[dataset]["global"]))
        vicreg_metrics = vicreg_metrics_all.get(dataset, {})
        for metric_name, value in vicreg_metrics.items():
            dataset_metrics[metric_name] = float(value)
        if dataset_metrics:
            metrics_by_dataset[dataset] = dataset_metrics
    return metrics_by_dataset

def train(
    args: DotMap,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    logger: Optional[Run],
    device: torch.device,
) -> None:
    """
    Train the given model with the strategy proposed in the paper https://arxiv.org/abs/2310.14918.

    Args:
        args (dotmap.DotMap): the training arguments
        model (torch.nn.Module): the model to train
        train_dataloader (torch.utils.data.Dataloader): the training data loader
        optimizer (torch.optim.Optimizer): the optimizer to use
        lr_scheduler (Optional[torch.optim.lr_scheduler]): the learning rate scheduler to use
        scaler (torch.cuda.amp.GradScaler): the scaler to use for mixed precision training
        logger (Optional[wandb.wandb_run.Run]): the logger to use
        device (torch.device): the device to use for training
    """

    # the following two functions create tensor from batch["dist_comps"] on gpu in parallel
    # def convert_to_tensor_on_gpu(lst):
    #     return torch.tensor(lst).to(device=device, non_blocking=True)

    # def dist_comps_conversion(batch):
    #     with ThreadPoolExecutor() as executor:
    #         futures = []
    #         for j in range(len(batch["dist_comps"])):
    #             for k in range(len(batch["dist_comps"][j])):
    #                 for l in range(1, len(batch["dist_comps"][j][k])):
    #                     # Submit the conversion task to the executor
    #                     futures.append(
    #                         executor.submit(
    #                             lambda idx_j=j, idx_k=k, idx_l=l: convert_to_tensor_on_gpu(
    #                                 batch["dist_comps"][idx_j][idx_k][idx_l]
    #                             )
    #                         )
    #                     )

    #         # Wait for all futures to complete
    #         for idx_j in range(len(batch["dist_comps"])):
    #             for idx_k in range(len(batch["dist_comps"][idx_j])):
    #                 for idx_l in range(
    #                     1, len(batch["dist_comps"][idx_j][idx_k])
    #                 ):
    #                     # Get the result from the future and update the batch in-place
    #                     batch["dist_comps"][idx_j][idx_k][idx_l] = futures.pop(
    #                         0
    #                     ).result()

    checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
    try:
        checkpoint_path.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        print(f"Exception: {e}")

    print("Saving checkpoints in folder: ", checkpoint_path)
    with open(
        args.checkpoint_base_path / args.experiment_name / "config.yaml", "w"
    ) as f:
        dumpable_args = args.copy()
        for (
            key,
            value,
        ) in dumpable_args.items():  # Convert PosixPath values of args to string
            if isinstance(value, Path):
                dumpable_args[key] = str(value)
        yaml.dump(dumpable_args.toDict(), f)

    # Initialize training parameters
    if args.training.resume_training:
        start_epoch = args.training.start_epoch
        max_epochs = args.training.epochs
        best_srocc = args.best_srocc
    else:
        start_epoch = 0
        max_epochs = args.training.epochs
        best_srocc = 0

    last_srocc = 0
    last_plcc = 0
    last_model_filename = ""
    best_model_filename = ""
    skip_validation_phase = bool(getattr(args.validation, "skip_phase", False))

    # For caching purposes
    if (
        args.training.data.cache_mode == "save"
        and args.training.data.cache_save_epoch is not None
    ):
        start_epoch = int(args.training.data.cache_save_epoch)
        max_epochs = start_epoch + 1

    steps_per_epoch = len(train_dataloader)
    metrics_cfg = _resolve_metrics_over_time_config(args, steps_per_epoch)
    metrics_snapshots: List[Dict[str, Any]] = []
    global_step = start_epoch * steps_per_epoch
    max_steps = args.training.get("max_steps", None)
    stop_training = False
    metrics_output_path = (
        args.checkpoint_base_path / args.experiment_name / "metrics_over_time.pt"
    )
    metrics_json_path = (
        args.checkpoint_base_path / args.experiment_name / "metrics_over_time.json"
    )
    total_train_steps = max_epochs * steps_per_epoch if steps_per_epoch else 0
    cache_save_max_batches = args.training.get("cache_save_max_batches", None)
    if cache_save_max_batches is not None:
        cache_save_max_batches = int(cache_save_max_batches)
        if cache_save_max_batches <= 0:
            cache_save_max_batches = None
    hp_eval_cfg = args.training.get("hp_eval", None) if hasattr(args.training, "get") else None
    hp_eval_enabled = bool(hp_eval_cfg.get("enabled", False)) if hp_eval_cfg else False
    hp_eval_steps: List[int] = []
    hp_eval_state: Optional[Dict[str, Any]] = None
    hp_eval_datasets: List[str] = []
    hp_eval_num_splits: Optional[int] = None
    hp_eval_fast_mode: Optional[bool] = None
    hp_eval_grid_search: Optional[bool] = None
    hp_eval_alpha: Optional[float] = None
    hp_eval_fast_mode_strategy: Optional[str] = None
    hp_eval_fast_mode_budget: Optional[float] = None
    hp_eval_fast_mode_feature_ratio: Optional[float] = None
    hp_eval_fast_mode_alpha_points: Optional[int] = None
    hp_eval_fast_mode_reduce_eval_splits: Optional[bool] = None
    hp_eval_fast_mode_alpha_cache_enabled: Optional[bool] = None
    hp_eval_fast_mode_cache_fallback_strategy: Optional[str] = None
    hp_eval_log_wandb = False
    hp_eval_triggered_steps: set = set()
    hp_eval_output_path = (
        args.checkpoint_base_path / args.experiment_name / "hp_metrics_over_time.pt"
    )
    hp_eval_last_path = (
        args.checkpoint_base_path / args.experiment_name / "hp_metrics_last.json"
    )

    def _coerce_none(value):
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return value

    def _resolve_int(value, fallback):
        value = _coerce_none(value)
        if value is None:
            return fallback
        return int(value)

    def _resolve_float(value, fallback):
        value = _coerce_none(value)
        if value is None:
            return fallback
        return float(value)

    def _resolve_bool(value, fallback):
        value = _coerce_none(value)
        if value is None:
            return fallback
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        return bool(value)

    def _resolve_str(value, fallback):
        value = _coerce_none(value)
        if value is None:
            return fallback
        return str(value)

    if hp_eval_enabled:
        hp_eval_datasets = [
            str(ds).lower() for ds in _resolve_hp_eval_datasets(args, hp_eval_cfg)
        ]
        hp_eval_num_splits = _resolve_int(
            hp_eval_cfg.get("num_splits", None), args.test.num_splits
        )
        hp_eval_fast_mode = _resolve_bool(
            hp_eval_cfg.get("fast_mode", None),
            bool(getattr(args.test, "fast_mode", False)),
        )
        hp_eval_grid_search = _resolve_bool(
            hp_eval_cfg.get("grid_search", None),
            bool(getattr(args.test, "grid_search", False)),
        )
        hp_eval_alpha = _resolve_float(
            hp_eval_cfg.get("alpha", None), float(args.test.alpha)
        )
        hp_eval_fast_mode_strategy = _resolve_str(
            hp_eval_cfg.get("fast_mode_strategy", None),
            _coerce_none(args.test.get("fast_mode_strategy", None)),
        )
        hp_eval_fast_mode_budget = _resolve_float(
            hp_eval_cfg.get("fast_mode_budget", None),
            _resolve_float(args.test.get("fast_mode_budget", 0.4), 0.4),
        )
        hp_eval_fast_mode_feature_ratio = _resolve_float(
            hp_eval_cfg.get("fast_mode_feature_ratio", None),
            _resolve_float(args.test.get("fast_mode_feature_ratio", 0.35), 0.35),
        )
        hp_eval_fast_mode_alpha_points = _resolve_int(
            hp_eval_cfg.get("fast_mode_alpha_points", None),
            _resolve_int(args.test.get("fast_mode_alpha_points", 24), 24),
        )
        hp_eval_fast_mode_reduce_eval_splits = _resolve_bool(
            hp_eval_cfg.get("fast_mode_reduce_eval_splits", None),
            _resolve_bool(args.test.get("fast_mode_reduce_eval_splits", False), False),
        )
        hp_eval_fast_mode_alpha_cache_enabled = _resolve_bool(
            hp_eval_cfg.get("fast_mode_alpha_cache_enabled", None),
            _resolve_bool(args.test.get("fast_mode_alpha_cache_enabled", True), True),
        )
        hp_eval_fast_mode_cache_fallback_strategy = _resolve_str(
            hp_eval_cfg.get("fast_mode_cache_fallback_strategy", None),
            _coerce_none(args.test.get("fast_mode_cache_fallback_strategy", None)),
        )
        hp_eval_log_wandb = _resolve_bool(
            hp_eval_cfg.get("log_wandb", True), True
        )
        hp_eval_steps = _resolve_hp_eval_steps(
            total_train_steps, _resolve_hp_eval_fractions(hp_eval_cfg)
        )
        hp_eval_state = {
            "hp_name": str(hp_eval_cfg.get("hp_name", "")),
            "hp_value": _coerce_hp_value(hp_eval_cfg.get("hp_value", None)),
            "eval_points": [],
            "metrics": {},
        }

    grad_probe_cfg = (
        args.training.get("grad_probe", DotMap())
        if hasattr(args.training, "get")
        else DotMap()
    )
    grad_probe_enabled = bool(grad_probe_cfg.get("enabled", False))
    grad_probe_interval = int(grad_probe_cfg.get("interval", 0) or 0)
    grad_probe_include_unweighted = bool(
        grad_probe_cfg.get("include_unweighted", True)
    )
    grad_probe_coeff = {
        "var_loss": float(getattr(args.model.coeff, "var", 1.0)),
        "inv_loss": float(getattr(args.model.coeff, "inv", 1.0)),
        "cov_loss": float(getattr(args.model.coeff, "cov", 1.0)),
    }
    grad_probe_targets = (
        _collect_grad_probe_targets(model) if grad_probe_enabled else dict()
    )
    if grad_probe_enabled and grad_probe_interval <= 0:
        grad_probe_enabled = False
        print("Grad probe disabled because interval <= 0.")
    elif grad_probe_enabled:
        print(
            "Grad probe enabled:"
            f" interval={grad_probe_interval},"
            f" include_unweighted={grad_probe_include_unweighted},"
            f" targets={list(grad_probe_targets.keys())}"
        )
        if not grad_probe_targets:
            print("Grad probe found no trainable encoder/projector params.")
    engine_examples = None
    viz_cfg = args.get("viz", None)
    if viz_cfg and bool(viz_cfg.get("save_engine_examples_enabled", False)):
        engine_examples = _EngineExamplesSaver(args, logger, device)
    engine_examples_only = bool(
        viz_cfg.get("save_engine_examples_only", False)
    ) if viz_cfg else False
    step_timing = _resolve_step_timing_recorder(args, checkpoint_path)
    step_timing_summary_path: Optional[Path] = None
    benchmark_cfg = args.get("benchmark", None) if hasattr(args, "get") else None
    if step_timing.enabled and benchmark_cfg is not None:
        summary_path_value = _coerce_none(benchmark_cfg.get("summary_path", None))
        if summary_path_value is None:
            step_timing_summary_path = step_timing.csv_path.with_suffix(".md")
        else:
            step_timing_summary_path = Path(summary_path_value)
    num_logging_steps = 0
    train_log_every_n_steps = 1
    if hasattr(args, "logging") and hasattr(args.logging, "wandb"):
        train_log_every_n_steps = max(
            1,
            _resolve_int(
                args.logging.wandb.get("train_log_every_n_steps", 1),
                1,
            ),
        )
    fail_fast_cfg = (
        args.training.get("fail_fast", DotMap(_dynamic=True))
        if hasattr(args.training, "get")
        else DotMap(_dynamic=True)
    )
    fail_fast_enabled = _resolve_bool(fail_fast_cfg.get("enabled", False), False)
    fail_fast_cov_loss_max = _resolve_float(
        fail_fast_cfg.get("cov_loss_max", 1e6), 1e6
    )
    dataloader_recovery_cfg = (
        args.training.get("dataloader_recovery", DotMap(_dynamic=True))
        if hasattr(args.training, "get")
        else DotMap(_dynamic=True)
    )
    dataloader_recovery_enabled = _resolve_bool(
        dataloader_recovery_cfg.get("enabled", True), True
    )
    dataloader_recovery_max_restarts_per_epoch = max(
        0, _resolve_int(dataloader_recovery_cfg.get("max_restarts_per_epoch", 2), 2)
    )
    dataloader_recovery_error_markers = (
        "Shared memory manager connection has timed out",
        "DataLoader worker",
        "is killed by signal",
    )

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        if max_steps is not None and global_step >= int(max_steps):
            break
        train_dataloader.dataset.set_epoch(epoch)
        # if epoch == 1:
        #     break

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch [{epoch + 1}/{max_epochs}]",
            disable=_env_bool("SHAMISA_DISABLE_TQDM", False),
        )
        train_iterator = iter(train_dataloader)
        dataloader_restarts_this_epoch = 0

        for i in range(len(train_dataloader)):
            data_wait_start = time.perf_counter()
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            except RuntimeError as exc:
                err_msg = str(exc)
                recoverable = (
                    dataloader_recovery_enabled
                    and dataloader_restarts_this_epoch
                    < dataloader_recovery_max_restarts_per_epoch
                    and any(marker in err_msg for marker in dataloader_recovery_error_markers)
                )
                if recoverable:
                    dataloader_restarts_this_epoch += 1
                    if _is_rank0():
                        print(
                            "Recoverable DataLoader RuntimeError encountered; "
                            f"restarting iterator ({dataloader_restarts_this_epoch}/"
                            f"{dataloader_recovery_max_restarts_per_epoch}) at "
                            f"epoch={epoch} step={global_step}: {err_msg}"
                        )
                    shutdown_workers = getattr(train_iterator, "_shutdown_workers", None)
                    if callable(shutdown_workers):
                        try:
                            shutdown_workers()
                        except Exception:
                            pass
                    train_iterator = iter(train_dataloader)
                    continue
                raise
            data_wait_s = time.perf_counter() - data_wait_start

            # if i == 1:
            #     break
            # Keep wandb step monotonic and aligned with training progress.
            num_logging_steps = int(global_step)
            if train_dataloader.dataset.cache_mode == "save":
                progress_bar.update(1)
                if cache_save_max_batches is not None and (i + 1) >= cache_save_max_batches:
                    stop_training = True
                    break
                continue

            # Initialize inputs
            batch_prep_start = time.perf_counter()
            batch["dist_imgs"] = batch["dist_imgs"].to(device=device, non_blocking=True)
            batch["ref_imgs"] = batch["ref_imgs"].to(device=device, non_blocking=True)

            if engine_examples:
                should_stop = engine_examples.maybe_save(
                    batch=batch,
                    epoch=epoch,
                    batch_idx=i,
                    global_step=global_step,
                )
                if should_stop:
                    if _is_rank0():
                        print("Early stop after saving engine examples.")
                    if step_timing_summary_path is not None:
                        step_timing.write_summary_markdown(step_timing_summary_path)
                    step_timing.close()
                    progress_bar.close()
                    return
                if engine_examples_only:
                    global_step += 1
                    progress_bar.update(1)
                    continue

            # dist_comps_conversion(batch)  # tested to make sure it works
            for key in batch["dist_comps"].keys():
                batch["dist_comps"][key] = batch["dist_comps"][key].to(
                    device=device, non_blocking=True
                )
            batch_prep_s = time.perf_counter() - batch_prep_start

            if args.training.loop_break:
                stop_training = True
                break

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            if step_timing.should_sync_cuda(device):
                torch.cuda.synchronize(device)
            fwd_bwd_optim_start = time.perf_counter()
            with torch.cuda.amp.autocast():  # For fp16 training
                loss, loss_terms_dict = model(batch, batch_style_inp=True)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}: {loss}")

            if loss_terms_dict is None:
                loss_terms_dict = dict()
            if fail_fast_enabled:
                numeric_loss_terms: Dict[str, float] = {}
                for key, value in loss_terms_dict.items():
                    try:
                        if isinstance(value, torch.Tensor):
                            if value.numel() != 1:
                                continue
                            numeric_value = float(value.detach().item())
                        else:
                            numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    numeric_loss_terms[key] = numeric_value
                    if not math.isfinite(numeric_value):
                        raise RuntimeError(
                            f"Non-finite {key} detected at step {global_step}: {numeric_value}"
                        )
                cov_value = numeric_loss_terms.get("cov_loss", None)
                if cov_value is not None and cov_value > float(fail_fast_cov_loss_max):
                    raise RuntimeError(
                        "Fail-fast: cov_loss exceeded threshold "
                        f"({cov_value:.6g} > {float(fail_fast_cov_loss_max):.6g}) "
                        f"at step {global_step}."
                    )
            if (
                grad_probe_enabled
                and grad_probe_targets
                and ((global_step + 1) % grad_probe_interval == 0)
            ):
                grad_probe_metrics = _compute_grad_probe_metrics(
                    loss=loss,
                    loss_terms_dict=loss_terms_dict,
                    coeff=grad_probe_coeff,
                    probe_targets=grad_probe_targets,
                    include_unweighted=grad_probe_include_unweighted,
                )
                loss_terms_dict.update(grad_probe_metrics)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step_timing.should_sync_cuda(device):
                torch.cuda.synchronize(device)
            fwd_bwd_optim_s = time.perf_counter() - fwd_bwd_optim_start
            global_step += 1
            num_logging_steps = int(global_step)
            cur_loss = loss.item()
            total_step_s = data_wait_s + batch_prep_s + fwd_bwd_optim_s
            step_timing.record(
                global_step=global_step,
                epoch=epoch,
                batch_idx=i,
                loss=cur_loss,
                data_wait_s=data_wait_s,
                batch_prep_s=batch_prep_s,
                fwd_bwd_optim_s=fwd_bwd_optim_s,
                total_step_s=total_step_s,
            )

            if (
                lr_scheduler
                and lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts"
            ):
                lr_scheduler.step(epoch + i / len(train_dataloader))

            if step_timing.should_stop_training():
                stop_training = True
            if (
                not stop_training
                and hp_eval_enabled
                and hp_eval_state is not None
                and hp_eval_datasets
                and global_step in hp_eval_steps
                and global_step not in hp_eval_triggered_steps
            ):
                if _is_distributed():
                    torch.distributed.barrier()
                if _is_rank0():
                    metrics_by_dataset = _compute_hp_eval_metrics(
                        args=args,
                        model=model,
                        device=device,
                        datasets=hp_eval_datasets,
                        num_splits=hp_eval_num_splits,
                        grid_search=hp_eval_grid_search,
                        alpha=hp_eval_alpha,
                        fast_mode=hp_eval_fast_mode,
                        fast_mode_strategy=hp_eval_fast_mode_strategy,
                        fast_mode_budget=hp_eval_fast_mode_budget,
                        fast_mode_feature_ratio=hp_eval_fast_mode_feature_ratio,
                        fast_mode_alpha_points=hp_eval_fast_mode_alpha_points,
                        fast_mode_reduce_eval_splits=hp_eval_fast_mode_reduce_eval_splits,
                        fast_mode_alpha_cache_enabled=hp_eval_fast_mode_alpha_cache_enabled,
                        fast_mode_cache_fallback_strategy=hp_eval_fast_mode_cache_fallback_strategy,
                        alpha_cache_signature=(
                            f"{args.experiment_name}:hp_eval:{global_step}"
                        ),
                    )
                    _update_hp_eval_state(
                        hp_state=hp_eval_state,
                        eval_point=global_step,
                        metrics_by_dataset=metrics_by_dataset,
                    )
                    if logger and hp_eval_log_wandb and args.logging.use_wandb:
                        _log_hp_eval_metrics(
                            logger=logger,
                            hp_eval=hp_eval_cfg,
                            metrics_by_dataset=metrics_by_dataset,
                            step=global_step,
                            final_only=False,
                        )
                hp_eval_triggered_steps.add(global_step)
                if _is_distributed():
                    torch.distributed.barrier()

            if (
                not stop_training
                and metrics_cfg["enabled"]
                and metrics_cfg["datasets"]
                and global_step % metrics_cfg["interval"] == 0
            ):
                if _is_distributed():
                    torch.distributed.barrier()
                if _is_rank0():
                    was_training = model.training
                    model.eval()
                    try:
                        with torch.no_grad():
                            snapshot, log_payload = _evaluate_over_time(
                                args=args,
                                model=model,
                                device=device,
                                datasets=metrics_cfg["datasets"],
                                num_splits=metrics_cfg["trials"],
                                epoch=epoch,
                                global_step=global_step,
                                compute_h=metrics_cfg["compute_h"],
                                compute_z=metrics_cfg["compute_z"],
                                save_raw=metrics_cfg["save_raw"],
                                eval_dtype=metrics_cfg["eval_dtype"],
                            )
                        metrics_snapshots.append(snapshot)
                        if (
                            metrics_cfg["save_every"] > 0
                            and len(metrics_snapshots) % metrics_cfg["save_every"] == 0
                        ):
                            torch.save(metrics_snapshots, metrics_output_path)
                            try:
                                with open(metrics_json_path, "w") as json_file:
                                    json.dump(metrics_snapshots, json_file, indent=2)
                            except Exception as exc:
                                print(
                                    f"Exception while saving metrics_over_time.json: {exc}"
                                )
                        if logger:
                            logger.log(log_payload, step=global_step)
                    finally:
                        model.train(was_training)
                if _is_distributed():
                    torch.distributed.barrier()

            if max_steps is not None and global_step >= int(max_steps):
                stop_training = True

            running_loss += cur_loss
            progress_bar.set_postfix(
                loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc
            )

            # Logging
            if logger:
                if loss_terms_dict is None:
                    loss_terms_dict = dict()

                should_log_step = (
                    num_logging_steps <= 1
                    or stop_training
                    or (num_logging_steps % train_log_every_n_steps == 0)
                )
                if should_log_step:
                    logger.log(
                        {"loss": cur_loss, "lr": optimizer.param_groups[0]["lr"]}
                        | loss_terms_dict,
                        step=num_logging_steps,
                    )

                # Log images
                # if i % args.training.log_images_frequency == 0:
                #     log_input = []
                #     for j, (
                #         img_A,
                #         name_A,
                #         img_B,
                #         name_B,
                #         dist_funcs,
                #         dist_values,
                #     ) in enumerate(
                #         zip(
                #             inputs_A_orig,
                #             img_A_name,
                #             inputs_B_orig,
                #             img_B_name,
                #             distortion_functions,
                #             distortion_values,
                #         )
                #     ):
                #         caption = (
                #             "A_orig_"
                #             + name_A
                #             + "_"
                #             + "_".join(
                #                 [
                #                     f"{value:.2f}{dist}"
                #                     for dist, value in zip(dist_funcs, dist_values)
                #                 ]
                #             )
                #         )
                #         log_img = wandb.Image(torch.clip(img_A, 0, 1), caption=caption)
                #         log_input.append(log_img)
                #         caption = (
                #             "B_orig_"
                #             + name_B
                #             + "_"
                #             + "_".join(
                #                 [
                #                     f"{value:.2f}{dist}"
                #                     for dist, value in zip(dist_funcs, dist_values)
                #                 ]
                #             )
                #         )
                #         log_img = wandb.Image(torch.clip(img_B, 0, 1), caption=caption)
                #         log_input.append(log_img)
                #         caption = (
                #             "A_ds_"
                #             + name_A
                #             + "_"
                #             + "_".join(
                #                 [
                #                     f"{value:.2f}{dist}"
                #                     for dist, value in zip(dist_funcs, dist_values)
                #                 ]
                #             )
                #         )
                #         log_img = wandb.Image(
                #             torch.clip(inputs_A_ds[j], 0, 1), caption=caption
                #         )
                #         log_input.append(log_img)
                #         caption = (
                #             "B_ds_"
                #             + name_B
                #             + "_"
                #             + "_".join(
                #                 [
                #                     f"{value:.2f}{dist}"
                #                     for dist, value in zip(dist_funcs, dist_values)
                #                 ]
                #             )
                #         )
                #         log_img = wandb.Image(
                #             torch.clip(inputs_B_ds[j], 0, 1), caption=caption
                #         )
                #         log_input.append(log_img)
                #     logger.log({"input": log_input}, step=num_logging_steps
            progress_bar.update(1)
            if stop_training:
                break

        try:
            shutdown_workers = getattr(train_iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
        except Exception as exc:
            if _is_rank0():
                print(f"Non-fatal DataLoader shutdown exception: {exc}")
        finally:
            del train_iterator
        progress_bar.close()

        if train_dataloader.dataset.cache_mode == "save":
            continue

        if (
            lr_scheduler
            and lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts"
        ):
            lr_scheduler.step()

        stop_after_epoch = stop_training
        benchmark_early_stop = (
            stop_after_epoch
            and step_timing.enabled
            and step_timing.stop_after_benchmark
            and step_timing.reached_budget()
        )
        if benchmark_early_stop:
            if _is_rank0():
                print(
                    "Benchmark budget reached; skipping validation/checkpoints and stopping training."
                )
            break

        # Validation
        if epoch % args.validation.frequency == 0:
            if skip_validation_phase:
                print("Skipping validation phase (validation.skip_phase=true).")
            else:
                # Persist latest train state before entering validation so
                # progress survives timeouts/errors in eval.
                pre_val_snapshot_path = checkpoint_path / "pre_val_snapshot.pth"
                _atomic_torch_save(
                    _build_train_resume_payload(
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        args=args,
                        global_step=global_step,
                        tag="pre_validation",
                    ),
                    pre_val_snapshot_path,
                )
                print(f"Saved pre-validation snapshot: {pre_val_snapshot_path.name}")
                print("Starting validation...")

                last_srocc, last_plcc = validate(
                    args, model, logger, num_logging_steps, device
                )

                # Log embeddings visualizations
                if args.validation.visualize and logger:
                    kadid10k_val = KADID10KDataset(
                        args.data_base_path / "KADID10K", phase="val"
                    )
                    val_dataloader = DataLoader(
                        kadid10k_val,
                        batch_size=args.test.batch_size,
                        shuffle=False,
                        num_workers=args.test.num_workers,
                    )
                    try:
                        figures = visualize_tsne_umap_mos(
                            model,
                            val_dataloader,
                            tsne_args=args.validation.visualization.tsne,
                            umap_args=args.validation.visualization.umap,
                            device=device,
                        )
                        logger.log(figures, step=num_logging_steps)
                    except Exception as e:
                        print(f"Exception in Visualization: {e}")

                progress_bar.set_postfix(
                    loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc
                )

        # Save checkpoints
        print("Saving checkpoint")

        # Save best checkpoint weights
        if (last_srocc > best_srocc) or (not best_model_filename):
            best_srocc = last_srocc
            best_plcc = last_plcc
            # Save best metrics in arguments for resuming training
            args.best_srocc = best_srocc
            args.best_plcc = best_plcc
            if best_model_filename:
                os.remove(
                    checkpoint_path / best_model_filename
                )  # Remove previous best model
            best_model_filename = (
                f"best_epoch_{epoch}_srocc_{best_srocc:.3f}_plcc_{best_plcc:.3f}.pth"
            )
            _atomic_torch_save(model.state_dict(), checkpoint_path / best_model_filename)

        # Save last checkpoint
        if last_model_filename:
            os.remove(
                checkpoint_path / last_model_filename
            )  # Remove previous last model
        last_model_filename = (
            f"last_epoch_{epoch}_srocc_{last_srocc:.3f}_plcc_{last_plcc:.3f}.pth"
        )
        args.last_srocc = last_srocc
        args.global_step = int(global_step)
        args.training.last_finished_epoch = int(epoch)
        _atomic_torch_save(
            _build_train_resume_payload(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                args=args,
                global_step=global_step,
                tag="last_epoch",
            ),
            checkpoint_path / last_model_filename,
        )

        if stop_after_epoch:
            break

    if (
        metrics_cfg["enabled"]
        and metrics_snapshots
        and _is_rank0()
        and train_dataloader.dataset.cache_mode != "save"
    ):
        torch.save(metrics_snapshots, metrics_output_path)
        try:
            with open(metrics_json_path, "w") as json_file:
                json.dump(metrics_snapshots, json_file, indent=2)
        except Exception as exc:
            print(f"Exception while saving metrics_over_time.json: {exc}")

    if (
        hp_eval_enabled
        and hp_eval_state is not None
        and hp_eval_datasets
        and _is_rank0()
        and train_dataloader.dataset.cache_mode != "save"
    ):
        final_step = global_step
        run_final_eval = (
            not hp_eval_state["eval_points"]
            or hp_eval_state["eval_points"][-1] != final_step
        )
        if run_final_eval:
            metrics_by_dataset = _compute_hp_eval_metrics(
                args=args,
                model=model,
                device=device,
                datasets=hp_eval_datasets,
                num_splits=hp_eval_num_splits,
                grid_search=hp_eval_grid_search,
                alpha=hp_eval_alpha,
                fast_mode=hp_eval_fast_mode,
                fast_mode_strategy=hp_eval_fast_mode_strategy,
                fast_mode_budget=hp_eval_fast_mode_budget,
                fast_mode_feature_ratio=hp_eval_fast_mode_feature_ratio,
                fast_mode_alpha_points=hp_eval_fast_mode_alpha_points,
                fast_mode_reduce_eval_splits=hp_eval_fast_mode_reduce_eval_splits,
                fast_mode_alpha_cache_enabled=hp_eval_fast_mode_alpha_cache_enabled,
                fast_mode_cache_fallback_strategy=hp_eval_fast_mode_cache_fallback_strategy,
                alpha_cache_signature=(
                    f"{args.experiment_name}:hp_eval:{final_step}"
                ),
            )
            _update_hp_eval_state(
                hp_state=hp_eval_state,
                eval_point=final_step,
                metrics_by_dataset=metrics_by_dataset,
            )
            if logger and hp_eval_log_wandb and args.logging.use_wandb:
                _log_hp_eval_metrics(
                    logger=logger,
                    hp_eval=hp_eval_cfg,
                    metrics_by_dataset=metrics_by_dataset,
                    step=final_step,
                    final_only=False,
                )
                _log_hp_eval_metrics(
                    logger=logger,
                    hp_eval=hp_eval_cfg,
                    metrics_by_dataset=metrics_by_dataset,
                    step=final_step,
                    final_only=True,
                )
        else:
            if logger and hp_eval_log_wandb and args.logging.use_wandb:
                last_metrics = _extract_last_metrics(hp_eval_state)
                _log_hp_eval_metrics(
                    logger=logger,
                    hp_eval=hp_eval_cfg,
                    metrics_by_dataset=last_metrics,
                    step=final_step,
                    final_only=True,
                )

        hp_payload = {
            "hp_name": hp_eval_state.get("hp_name", ""),
            "hp_value": hp_eval_state.get("hp_value", None),
            "eval_points": hp_eval_state.get("eval_points", []),
            "metrics": hp_eval_state.get("metrics", {}),
        }
        torch.save(hp_payload, hp_eval_output_path)
        try:
            last_metrics = _extract_last_metrics(hp_eval_state)
            last_eval_point = None
            if hp_eval_state.get("eval_points"):
                last_eval_point = hp_eval_state["eval_points"][-1]
            hp_last_payload = {
                "hp_name": hp_payload["hp_name"],
                "hp_value": hp_payload["hp_value"],
                "eval_point": last_eval_point,
                "metrics": last_metrics,
            }
            with open(hp_eval_last_path, "w") as json_file:
                json.dump(hp_last_payload, json_file, indent=2)
        except Exception as exc:
            print(f"Exception while saving hp_metrics_last.json: {exc}")

    if step_timing_summary_path is not None:
        step_timing.write_summary_markdown(step_timing_summary_path)
    step_timing.close()

    print("Finished training")


def validate(
    args: DotMap,
    model: torch.nn.Module,
    logger: Optional[Run],
    num_logging_steps: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate the given model on the validation datasets.

    Args:
        args (dotmap.DotMap): the training arguments
        model (torch.nn.Module): the model to validate
        logger (Optional[wandb.wandb_run.Run]): the logger to use
        num_logging_steps (int): the number of logging steps
        device (torch.device): the device to use
    """
    model.eval()
    validation_grid_search = getattr(args.validation, "grid_search", False)
    if isinstance(validation_grid_search, str):
        lowered = validation_grid_search.strip().lower()
        if lowered in {"true", "false"}:
            validation_grid_search = lowered == "true"
    validation_grid_search = bool(validation_grid_search)
    eval_args = copy.deepcopy(args)
    eval_args.alpha_cache_signature = (
        f"{getattr(args, 'experiment_name', 'runtime')}:validation:{num_logging_steps}"
    )

    (
        srocc_all,
        plcc_all,
        _,
        _,
        _,
        vicreg_metrics_all,
        _,
    ) = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=args.validation.datasets,
        num_splits=args.validation.num_splits,
        phase="val",
        alpha=args.validation.alpha,
        grid_search=validation_grid_search,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device,
        logger=logger,
        num_logging_steps=num_logging_steps,
        args=eval_args,
    )

    # Compute the median for each list in srocc_all and plcc_all
    srocc_all_median = {
        key: np.median(value["global"]) for key, value in srocc_all.items()
    }
    plcc_all_median = {
        key: np.median(value["global"]) for key, value in plcc_all.items()
    }

    # Compute the synthetic and authentic averages
    srocc_synthetic_avg = np.mean(
        [
            srocc_all_median[key]
            for key in srocc_all_median.keys()
            if key in synthetic_datasets
        ]
    )
    plcc_synthetic_avg = np.mean(
        [
            plcc_all_median[key]
            for key in plcc_all_median.keys()
            if key in synthetic_datasets
        ]
    )
    srocc_authentic_avg = np.mean(
        [
            srocc_all_median[key]
            for key in srocc_all_median.keys()
            if key in authentic_datasets
        ]
    )
    plcc_authentic_avg = np.mean(
        [
            plcc_all_median[key]
            for key in plcc_all_median.keys()
            if key in authentic_datasets
        ]
    )

    # Compute the global average
    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    # Optional: report test-phase metrics computed during validation, using the
    # same per-split train-fit regressors as validation (no extra fitting).
    val_phase_test_srocc = {
        key: float(meta.get("test_srocc_median", float("nan")))
        for key, meta in result_metadata.items()
    }
    val_phase_test_plcc = {
        key: float(meta.get("test_plcc_median", float("nan")))
        for key, meta in result_metadata.items()
    }
    val_phase_test_srocc_values = [
        value for value in val_phase_test_srocc.values() if not np.isnan(value)
    ]
    val_phase_test_plcc_values = [
        value for value in val_phase_test_plcc.values() if not np.isnan(value)
    ]
    val_phase_test_srocc_synthetic_values = [
        val_phase_test_srocc[key]
        for key in val_phase_test_srocc.keys()
        if key in synthetic_datasets and not np.isnan(val_phase_test_srocc[key])
    ]
    val_phase_test_plcc_synthetic_values = [
        val_phase_test_plcc[key]
        for key in val_phase_test_plcc.keys()
        if key in synthetic_datasets and not np.isnan(val_phase_test_plcc[key])
    ]
    val_phase_test_srocc_authentic_values = [
        val_phase_test_srocc[key]
        for key in val_phase_test_srocc.keys()
        if key in authentic_datasets and not np.isnan(val_phase_test_srocc[key])
    ]
    val_phase_test_plcc_authentic_values = [
        val_phase_test_plcc[key]
        for key in val_phase_test_plcc.keys()
        if key in authentic_datasets and not np.isnan(val_phase_test_plcc[key])
    ]
    val_phase_test_srocc_avg = (
        float(np.mean(val_phase_test_srocc_values))
        if val_phase_test_srocc_values
        else float("nan")
    )
    val_phase_test_plcc_avg = (
        float(np.mean(val_phase_test_plcc_values))
        if val_phase_test_plcc_values
        else float("nan")
    )
    val_phase_test_srocc_synthetic_avg = (
        float(np.mean(val_phase_test_srocc_synthetic_values))
        if val_phase_test_srocc_synthetic_values
        else float("nan")
    )
    val_phase_test_plcc_synthetic_avg = (
        float(np.mean(val_phase_test_plcc_synthetic_values))
        if val_phase_test_plcc_synthetic_values
        else float("nan")
    )
    val_phase_test_srocc_authentic_avg = (
        float(np.mean(val_phase_test_srocc_authentic_values))
        if val_phase_test_srocc_authentic_values
        else float("nan")
    )
    val_phase_test_plcc_authentic_avg = (
        float(np.mean(val_phase_test_plcc_authentic_values))
        if val_phase_test_plcc_authentic_values
        else float("nan")
    )

    val_cross_srocc_avg = float("nan")
    val_cross_plcc_avg = float("nan")
    val_within_srocc_avg = float("nan")
    val_within_plcc_avg = float("nan")
    validation_cross_cfg = (
        args.validation.get("cross_eval", DotMap(_dynamic=True))
        if hasattr(args.validation, "get")
        else DotMap(_dynamic=True)
    )
    validation_cross_enabled = bool(validation_cross_cfg.get("enabled", False))
    if validation_cross_enabled:
        train_datasets = list(validation_cross_cfg.get("train_datasets", []))
        test_datasets = list(validation_cross_cfg.get("test_datasets", []))
        if train_datasets and test_datasets:
            val_cross_args = copy.deepcopy(eval_args)
            if "test" not in val_cross_args or val_cross_args.test is None:
                val_cross_args.test = DotMap(_dynamic=True)
            val_cross_args.test.cross_eval = DotMap(_dynamic=True)

            cross_grid_search = validation_cross_cfg.get(
                "grid_search", validation_grid_search
            )
            if isinstance(cross_grid_search, str):
                lowered = cross_grid_search.strip().lower()
                if lowered in {"true", "false"}:
                    cross_grid_search = lowered == "true"
            cross_grid_search = bool(cross_grid_search)

            cross_alpha = validation_cross_cfg.get("alpha", args.validation.alpha)
            if cross_alpha is None:
                cross_alpha = args.validation.alpha
            try:
                cross_alpha = float(cross_alpha)
            except (TypeError, ValueError):
                cross_alpha = float(args.validation.alpha)

            cross_override_keys = [
                "fast_mode",
                "fast_mode_strategy",
                "fast_mode_budget",
                "fast_mode_feature_ratio",
                "fast_mode_alpha_points",
                "fast_mode_alpha_min",
                "fast_mode_alpha_max",
                "fast_mode_reduce_eval_splits",
                "fast_mode_eval_budget",
                "fast_mode_alpha_cache_enabled",
                "fast_mode_cache_fallback_strategy",
                "fast_mode_alpha_cache_path",
                "fast_mode_n_jobs",
            ]
            for key in cross_override_keys:
                value = validation_cross_cfg.get(key, None)
                if value is not None:
                    val_cross_args.test.cross_eval[key] = value
                elif hasattr(args.validation, "get"):
                    fallback = args.validation.get(key, None)
                    if fallback is not None:
                        val_cross_args.test.cross_eval[key] = fallback

            val_cross_args.alpha_cache_signature = (
                f"{getattr(args, 'experiment_name', 'runtime')}:validation_cross:{num_logging_steps}"
            )

            (
                cross_srocc_all,
                cross_plcc_all,
                _,
                _,
                _,
                _,
                cross_result_metadata,
            ) = get_cross_results(
                model=model,
                data_base_path=args.data_base_path,
                train_datasets=train_datasets,
                test_datasets=test_datasets,
                num_splits=args.validation.num_splits,
                phase="val",
                alpha=cross_alpha,
                grid_search=cross_grid_search,
                crop_size=args.test.crop_size,
                batch_size=args.test.batch_size,
                num_workers=args.test.num_workers,
                device=device,
                eval_type=args.get("eval_type", "scratch"),
                args=val_cross_args,
                logger=logger,
            )

            cross_srocc_median = {
                key: np.median(value["global"])
                for key, value in cross_srocc_all.items()
            }
            cross_plcc_median = {
                key: np.median(value["global"])
                for key, value in cross_plcc_all.items()
            }
            cross_keys = [
                key
                for key, meta in cross_result_metadata.items()
                if meta["train_dataset"] != meta["test_dataset"]
            ]
            within_keys = [
                key
                for key, meta in cross_result_metadata.items()
                if meta["train_dataset"] == meta["test_dataset"]
            ]
            if cross_keys:
                val_cross_srocc_avg = float(
                    np.mean([cross_srocc_median[key] for key in cross_keys])
                )
                val_cross_plcc_avg = float(
                    np.mean([cross_plcc_median[key] for key in cross_keys])
                )
            if within_keys:
                val_within_srocc_avg = float(
                    np.mean([cross_srocc_median[key] for key in within_keys])
                )
                val_within_plcc_avg = float(
                    np.mean([cross_plcc_median[key] for key in within_keys])
                )

    if logger:
        if np.isfinite(val_cross_srocc_avg):
            # Equal weight for within-dataset average and cross-dataset average.
            val_srocc_blend = float(0.5 * srocc_avg + 0.5 * val_cross_srocc_avg)
        else:
            val_srocc_blend = float(srocc_avg)
        logger.log(
            {
                f"val_srocc_{key}": srocc_all_median[key]
                for key in srocc_all_median.keys()
            },
            step=num_logging_steps,
        )
        logger.log(
            {f"val_plcc_{key}": plcc_all_median[key] for key in plcc_all_median.keys()},
            step=num_logging_steps,
        )
        logger.log(
            {
                f"val_phase_test_srocc_{key}": val_phase_test_srocc[key]
                for key in val_phase_test_srocc.keys()
                if not np.isnan(val_phase_test_srocc[key])
            },
            step=num_logging_steps,
        )
        logger.log(
            {
                f"val_phase_test_plcc_{key}": val_phase_test_plcc[key]
                for key in val_phase_test_plcc.keys()
                if not np.isnan(val_phase_test_plcc[key])
            },
            step=num_logging_steps,
        )
        logger.log(
            {
                "val_srocc_synthetic_avg": srocc_synthetic_avg,
                "val_plcc_synthetic_avg": plcc_synthetic_avg,
                "val_srocc_authentic_avg": srocc_authentic_avg,
                "val_plcc_authentic_avg": plcc_authentic_avg,
                "val_srocc_avg": srocc_avg,
                "val_plcc_avg": plcc_avg,
                "val_phase_test_srocc_synthetic_avg": val_phase_test_srocc_synthetic_avg,
                "val_phase_test_plcc_synthetic_avg": val_phase_test_plcc_synthetic_avg,
                "val_phase_test_srocc_authentic_avg": val_phase_test_srocc_authentic_avg,
                "val_phase_test_plcc_authentic_avg": val_phase_test_plcc_authentic_avg,
                "val_phase_test_srocc_avg": val_phase_test_srocc_avg,
                "val_phase_test_plcc_avg": val_phase_test_plcc_avg,
                "val_cross_srocc_avg": val_cross_srocc_avg,
                "val_cross_plcc_avg": val_cross_plcc_avg,
                "val_within_srocc_avg": val_within_srocc_avg,
                "val_within_plcc_avg": val_within_plcc_avg,
                "val_srocc_blend": val_srocc_blend,
            },
            step=num_logging_steps,
        )
        # Also persist validation targets in summary so W&B sweeps can always read them.
        logger.summary["val_srocc_avg"] = float(srocc_avg)
        logger.summary["val_plcc_avg"] = float(plcc_avg)
        logger.summary["val_phase_test_srocc_synthetic_avg"] = float(
            val_phase_test_srocc_synthetic_avg
        )
        logger.summary["val_phase_test_plcc_synthetic_avg"] = float(
            val_phase_test_plcc_synthetic_avg
        )
        logger.summary["val_phase_test_srocc_authentic_avg"] = float(
            val_phase_test_srocc_authentic_avg
        )
        logger.summary["val_phase_test_plcc_authentic_avg"] = float(
            val_phase_test_plcc_authentic_avg
        )
        logger.summary["val_phase_test_srocc_avg"] = float(val_phase_test_srocc_avg)
        logger.summary["val_phase_test_plcc_avg"] = float(val_phase_test_plcc_avg)
        logger.summary["val_cross_srocc_avg"] = float(val_cross_srocc_avg)
        logger.summary["val_cross_plcc_avg"] = float(val_cross_plcc_avg)
        logger.summary["val_within_srocc_avg"] = float(val_within_srocc_avg)
        logger.summary["val_within_plcc_avg"] = float(val_within_plcc_avg)
        logger.summary["val_srocc_blend"] = float(val_srocc_blend)

    return srocc_avg, plcc_avg
