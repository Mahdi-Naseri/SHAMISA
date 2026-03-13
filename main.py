import json
import argparse
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import string
import subprocess
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
import yaml

from utils.torch_amp_compat import patch_cuda_amp_custom_autocast

patch_cuda_amp_custom_autocast(device_type="cuda")

from train import train
from test import test
from models.simclr import SimCLR
from models.vicreg import Vicreg
from data import KADIS700Dataset, KADIS700StructuredDataset, structured_kadis_collate
from models.associations import finalize_association_config
from utils.utils import (
    PROJECT_ROOT,
    parse_config,
    parse_command_line_args,
    merge_configs,
    replace_none_string,
    prepare_wandb_config,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_WORKER_CPU_THREADS = None


def _dataloader_worker_init_fn(_worker_id: int) -> None:
    global _WORKER_CPU_THREADS
    if _WORKER_CPU_THREADS is None:
        return
    threads = int(_WORKER_CPU_THREADS)
    if threads <= 0:
        return
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _atomic_torch_save(payload, target_path: Path) -> None:
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


def _find_resume_checkpoint(pretrain_dir: Path) -> Path:
    last_candidates = sorted(
        [path for path in pretrain_dir.glob("*.pth") if "last" in path.name],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if last_candidates:
        return last_candidates[0]

    for fallback_name in ("pre_val_snapshot.pth", "pre_test_train_state.pth"):
        fallback_path = pretrain_dir / fallback_name
        if fallback_path.exists():
            return fallback_path

    raise FileNotFoundError(f"No resume checkpoint found in {pretrain_dir}")


def _build_train_state_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    args,
    tag: str,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_tag": tag,
        "config": args,
    }
    try:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        payload["rng_state"] = rng_state
    except Exception:
        pass
    return payload


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_resource_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cannot allocate memory",
        "std::bad_alloc",
    )
    return any(marker in msg for marker in markers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args, unknown = parser.parse_known_args()
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)

    args = replace_none_string(args)

    if args.experiment_name in ["debug", "sweep"]:
        now = datetime.now()
        current_time_str = now.strftime("_%Y-%m-%d_%H-%M-%S")
        random_suffix = "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )
        args.experiment_name += current_time_str + "_" + random_suffix

    if args.training.resume_training == 1:
        args.training.resume_training = True
    args = finalize_association_config(args)

    print(args)

    if args.device != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = PROJECT_ROOT / "experiments"

    def _dump_config_fingerprint(cfg):
        output_dir = args.checkpoint_base_path / args.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_cfg = prepare_wandb_config(cfg)
        config_yaml = yaml.safe_dump(
            resolved_cfg, sort_keys=True, default_flow_style=False
        )
        config_hash = hashlib.sha1(config_yaml.encode("utf-8")).hexdigest()
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = "unknown"

        config_path = output_dir / "config_resolved.yaml"
        fingerprint_path = output_dir / "config_fingerprint.json"
        config_path.write_text(config_yaml)
        fingerprint_payload = {
            "config_sha1": config_hash,
            "git_commit": git_commit,
            "config_path": str(config_path),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        fingerprint_path.write_text(json.dumps(fingerprint_payload, indent=2))

        print("=== RESOLVED CONFIG ===")
        print(config_yaml)
        print(f"CONFIG_SHA1={config_hash}")
        print(f"GIT_COMMIT={git_commit}")
        print(f"CONFIG_DUMP_PATH={config_path}")
        print(f"FINGERPRINT_PATH={fingerprint_path}")

    _dump_config_fingerprint(args)

    if args.training.data.distortion_pipeline == "classic":
        train_dataset = KADIS700Dataset(
            root=args.data_base_path / "KADIS700",
            patch_size=args.training.data.patch_size,
            max_distortions=args.training.data.max_distortions,
            num_levels=args.training.data.num_levels,
            pristine_prob=args.training.data.pristine_prob,
        )
    elif args.training.data.distortion_pipeline == "shamisa":
        severity_discrete = bool(
            getattr(args.training.data, "severity_discrete", False)
        )
        severity_dist = getattr(args.training.data, "severity_dist", "gaussian")
        fixed_order = bool(getattr(args.training.data, "fixed_order", False))
        cache_load_mode = args.training.data.get("cache_load_mode", "mmap")
        if not isinstance(cache_load_mode, str) or not cache_load_mode.strip():
            cache_load_mode = "mmap"
        no_cache_opt_variant = args.training.data.get(
            "no_cache_opt_variant", "variant_b"
        )
        if not isinstance(no_cache_opt_variant, str) or not no_cache_opt_variant.strip():
            no_cache_opt_variant = "variant_b"
        train_dataset = KADIS700StructuredDataset(
            root=args.data_base_path / "KADIS700",
            patch_size=args.training.data.patch_size,
            max_distortions=args.training.data.max_distortions,
            num_levels=args.training.data.num_levels,
            n_refs=args.training.data.n_refs,
            n_dist_comps=args.training.data.n_dist_comps,
            n_dist_comp_levels=args.training.data.n_dist_comp_levels,
            extended_int_distortions=args.training.data.extended_int_distortions,
            severity_discrete=severity_discrete,
            severity_dist=severity_dist,
            fixed_order=fixed_order,
            cache_path=args.training.data.cache_path,
            cache_mode=args.training.data.cache_mode,
            cache_load_max_avail_epoch=args.training.data.cache_load_max_avail_epoch,
            cache_load_mode=cache_load_mode,
            no_cache_opt_variant=no_cache_opt_variant,
        )
    else:
        assert False

    pin_memory = bool(getattr(args.training, "pin_memory", True))
    persistent_workers = bool(
        getattr(args.training, "persistent_workers", args.training.num_workers > 0)
    )
    prefetch_factor = getattr(args.training, "prefetch_factor", None)
    pin_memory_device = getattr(args.training, "pin_memory_device", None)
    dataloader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        shuffle=args.training.data.shuffle,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=structured_kadis_collate,
    )
    worker_cpu_threads = getattr(args.training, "worker_cpu_threads", None)
    if worker_cpu_threads is not None:
        try:
            worker_cpu_threads = int(worker_cpu_threads)
        except (TypeError, ValueError):
            worker_cpu_threads = None
    if args.training.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
        if worker_cpu_threads is not None and worker_cpu_threads > 0:
            global _WORKER_CPU_THREADS
            _WORKER_CPU_THREADS = worker_cpu_threads
            dataloader_kwargs["worker_init_fn"] = _dataloader_worker_init_fn
    if pin_memory and pin_memory_device:
        dataloader_kwargs["pin_memory_device"] = str(pin_memory_device)

    train_dataloader = DataLoader(**dataloader_kwargs)

    def _resolve_projector_dim(cfg, key: str):
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "projector"):
            return None
        value = getattr(cfg.model.projector, key, None)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"none", "null", ""}:
                return None
            try:
                value = float(value)
            except ValueError:
                return None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    projector_out_dim = _resolve_projector_dim(args, "out_dim")
    projector_hidden_dim = _resolve_projector_dim(args, "hidden_dim")

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
        assert False
    model = model.to(device)

    if args.training.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.training.lr,
        )
    elif args.training.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.training.lr,
        )
    elif args.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.training.lr,
            momentum=args.training.optimizer.momentum,
            weight_decay=args.training.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {args.training.optimizer.name} not implemented"
        )

    if (
        "lr_scheduler" in args.training
        and args.training.lr_scheduler.name == "CosineAnnealingWarmRestarts"
    ):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.training.lr_scheduler.T_0,
            T_mult=args.training.lr_scheduler.T_mult,
            eta_min=args.training.lr_scheduler.eta_min,
            verbose=False,
        )
    else:
        lr_scheduler = None

    scaler = torch.cuda.amp.GradScaler()  # Automatic mixed precision scaler

    run_id = None
    wandb_config = None
    if args.training.resume_training:
        try:
            pretrain_dir = args.checkpoint_base_path / args.experiment_name / "pretrain"
            checkpoint_path = _find_resume_checkpoint(pretrain_dir)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scaler_state_dict", None) is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            scheduler_state = checkpoint.get("lr_scheduler_state_dict", None)
            if lr_scheduler is not None and scheduler_state is not None:
                try:
                    lr_scheduler.load_state_dict(scheduler_state)
                except Exception as exc:
                    print(f"Warning: could not restore lr_scheduler state: {exc}")
            epoch = checkpoint["epoch"]
            args.training.start_epoch = epoch + 1
            args.global_step = int(checkpoint.get("global_step", 0))
            config_from_ckpt = checkpoint.get("config", {})
            run_id = (
                config_from_ckpt.get("logging", {})
                .get("wandb", {})
                .get("run_id", None)
            )
            args.best_srocc = config_from_ckpt.get(
                "best_srocc", getattr(args, "best_srocc", -1.0)
            )
            args.best_plcc = config_from_ckpt.get(
                "best_plcc", getattr(args, "best_plcc", 0)
            )

            rng_state = checkpoint.get("rng_state", None)
            if isinstance(rng_state, dict):
                try:
                    if "python" in rng_state:
                        random.setstate(rng_state["python"])
                    if "numpy" in rng_state:
                        np.random.set_state(rng_state["numpy"])
                    if "torch" in rng_state:
                        torch.set_rng_state(rng_state["torch"])
                    if torch.cuda.is_available() and "torch_cuda" in rng_state:
                        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
                except Exception as exc:
                    print(f"Warning: could not fully restore RNG state: {exc}")
            print(
                f"--- Resuming training after epoch {epoch + 1} "
                f"(from {checkpoint_path.name}) ---"
            )
        except Exception:
            print("ERROR: Could not resume training. Starting from scratch.")
    wandb_config = prepare_wandb_config(args)

    if args.logging.use_wandb:
        if args.training.resume_training:
            resume_mode = "must" if run_id else "allow"
        else:
            resume_mode = "never"
        logger = wandb.init(
            project=args.logging.wandb.project,
            entity=args.logging.wandb.entity,
            name=args.experiment_name if not args.training.resume_training else None,
            config=wandb_config,
            mode="online" if args.logging.wandb.online else "offline",
            resume=resume_mode,
            id=run_id,
        )
        args.logging.wandb.run_id = logger.id
    else:
        logger = None

    try:
        train(
            args, model, train_dataloader, optimizer, lr_scheduler, scaler, logger, device
        )

        print("--- Training finished ---")
        if train_dataloader.dataset.cache_mode == "save":
            return

        if args.training.test:
            pretrain_dir = args.checkpoint_base_path / args.experiment_name / "pretrain"
            best_candidates = sorted(
                [ckpt_path for ckpt_path in pretrain_dir.glob("*.pth") if "best" in ckpt_path.name],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            checkpoint_path = best_candidates[0]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

            pre_test_epoch = int(
                getattr(
                    args.training,
                    "last_finished_epoch",
                    max(int(getattr(args.training, "epochs", 1)) - 1, 0),
                )
            )
            pre_test_global_step = int(getattr(args, "global_step", 0))
            pre_test_train_state_path = (
                args.checkpoint_base_path
                / args.experiment_name
                / "pretrain"
                / "pre_test_train_state.pth"
            )
            _atomic_torch_save(
                _build_train_state_payload(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=pre_test_epoch,
                    global_step=pre_test_global_step,
                    args=args,
                    tag="pre_test_train_state",
                ),
                pre_test_train_state_path,
            )
            print(f"Saved pre-test train-state snapshot: {pre_test_train_state_path.name}")
            model.load_state_dict(checkpoint, strict=True)
            model.to(device)
            print(f"Starting testing with best checkpoint...")

            test(args, model, logger, device)
            print("--- Testing finished ---")
    except RuntimeError as exc:
        if _is_resource_failure(exc):
            penalty_obj = _env_float("SHAMISA_SWEEP_RESOURCE_FAIL_OBJECTIVE", -1.0)
            print(
                "[resource-fail] Runtime resource failure detected; "
                f"logging penalty objective {penalty_obj}: {exc}"
            )
            if logger is not None:
                payload = {
                    "run_resource_failure": 1,
                    "run_resource_failure_kind": "runtime_resource_error",
                    "run_resource_failure_message": str(exc)[:512],
                    "val_srocc_blend": float(penalty_obj),
                    "val_srocc_avg": float(penalty_obj),
                    "val_cross_srocc_avg": float(penalty_obj),
                }
                for key, value in payload.items():
                    logger.summary[key] = value
                logger.log(payload)
            return
        raise
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
