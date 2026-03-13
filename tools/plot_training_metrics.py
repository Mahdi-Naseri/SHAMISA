#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt
from cycler import cycler


DIAG_METRICS = [
    "corr_H",
    "corr_Z",
    "inv_H",
    "inv_Z",
    "std_H",
    "std_Z",
    "nstd_H",
    "nstd_Z",
    "rank_H",
    "rank_Z",
    "nrank_H",
    "nrank_Z",
]


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_snapshots(metrics_file: Path) -> List[Dict]:
    if metrics_file.suffix in {".pt", ".pth", ".ckpt"}:
        data = torch.load(metrics_file, map_location="cpu")
    else:
        with metrics_file.open("r") as handle:
            data = json.load(handle)

    if isinstance(data, dict) and "snapshots" in data:
        snapshots = data["snapshots"]
    else:
        snapshots = data

    if not isinstance(snapshots, list):
        raise ValueError("metrics_file does not contain a list of snapshots.")
    return snapshots


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _collect_dataset_names(snapshots: List[Dict]) -> List[str]:
    names: List[str] = []
    for snapshot in snapshots:
        datasets = snapshot.get("datasets", {})
        for name in datasets.keys():
            if name not in names:
                names.append(name)
    return names


def _select_datasets(all_names: List[str], selection: str) -> List[str]:
    if not selection or selection.lower() == "all":
        return all_names
    requested = [item.strip() for item in selection.split(",") if item.strip()]
    name_map = {_normalize_name(name): name for name in all_names}
    selected: List[str] = []
    for item in requested:
        key = _normalize_name(item)
        if key in name_map:
            selected.append(name_map[key])
        elif item in all_names:
            selected.append(item)
    return selected


def _grid_for_n(n_items: int) -> Tuple[int, int]:
    cols = max(1, int(math.ceil(math.sqrt(n_items))))
    rows = max(1, int(math.ceil(n_items / cols)))
    return rows, cols


def _extract_downstream_series(
    snapshots: List[Dict], dataset: str, metric: str
) -> Tuple[List[int], List[float], List[Optional[float]]]:
    steps: List[int] = []
    values: List[float] = []
    stds: List[Optional[float]] = []
    mean_key = f"{metric}/mean"
    std_key = f"{metric}/std"

    for snapshot in snapshots:
        dataset_entry = snapshot.get("datasets", {}).get(dataset)
        if not dataset_entry:
            continue
        downstream = dataset_entry.get("downstream", {})
        if mean_key not in downstream:
            continue
        step = snapshot.get("global_step")
        if step is None:
            continue
        steps.append(int(step))
        values.append(float(downstream[mean_key]))
        std_value = downstream.get(std_key)
        stds.append(float(std_value) if std_value is not None else None)

    if steps:
        order = np.argsort(steps)
        steps = [steps[idx] for idx in order]
        values = [values[idx] for idx in order]
        stds = [stds[idx] for idx in order]

    return steps, values, stds


def _extract_diag_series(
    snapshots: List[Dict], dataset: str, metric: str
) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    values: List[float] = []
    for snapshot in snapshots:
        dataset_entry = snapshot.get("datasets", {}).get(dataset)
        if not dataset_entry:
            continue
        diag = dataset_entry.get("diag", {})
        if metric not in diag:
            continue
        step = snapshot.get("global_step")
        if step is None:
            continue
        steps.append(int(step))
        values.append(float(diag[metric]))
    if steps:
        order = np.argsort(steps)
        steps = [steps[idx] for idx in order]
        values = [values[idx] for idx in order]
    return steps, values


def _infer_main_metric(snapshots: List[Dict]) -> str:
    for metric in ("SRCC", "ACC"):
        for snapshot in snapshots:
            for dataset_entry in snapshot.get("datasets", {}).values():
                downstream = dataset_entry.get("downstream", {})
                if f"{metric}/mean" in downstream:
                    return metric
    return "SRCC"


def _setup_style() -> None:
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.1,
            "lines.markersize": 4.2,
            "lines.solid_capstyle": "round",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )
    plt.rcParams["axes.prop_cycle"] = cycler(color=palette)


def _metric_stem(metric: str) -> str:
    metric = metric.upper()
    if metric == "ACC":
        return "accuracy"
    return metric.lower()


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def _maybe_log_wandb(run, key: str, path: Path) -> None:
    if run is None:
        return
    import wandb

    run.log({key: wandb.Image(str(path))})


def _plot_main_metric(
    snapshots: List[Dict],
    datasets: List[str],
    main_metric: str,
    secondary_metric: Optional[str],
    output_dir: Path,
    log_run=None,
) -> Optional[Tuple[Path, Path]]:
    if not datasets:
        return None

    rows, cols = _grid_for_n(len(datasets))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3.6, rows * 2.8), constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        steps, values, stds = _extract_downstream_series(
            snapshots, dataset, main_metric
        )
        if not steps:
            ax.set_title(dataset)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            continue

        stds_arr = np.array(
            [np.nan if val is None else val for val in stds], dtype=float
        )
        has_err = np.isfinite(stds_arr).any()
        if has_err:
            yerr = np.where(np.isfinite(stds_arr), stds_arr, 0.0)
            ax.errorbar(
                steps,
                values,
                yerr=yerr,
                marker="o",
                capsize=3,
                elinewidth=1.2,
            )
        else:
            ax.plot(steps, values, marker="o")

        if secondary_metric:
            sec_steps, sec_values, _ = _extract_downstream_series(
                snapshots, dataset, secondary_metric
            )
            if sec_steps:
                ax.plot(sec_steps, sec_values, linestyle="--", marker="s")

        ax.set_title(dataset)
        ax.margins(x=0.03)
        ax.set_axisbelow(True)

    for idx in range(len(datasets), rows * cols):
        axes[idx].axis("off")

    fig.supxlabel("Iters")
    fig.supylabel(main_metric)

    if secondary_metric:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color="C0", marker="o", label=main_metric),
            Line2D(
                [0],
                [0],
                color="C1",
                linestyle="--",
                marker="s",
                label=secondary_metric,
            ),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)

    stem = f"{_metric_stem(main_metric)}_{rows}-{cols}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    _save_figure(fig, png_path)
    _save_figure(fig, pdf_path)
    _maybe_log_wandb(log_run, f"plots/{stem}", png_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_diag_metric(
    snapshots: List[Dict],
    datasets: List[str],
    metric: str,
    output_dir: Path,
    log_run=None,
) -> Optional[Tuple[Path, Path]]:
    if not datasets:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for idx, dataset in enumerate(datasets):
        steps, values = _extract_diag_series(snapshots, dataset, metric)
        if not steps:
            continue
        ax.plot(steps, values, marker="o", label=dataset)

    ax.set_xlabel("Iters")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.margins(x=0.03)
    ax.set_axisbelow(True)

    if datasets:
        ncol = min(4, max(1, len(datasets)))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=ncol,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.3)

    stem = f"{metric}_1"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    _save_figure(fig, png_path)
    _save_figure(fig, pdf_path)
    _maybe_log_wandb(log_run, f"plots/{metric}", png_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics over training steps.")
    parser.add_argument("--metrics_file", required=True, help="Path to metrics_over_time.pt")
    parser.add_argument("--output_dir", required=True, help="Directory to save figures")
    parser.add_argument("--main_metric", default="SRCC", help="Main metric (SRCC or ACC)")
    parser.add_argument("--main_metric_secondary", default=None, help="Secondary metric (e.g., PLCC)")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'")
    parser.add_argument(
        "--make_multipanel_main_metric",
        type=_parse_bool,
        default=True,
        help="Whether to generate multi-panel main metric figure",
    )
    parser.add_argument("--log_wandb", action="store_true", help="Log plots to wandb")
    parser.add_argument("--wandb_project", default=None, help="wandb project name")
    parser.add_argument("--wandb_name", default=None, help="wandb run name")
    args = parser.parse_args()

    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    snapshots = _load_snapshots(metrics_file)
    if not snapshots:
        raise ValueError("No snapshots found in metrics file.")

    _setup_style()

    main_metric = args.main_metric.upper()
    if main_metric == "SRCC":
        inferred = _infer_main_metric(snapshots)
        if inferred != main_metric:
            main_metric = inferred

    secondary_metric = args.main_metric_secondary
    if secondary_metric:
        secondary_metric = secondary_metric.upper()

    all_dataset_names = _collect_dataset_names(snapshots)
    selected_datasets = _select_datasets(all_dataset_names, args.datasets)

    log_run = None
    if args.log_wandb:
        import wandb

        log_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
        )

    if args.make_multipanel_main_metric:
        _plot_main_metric(
            snapshots=snapshots,
            datasets=selected_datasets,
            main_metric=main_metric,
            secondary_metric=secondary_metric,
            output_dir=output_dir,
            log_run=log_run,
        )

    for metric in DIAG_METRICS:
        _plot_diag_metric(
            snapshots=snapshots,
            datasets=selected_datasets,
            metric=metric,
            output_dir=output_dir,
            log_run=log_run,
        )

    if log_run is not None:
        log_run.finish()


if __name__ == "__main__":
    main()
