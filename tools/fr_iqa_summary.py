#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


PLCC_SUFFIXES = ("_plcc_raw", "_plcc_logistic")


def strip_plcc_suffix(name: str) -> str:
    for suffix in PLCC_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_plcc_mode(run_name: str, summary: dict) -> str:
    if "plcc_mode" in summary:
        return summary["plcc_mode"]
    if run_name.endswith("_plcc_logistic"):
        return "plcc_logistic"
    if run_name.endswith("_plcc_raw"):
        return "plcc_raw"
    return "unknown"


def read_key_values(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def infer_model_label(experiment_name: str, ckpt_label: str) -> str:
    experiment_key = experiment_name.lower()
    if experiment_key.startswith("shamisa") or "shamisa" in experiment_key:
        return "SHAMISA_A0"
    if experiment_key.startswith("a0_") or experiment_key.endswith("_a0"):
        return "SHAMISA_A0"
    return experiment_name


def load_csv_rows(path: Path) -> list:
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def summarize_run(run_dir: Path) -> tuple[list, list]:
    fr_results_path = run_dir / "fr_results.csv"
    per_seed_path = run_dir / "per_seed.csv"
    if not fr_results_path.exists() or not per_seed_path.exists():
        return [], []

    summary_meta = read_key_values(run_dir / "run_summary.txt")
    run_name = run_dir.name
    experiment_name = summary_meta.get("experiment_name", strip_plcc_suffix(run_name))
    plcc_mode = infer_plcc_mode(run_name, summary_meta)

    fr_rows = load_csv_rows(fr_results_path)
    per_seed_rows = load_csv_rows(per_seed_path)

    dataset_stats = {}
    for row in per_seed_rows:
        dataset = row.get("dataset", "")
        dataset_stats.setdefault(dataset, {"srcc": [], "plcc": []})
        dataset_stats[dataset]["srcc"].append(to_float(row.get("srcc", "")))
        dataset_stats[dataset]["plcc"].append(to_float(row.get("plcc", "")))

    median_rows = []
    seed_stat_rows = []
    for row in fr_rows:
        dataset = row.get("dataset", "")
        ckpt_label = row.get("ckpt", "")
        model_label = infer_model_label(experiment_name, ckpt_label)

        median_rows.append(
            {
                "model": model_label,
                "checkpoint_label": ckpt_label,
                "plcc_mode": plcc_mode,
                "dataset": dataset,
                "srcc_median": row.get("srcc", ""),
                "plcc_median": row.get("plcc", ""),
            }
        )

        stats = dataset_stats.get(dataset, {"srcc": [], "plcc": []})
        srcc_values = np.array(stats["srcc"], dtype=np.float64)
        plcc_values = np.array(stats["plcc"], dtype=np.float64)
        srcc_values = srcc_values[np.isfinite(srcc_values)]
        plcc_values = plcc_values[np.isfinite(plcc_values)]

        srcc_mean = float(np.mean(srcc_values)) if srcc_values.size else float("nan")
        srcc_std = float(np.std(srcc_values)) if srcc_values.size else float("nan")
        plcc_mean = float(np.mean(plcc_values)) if plcc_values.size else float("nan")
        plcc_std = float(np.std(plcc_values)) if plcc_values.size else float("nan")

        seed_stat_rows.append(
            {
                "model": model_label,
                "checkpoint_label": ckpt_label,
                "plcc_mode": plcc_mode,
                "dataset": dataset,
                "srcc_mean": srcc_mean,
                "srcc_std": srcc_std,
                "plcc_mean": plcc_mean,
                "plcc_std": plcc_std,
            }
        )

    return median_rows, seed_stat_rows


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FR-IQA runs.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/fr_iqa",
        help="Root directory containing FR-IQA run outputs.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_median_rows = []
    all_seed_rows = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "summary":
            continue
        median_rows, seed_rows = summarize_run(run_dir)
        all_median_rows.extend(median_rows)
        all_seed_rows.extend(seed_rows)

    median_path = summary_dir / "fr_summary_median.csv"
    seed_path = summary_dir / "fr_summary_seed_stats.csv"
    write_csv(
        median_path,
        all_median_rows,
        [
            "model",
            "checkpoint_label",
            "plcc_mode",
            "dataset",
            "srcc_median",
            "plcc_median",
        ],
    )
    write_csv(
        seed_path,
        all_seed_rows,
        [
            "model",
            "checkpoint_label",
            "plcc_mode",
            "dataset",
            "srcc_mean",
            "srcc_std",
            "plcc_mean",
            "plcc_std",
        ],
    )

    print(f"Wrote {median_path}")
    print(f"Wrote {seed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
