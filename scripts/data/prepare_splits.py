#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SYNTHETIC_DATASETS = {"live", "csiq", "tid2013", "kadid10k"}
ALL_DATASETS = ["live", "csiq", "tid2013", "kadid10k", "flive", "spaq"]


def _padded_split_array(split_indices: list[np.ndarray]) -> np.ndarray:
    max_len = max((arr.size for arr in split_indices), default=0)
    output = np.full((len(split_indices), max_len), -1, dtype=np.int64)
    for row, indices in enumerate(split_indices):
        output[row, : indices.size] = indices
    return output


def _write_split_bundle(out_dir: Path, train_splits: list[np.ndarray], val_splits: list[np.ndarray], test_splits: list[np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", _padded_split_array(train_splits))
    np.save(out_dir / "val.npy", _padded_split_array(val_splits))
    np.save(out_dir / "test.npy", _padded_split_array(test_splits))


def _split_counts(n_items: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> tuple[int, int, int]:
    train_count = int(round(train_ratio * n_items))
    val_count = int(round(val_ratio * n_items))
    if train_count >= n_items:
        train_count = max(n_items - 2, 1)
    if train_count + val_count >= n_items:
        val_count = max(n_items - train_count - 1, 1)
    test_count = n_items - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count = max(train_count - 1, 1)
    return train_count, val_count, test_count


def _reference_series(dataset_key: str, dataset_root: Path) -> pd.Series:
    if dataset_key == "live":
        return pd.read_csv(dataset_root / "LIVE.txt")["ref_img_path"].astype(str)
    if dataset_key == "csiq":
        return pd.read_csv(dataset_root / "CSIQ.txt")["ref_img_path"].astype(str)
    if dataset_key == "tid2013":
        names = pd.read_csv(
            dataset_root / "mos_with_names.txt",
            sep=" ",
            header=None,
            names=["mos", "img_name"],
        )["img_name"].astype(str)
        return names.map(lambda name: name.split("_")[0].upper() + ".BMP")
    if dataset_key == "kadid10k":
        return pd.read_csv(dataset_root / "dmos.csv")["ref_img"].astype(str)
    raise ValueError(f"Unsupported reference-disjoint dataset '{dataset_key}'")


def _build_reference_disjoint_splits(dataset_key: str, dataset_root: Path, num_splits: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    references = _reference_series(dataset_key, dataset_root)
    unique_refs = np.array(sorted(references.unique().tolist()))
    train_splits, val_splits, test_splits = [], [], []
    train_count, val_count, _ = _split_counts(len(unique_refs))

    for split_idx in range(num_splits):
        rng = np.random.default_rng(seed + split_idx)
        shuffled_refs = unique_refs[rng.permutation(len(unique_refs))]
        train_refs = set(shuffled_refs[:train_count])
        val_refs = set(shuffled_refs[train_count : train_count + val_count])
        test_refs = set(shuffled_refs[train_count + val_count :])

        train_idx = np.flatnonzero(references.isin(train_refs).to_numpy())
        val_idx = np.flatnonzero(references.isin(val_refs).to_numpy())
        test_idx = np.flatnonzero(references.isin(test_refs).to_numpy())

        train_splits.append(np.sort(train_idx))
        val_splits.append(np.sort(val_idx))
        test_splits.append(np.sort(test_idx))

    return train_splits, val_splits, test_splits


def _build_random_splits(item_count: int, num_splits: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    train_splits, val_splits, test_splits = [], [], []
    train_count, val_count, _ = _split_counts(item_count)

    for split_idx in range(num_splits):
        rng = np.random.default_rng(seed + split_idx)
        order = rng.permutation(item_count)
        train_idx = np.sort(order[:train_count])
        val_idx = np.sort(order[train_count : train_count + val_count])
        test_idx = np.sort(order[train_count + val_count :])
        train_splits.append(train_idx)
        val_splits.append(val_idx)
        test_splits.append(test_idx)

    return train_splits, val_splits, test_splits


def _infer_flive_official_split(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    normalized_columns = {str(col).strip().lower(): col for col in labels.columns}
    for column_name in ["split", "set", "phase"]:
        if column_name not in normalized_columns:
            continue
        column = labels[normalized_columns[column_name]].astype(str).str.strip().str.lower()
        train_idx = np.flatnonzero(column.isin(["train", "training"]).to_numpy())
        val_idx = np.flatnonzero(column.isin(["val", "valid", "validation"]).to_numpy())
        test_idx = np.flatnonzero(column.isin(["test", "testing"]).to_numpy())
        if train_idx.size and val_idx.size and test_idx.size:
            return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    if "is_test" in normalized_columns:
        is_test = labels[normalized_columns["is_test"]].astype(int).to_numpy() != 0
        test_idx = np.flatnonzero(is_test)
        train_pool = np.flatnonzero(~is_test)
        if train_pool.size and test_idx.size:
            val_count = max(int(round(0.125 * train_pool.size)), 1)
            val_idx = np.sort(train_pool[:val_count])
            train_idx = np.sort(train_pool[val_count:])
            return train_idx, val_idx, np.sort(test_idx)

    return None


def _prepare_dataset(dataset_key: str, data_root: Path, num_splits: int, seed: int, overwrite: bool, allow_random_flive: bool) -> None:
    dataset_dir = data_root / dataset_key.upper().replace("10K", "10K")
    lookup = {
        "live": data_root / "LIVE",
        "csiq": data_root / "CSIQ",
        "tid2013": data_root / "TID2013",
        "kadid10k": data_root / "KADID10K",
        "flive": data_root / "FLIVE",
        "spaq": data_root / "SPAQ",
    }
    dataset_dir = lookup[dataset_key]
    split_dir = dataset_dir / "splits"

    if split_dir.exists() and not overwrite:
        print(f"[skip] {dataset_key}: splits already exist at {split_dir}")
        return

    if dataset_key in SYNTHETIC_DATASETS:
        train_splits, val_splits, test_splits = _build_reference_disjoint_splits(dataset_key, dataset_dir, num_splits, seed)
        _write_split_bundle(split_dir, train_splits, val_splits, test_splits)
        print(f"[ok] {dataset_key}: wrote {num_splits} reference-disjoint splits to {split_dir}")
        return

    if dataset_key == "spaq":
        item_count = len(pd.read_excel(dataset_dir / "Annotations" / "MOS and Image attribute scores.xlsx"))
        train_splits, val_splits, test_splits = _build_random_splits(item_count, num_splits, seed)
        _write_split_bundle(split_dir, train_splits, val_splits, test_splits)
        print(f"[ok] spaq: wrote {num_splits} random 70/10/20 splits to {split_dir}")
        return

    if dataset_key == "flive":
        labels = pd.read_csv(dataset_dir / "labels_image.csv")
        inferred = _infer_flive_official_split(labels)
        if inferred is None:
            if not allow_random_flive:
                raise ValueError(
                    "Could not infer the official FLIVE split from labels_image.csv. "
                    "Run again with --allow-random-flive to create an exploratory fallback split."
                )
            train_splits, val_splits, test_splits = _build_random_splits(len(labels), 1, seed)
        else:
            train_idx, val_idx, test_idx = inferred
            train_splits, val_splits, test_splits = [train_idx], [val_idx], [test_idx]
        _write_split_bundle(split_dir, train_splits, val_splits, test_splits)
        print(f"[ok] flive: wrote split bundle to {split_dir}")
        return

    raise ValueError(f"Unsupported dataset '{dataset_key}'")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset split files required by SHAMISA.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing LIVE/CSIQ/TID2013/KADID10K/FLIVE/SPAQ.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Subset of datasets to prepare. Default: all.")
    parser.add_argument("--num-splits", type=int, default=10, help="Number of repeated train/val/test splits for synthetic datasets and SPAQ.")
    parser.add_argument("--seed", type=int, default=27, help="Base RNG seed for split generation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--allow-random-flive", action="store_true", help="Generate a fallback FLIVE split when the official split cannot be inferred automatically.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = ALL_DATASETS if args.datasets == ["all"] else [item.lower() for item in args.datasets]
    for dataset_key in datasets:
        _prepare_dataset(
            dataset_key=dataset_key,
            data_root=args.data_root,
            num_splits=int(args.num_splits),
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
            allow_random_flive=bool(args.allow_random_flive),
        )


if __name__ == "__main__":
    main()
