"""
Select gMAD pairs from two score files and optionally export images.

Usage:
    python tools/gmad_pairs.py \
        --scores_a /path/to/shamisa_scores.npz \
        --scores_b /path/to/comparison_scores.npz \
        --output_dir /path/to/gmad_out \
        --image_root /path/to/WaterlooExploration \
        --copy_images
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from distutils.util import strtobool


def _sanitize_identifier(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars)
    return sanitized.strip("_") or "result"


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _parse_wandb_tags(tags: Optional[str]) -> Optional[List[str]]:
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


def _init_wandb(args, summary: Dict) -> Optional[object]:
    if not args.enable_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but --enable_wandb was set.") from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT") or "shamisa"
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    run_name = args.wandb_run_name or f"gmad_{args.name_a}_vs_{args.name_b}"
    mode = _resolve_wandb_mode(args.wandb_mode)
    tags = _parse_wandb_tags(args.wandb_tags)

    init_kwargs = {
        "project": project,
        "entity": entity,
        "name": run_name,
        "group": args.wandb_group,
        "tags": tags,
        "config": summary.get("config", {}),
        "mode": mode,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def _log_wandb_results(
    run,
    summary: Dict,
    summary_path: Path,
    output_dir: Path,
    args,
) -> None:
    if run is None:
        return

    import wandb

    metrics = {}
    score_stats = summary.get("score_stats", {})
    for model_name, stats in score_stats.items():
        for key, value in stats.items():
            metrics[f"score/{model_name}_{key}"] = value

    for case_key, case in summary.get("cases", {}).items():
        gaps = [pair["attacker_gap"] for pair in case.get("pairs", [])]
        if gaps:
            metrics[f"gmad/{case_key}_gap_max"] = float(max(gaps))
            metrics[f"gmad/{case_key}_gap_min"] = float(min(gaps))
            metrics[f"gmad/{case_key}_gap_mean"] = float(np.mean(gaps))
        metrics[f"gmad/{case_key}_bin_size"] = case.get("bin_size", 0)

    if metrics:
        run.log(metrics)

    artifact = wandb.Artifact(
        args.wandb_artifact_name or f"gmad_{summary_path.stem}",
        type="gmad_results",
    )
    artifact.add_file(str(summary_path))

    images_dir = output_dir / "images"
    if images_dir.exists():
        for image_path in sorted(images_dir.glob("*.png")):
            artifact.add_file(str(image_path))

    run.log_artifact(artifact)
    run.finish()


def _load_scores(path: Path) -> Tuple[List[str], np.ndarray, Optional[str]]:
    data = np.load(path, allow_pickle=True)
    if "paths" not in data or "scores" not in data:
        raise ValueError(f"Score file {path} must contain 'paths' and 'scores'.")

    paths = [str(p) for p in data["paths"].tolist()]
    scores = np.asarray(data["scores"], dtype=np.float64)
    if scores.shape[0] != len(paths):
        raise ValueError(
            f"Score file {path} has {len(paths)} paths but {scores.shape[0]} scores."
        )
    model_name = None
    if "model_name" in data:
        raw = data["model_name"]
        model_name = str(raw.tolist()) if hasattr(raw, "tolist") else str(raw)
    return paths, scores, model_name


def _align_scores(
    paths_a: List[str],
    scores_a: np.ndarray,
    paths_b: List[str],
    scores_b: np.ndarray,
    strict: bool,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if len(set(paths_a)) != len(paths_a) or len(set(paths_b)) != len(paths_b):
        raise ValueError("Duplicate paths detected in score files; cannot align.")

    if paths_a == paths_b:
        return paths_a, scores_a, scores_b

    map_a = {p: i for i, p in enumerate(paths_a)}
    map_b = {p: i for i, p in enumerate(paths_b)}
    missing_in_b = [p for p in paths_a if p not in map_b]
    missing_in_a = [p for p in paths_b if p not in map_a]

    if (missing_in_b or missing_in_a) and strict:
        raise ValueError(
            "Score files do not align.\n"
            f"Missing in B: {missing_in_b[:5]}{'...' if len(missing_in_b) > 5 else ''}\n"
            f"Missing in A: {missing_in_a[:5]}{'...' if len(missing_in_a) > 5 else ''}"
        )

    common = [p for p in paths_a if p in map_b]
    if not common:
        raise ValueError("No overlapping paths found between score files.")

    aligned_scores_a = scores_a[[map_a[p] for p in common]]
    aligned_scores_b = scores_b[[map_b[p] for p in common]]
    return common, aligned_scores_a, aligned_scores_b


def _compute_bin_edges(scores: np.ndarray, num_bins: int, binning: str) -> np.ndarray:
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")
    if scores.size == 0:
        raise ValueError("No scores provided for binning.")
    if binning == "quantile":
        edges = np.quantile(scores, np.linspace(0, 1, num_bins + 1))
    elif binning == "uniform":
        min_val = float(scores.min())
        max_val = float(scores.max())
        if min_val == max_val:
            edges = np.array([min_val] * (num_bins + 1), dtype=np.float64)
        else:
            edges = np.linspace(min_val, max_val, num_bins + 1)
    else:
        raise ValueError(f"Unsupported binning type: {binning}")
    return edges


def _assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.zeros_like(scores, dtype=np.int64)
    if np.allclose(edges[0], edges[-1]):
        return np.zeros_like(scores, dtype=np.int64)
    return np.digitize(scores, edges[1:-1], right=False)


def _select_pairs(
    indices: np.ndarray,
    def_scores: np.ndarray,
    att_scores: np.ndarray,
    paths: List[str],
    top_k: int,
) -> List[Dict]:
    if indices.size < 2:
        raise ValueError("Not enough images in selected defender level to form pairs.")

    ordered = indices[np.argsort(att_scores[indices])]
    max_pairs = ordered.size // 2
    if max_pairs < 1:
        raise ValueError("Not enough images to form a gMAD pair.")

    use_k = min(top_k, max_pairs)
    pairs = []
    for k in range(use_k):
        worst_idx = int(ordered[k])
        best_idx = int(ordered[-k - 1])
        if best_idx == worst_idx:
            raise ValueError("Selected pair has duplicate images.")

        pair = {
            "pair_index": k,
            "img_best": paths[best_idx],
            "img_worst": paths[worst_idx],
            "img1": paths[best_idx],
            "img2": paths[worst_idx],
            "q_def_best": float(def_scores[best_idx]),
            "q_def_worst": float(def_scores[worst_idx]),
            "q_att_best": float(att_scores[best_idx]),
            "q_att_worst": float(att_scores[worst_idx]),
            "q_def_img1": float(def_scores[best_idx]),
            "q_def_img2": float(def_scores[worst_idx]),
            "q_att_img1": float(att_scores[best_idx]),
            "q_att_img2": float(att_scores[worst_idx]),
            "attacker_gap": float(
                abs(att_scores[best_idx] - att_scores[worst_idx])
            ),
        }
        pairs.append(pair)
    return pairs


def _resolve_image_path(path: str, image_root: Optional[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if image_root is None:
        raise ValueError(
            "image_root must be provided when score paths are relative."
        )
    return image_root / candidate


def _export_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.save(dst, format="PNG")


def _build_case_key(def_name: str, level: str) -> str:
    return f"{def_name}_def_{level}"


def _build_export_name(
    def_name: str, level: str, att_name: str, pair_idx: int, kind: str
) -> str:
    return (
        f"gmad_{_sanitize_identifier(def_name)}_def_{level}_att_"
        f"{_sanitize_identifier(att_name)}_pair{pair_idx}_{kind}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select gMAD pairs.")
    parser.add_argument("--scores_a", type=str, required=True, help="Scores npz for model A")
    parser.add_argument("--scores_b", type=str, required=True, help="Scores npz for model B")
    parser.add_argument("--name_a", type=str, default="shamisa", help="Name for model A")
    parser.add_argument("--name_b", type=str, default="comparison_model", help="Name for model B")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins")
    parser.add_argument("--low_bin", type=int, default=0, help="Low bin index")
    parser.add_argument("--high_bin", type=int, default=None, help="High bin index")
    parser.add_argument("--top_k", type=int, default=1, help="Top K pairs per case")
    parser.add_argument(
        "--binning",
        type=str,
        default="quantile",
        choices=["quantile", "uniform"],
        help="Binning strategy",
    )
    parser.add_argument(
        "--panel_style",
        type=str,
        default="debug",
        choices=["debug", "paper"],
        help="Panel style hint for downstream rendering",
    )
    parser.add_argument(
        "--low_percentile",
        type=float,
        default=None,
        help="Optional low percentile selection (0-100)",
    )
    parser.add_argument(
        "--high_percentile",
        type=float,
        default=None,
        help="Optional high percentile selection (0-100)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--copy_images", action="store_true", help="Copy images for selected pairs"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory for images if paths are relative",
    )
    parser.add_argument(
        "--strict_alignment",
        type=strtobool,
        default=True,
        help="Require identical image sets",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=True,
        help="Log gMAD results to Weights & Biases",
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

    args = parser.parse_args()
    if args.disable_wandb:
        args.enable_wandb = False

    scores_a_path = Path(args.scores_a).expanduser()
    scores_b_path = Path(args.scores_b).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_a, scores_a, model_name_a = _load_scores(scores_a_path)
    paths_b, scores_b, model_name_b = _load_scores(scores_b_path)

    paths, scores_a, scores_b = _align_scores(
        paths_a, scores_a, paths_b, scores_b, bool(args.strict_alignment)
    )
    print(f"[Sanity] Aligned {len(paths)} images between score files.")
    if paths_a != paths_b:
        print("[Sanity] Paths differ between score files; aligned by path key.")

    if args.high_bin is None:
        args.high_bin = args.num_bins - 1

    if args.low_bin < 0 or args.low_bin >= args.num_bins:
        raise ValueError("low_bin is out of range for num_bins.")
    if args.high_bin < 0 or args.high_bin >= args.num_bins:
        raise ValueError("high_bin is out of range for num_bins.")
    if args.low_percentile is not None or args.high_percentile is not None:
        if args.low_percentile is None or args.high_percentile is None:
            raise ValueError("Both low_percentile and high_percentile must be set.")
        if not 0 <= args.low_percentile <= 100:
            raise ValueError("low_percentile must be between 0 and 100.")
        if not 0 <= args.high_percentile <= 100:
            raise ValueError("high_percentile must be between 0 and 100.")
        if args.low_percentile >= args.high_percentile:
            raise ValueError("low_percentile must be less than high_percentile.")

    image_root = Path(args.image_root).expanduser() if args.image_root else None

    summary = {
        "config": {
            "scores_a": str(scores_a_path),
            "scores_b": str(scores_b_path),
            "name_a": args.name_a,
            "name_b": args.name_b,
            "model_name_a": model_name_a,
            "model_name_b": model_name_b,
            "num_bins": args.num_bins,
            "low_bin": args.low_bin,
            "high_bin": args.high_bin,
            "top_k": args.top_k,
            "binning": args.binning,
            "panel_style": args.panel_style,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "strict_alignment": bool(args.strict_alignment),
            "timestamp": datetime.now().isoformat(),
        },
        "alignment": {
            "num_images": int(len(paths)),
            "paths": paths,
            "scores": {
                args.name_a: scores_a.tolist(),
                args.name_b: scores_b.tolist(),
            },
        },
        "score_stats": {
            args.name_a: {
                "mean": float(np.mean(scores_a)),
                "std": float(np.std(scores_a)),
                "min": float(np.min(scores_a)),
                "max": float(np.max(scores_a)),
            },
            args.name_b: {
                "mean": float(np.mean(scores_b)),
                "std": float(np.std(scores_b)),
                "min": float(np.min(scores_b)),
                "max": float(np.max(scores_b)),
            },
        },
        "bin_edges": {},
        "cases": {},
    }

    def _build_cases(def_scores, att_scores, def_name, att_name):
        edges = _compute_bin_edges(def_scores, args.num_bins, args.binning)
        summary["bin_edges"][def_name] = edges.tolist()
        bin_ids = _assign_bins(def_scores, edges)
        print(
            f"[Sanity] Binning={args.binning}, num_bins={args.num_bins}, "
            f"low_bin={args.low_bin}, high_bin={args.high_bin}"
        )

        if args.low_percentile is not None or args.high_percentile is not None:
            if args.low_percentile is None or args.high_percentile is None:
                raise ValueError("Both low_percentile and high_percentile must be set.")
            low_thr = np.percentile(def_scores, args.low_percentile)
            high_thr = np.percentile(def_scores, args.high_percentile)
            low_indices = np.where(def_scores <= low_thr)[0]
            high_indices = np.where(def_scores >= high_thr)[0]
            low_edges = [float(low_thr), float(low_thr)]
            high_edges = [float(high_thr), float(high_thr)]
            selection_note = "percentile"
        else:
            low_indices = np.where(bin_ids == args.low_bin)[0]
            high_indices = np.where(bin_ids == args.high_bin)[0]
            low_edges = [float(edges[args.low_bin]), float(edges[args.low_bin + 1])]
            high_edges = [float(edges[args.high_bin]), float(edges[args.high_bin + 1])]
            selection_note = "bin"

        def _log_bin_stats(level, indices, edges_range):
            if indices.size == 0:
                print(
                    f"[Sanity] {def_name} {level}: empty selection "
                    f"(selection={selection_note})"
                )
                return
            def_vals = def_scores[indices]
            print(
                f"[Sanity] {def_name} {level}: selection={selection_note} "
                f"bin_edges={edges_range} bin_size={indices.size} "
                f"def_min={float(def_vals.min()):.5f} "
                f"def_max={float(def_vals.max()):.5f}"
            )

        def _in_bin(def_value, level, edges_range):
            if selection_note == "percentile":
                if level == "low":
                    return def_value <= edges_range[0] + 1e-8
                return def_value >= edges_range[0] - 1e-8
            low, high = edges_range
            return (def_value >= low - 1e-8) and (def_value <= high + 1e-8)

        cases = []
        for level, indices, edges_range in [
            ("low", low_indices, low_edges),
            ("high", high_indices, high_edges),
        ]:
            _log_bin_stats(level, indices, edges_range)
            pairs = _select_pairs(indices, def_scores, att_scores, paths, args.top_k)
            for pair in pairs:
                def1 = pair["q_def_img1"]
                def2 = pair["q_def_img2"]
                att1 = pair["q_att_img1"]
                att2 = pair["q_att_img2"]
                defdiff = abs(def1 - def2)
                gap = abs(att1 - att2)
                in_bin_1 = _in_bin(def1, level, edges_range)
                in_bin_2 = _in_bin(def2, level, edges_range)
                best_is_high = att1 >= att2
                print(
                    f"[Sanity] {def_name} {level} pair{pair['pair_index']}: "
                    f"defdiff={defdiff:.5f} gap={gap:.5f} "
                    f"in_bin=({in_bin_1},{in_bin_2}) "
                    f"best_is_high={best_is_high}"
                )
            case = {
                "defender_name": def_name,
                "attacker_name": att_name,
                "level": level,
                "selection": selection_note,
                "bin_edges": edges_range,
                "bin_size": int(indices.size),
                "pairs": pairs,
            }
            cases.append(case)
        return cases

    cases_a = _build_cases(scores_a, scores_b, args.name_a, args.name_b)
    cases_b = _build_cases(scores_b, scores_a, args.name_b, args.name_a)

    for case in cases_a + cases_b:
        case_key = _build_case_key(case["defender_name"], case["level"])
        summary["cases"][case_key] = case

    if args.copy_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for case in summary["cases"].values():
            def_name = case["defender_name"]
            att_name = case["attacker_name"]
            level = case["level"]
            for pair in case["pairs"]:
                pair_idx = pair["pair_index"]
                best_src = _resolve_image_path(pair["img_best"], image_root)
                worst_src = _resolve_image_path(pair["img_worst"], image_root)
                best_name = _build_export_name(
                    def_name, level, att_name, pair_idx, "best"
                )
                worst_name = _build_export_name(
                    def_name, level, att_name, pair_idx, "worst"
                )
                best_dst = images_dir / best_name
                worst_dst = images_dir / worst_name
                _export_image(best_src, best_dst)
                _export_image(worst_src, worst_dst)
                pair["exported_best"] = str(best_dst.relative_to(output_dir))
                pair["exported_worst"] = str(worst_dst.relative_to(output_dir))

    summary_path = output_dir / "gmad_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(summary), handle, indent=2)

    print(f"Saved gMAD summary to {summary_path}")

    if args.wandb_mode == "disabled":
        return

    wandb_run = _init_wandb(args, summary)
    if wandb_run is not None:
        _log_wandb_results(wandb_run, summary, summary_path, output_dir, args)


if __name__ == "__main__":
    main()
