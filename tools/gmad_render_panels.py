"""
Render stacked gMAD panel images from a gmad_summary.json.

Usage:
    python tools/gmad_render_panels.py \
        --summary /path/to/gmad_summary.json \
        --images_dir /path/to/gmad_out/images \
        --output_dir /path/to/gmad_out/panels \
        --panel_style debug
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont


def _sanitize_identifier(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars)
    return sanitized.strip("_") or "result"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    new_height = int(round(image.height * (width / image.width)))
    return image.resize((width, new_height), Image.BICUBIC)


def _resolve_exported_path(
    summary_dir: Path, images_dir: Path, entry: Dict, key: str
) -> Path:
    if key not in entry:
        raise FileNotFoundError(
            f"Missing {key} in gmad_summary.json. "
            "Run gmad_pairs.py with --copy_images first."
        )
    path = Path(entry[key])
    if path.is_absolute():
        return path
    candidate = summary_dir / path
    if candidate.exists():
        return candidate
    if path.parts and path.parts[0] == "images":
        return images_dir / Path(*path.parts[1:])
    return images_dir / path


def _parse_wandb_tags(tags: str):
    if not tags:
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _resolve_wandb_mode(arg_mode: str) -> str:
    if arg_mode:
        return arg_mode
    env_mode = os.getenv("WANDB_MODE")
    if env_mode:
        return env_mode
    if os.getenv("WANDB_OFFLINE"):
        return "offline"
    return "online"


def _init_wandb(args, summary: Dict):
    if not args.enable_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but --enable_wandb was set.") from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT") or "shamisa"
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    run_name = args.wandb_run_name or f"gmad_panels_{summary.get('config', {}).get('name_a', 'run')}"
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


def _log_wandb_panels(run, panel_paths, artifact_name: str) -> None:
    if run is None:
        return

    import wandb

    log_dict = {}
    for panel_path in panel_paths:
        key = f"gmad_panel/{panel_path.stem}"
        log_dict[key] = wandb.Image(str(panel_path))
    if log_dict:
        run.log(log_dict)

    artifact = wandb.Artifact(artifact_name, type="gmad_panels")
    for panel_path in panel_paths:
        artifact.add_file(str(panel_path))
    run.log_artifact(artifact)
    run.finish()


def _load_font(size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_panel_debug(
    best_img: Image.Image,
    worst_img: Image.Image,
    defender_name: str,
    attacker_name: str,
    output_path: Path,
) -> None:
    margin = 20
    text_gap = 8
    block_gap = 16
    arrow_gap = 4

    font = _load_font(24)
    top_label = f"Best {attacker_name}"
    bottom_label = f"Worst {attacker_name}"
    mid_label = f"Fixed defender: {defender_name}"
    arrow_label = "v"

    width = max(best_img.width, worst_img.width)
    best_img = _resize_to_width(best_img, width)
    worst_img = _resize_to_width(worst_img, width)

    dummy = Image.new("RGB", (width, 10), "white")
    draw = ImageDraw.Draw(dummy)
    top_w, top_h = _text_size(draw, top_label, font)
    bottom_w, bottom_h = _text_size(draw, bottom_label, font)
    mid_w, mid_h = _text_size(draw, mid_label, font)
    arrow_w, arrow_h = _text_size(draw, arrow_label, font)

    total_height = (
        margin
        + top_h
        + text_gap
        + best_img.height
        + block_gap
        + mid_h
        + arrow_gap
        + arrow_h
        + block_gap
        + worst_img.height
        + text_gap
        + bottom_h
        + margin
    )
    total_width = width + 2 * margin

    panel = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(panel)

    y = margin
    draw.text(((total_width - top_w) // 2, y), top_label, font=font, fill="black")
    y += top_h + text_gap

    panel.paste(best_img, (margin, y))
    y += best_img.height + block_gap

    draw.text(((total_width - mid_w) // 2, y), mid_label, font=font, fill="black")
    y += mid_h + arrow_gap

    draw.text(((total_width - arrow_w) // 2, y), arrow_label, font=font, fill="black")
    y += arrow_h + block_gap

    panel.paste(worst_img, (margin, y))
    y += worst_img.height + text_gap

    draw.text(
        ((total_width - bottom_w) // 2, y),
        bottom_label,
        font=font,
        fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path, format="PNG")


def _render_panel_paper(
    best_img: Image.Image,
    worst_img: Image.Image,
    output_path: Path,
) -> None:
    margin = 6
    separator = 4
    width = max(best_img.width, worst_img.width)
    best_img = _resize_to_width(best_img, width)
    worst_img = _resize_to_width(worst_img, width)

    total_width = width + 2 * margin
    total_height = best_img.height + worst_img.height + separator + 2 * margin
    panel = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(panel)

    y = margin
    panel.paste(best_img, (margin, y))
    y += best_img.height
    draw.rectangle(
        [margin, y, margin + width, y + separator - 1],
        fill=(200, 200, 200),
    )
    y += separator
    panel.paste(worst_img, (margin, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gMAD panel images.")
    parser.add_argument(
        "--summary", type=str, required=True, help="Path to gmad_summary.json"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Images directory (defaults to summary parent / images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to summary parent / panels)",
    )
    parser.add_argument(
        "--panel_style",
        type=str,
        default=None,
        choices=["debug", "paper"],
        help="Panel style: debug (text) or paper (no text)",
    )
    parser.add_argument(
        "--export_individual_paper",
        action="store_true",
        help="If set with paper style, export individual top/bottom images",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=True,
        help="Log panels to Weights & Biases",
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

    summary_path = Path(args.summary).expanduser()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    base_dir = summary_path.parent
    images_dir = (
        Path(args.images_dir).expanduser() if args.images_dir else base_dir / "images"
    )
    panel_style = (
        args.panel_style
        or summary.get("config", {}).get("panel_style")
        or "debug"
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else base_dir / ("panels_paper" if panel_style == "paper" else "panels")
    )
    export_individual = bool(args.export_individual_paper) and panel_style == "paper"

    cases = summary.get("cases", {})
    if not cases:
        raise ValueError("No cases found in summary file.")

    panel_paths = []

    for case in cases.values():
        defender = case["defender_name"]
        attacker = case["attacker_name"]
        level = case["level"]
        pairs = case.get("pairs", [])
        if not pairs:
            continue

        for pair in pairs:
            best_path = _resolve_exported_path(
                base_dir, images_dir, pair, "exported_best"
            )
            worst_path = _resolve_exported_path(
                base_dir, images_dir, pair, "exported_worst"
            )
            if not best_path.exists() or not worst_path.exists():
                raise FileNotFoundError(
                    f"Missing exported images: {best_path} / {worst_path}"
                )

            with Image.open(best_path) as best_img, Image.open(worst_path) as worst_img:
                best_img = best_img.convert("RGB")
                worst_img = worst_img.convert("RGB")
                pair_idx = pair.get("pair_index", 0)
                name_parts = [
                    "panel",
                    _sanitize_identifier(defender),
                    "def",
                    level,
                    "att",
                    _sanitize_identifier(attacker),
                ]
                if len(pairs) > 1:
                    name_parts.append(f"pair{pair_idx}")
                out_name = "_".join(name_parts) + ".png"
                out_path = output_dir / out_name
                if panel_style == "paper":
                    _render_panel_paper(
                        best_img=best_img,
                        worst_img=worst_img,
                        output_path=out_path,
                    )
                    if export_individual:
                        case_key = f"{_sanitize_identifier(defender)}_def_{level}"
                        case_dir = base_dir / "images_paper" / f"case_{case_key}"
                        case_dir.mkdir(parents=True, exist_ok=True)
                        best_img.save(
                            case_dir / f"pair{pair_idx}_top.png",
                            format="PNG",
                        )
                        worst_img.save(
                            case_dir / f"pair{pair_idx}_bottom.png",
                            format="PNG",
                        )
                else:
                    _render_panel_debug(
                        best_img=best_img,
                        worst_img=worst_img,
                        defender_name=defender,
                        attacker_name=attacker,
                        output_path=out_path,
                    )
                panel_paths.append(out_path)

    print(f"Saved panel images to {output_dir}")

    if args.wandb_mode == "disabled":
        return

    wandb_run = _init_wandb(args, summary)
    if wandb_run is not None and panel_paths:
        artifact_name = args.wandb_artifact_name or f"gmad_panels_{output_dir.name}"
        _log_wandb_panels(wandb_run, panel_paths, artifact_name)


if __name__ == "__main__":
    main()
