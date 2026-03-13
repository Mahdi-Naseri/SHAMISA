#!/usr/bin/env python3
"""
Utility to merge experiment overrides into the default SHAMISA config while
preserving ordering and comments.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import List

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ruamel.yaml is required to run this script. Install it via "
        "`pip install ruamel.yaml`.\n",
    )
    raise SystemExit(1) from exc


def _load_default_config(path: Path) -> CommentedMap:
    yaml = YAML()
    yaml.preserve_quotes = True
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)
    if not isinstance(data, CommentedMap):
        raise ValueError(f"Unexpected YAML structure in {path}")
    return data


def _load_overrides(path: Path):
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle) or {}


def _merge_overrides(
    base,
    overrides,
    *,
    path: str = "",
    strict: bool = False,
) -> List[str]:
    warnings: List[str] = []
    if not isinstance(overrides, Mapping):
        raise TypeError("Overrides must be a mapping at the top level.")

    for key, value in overrides.items():
        dotted = f"{path}.{key}" if path else str(key)
        if key not in base:
            message = f"Skipping unknown key '{dotted}' (not in default config)."
            if strict:
                raise KeyError(message)
            warnings.append(message)
            continue

        base_value = base[key]
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            warnings.extend(
                _merge_overrides(
                    base_value,
                    value,
                    path=dotted,
                    strict=strict,
                ),
            )
        else:
            base[key] = value
    return warnings


def _dump_config(data: CommentedMap, path: Path) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096  # avoid re-wrapping long inline entries
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge an experiment config into the default SHAMISA config while "
            "preserving comments."
        ),
    )
    parser.add_argument(
        "experiment_config",
        type=Path,
        help="Path to the experiment checkpoint config YAML file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the merged config (default: ./config.merged.yaml).",
    )
    parser.add_argument(
        "-d",
        "--default-config",
        type=Path,
        help="Path to the default SHAMISA config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when overrides contain unknown keys.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    default_path = args.default_config or script_dir / "configs" / "default.yaml"
    experiment_path = args.experiment_config
    output_path = args.output or script_dir / "config.merged.yaml"

    if not default_path.is_file():
        sys.stderr.write(f"Default config not found: {default_path}\n")
        return 1
    if not experiment_path.is_file():
        sys.stderr.write(f"Experiment config not found: {experiment_path}\n")
        return 1

    default_config = _load_default_config(default_path)
    overrides = _load_overrides(experiment_path)

    warnings = _merge_overrides(
        default_config,
        overrides,
        strict=args.strict,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_config(default_config, output_path)

    sys.stdout.write(f"Merged config written to {output_path}\n")
    if warnings:
        sys.stdout.write(
            "\n".join(
                ["Warnings:"] + [f"  - {warning}" for warning in warnings],
            )
            + "\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
