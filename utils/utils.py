import argparse
import ast
import yaml
from dotmap import DotMap
from distutils.util import strtobool
from pathlib import Path
from typing import Any
import re
import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_SIMPLE4_DATASETS = ["live", "csiq", "tid2013", "kadid10k"]


def _gather_bool_paths(section, prefix=""):
    """Collect dotted key paths that are booleans in the canonical config."""
    bool_paths = set()
    if isinstance(section, dict):
        for key, value in section.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            bool_paths.update(_gather_bool_paths(value, new_prefix))
    elif isinstance(section, list):
        # Lists are keyed by their parent name; we don't track indices here.
        for item in section:
            bool_paths.update(_gather_bool_paths(item, prefix))
    elif isinstance(section, bool):
        bool_paths.add(prefix)
    return bool_paths


def _normalize_scalar(value, path="", bool_paths=None):
    if bool_paths is None:
        bool_paths = set()

    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"true", "false", "yes", "no"}:
            as_bool = bool(strtobool(lowered))
            return as_bool
        if _NUMERIC_RE.match(stripped):
            try:
                numeric = float(stripped)
                if numeric.is_integer():
                    numeric_int = int(numeric)
                    if path in bool_paths:
                        return bool(numeric_int)
                    return numeric_int
                return numeric
            except ValueError:
                return value
        return value

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if path in bool_paths:
            return bool(value)
        return value
    if isinstance(value, float):
        if path in bool_paths:
            return bool(value)
        return value
    return value


def _normalize_types(section, prefix="", bool_paths=None):
    """Recursively normalize strings and numeric-like values to the expected types."""
    if bool_paths is None:
        bool_paths = set()

    if isinstance(section, dict):
        return {
            key: _normalize_types(
                value, f"{prefix}.{key}" if prefix else key, bool_paths
            )
            for key, value in section.items()
        }
    if isinstance(section, list):
        return [
            _normalize_types(value, prefix, bool_paths)
            for value in section
        ]
    return _normalize_scalar(section, prefix, bool_paths)


def parse_config(config_file_path: str) -> DotMap:
    """Parse the YAML configuration file"""
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    canonical_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    canonical_bool_paths = (
        _gather_bool_paths(yaml.safe_load(canonical_config_path.read_text()))
        if canonical_config_path.exists()
        else set()
    )

    normalized = _normalize_types(config, bool_paths=canonical_bool_paths)
    return DotMap(normalized, _dynamic=False)

def _ensure_cross_eval_defaults(config: DotMap) -> None:
    if "test" not in config:
        return

    test_section = config["test"]
    if not isinstance(test_section, DotMap):
        return

    if "cross_eval" not in test_section:
        test_section["cross_eval"] = DotMap(
            {
                "enabled": False,
                "include_standard": True,
                "train_datasets": list(_SIMPLE4_DATASETS),
                "test_datasets": list(_SIMPLE4_DATASETS),
            },
            _dynamic=False,
        )
    else:
        cross_eval = test_section["cross_eval"]
        if not isinstance(cross_eval, DotMap):
            cross_eval = DotMap(cross_eval, _dynamic=False)
            test_section["cross_eval"] = cross_eval
        cross_eval.setdefault("enabled", False)
        cross_eval.setdefault("include_standard", True)
        cross_eval.setdefault("train_datasets", list(_SIMPLE4_DATASETS))
        cross_eval.setdefault("test_datasets", list(_SIMPLE4_DATASETS))


def parse_command_line_args(config: DotMap) -> DotMap:
    """Parse the command-line arguments"""
    parser = argparse.ArgumentParser()

    _ensure_cross_eval_defaults(config)

    def _coerce_value(value):
        # Parse CLI strings into scalar Python values where possible.
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        lowered = stripped.lower()

        # Allow W&B agent style list overrides such as "['live', 'csiq']".
        if (
            (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("(") and stripped.endswith(")"))
        ):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, tuple):
                parsed = list(parsed)
            if isinstance(parsed, list):
                return [_coerce_value(item) for item in parsed]

        if lowered in {"true", "false"}:
            return bool(strtobool(lowered))
        if lowered in {"none", "null"}:
            return None
        try:
            if "." in stripped:
                float_value = float(stripped)
                # Preserve integers that arrive as floats like "1.0"
                if float_value.is_integer():
                    return int(float_value)
                return float_value
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return value

    def _int_or_bool(val):
        # Allow integer-like toggles to be set via True/False as well as 1/0
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
        try:
            return int(strtobool(str(val)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid value for int/bool argument: {val}"
            ) from exc

    def _infer_type(value):
        if isinstance(value, bool):
            return strtobool
        if isinstance(value, int):
            return _int_or_bool
        if value is None:
            # Avoid argparse.NoneType conversion failures for sweep overrides.
            return _coerce_value
        return type(value)

    # Automatically add command-line arguments based on the config structure
    def add_arguments(section, prefix=""):
        for key, value in section.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                # Check if the value is a list
                if isinstance(value, list):
                    elem_type = _infer_type(value[0]) if len(value) > 0 else str
                    parser.add_argument(
                        f"--{full_key}",
                        default=value,
                        type=elem_type,
                        nargs="+",
                        help=f"Value for {full_key}",
                    )
                else:
                    parser.add_argument(
                        f"--{full_key}",
                        default=value,
                        type=_infer_type(value),
                        help=f"Value for {full_key}",
                    )

    add_arguments(config)

    def _set_arg(target: DotMap, dotted_key: str, value):
        keys = dotted_key.split(".")
        current = target
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], DotMap):
                current[key] = DotMap(_dynamic=True)
            else:
                # Allow nested updates on non-dynamic DotMaps
                current[key]._dynamic = True
            current = current[key]
        current[keys[-1]] = value

    def _normalize_cli_collections(value):
        if isinstance(value, list):
            # argparse list options with a single list-literal token come in as:
            # ["['live', 'csiq', ...]"].
            if len(value) == 1 and isinstance(value[0], str):
                parsed_single = _coerce_value(value[0])
                if isinstance(parsed_single, list):
                    return [_normalize_cli_collections(item) for item in parsed_single]
            return [_normalize_cli_collections(item) for item in value]
        if isinstance(value, dict):
            return {k: _normalize_cli_collections(v) for k, v in value.items()}
        return value

    parsed_args, unknown_args = parser.parse_known_args()
    normalized_vars = {
        key: _normalize_cli_collections(value)
        for key, value in vars(parsed_args).items()
    }
    args = DotMap(normalized_vars, _dynamic=True)

    # Normalize argparse keys with dots into nested DotMaps for consistency
    for key, value in list(args.items()):
        if isinstance(key, str) and "." in key:
            _set_arg(args, key, value)
            del args[key]

    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if not token.startswith("--"):
            raise ValueError(f"Invalid command-line argument format: {token}")

        key_value = token[2:]
        if "=" in key_value:
            key, initial_value = key_value.split("=", 1)
            values = [initial_value]
            idx += 1
        else:
            key = key_value
            values = []
            idx += 1
            while idx < len(unknown_args) and not unknown_args[idx].startswith("--"):
                values.append(unknown_args[idx])
                idx += 1

        if not values:
            coerced_value = True
        elif len(values) == 1:
            coerced_value = _coerce_value(values[0])
        else:
            coerced_value = [_coerce_value(v) for v in values]

        _set_arg(args, key, coerced_value)

    return args


def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """Merge the command-line arguments into the config. The command-line arguments take precedence over the config file
    :rtype: object
    """
    def flatten(dot_map: DotMap, prefix=""):
        items = []
        for key, value in dot_map.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, DotMap):
                items.extend(flatten(value, full_key))
            else:
                items.append((full_key, value))
        return items

    def set_value(target: DotMap, dotted_key: str, value):
        keys = dotted_key.split(".")
        current = target
        for idx, key in enumerate(keys):
            is_last = idx == len(keys) - 1
            if is_last:
                current[key] = value
            else:
                if key not in current or not isinstance(current[key], DotMap):
                    current[key] = DotMap(_dynamic=True)
                else:
                    current[key]._dynamic = True
                current = current[key]

    original_dynamic = getattr(config, "_dynamic", True)
    config._dynamic = True

    for dotted_key, value in flatten(args):
        if dotted_key in {"config", "eval_type"}:
            continue
        set_value(config, dotted_key, value)

    config._dynamic = original_dynamic

    return config


def replace_none_string(args):
    for key, value in args.items():
        if isinstance(value, str) and value == "None":
            args[key] = None
        elif isinstance(value, DotMap):
            replace_none_string(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str) and item == "None":
                    value[i] = None
                elif isinstance(item, DotMap):
                    replace_none_string(item)

    return args


def prepare_wandb_config(config: Any) -> dict:
    """
    Convert a nested config (DotMap/dict) into a plain, wandb-friendly dict with
    JSON-serializable values. Non-primitive types are coerced to strings to avoid
    serialization errors introduced in newer wandb versions.
    """

    def _convert(value: Any):
        if isinstance(value, DotMap):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        return str(value)

    return _convert(config)
