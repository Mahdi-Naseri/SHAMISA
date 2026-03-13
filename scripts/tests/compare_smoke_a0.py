#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing smoke summary: {p}")
    return json.loads(p.read_text())


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs cleaned SHAMISA smoke outputs.")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON summary")
    parser.add_argument("--cleaned", required=True, help="Path to cleaned JSON summary")
    parser.add_argument(
        "--metric-tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for metric deltas",
    )
    parser.add_argument(
        "--allow-checkpoint-sha-mismatch",
        action="store_true",
        help="Do not fail if checkpoint SHA256 differs",
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    cleaned = _load(args.cleaned)

    compare_fields = [
        "global_avg_srocc",
        "global_avg_plcc",
        "val_avg_srocc",
        "val_avg_plcc",
    ]

    failures: list[str] = []
    print("Smoke Comparison")
    print(f"- baseline: {args.baseline}")
    print(f"- cleaned:  {args.cleaned}")
    print(f"- metric_tolerance: {args.metric_tol}")

    for field in compare_fields:
        b = float(baseline.get(field, float("nan")))
        c = float(cleaned.get(field, float("nan")))
        if math.isnan(b) or math.isnan(c):
            failures.append(f"{field}: nan encountered (baseline={b}, cleaned={c})")
            continue
        delta = abs(b - c)
        print(f"- {field}: baseline={_fmt(b)} cleaned={_fmt(c)} delta={delta:.6g}")
        if delta > args.metric_tol:
            failures.append(
                f"{field}: delta {delta:.6g} exceeds tolerance {args.metric_tol:.6g}"
            )

    b_sha = baseline.get("best_checkpoint_sha256")
    c_sha = cleaned.get("best_checkpoint_sha256")
    if not args.allow_checkpoint_sha_mismatch:
        if not b_sha or not c_sha:
            failures.append("best_checkpoint_sha256 missing in baseline/cleaned summary")
        elif b_sha != c_sha:
            failures.append("best_checkpoint_sha256 mismatch")
    print(f"- best_checkpoint_sha256 baseline={b_sha} cleaned={c_sha}")

    if failures:
        print("\nResult: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
