#!/usr/bin/env python3
"""Microbenchmark the 32-parameter GAFA output layer.

This script measures only the GAFA layer: a 13-feature ridge residual head and a
17-feature ridge gate head. It uses the benchmark's actual calibration and test
point counts, but does not run a frozen foundation backbone.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_ADAPT = ROOT / "scripts" / "selective_residual_adaptation.py"


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("selective_residual_adaptation_for_microbench", RESIDUAL_ADAPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RESIDUAL_ADAPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_layer(module: Any, calibration_points: int, test_points: int, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_fit = rng.normal(size=(calibration_points, 13)).astype(np.float64)
    p_fit = rng.normal(scale=0.2, size=calibration_points).astype(np.float64)
    y_fit = p_fit + rng.normal(scale=0.05, size=calibration_points).astype(np.float64)
    u_fit = np.abs(rng.normal(scale=0.1, size=calibration_points)).astype(np.float64)

    x_test = rng.normal(size=(test_points, 13)).astype(np.float64)
    p_test = rng.normal(scale=0.2, size=test_points).astype(np.float64)
    u_test = np.abs(rng.normal(scale=0.1, size=test_points)).astype(np.float64)

    # Warm up NumPy's linear algebra path so the first setting does not absorb
    # one-time library initialization cost.
    warmup_n = min(calibration_points, 256)
    warmup_head = module.fit_residual_head(
        y_fit[:warmup_n],
        p_fit[:warmup_n],
        u_fit[:warmup_n],
        x_fit[:warmup_n],
        "residual_all",
        alpha=1.0,
        uncertainty_keep_ratio=1.0,
        residual_keep_ratio=1.0,
    )
    assert warmup_head is not None

    fit_start = time.perf_counter()
    residual = y_fit - p_fit
    residual_head = module.fit_residual_head(
        y_fit,
        p_fit,
        u_fit,
        x_fit,
        "residual_all",
        alpha=1.0,
        uncertainty_keep_ratio=1.0,
        residual_keep_ratio=1.0,
    )
    assert residual_head is not None
    correction_fit = module.predict_ridge(x_fit, residual_head["weights"], residual_head["mean"], residual_head["std"])
    gate_head = module.fit_gate_head(y_fit, p_fit, correction_fit, x_fit, u_fit, alpha=1.0)
    fit_seconds = time.perf_counter() - fit_start

    def apply_once() -> np.ndarray:
        correction_test = module.predict_ridge(x_test, residual_head["weights"], residual_head["mean"], residual_head["std"])
        return module.apply_gate_head(p_test, correction_test, x_test, u_test, gate_head, blend_ratio=1.0)

    apply_once()
    infer_start = time.perf_counter()
    for _ in range(repeats):
        pred = apply_once()
    inference_seconds = (time.perf_counter() - infer_start) / repeats
    # Keep the result live so Python cannot discard the timed computation.
    checksum = float(np.mean(pred))

    return {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "overhead_per_point_us": inference_seconds / max(1, test_points) * 1_000_000,
        "checksum": checksum,
    }


def run(args: argparse.Namespace) -> None:
    module = load_residual_module()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_path = ROOT / args.metrics_csv
    predictions_path = ROOT / args.predictions_csv
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)

    rows: list[dict[str, float | int | str]] = []
    for index, setting in enumerate(args.settings.split(",")):
        setting = setting.strip()
        if not setting:
            continue
        row = metrics[
            (metrics["model"] == args.model)
            & (metrics["mode"] == "adaptive_selective")
            & (metrics["description"] == setting)
        ]
        if row.empty:
            raise ValueError(f"No adaptive row found for {args.model} / {setting}")
        metric_row = row.iloc[0]
        pred_rows = predictions[
            (predictions["model"] == args.model)
            & (predictions["description"] == setting)
        ]
        if pred_rows.empty:
            raise ValueError(f"No prediction rows found for {args.model} / {setting}")

        calibration_points = int(metric_row["calibration_points"])
        test_points = int(len(pred_rows))
        test_windows = int(pred_rows[["site", "window_index"]].drop_duplicates().shape[0])
        result = benchmark_layer(module, calibration_points, test_points, args.repeats, args.seed + index)
        rows.append(
            {
                "setting": setting,
                "model": args.model,
                "parameters": 32,
                "calibration_points": calibration_points,
                "test_windows": test_windows,
                "test_points": test_points,
                **result,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "gafa_overhead_microbenchmark.csv", index=False)
    write_summary(df, output_root / "gafa_overhead_microbenchmark.md")
    print(df.to_string(index=False))
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# GAFA Output-Layer Microbenchmark",
        "",
        "The benchmark times the full 32-parameter residual-plus-gate layer using the paper's calibration and test point counts. Frozen backbone inference is excluded.",
        "",
        "| Setting | Params | Calibration points | Test points | Fit time (s) | Test overhead (ms) | Overhead / point (us) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.parameters} | {row.calibration_points} | {row.test_points} | "
            f"{row.fit_seconds:.4f} | {row.inference_seconds * 1000:.4f} | {row.overhead_per_point_us:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microbenchmark GAFA output-layer overhead.")
    parser.add_argument("--metrics_csv", default="carbon/exp_allscales_gated/selective_residual_metrics.csv")
    parser.add_argument("--predictions_csv", default="carbon/exp_allscales_gated/selective_residual_predictions.csv")
    parser.add_argument("--model", default="chronos_base")
    parser.add_argument("--settings", default="30天输入 -> 7天预测,60天输入 -> 14天预测,90天输入 -> 30天预测")
    parser.add_argument("--output_root", default="carbon/gafa_overhead_microbenchmark")
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
