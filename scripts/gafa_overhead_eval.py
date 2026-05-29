#!/usr/bin/env python3
"""Measure output-space overhead for GAFA.

The timer starts after frozen foundation forecasts have been produced. This
isolates the cost added by GAFA: validation selection, final ridge-head fitting,
and applying the fitted output-space heads to test forecasts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_ADAPT = ROOT / "scripts" / "selective_residual_adaptation.py"


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("selective_residual_adaptation_for_overhead", RESIDUAL_ADAPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RESIDUAL_ADAPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_final_heads(module: Any, fit_items: list[dict[str, Any]], best: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    y_fit, p_fit, u_fit, x_fit = module.flatten(fit_items)
    head = module.fit_residual_head(
        y_fit,
        p_fit,
        u_fit,
        x_fit,
        best["strategy"],
        best.get("alpha", args.ridge_alpha),
        best.get("uncertainty_keep_ratio", args.uncertainty_keep_ratio),
        best.get("residual_keep_ratio", args.residual_keep_ratio),
    )
    gate_head = None
    if head is not None and best.get("use_gate", False):
        correction_fit = module.predict_ridge(x_fit, head["weights"], head["mean"], head["std"])
        gate_head = module.fit_gate_head(
            y_fit,
            p_fit,
            correction_fit,
            x_fit,
            u_fit,
            best.get("gate_alpha", best.get("alpha", args.ridge_alpha)),
        )
    return head, gate_head


def apply_final_heads(
    module: Any,
    test_items: list[dict[str, Any]],
    best: dict[str, Any],
    head: dict[str, Any] | None,
    gate_head: dict[str, Any] | None,
) -> np.ndarray:
    _, p_test, u_test, x_test = module.flatten(test_items)
    if head is None:
        return p_test.copy()
    correction = module.predict_ridge(x_test, head["weights"], head["mean"], head["std"])
    if best.get("use_gate", False):
        return module.apply_gate_head(p_test, correction, x_test, u_test, gate_head, best["blend_ratio"])
    return p_test + best["blend_ratio"] * correction


def time_apply(
    module: Any,
    test_items: list[dict[str, Any]],
    best: dict[str, Any],
    head: dict[str, Any] | None,
    gate_head: dict[str, Any] | None,
    repeats: int,
) -> tuple[np.ndarray, float]:
    repeats = max(1, repeats)
    pred = apply_final_heads(module, test_items, best, head, gate_head)
    start = time.perf_counter()
    for _ in range(repeats):
        pred = apply_final_heads(module, test_items, best, head, gate_head)
    elapsed = (time.perf_counter() - start) / repeats
    return pred, elapsed


def run(args: argparse.Namespace) -> None:
    module = load_residual_module()
    module.set_global_seed(args.seed)
    unified = module.load_unified_module()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for attr in ["timefm_path", "chronos_path"]:
        value = Path(getattr(args, attr))
        if not value.is_absolute():
            setattr(args, attr, str(ROOT / value))

    args.models = [args.model]
    forecasters = unified.make_forecasters(args)
    rows: list[dict[str, Any]] = []

    for spec in unified.iter_specs("daily", args.custom_daily_specs):
        print(f"\n[spec] {spec.description}")
        for forecaster in forecasters:
            print(f"[model] {forecaster.name} (raw forecast generation excluded from overhead)")
            raw_start = time.perf_counter()
            split_items = module.collect_predictions(args, unified, spec, forecaster)
            raw_collection_seconds = time.perf_counter() - raw_start

            train_items = split_items["train"]
            val_items = split_items["val"]
            test_items = split_items["test"]
            fit_items = module.calibration_items_from_splits(split_items, args.calibration_splits)
            if not train_items or not val_items or not test_items:
                print("  [skip] not enough windows")
                continue

            selection_start = time.perf_counter()
            best = module.tune_strategy(train_items, val_items, args)
            selection_seconds = time.perf_counter() - selection_start

            refit_start = time.perf_counter()
            head, gate_head = fit_final_heads(module, fit_items, best, args)
            refit_seconds = time.perf_counter() - refit_start

            pred, inference_seconds = time_apply(module, test_items, best, head, gate_head, args.inference_repeats)
            y_test, p_test, _, _ = module.flatten(test_items)
            raw_metrics = module.metrics(y_test, p_test)
            aligned_metrics = module.metrics(y_test, pred)

            test_points = len(y_test)
            test_windows = len(test_items)
            selected_points = len(module.flatten(fit_items)[0]) if head is None else int(head["selected_points"])
            selected_ratio = 1.0 if head is None else float(head["selected_ratio"])
            params = 32
            row = {
                "model": forecaster.name,
                "description": spec.description,
                "context_length": spec.seq_len,
                "prediction_length": spec.pred_len,
                "strategy": best["strategy"],
                "use_gate": bool(best.get("use_gate", False)),
                "parameters": params,
                "calibration_points": len(module.flatten(fit_items)[0]),
                "selected_points": selected_points,
                "selected_ratio": selected_ratio,
                "test_windows": test_windows,
                "test_points": test_points,
                "raw_collection_seconds_excluded": raw_collection_seconds,
                "selection_seconds": selection_seconds,
                "refit_seconds": refit_seconds,
                "fit_seconds": selection_seconds + refit_seconds,
                "inference_seconds": inference_seconds,
                "overhead_per_point_us": inference_seconds / max(1, test_points) * 1_000_000,
                "overhead_per_window_ms": inference_seconds / max(1, test_windows) * 1_000,
                "raw_r2": raw_metrics["r2"],
                "aligned_r2": aligned_metrics["r2"],
                "tuning_config": json.dumps(
                    {
                        "strategy": best["strategy"],
                        "alpha": best.get("alpha"),
                        "uncertainty_keep_ratio": best.get("uncertainty_keep_ratio"),
                        "residual_keep_ratio": best.get("residual_keep_ratio"),
                        "blend_ratio": best.get("blend_ratio"),
                        "use_gate": bool(best.get("use_gate", False)),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
            rows.append(row)
            print(
                "  "
                f"strategy={row['strategy']} gate={int(row['use_gate'])} "
                f"fit={row['fit_seconds']:.4f}s infer={row['inference_seconds']:.6f}s "
                f"point={row['overhead_per_point_us']:.3f}us"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "gafa_overhead_metrics.csv", index=False)
    write_summary(df, output_root / "gafa_overhead_summary.md")
    print(f"\n[done] wrote overhead results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# GAFA Output-Space Overhead",
        "",
        "Raw foundation forecast generation is excluded. Fit time includes validation selection plus final refit.",
        "",
        "| Setting | Model | Strategy | Gate | Params | Fit time (s) | Test overhead (ms) | Overhead / point (us) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.description} | {row.model} | {row.strategy} | {int(bool(row.use_gate))} | "
            f"{row.parameters} | {row.fit_seconds:.4f} | {row.inference_seconds * 1000:.4f} | "
            f"{row.overhead_per_point_us:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure GAFA output-space fit and inference overhead.")
    parser.add_argument("--model", choices=["timefm_v2", "chronos_base", "moirai2"], default="chronos_base")
    parser.add_argument("--custom_daily_specs", default="30:7,60:14,90:30")
    parser.add_argument("--output_root", default="carbon/gafa_overhead")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=20)
    parser.add_argument("--moirai2_model_id", default="ori/moirai")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--calibration_splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--uncertainty_keep_ratio", type=float, default=0.8)
    parser.add_argument("--residual_keep_ratio", type=float, default=0.9)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--uncertainty_grid", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--residual_grid", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--blend_grid", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--use_gate_grid", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--site_guard_grid", nargs="+", type=int, default=[0])
    parser.add_argument("--min_relative_gain", type=float, default=0.03)
    parser.add_argument("--site_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--inference_repeats", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
