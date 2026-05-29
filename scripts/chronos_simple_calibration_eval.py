#!/usr/bin/env python3
"""Conservative Chronos calibration experiments for the carbon benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SELECTIVE = ROOT / "scripts" / "selective_residual_adaptation.py"


def load_selective_module() -> Any:
    spec = importlib.util.spec_from_file_location("selective_residual_adaptation", SELECTIVE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SELECTIVE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "nse": r2}


def fit_affine(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(2) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def apply_affine(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights[0] + weights[1] * x


def fit_site_bias(y_true: np.ndarray, y_pred: np.ndarray, sites: np.ndarray, shrink: float) -> tuple[dict[str, float], float]:
    residual = y_true - y_pred
    global_bias = float(residual.mean())
    site_bias: dict[str, float] = {}
    for site in np.unique(sites):
        mask = sites == site
        n = int(mask.sum())
        local = float(residual[mask].mean())
        weight = n / (n + shrink)
        site_bias[str(site)] = weight * local + (1.0 - weight) * global_bias
    return site_bias, global_bias


def apply_site_bias(y_pred: np.ndarray, sites: np.ndarray, site_bias: dict[str, float], global_bias: float, blend: float) -> np.ndarray:
    correction = np.array([site_bias.get(str(site), global_bias) for site in sites], dtype=np.float64)
    return y_pred + blend * correction


def fit_site_affine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    alpha: float,
    shrink: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    global_weights = fit_affine(y_pred, y_true, alpha)
    site_weights: dict[str, np.ndarray] = {}
    for site in np.unique(sites):
        mask = sites == site
        n = int(mask.sum())
        local = fit_affine(y_pred[mask], y_true[mask], alpha)
        weight = n / (n + shrink)
        site_weights[str(site)] = weight * local + (1.0 - weight) * global_weights
    return site_weights, global_weights


def apply_site_affine(y_pred: np.ndarray, sites: np.ndarray, site_weights: dict[str, np.ndarray], global_weights: np.ndarray, blend: float) -> np.ndarray:
    calibrated = np.empty_like(y_pred, dtype=np.float64)
    for site in np.unique(sites):
        mask = sites == site
        weights = site_weights.get(str(site), global_weights)
        calibrated[mask] = apply_affine(y_pred[mask], weights)
    return y_pred + blend * (calibrated - y_pred)


def tune(train: dict[str, np.ndarray], val: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = [{"strategy": "raw"}]
    for blend in args.blend_grid:
        candidates.append({"strategy": "global_bias", "blend": blend})
        for shrink in args.shrink_grid:
            candidates.append({"strategy": "site_bias", "blend": blend, "shrink": shrink})
            for alpha in args.alpha_grid:
                candidates.append({"strategy": "site_affine", "blend": blend, "shrink": shrink, "alpha": alpha})
        for alpha in args.alpha_grid:
            candidates.append({"strategy": "global_affine", "blend": blend, "alpha": alpha})

    best: dict[str, Any] | None = None
    for candidate in candidates:
        pred_val = predict_candidate(train, val, candidate)
        val_metrics = metric_values(val["y"], pred_val)
        current = {**candidate, "val_metrics": val_metrics}
        if best is None or val_metrics["mse"] < best["val_metrics"]["mse"] - 1e-12:
            best = current
    assert best is not None
    return best


def predict_candidate(fit: dict[str, np.ndarray], target: dict[str, np.ndarray], candidate: dict[str, Any]) -> np.ndarray:
    strategy = candidate["strategy"]
    if strategy == "raw":
        return target["p"].copy()
    if strategy == "global_bias":
        bias = float(np.mean(fit["y"] - fit["p"]))
        return target["p"] + candidate["blend"] * bias
    if strategy == "site_bias":
        site_bias, global_bias = fit_site_bias(fit["y"], fit["p"], fit["sites"], candidate["shrink"])
        return apply_site_bias(target["p"], target["sites"], site_bias, global_bias, candidate["blend"])
    if strategy == "global_affine":
        weights = fit_affine(fit["p"], fit["y"], candidate["alpha"])
        calibrated = apply_affine(target["p"], weights)
        return target["p"] + candidate["blend"] * (calibrated - target["p"])
    if strategy == "site_affine":
        site_weights, global_weights = fit_site_affine(fit["y"], fit["p"], fit["sites"], candidate["alpha"], candidate["shrink"])
        return apply_site_affine(target["p"], target["sites"], site_weights, global_weights, candidate["blend"])
    raise ValueError(f"Unknown strategy: {strategy}")


def as_arrays(module: Any, items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    y, p, _u, _x = module.flatten(items)
    return {"y": y, "p": p, "sites": module.point_sites(items)}


def run(args: argparse.Namespace) -> None:
    module = load_selective_module()
    module.set_global_seed(args.seed)
    unified = module.load_unified_module()

    if not Path(args.chronos_path).is_absolute():
        args.chronos_path = str(ROOT / args.chronos_path)
    if not Path(args.timefm_path).is_absolute():
        args.timefm_path = str(ROOT / args.timefm_path)

    args.models = ["chronos_base"]
    spec = unified.iter_specs("daily", "365:60")[0]
    forecaster = unified.make_forecasters(args)[0]
    split_items = module.collect_predictions(args, unified, spec, forecaster)

    train = as_arrays(module, split_items["train"])
    val = as_arrays(module, split_items["val"])
    test = as_arrays(module, split_items["test"])
    fit = as_arrays(module, module.combine_items(split_items["train"], split_items["val"]))

    best = tune(train, val, args)
    raw_metrics = metric_values(test["y"], test["p"])
    calibrated = predict_candidate(fit, test, best)
    calibrated_metrics = metric_values(test["y"], calibrated)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [
        {"model": "chronos_base", "mode": "foundation_raw", "tuning_config": "{}", **raw_metrics},
        {
            "model": "chronos_base",
            "mode": "simple_calibrated",
            "tuning_config": json.dumps(best, ensure_ascii=True, default=float, sort_keys=True),
            **calibrated_metrics,
        },
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_root / "simple_calibration_metrics.csv", index=False)

    pred_rows = pd.DataFrame(
        {
            "site": test["sites"],
            "y_true": test["y"],
            "foundation_raw": test["p"],
            "simple_calibrated": calibrated,
        }
    )
    pred_rows.to_csv(output_root / "simple_calibration_predictions.csv", index=False)

    lines = [
        "# Chronos Simple Calibration",
        "",
        f"Best validation strategy: `{best['strategy']}`",
        "",
        "| Mode | MSE | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.mode} | {row.mse:.6f} | {row.mae:.6f} | {row.rmse:.6f} | {row.r2:.6f} |")
    (output_root / "simple_calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conservative Chronos calibration on 365 -> 60.")
    parser.add_argument("--output_root", default="carbon/chronos_simple_calibration_365_60")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=20)
    parser.add_argument("--moirai2_model_id", default="ori/moirai")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--blend_grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--shrink_grid", nargs="+", type=float, default=[0.0, 30.0, 60.0, 120.0, 300.0])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
