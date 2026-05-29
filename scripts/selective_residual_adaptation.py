#!/usr/bin/env python3
"""Selective residual adaptation for time-series foundation models on carbon NEE.

The foundation model is kept frozen. We fit a small ridge residual head on
calibration windows:

    final_prediction = foundation_prediction + residual_head(features)

The residual head is trained only on selected calibration timesteps with lower
model uncertainty and non-extreme residuals, following the practical spirit of
selective learning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_EVAL = ROOT / "scripts" / "unified_foundation_eval.py"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_unified_module() -> Any:
    spec = importlib.util.spec_from_file_location("unified_foundation_eval", UNIFIED_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {UNIFIED_EVAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(x), 1), dtype=np.float64), x], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = standardize_fit(x)
    xs = (x - mean) / std
    xb = add_bias(xs)
    reg = np.eye(xb.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return weights, mean, std


def predict_ridge(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x - mean) / std
    return add_bias(xs) @ weights


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "nse": r2}


def build_features(context: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, spec: Any) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    steps = np.arange(1, len(pred) + 1, dtype=np.float64)
    step_norm = steps / max(1, len(pred))
    period = max(1, int(spec.period))
    seasonal = np.resize(context[-period:], len(pred)) if len(context) >= period else np.repeat(context[-1], len(pred))
    last_value = np.repeat(context[-1], len(pred))
    ctx_mean = np.repeat(context.mean(), len(pred))
    ctx_std = np.repeat(context.std(), len(pred))
    recent = context[-min(len(context), period) :]
    recent_mean = np.repeat(recent.mean(), len(pred))
    recent_std = np.repeat(recent.std(), len(pred))
    return np.column_stack(
        [
            pred,
            uncertainty,
            step_norm,
            np.sin(2 * np.pi * step_norm),
            np.cos(2 * np.pi * step_norm),
            last_value,
            seasonal,
            pred - seasonal,
            pred - last_value,
            ctx_mean,
            ctx_std,
            recent_mean,
            recent_std,
        ]
    )


def selection_mask(
    uncertainty: np.ndarray,
    residual: np.ndarray,
    uncertainty_keep_ratio: float,
    residual_keep_ratio: float,
) -> np.ndarray:
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if np.allclose(uncertainty, uncertainty[0]):
        uncertainty_ok = np.ones_like(uncertainty, dtype=bool)
    else:
        uncertainty_ok = uncertainty <= np.quantile(uncertainty, uncertainty_keep_ratio)
    residual_abs = np.abs(residual)
    residual_ok = residual_abs <= np.quantile(residual_abs, residual_keep_ratio)
    mask = uncertainty_ok & residual_ok
    if mask.sum() < max(32, int(0.05 * len(mask))):
        # Avoid an unstable residual head when selection is too aggressive.
        mask = residual_ok
    return mask


def collect_predictions(args: argparse.Namespace, unified: Any, spec: Any, forecaster: Any) -> dict[str, list[dict[str, Any]]]:
    site_data = unified.load_clean(spec.data_path)
    sites = sorted(site_data)
    if args.sites:
        wanted = set(args.sites)
        sites = [s for s in sites if s in wanted]
    if args.max_sites:
        sites = sites[: args.max_sites]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for site in sites:
        windows = unified.build_windows(site_data[site], spec, args.stride or spec.pred_len, args.max_windows_per_site)
        if len(windows) < 3:
            continue
        for idx, window in enumerate(windows):
            split_name = unified.split_name_for_index(idx, len(windows), args.train_ratio, args.val_ratio)
            pred, unc = forecaster.predict(window["context"], spec)
            features = build_features(window["context"], pred, unc, spec)
            item = {
                "site": site,
                "window_index": idx,
                "dates": window["dates"],
                "target": window["target"].astype(np.float64),
                "pred": pred.astype(np.float64),
                "uncertainty": unc.astype(np.float64),
                "features": features,
            }
            splits[split_name].append(item)
    return splits


def flatten(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.concatenate([x["target"] for x in items])
    y_pred = np.concatenate([x["pred"] for x in items])
    uncertainty = np.concatenate([x["uncertainty"] for x in items])
    features = np.concatenate([x["features"] for x in items], axis=0)
    return y_true, y_pred, uncertainty, features


def point_sites(items: list[dict[str, Any]]) -> np.ndarray:
    return np.concatenate([np.repeat(str(x["site"]), len(x["target"])) for x in items])


def apply_site_guard(
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    adaptive_pred: np.ndarray,
    sites: np.ndarray,
    min_relative_gain: float,
) -> tuple[np.ndarray, set[str]]:
    guarded = raw_pred.copy()
    allowed_sites: set[str] = set()
    for site in np.unique(sites):
        mask = sites == site
        raw_mse = float(np.mean((raw_pred[mask] - y_true[mask]) ** 2))
        adaptive_mse = float(np.mean((adaptive_pred[mask] - y_true[mask]) ** 2))
        relative_gain = (raw_mse - adaptive_mse) / max(abs(raw_mse), 1e-12)
        if relative_gain >= min_relative_gain:
            guarded[mask] = adaptive_pred[mask]
            allowed_sites.add(str(site))
    return guarded, allowed_sites


def apply_allowed_site_mask(
    raw_pred: np.ndarray,
    adaptive_pred: np.ndarray,
    sites: np.ndarray,
    allowed_sites: set[str],
) -> np.ndarray:
    if not allowed_sites:
        return raw_pred.copy()
    guarded = raw_pred.copy()
    mask = np.isin(sites, list(allowed_sites))
    guarded[mask] = adaptive_pred[mask]
    return guarded


def combine_items(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        out.extend(group)
    return out


def calibration_items_from_splits(
    split_items: dict[str, list[dict[str, Any]]],
    split_names: list[str],
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for split_name in split_names:
        if split_name not in split_items:
            raise ValueError(f"Unknown calibration split: {split_name}")
        groups.append(split_items[split_name])
    return combine_items(*groups)


def fit_residual_head(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    features: np.ndarray,
    strategy: str,
    alpha: float,
    uncertainty_keep_ratio: float,
    residual_keep_ratio: float,
) -> dict[str, Any] | None:
    if strategy == "foundation_raw":
        return None

    residual = y_true - y_pred
    if strategy == "residual_all":
        mask = np.ones_like(residual, dtype=bool)
    elif strategy == "selective_residual":
        mask = selection_mask(uncertainty, residual, uncertainty_keep_ratio, residual_keep_ratio)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    weights, mean, std = fit_ridge(features[mask], residual[mask], alpha)
    return {
        "strategy": strategy,
        "alpha": float(alpha),
        "uncertainty_keep_ratio": float(uncertainty_keep_ratio),
        "residual_keep_ratio": float(residual_keep_ratio),
        "selected_points": int(mask.sum()),
        "selected_ratio": float(mask.mean()),
        "weights": weights,
        "mean": mean,
        "std": std,
    }


def predict_with_head(
    base_pred: np.ndarray,
    features: np.ndarray,
    head: dict[str, Any] | None,
    blend_ratio: float,
) -> np.ndarray:
    if head is None:
        return base_pred.copy()
    correction = predict_ridge(features, head["weights"], head["mean"], head["std"])
    return base_pred + blend_ratio * correction


def build_gate_features(features: np.ndarray, correction: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.float64).reshape(-1, 1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1, 1)
    return np.concatenate(
        [
            features,
            correction,
            np.abs(correction),
            uncertainty,
            np.abs(correction) * uncertainty,
        ],
        axis=1,
    )


def fit_gate_head(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    correction: np.ndarray,
    features: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float,
) -> dict[str, Any] | None:
    correction = np.asarray(correction, dtype=np.float64)
    usable = np.abs(correction) > 1e-8
    if usable.sum() < max(32, int(0.05 * len(correction))):
        return None

    target_blend = np.zeros_like(correction, dtype=np.float64)
    target_blend[usable] = (y_true[usable] - base_pred[usable]) / correction[usable]
    target_blend = np.clip(target_blend, 0.0, 1.0)

    gate_x = build_gate_features(features[usable], correction[usable], uncertainty[usable])
    weights, mean, std = fit_ridge(gate_x, target_blend[usable], alpha)
    return {"weights": weights, "mean": mean, "std": std}


def apply_gate_head(
    base_pred: np.ndarray,
    correction: np.ndarray,
    features: np.ndarray,
    uncertainty: np.ndarray,
    gate_head: dict[str, Any] | None,
    blend_ratio: float,
) -> np.ndarray:
    if gate_head is None:
        return base_pred + blend_ratio * correction
    gate_x = build_gate_features(features, correction, uncertainty)
    gate = predict_ridge(gate_x, gate_head["weights"], gate_head["mean"], gate_head["std"])
    gate = np.clip(gate, 0.0, 1.0)
    return base_pred + blend_ratio * gate * correction


def iter_candidate_params(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{"strategy": "foundation_raw", "blend_ratio": 0.0}]
    for alpha in args.alpha_grid:
        for blend_ratio in args.blend_grid:
            for use_gate in args.use_gate_grid:
                for site_guard in args.site_guard_grid:
                    candidates.append(
                        {
                            "strategy": "residual_all",
                            "alpha": float(alpha),
                            "uncertainty_keep_ratio": 1.0,
                            "residual_keep_ratio": 1.0,
                            "blend_ratio": float(blend_ratio),
                            "use_gate": bool(use_gate),
                            "site_guard": bool(site_guard),
                        }
                    )
    for alpha in args.alpha_grid:
        for uncertainty_keep_ratio in args.uncertainty_grid:
            for residual_keep_ratio in args.residual_grid:
                for blend_ratio in args.blend_grid:
                    for use_gate in args.use_gate_grid:
                        for site_guard in args.site_guard_grid:
                            candidates.append(
                                {
                                    "strategy": "selective_residual",
                                    "alpha": float(alpha),
                                    "uncertainty_keep_ratio": float(uncertainty_keep_ratio),
                                    "residual_keep_ratio": float(residual_keep_ratio),
                                    "blend_ratio": float(blend_ratio),
                                    "use_gate": bool(use_gate),
                                    "site_guard": bool(site_guard),
                                }
                            )
    return candidates


def tune_strategy(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    y_train, p_train, u_train, x_train = flatten(train_items)
    y_val, p_val, u_val, x_val = flatten(val_items)
    val_sites = point_sites(val_items)

    best: dict[str, Any] | None = None
    raw_candidate: dict[str, Any] | None = None
    for candidate in iter_candidate_params(args):
        strategy = candidate["strategy"]
        head = fit_residual_head(
            y_train,
            p_train,
            u_train,
            x_train,
            strategy,
            candidate.get("alpha", args.ridge_alpha),
            candidate.get("uncertainty_keep_ratio", args.uncertainty_keep_ratio),
            candidate.get("residual_keep_ratio", args.residual_keep_ratio),
        )
        gate_head = None
        if head is not None:
            correction_train = predict_ridge(x_train, head["weights"], head["mean"], head["std"])
            gate_head = fit_gate_head(
                y_train,
                p_train,
                correction_train,
                x_train,
                u_train,
                candidate.get("gate_alpha", candidate.get("alpha", args.ridge_alpha)),
            )
        allowed_sites = None
        if head is None:
            pred_val = p_val.copy()
        else:
            correction_val = predict_ridge(x_val, head["weights"], head["mean"], head["std"])
            if candidate.get("use_gate", False):
                pred_val = apply_gate_head(p_val, correction_val, x_val, u_val, gate_head, candidate["blend_ratio"])
            else:
                pred_val = p_val + candidate["blend_ratio"] * correction_val
            if candidate.get("site_guard", False):
                pred_val, allowed_sites = apply_site_guard(
                    y_val,
                    p_val,
                    pred_val,
                    val_sites,
                    args.site_guard_min_relative_gain,
                )
        val_metrics = metrics(y_val, pred_val)
        current = {
            **candidate,
            "head": head,
            "gate_head": gate_head,
            "val_metrics": val_metrics,
            "val_pred": pred_val,
            "allowed_sites": allowed_sites if head is not None and candidate.get("site_guard", False) else None,
        }
        if strategy == "foundation_raw":
            raw_candidate = current
        if best is None:
            best = current
            continue
        if val_metrics["mse"] < best["val_metrics"]["mse"] - 1e-12:
            best = current
            continue
        if abs(val_metrics["mse"] - best["val_metrics"]["mse"]) <= 1e-12 and val_metrics["mae"] < best["val_metrics"]["mae"]:
            best = current
    assert best is not None
    assert raw_candidate is not None
    raw_mse = raw_candidate["val_metrics"]["mse"]
    best_mse = best["val_metrics"]["mse"]
    if best["strategy"] != "foundation_raw":
        relative_gain = (raw_mse - best_mse) / max(abs(raw_mse), 1e-12)
        if relative_gain < args.min_relative_gain:
            return raw_candidate
    return best


def refit_and_predict(
    fit_items: list[dict[str, Any]],
    target_items: list[dict[str, Any]],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    y_fit, p_fit, u_fit, x_fit = flatten(fit_items)
    _, p_target, _, x_target = flatten(target_items)
    target_sites = point_sites(target_items)
    head = fit_residual_head(
        y_fit,
        p_fit,
        u_fit,
        x_fit,
        strategy_config["strategy"],
        strategy_config.get("alpha", args.ridge_alpha),
        strategy_config.get("uncertainty_keep_ratio", args.uncertainty_keep_ratio),
        strategy_config.get("residual_keep_ratio", args.residual_keep_ratio),
    )
    if head is None:
        return p_target.copy(), head
    correction_fit = predict_ridge(x_fit, head["weights"], head["mean"], head["std"])
    gate_head = None
    if strategy_config.get("use_gate", False):
        gate_head = fit_gate_head(
            y_fit,
            p_fit,
            correction_fit,
            x_fit,
            u_fit,
            strategy_config.get("gate_alpha", strategy_config.get("alpha", args.ridge_alpha)),
        )
    correction_target = predict_ridge(x_target, head["weights"], head["mean"], head["std"])
    _, _, u_target, _ = flatten(target_items)
    if strategy_config.get("use_gate", False):
        pred_target = apply_gate_head(
            p_target,
            correction_target,
            x_target,
            u_target,
            gate_head,
            strategy_config["blend_ratio"],
        )
    else:
        pred_target = p_target + strategy_config["blend_ratio"] * correction_target
    if strategy_config.get("site_guard", False):
        pred_target = apply_allowed_site_mask(
            p_target,
            pred_target,
            target_sites,
            strategy_config.get("allowed_sites") or set(),
        )
    if gate_head is not None:
        head = {**head, "gate_head": gate_head}
    return pred_target, head


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    unified = load_unified_module()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    for attr in ["timefm_path", "chronos_path"]:
        value = Path(getattr(args, attr))
        if not value.is_absolute():
            setattr(args, attr, str(ROOT / value))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    specs = unified.iter_specs(args.task, args.custom_daily_specs)
    forecasters = unified.make_forecasters(args)

    for spec in specs:
        print(f"\n[spec] {spec.task} | {spec.description}")
        for forecaster in forecasters:
            if forecaster.name not in {"timefm_v2", "chronos_base", "moirai2"}:
                continue
            print(f"[model] {forecaster.name}")
            split_items = collect_predictions(args, unified, spec, forecaster)
            train_items = split_items["train"]
            val_items = split_items["val"]
            test_items = split_items["test"]
            if not train_items or not val_items or not test_items:
                print("  [skip] not enough windows")
                continue

            y_train, p_train, u_train, x_train = flatten(train_items)
            y_val, p_val, _, _ = flatten(val_items)
            y_test, p_test, u_test, x_test = flatten(test_items)

            raw_val_metrics = metrics(y_val, p_val)
            raw_metrics = metrics(y_test, p_test)
            rows.append(
                {
                    "model": forecaster.name,
                    "task": spec.task,
                    "description": spec.description,
                    "mode": "foundation_raw",
                    "calibration_points": len(y_train),
                    "selected_points": len(y_train),
                    "selected_ratio": 1.0,
                    "blend_ratio": 0.0,
                    "selected_strategy": "foundation_raw",
                    "val_mse": raw_val_metrics["mse"],
                    "val_r2": raw_val_metrics["r2"],
                    "tuning_config": "{}",
                    **raw_metrics,
                }
            )

            best = tune_strategy(train_items, val_items, args)
            fit_items = calibration_items_from_splits(split_items, args.calibration_splits)
            adaptive_pred, adaptive_head = refit_and_predict(fit_items, test_items, best, args)
            adaptive_metrics = metrics(y_test, adaptive_pred)
            selected_points = len(y_train) if adaptive_head is None else adaptive_head["selected_points"]
            selected_ratio = 1.0 if adaptive_head is None else adaptive_head["selected_ratio"]
            rows.append(
                {
                    "model": forecaster.name,
                    "task": spec.task,
                    "description": spec.description,
                    "mode": "adaptive_selective",
                    "calibration_points": len(flatten(fit_items)[0]),
                    "selected_points": selected_points,
                    "selected_ratio": selected_ratio,
                    "blend_ratio": float(best["blend_ratio"]),
                    "selected_strategy": best["strategy"],
                    "use_gate": bool(best.get("use_gate", False)),
                    "site_guard": bool(best.get("site_guard", False)),
                    "val_mse": float(best["val_metrics"]["mse"]),
                    "val_r2": float(best["val_metrics"]["r2"]),
                    "raw_val_mse": float(raw_val_metrics["mse"]),
                    "raw_val_r2": float(raw_val_metrics["r2"]),
                    "tuning_config": json.dumps(
                        {
                            "strategy": best["strategy"],
                            "alpha": best.get("alpha"),
                            "uncertainty_keep_ratio": best.get("uncertainty_keep_ratio"),
                            "residual_keep_ratio": best.get("residual_keep_ratio"),
                            "blend_ratio": best["blend_ratio"],
                            "use_gate": bool(best.get("use_gate", False)),
                            "site_guard": bool(best.get("site_guard", False)),
                            "allowed_site_count": len(best.get("allowed_sites") or []),
                            "calibration_splits": args.calibration_splits,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    **adaptive_metrics,
                }
            )
            print(
                "  "
                f"best={best['strategy']} blend={best['blend_ratio']:.2f} "
                f"gate={int(bool(best.get('use_gate', False)))} "
                f"site_guard={int(bool(best.get('site_guard', False)))} "
                f"val_mse={best['val_metrics']['mse']:.4f} "
                f"test_mse={adaptive_metrics['mse']:.4f} test_r2={adaptive_metrics['r2']:.4f}"
            )

            if args.save_predictions:
                offset = 0
                for item in test_items:
                    n = len(item["target"])
                    raw = p_test[offset : offset + n]
                    adaptive = adaptive_pred[offset : offset + n]
                    for step, (date, y, raw_p, adaptive_p, unc) in enumerate(
                        zip(item["dates"], item["target"], raw, adaptive, item["uncertainty"]),
                        start=1,
                    ):
                        prediction_rows.append(
                            {
                                "model": forecaster.name,
                                "task": spec.task,
                                "description": spec.description,
                                "site": item["site"],
                                "window_index": item["window_index"],
                                "step": step,
                                "date": date,
                                "y_true": float(y),
                                "foundation_raw": float(raw_p),
                                "adaptive_selective": float(adaptive_p),
                                "uncertainty": float(unc),
                            }
                        )
                    offset += n

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "selective_residual_metrics.csv", index=False)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_root / "selective_residual_predictions.csv", index=False)
    write_summary(results, output_root / "selective_residual_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Selective Residual Adaptation Results", ""]
    if df.empty:
        lines.append("No results.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
            "Foundation model frozen. The new `adaptive_selective` mode tunes strategy on the validation split and then refits on train+val.",
            "Candidate strategies include `foundation_raw`, `residual_all`, and `selective_residual`, each with blend-ratio and ridge regularization search.",
            "",
            "| Task | Description | Model | Mode | Selected Strategy | Gate | Blend | Selected Ratio | MSE | MAE | RMSE | R2 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in df.sort_values(["task", "description", "model", "mse"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.description} | {row.model} | {row.mode} | "
            f"{getattr(row, 'selected_strategy', '')} | {int(bool(getattr(row, 'use_gate', False)))} | "
            f"{getattr(row, 'blend_ratio', 0.0):.2f} | "
            f"{row.selected_ratio:.3f} | {row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective residual adaptation for foundation forecasting models.")
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="daily")
    parser.add_argument("--models", nargs="+", choices=["timefm_v2", "chronos_base", "moirai2"], default=["timefm_v2", "chronos_base"])
    parser.add_argument("--output_root", default="carbon/selective_residual_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
    parser.add_argument("--moirai2_model_id", default="Salesforce/moirai-2.0-R-small")
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
    parser.add_argument("--use_gate_grid", nargs="+", type=int, default=[0, 1], help="Whether to search gated residual blending.")
    parser.add_argument("--site_guard_grid", nargs="+", type=int, default=[0, 1], help="Whether to search per-site validation guarding.")
    parser.add_argument("--min_relative_gain", type=float, default=0.03, help="Require this relative val-MSE gain before leaving foundation_raw.")
    parser.add_argument("--site_guard_min_relative_gain", type=float, default=0.0, help="Require this per-site val-MSE gain before applying adaptation to that site's test windows.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling-based foundation models.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--custom_daily_specs", default=None, help="Comma-separated seq:pred pairs, e.g. 180:60,365:90")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
