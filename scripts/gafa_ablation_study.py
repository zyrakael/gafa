#!/usr/bin/env python3
"""Ablation study for GAFA: gated vs ungated residual adaptation.

This script compares the performance of:
1. Foundation raw (baseline)
2. Ungated residual adaptation (residual head only)
3. Gated residual adaptation (residual head + gate head)

The gate head controls how much of the residual correction is applied,
allowing selective adaptation. This ablation isolates the gate's contribution.
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
GAFA_EVAL = ROOT / "scripts" / "selective_residual_adaptation.py"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_ablation(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    unified = load_module(UNIFIED_EVAL, "unified_foundation_eval")
    gafa = load_module(GAFA_EVAL, "selective_residual_adaptation")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    for attr in ["timefm_path", "chronos_path"]:
        value = Path(getattr(args, attr))
        if not value.is_absolute():
            setattr(args, attr, str(ROOT / value))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    specs = unified.iter_specs(args.task, args.custom_daily_specs)
    forecasters = unified.make_forecasters(args)

    for spec in specs:
        print(f"\n[spec] {spec.task} | {spec.description}")
        for forecaster in forecasters:
            if forecaster.name not in {"timefm_v2", "chronos_base", "moirai2"}:
                continue
            print(f"[model] {forecaster.name}")
            
            # Collect predictions
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
                    features = gafa.build_features(window["context"], pred, unc, spec)
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

            train_items = splits["train"]
            val_items = splits["val"]
            test_items = splits["test"]
            if not train_items or not val_items or not test_items:
                print("  [skip] not enough windows")
                continue

            y_train, p_train, u_train, x_train = gafa.flatten(train_items)
            y_val, p_val, u_val, x_val = gafa.flatten(val_items)
            y_test, p_test, u_test, x_test = gafa.flatten(test_items)

            # Baseline: foundation raw
            raw_val_metrics = gafa.metrics(y_val, p_val)
            raw_metrics = gafa.metrics(y_test, p_test)
            rows.append({
                "model": forecaster.name,
                "task": spec.task,
                "description": spec.description,
                "ablation_mode": "foundation_raw",
                "use_gate": False,
                "calibration_points": len(y_train),
                "selected_points": len(y_train),
                "selected_ratio": 1.0,
                "blend_ratio": 0.0,
                "val_mse": raw_val_metrics["mse"],
                "val_r2": raw_val_metrics["r2"],
                **raw_metrics,
            })

            # Ablation 1: Ungated residual adaptation (residual head only)
            print("  [ablation] ungated residual adaptation")
            best_ungated: dict[str, Any] | None = None
            for alpha in args.alpha_grid:
                for blend_ratio in args.blend_grid:
                    for uncertainty_keep_ratio in args.uncertainty_grid:
                        for residual_keep_ratio in args.residual_grid:
                            head = gafa.fit_residual_head(
                                y_train,
                                p_train,
                                u_train,
                                x_train,
                                "selective_residual",
                                alpha,
                                uncertainty_keep_ratio,
                                residual_keep_ratio,
                            )
                            if head is None:
                                continue
                            
                            correction_val = gafa.predict_ridge(x_val, head["weights"], head["mean"], head["std"])
                            pred_val = p_val + blend_ratio * correction_val
                            val_metrics = gafa.metrics(y_val, pred_val)
                            
                            if best_ungated is None or val_metrics["mse"] < best_ungated["val_metrics"]["mse"]:
                                best_ungated = {
                                    "alpha": alpha,
                                    "blend_ratio": blend_ratio,
                                    "uncertainty_keep_ratio": uncertainty_keep_ratio,
                                    "residual_keep_ratio": residual_keep_ratio,
                                    "head": head,
                                    "val_metrics": val_metrics,
                                }

            if best_ungated is not None:
                # Refit on train+val
                fit_items = train_items + val_items
                y_fit, p_fit, u_fit, x_fit = gafa.flatten(fit_items)
                head = gafa.fit_residual_head(
                    y_fit,
                    p_fit,
                    u_fit,
                    x_fit,
                    "selective_residual",
                    best_ungated["alpha"],
                    best_ungated["uncertainty_keep_ratio"],
                    best_ungated["residual_keep_ratio"],
                )
                correction_test = gafa.predict_ridge(x_test, head["weights"], head["mean"], head["std"])
                pred_test_ungated = p_test + best_ungated["blend_ratio"] * correction_test
                ungated_metrics = gafa.metrics(y_test, pred_test_ungated)
                
                rows.append({
                    "model": forecaster.name,
                    "task": spec.task,
                    "description": spec.description,
                    "ablation_mode": "ungated_residual",
                    "use_gate": False,
                    "calibration_points": len(y_fit),
                    "selected_points": best_ungated["head"]["selected_points"],
                    "selected_ratio": best_ungated["head"]["selected_ratio"],
                    "blend_ratio": best_ungated["blend_ratio"],
                    "val_mse": best_ungated["val_metrics"]["mse"],
                    "val_r2": best_ungated["val_metrics"]["r2"],
                    **ungated_metrics,
                })
                print(f"    ungated: val_mse={best_ungated['val_metrics']['mse']:.4f}, test_mse={ungated_metrics['mse']:.4f}, test_r2={ungated_metrics['r2']:.4f}")

            # Ablation 2: Gated residual adaptation (residual head + gate head)
            print("  [ablation] gated residual adaptation")
            best_gated: dict[str, Any] | None = None
            for alpha in args.alpha_grid:
                for blend_ratio in args.blend_grid:
                    for uncertainty_keep_ratio in args.uncertainty_grid:
                        for residual_keep_ratio in args.residual_grid:
                            head = gafa.fit_residual_head(
                                y_train,
                                p_train,
                                u_train,
                                x_train,
                                "selective_residual",
                                alpha,
                                uncertainty_keep_ratio,
                                residual_keep_ratio,
                            )
                            if head is None:
                                continue
                            
                            correction_train = gafa.predict_ridge(x_train, head["weights"], head["mean"], head["std"])
                            gate_head = gafa.fit_gate_head(
                                y_train,
                                p_train,
                                correction_train,
                                x_train,
                                u_train,
                                alpha,
                            )
                            
                            correction_val = gafa.predict_ridge(x_val, head["weights"], head["mean"], head["std"])
                            if gate_head is not None:
                                pred_val = gafa.apply_gate_head(p_val, correction_val, x_val, u_val, gate_head, blend_ratio)
                            else:
                                pred_val = p_val + blend_ratio * correction_val
                            val_metrics = gafa.metrics(y_val, pred_val)
                            
                            if best_gated is None or val_metrics["mse"] < best_gated["val_metrics"]["mse"]:
                                best_gated = {
                                    "alpha": alpha,
                                    "blend_ratio": blend_ratio,
                                    "uncertainty_keep_ratio": uncertainty_keep_ratio,
                                    "residual_keep_ratio": residual_keep_ratio,
                                    "head": head,
                                    "gate_head": gate_head,
                                    "val_metrics": val_metrics,
                                }

            if best_gated is not None:
                # Refit on train+val
                fit_items = train_items + val_items
                y_fit, p_fit, u_fit, x_fit = gafa.flatten(fit_items)
                head = gafa.fit_residual_head(
                    y_fit,
                    p_fit,
                    u_fit,
                    x_fit,
                    "selective_residual",
                    best_gated["alpha"],
                    best_gated["uncertainty_keep_ratio"],
                    best_gated["residual_keep_ratio"],
                )
                correction_fit = gafa.predict_ridge(x_fit, head["weights"], head["mean"], head["std"])
                gate_head = gafa.fit_gate_head(
                    y_fit,
                    p_fit,
                    correction_fit,
                    x_fit,
                    u_fit,
                    best_gated["alpha"],
                )
                
                correction_test = gafa.predict_ridge(x_test, head["weights"], head["mean"], head["std"])
                if gate_head is not None:
                    pred_test_gated = gafa.apply_gate_head(p_test, correction_test, x_test, u_test, gate_head, best_gated["blend_ratio"])
                else:
                    pred_test_gated = p_test + best_gated["blend_ratio"] * correction_test
                gated_metrics = gafa.metrics(y_test, pred_test_gated)
                
                rows.append({
                    "model": forecaster.name,
                    "task": spec.task,
                    "description": spec.description,
                    "ablation_mode": "gated_residual",
                    "use_gate": gate_head is not None,
                    "calibration_points": len(y_fit),
                    "selected_points": head["selected_points"],
                    "selected_ratio": head["selected_ratio"],
                    "blend_ratio": best_gated["blend_ratio"],
                    "val_mse": best_gated["val_metrics"]["mse"],
                    "val_r2": best_gated["val_metrics"]["r2"],
                    **gated_metrics,
                })
                print(f"    gated: val_mse={best_gated['val_metrics']['mse']:.4f}, test_mse={gated_metrics['mse']:.4f}, test_r2={gated_metrics['r2']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "gafa_ablation_metrics.csv", index=False)
    write_summary(results, output_root / "gafa_ablation_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# GAFA Ablation Study: Gated vs Ungated Residual Adaptation", ""]
    if df.empty:
        lines.append("No results.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    
    lines.extend([
        "## Overview",
        "This ablation study isolates the contribution of the gate head in GAFA.",
        "- **foundation_raw**: Frozen foundation forecast (baseline)",
        "- **ungated_residual**: Residual head only (no gate)",
        "- **gated_residual**: Residual head + gate head (full GAFA)",
        "",
        "The gate head learns to control how much of the residual correction is applied,",
        "enabling selective adaptation. A positive gate contribution indicates that the gate",
        "improves performance by selectively applying corrections.",
        "",
        "## Results by Task and Model",
        "",
        "| Task | Description | Model | Mode | Gate | Blend | Selected Ratio | MSE | MAE | RMSE | R2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    
    for row in df.sort_values(["task", "description", "model", "ablation_mode"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.description} | {row.model} | {row.ablation_mode} | "
            f"{int(bool(getattr(row, 'use_gate', False)))} | {getattr(row, 'blend_ratio', 0.0):.2f} | "
            f"{row.selected_ratio:.3f} | {row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    
    # Add gate contribution analysis
    lines.extend(["", "## Gate Contribution Analysis", ""])
    lines.append("| Task | Description | Model | Ungated MSE | Gated MSE | Gate Gain (%) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    
    for (task, desc, model), group in df.groupby(["task", "description", "model"]):
        ungated = group[group["ablation_mode"] == "ungated_residual"]
        gated = group[group["ablation_mode"] == "gated_residual"]
        
        if not ungated.empty and not gated.empty:
            ungated_mse = ungated.iloc[0]["mse"]
            gated_mse = gated.iloc[0]["mse"]
            gate_gain = (ungated_mse - gated_mse) / max(abs(ungated_mse), 1e-12) * 100
            lines.append(
                f"| {task} | {desc} | {model} | {ungated_mse:.4f} | {gated_mse:.4f} | {gate_gain:+.2f}% |"
            )
    
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAFA ablation study: gated vs ungated residual adaptation.")
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="daily")
    parser.add_argument("--models", nargs="+", choices=["timefm_v2", "chronos_base", "moirai2"], default=["timefm_v2", "chronos_base"])
    parser.add_argument("--output_root", default="carbon/gafa_ablation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
    parser.add_argument("--moirai2_model_id", default="Salesforce/moirai-2.0-R-small")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--uncertainty_keep_ratio", type=float, default=0.8)
    parser.add_argument("--residual_keep_ratio", type=float, default=0.9)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--uncertainty_grid", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--residual_grid", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--blend_grid", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--custom_daily_specs", default=None, help="Comma-separated seq:pred pairs, e.g. 180:60,365:90")
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
