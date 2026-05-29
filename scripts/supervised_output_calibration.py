#!/usr/bin/env python3
"""Output-space calibration control for target-trained supervised models.

This experiment answers whether GAFA-style residual calibration is simply a
generic post-processing trick. Supervised models are first trained on the target
benchmark exactly as in `unified_supervised_eval.py`; then the frozen supervised
predictions are passed through the same low-capacity residual/gate heads used by
the foundation alignment experiment.
"""

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
SUPERVISED_EVAL = ROOT / "scripts" / "unified_supervised_eval.py"
RESIDUAL_ADAPT = ROOT / "scripts" / "selective_residual_adaptation.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_items(contexts: np.ndarray, targets: np.ndarray, preds: np.ndarray, spec: Any, split_name: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx in range(len(targets)):
        pred = preds[idx].astype(np.float64)
        uncertainty = np.zeros_like(pred, dtype=np.float64)
        items.append(
            {
                "site": split_name,
                "window_index": idx,
                "target": targets[idx].astype(np.float64),
                "pred": pred,
                "uncertainty": uncertainty,
                "features": RESIDUAL.build_features(contexts[idx], pred, uncertainty, spec),
            }
        )
    return items


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Supervised Output Calibration Control",
        "",
        "Target-trained supervised models are frozen, then passed through the same residual/gate calibration used for foundation alignment.",
        "",
        "| Task | Description | Model | Mode | Selected Strategy | Gate | Blend | MSE | MAE | RMSE | R2 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.sort_values(["task", "description", "model", "mode"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.description} | {row.model} | {row.mode} | "
            f"{getattr(row, 'selected_strategy', '')} | {int(bool(getattr(row, 'use_gate', False)))} | "
            f"{getattr(row, 'blend_ratio', 0.0):.2f} | {row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    supervised = load_module(SUPERVISED_EVAL, "unified_supervised_eval_for_calibration")
    unified = supervised.load_unified_module()

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for spec in unified.iter_specs(args.task, args.custom_daily_specs):
        print(f"\n[spec] {spec.task} | {spec.description}")
        (
            train_x,
            train_y,
            train_x_mark,
            train_dec_mark,
            val_x,
            val_y,
            val_x_mark,
            val_dec_mark,
            test_x,
            test_y,
            test_x_mark,
            test_dec_mark,
        ) = supervised.collect_windows(args, unified, spec)
        print(f"  windows train={len(train_x)} val={len(val_x)} test={len(test_x)}")

        for model_name in args.models:
            print(f"  [model] {model_name}")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            model = supervised.make_model(model_name, spec, args)
            model, mean, std = supervised.train_model(
                model,
                train_x,
                train_y,
                train_x_mark,
                train_dec_mark,
                val_x,
                val_y,
                val_x_mark,
                val_dec_mark,
                args,
            )
            train_pred = supervised.predict_model(model, train_x, train_y, train_x_mark, train_dec_mark, mean, std, args)
            val_pred = supervised.predict_model(model, val_x, val_y, val_x_mark, val_dec_mark, mean, std, args)
            test_pred = supervised.predict_model(model, test_x, test_y, test_x_mark, test_dec_mark, mean, std, args)

            train_items = build_items(train_x, train_y, train_pred, spec, "train")
            val_items = build_items(val_x, val_y, val_pred, spec, "val")
            test_items = build_items(test_x, test_y, test_pred, spec, "test")

            y_test, p_test, _, _ = RESIDUAL.flatten(test_items)
            raw_metrics = RESIDUAL.metrics(y_test, p_test)
            rows.append(
                {
                    "model": model_name,
                    "task": spec.task,
                    "description": spec.description,
                    "mode": "supervised_raw",
                    "train_windows": len(train_x),
                    "val_windows": len(val_x),
                    "test_windows": len(test_x),
                    "blend_ratio": 0.0,
                    "selected_strategy": "foundation_raw",
                    "use_gate": False,
                    "val_mse": np.nan,
                    "tuning_config": "{}",
                    **raw_metrics,
                }
            )

            best = RESIDUAL.tune_strategy(train_items, val_items, args)
            calibrated_pred, calibrated_head = RESIDUAL.refit_and_predict(train_items + val_items, test_items, best, args)
            calibrated_metrics = RESIDUAL.metrics(y_test, calibrated_pred)
            rows.append(
                {
                    "model": model_name,
                    "task": spec.task,
                    "description": spec.description,
                    "mode": "supervised_calibrated",
                    "train_windows": len(train_x),
                    "val_windows": len(val_x),
                    "test_windows": len(test_x),
                    "selected_points": len(RESIDUAL.flatten(train_items)[0]) if calibrated_head is None else calibrated_head["selected_points"],
                    "selected_ratio": 1.0 if calibrated_head is None else calibrated_head["selected_ratio"],
                    "blend_ratio": float(best["blend_ratio"]),
                    "selected_strategy": best["strategy"],
                    "use_gate": bool(best.get("use_gate", False)),
                    "val_mse": float(best["val_metrics"]["mse"]),
                    "tuning_config": json.dumps(
                        {
                            "strategy": best["strategy"],
                            "alpha": best.get("alpha"),
                            "uncertainty_keep_ratio": best.get("uncertainty_keep_ratio"),
                            "residual_keep_ratio": best.get("residual_keep_ratio"),
                            "blend_ratio": best["blend_ratio"],
                            "use_gate": bool(best.get("use_gate", False)),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    **calibrated_metrics,
                }
            )
            print(
                "    "
                f"raw_r2={raw_metrics['r2']:.4f} calibrated_r2={calibrated_metrics['r2']:.4f} "
                f"best={best['strategy']} blend={best['blend_ratio']:.2f} gate={int(bool(best.get('use_gate', False)))}"
            )

            if args.save_predictions:
                offset = 0
                for window_idx in range(len(test_y)):
                    n = spec.pred_len
                    raw = p_test[offset : offset + n]
                    calibrated = calibrated_pred[offset : offset + n]
                    for step in range(n):
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "task": spec.task,
                                "description": spec.description,
                                "window_index": window_idx,
                                "step": step + 1,
                                "y_true": float(test_y[window_idx, step]),
                                "supervised_raw": float(raw[step]),
                                "supervised_calibrated": float(calibrated[step]),
                            }
                        )
                    offset += n

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "supervised_output_calibration_metrics.csv", index=False)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_root / "supervised_output_calibration_predictions.csv", index=False)
    write_summary(df, output_root / "supervised_output_calibration_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAFA-style output calibration control for supervised models.")
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="daily")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["DLinear", "PatchTST", "iTransformer", "TSMixer", "TimeMixer", "PatchMLP", "TimeBridge", "TimeXer", "Transformer"],
        default=["DLinear"],
    )
    parser.add_argument("--output_root", default="carbon/supervised_output_calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--custom_daily_specs", default=None, help="Comma-separated seq:pred pairs, e.g. 180:60,365:60")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--revin", action="store_true")
    parser.add_argument("--ia_layers", type=int, default=1)
    parser.add_argument("--pd_layers", type=int, default=0)
    parser.add_argument("--ca_layers", type=int, default=1)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--uncertainty_grid", nargs="+", type=float, default=[1.0])
    parser.add_argument("--residual_grid", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    parser.add_argument("--blend_grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--use_gate_grid", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--site_guard_grid", nargs="+", type=int, default=[0])
    parser.add_argument("--min_relative_gain", type=float, default=0.0)
    parser.add_argument("--site_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--uncertainty_keep_ratio", type=float, default=1.0)
    parser.add_argument("--residual_keep_ratio", type=float, default=0.9)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


RESIDUAL = load_module(RESIDUAL_ADAPT, "selective_residual_adaptation_for_supervised_control")


if __name__ == "__main__":
    run(parse_args())
