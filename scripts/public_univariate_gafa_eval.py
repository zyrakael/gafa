#!/usr/bin/env python3
"""Public benchmark univariate evaluation for frozen forecasters and GAFA.

The public CSV files in `data/` follow the common long-term forecasting layout:
the first column is time and the final target column is `OT`. This script keeps
only that target series, builds chronological rolling windows, and evaluates the
same raw foundation forecast vs. GAFA alignment protocol used in the CIKM paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_EVAL = ROOT / "scripts" / "unified_foundation_eval.py"
GAFA_EVAL = ROOT / "scripts" / "selective_residual_adaptation.py"


@dataclass(frozen=True)
class PublicDataset:
    name: str
    path: Path
    freq: str
    period: int
    target_col: str = "OT"


@dataclass(frozen=True)
class PublicSpec:
    task: str
    dataset: str
    data_path: Path
    freq: str
    seq_len: int
    pred_len: int
    period: int
    description: str
    target_col: str


PUBLIC_DATASETS = {
    "ETTh1": PublicDataset("ETTh1", ROOT / "data" / "ETTh1.csv", "h", 24),
    "ETTh2": PublicDataset("ETTh2", ROOT / "data" / "ETTh2.csv", "h", 24),
    "ETTm1": PublicDataset("ETTm1", ROOT / "data" / "ETTm1.csv", "15min", 96),
    "ETTm2": PublicDataset("ETTm2", ROOT / "data" / "ETTm2.csv", "15min", 96),
    "Electricity": PublicDataset("Electricity", ROOT / "data" / "electricity.csv", "h", 24),
    "Weather": PublicDataset("Weather", ROOT / "data" / "weather.csv", "10min", 144),
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_public_series(dataset: PublicDataset) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(dataset.path)
    if df.empty:
        raise ValueError(f"{dataset.path} is empty")
    date_col = df.columns[0]
    target_col = dataset.target_col if dataset.target_col in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "site": dataset.name,
            "NEE_clean": pd.to_numeric(df[target_col], errors="coerce"),
            "segment_id": 0,
        }
    )
    out = out.dropna(subset=["date", "NEE_clean"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable {target_col} rows found in {dataset.path}")
    return {dataset.name: out}


def normalize_context_for_forecaster(context: np.ndarray, mode: str) -> tuple[np.ndarray, float, float]:
    if mode == "none":
        return context.astype(np.float32), 0.0, 1.0
    if mode != "context":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    context64 = np.asarray(context, dtype=np.float64)
    mean = float(np.mean(context64))
    std = float(np.std(context64))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    normalized = ((context64 - mean) / std).astype(np.float32)
    return normalized, mean, std


def invert_forecast_scale(pred: np.ndarray, unc: np.ndarray, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
    pred_out = np.asarray(pred, dtype=np.float64) * std + mean
    unc_out = np.asarray(unc, dtype=np.float64) * abs(std)
    return pred_out, unc_out


def iter_specs(args: argparse.Namespace) -> list[PublicSpec]:
    datasets = [PUBLIC_DATASETS[name] for name in args.datasets]
    specs: list[PublicSpec] = []
    for dataset in datasets:
        for pred_len in args.pred_lens:
            specs.append(
                PublicSpec(
                    task="public_univariate",
                    dataset=dataset.name,
                    data_path=dataset.path,
                    freq="30min",
                    seq_len=args.seq_len,
                    pred_len=int(pred_len),
                    period=dataset.period,
                    description=f"{dataset.name} OT {args.seq_len}->{int(pred_len)}",
                    target_col=dataset.target_col,
                )
            )
    return specs


def flatten(items: list[dict[str, Any]], gafa: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return gafa.flatten(items)


def point_sites(items: list[dict[str, Any]], gafa: Any) -> np.ndarray:
    return gafa.point_sites(items)


def collect_predictions(
    args: argparse.Namespace,
    unified: Any,
    gafa: Any,
    spec: PublicSpec,
    forecaster: Any,
    series_cache: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, list[dict[str, Any]]]:
    if spec.dataset not in series_cache:
        series_cache[spec.dataset] = normalize_public_series(PUBLIC_DATASETS[spec.dataset])
    site_data = series_cache[spec.dataset]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    stride = args.stride or spec.pred_len
    for site, site_df in site_data.items():
        windows = unified.build_windows(site_df, spec, stride, args.max_windows_per_dataset)
        if len(windows) < 3:
            continue
        for idx, window in enumerate(windows):
            split_name = unified.split_name_for_index(idx, len(windows), args.train_ratio, args.val_ratio)
            model_context, scale_mean, scale_std = normalize_context_for_forecaster(window["context"], args.normalization)
            pred, unc = forecaster.predict(model_context, spec)
            pred, unc = invert_forecast_scale(pred, unc, scale_mean, scale_std)
            features = gafa.build_features(window["context"], pred, unc, spec)
            splits[split_name].append(
                {
                    "site": site,
                    "window_index": idx,
                    "dates": window["dates"],
                    "target": window["target"].astype(np.float64),
                    "pred": pred.astype(np.float64),
                    "uncertainty": unc.astype(np.float64),
                    "features": features,
                    "scale_mean": scale_mean,
                    "scale_std": scale_std,
                }
            )
    return splits


def relative_gain(raw: float, adapted: float) -> float:
    return (raw - adapted) / max(abs(raw), 1e-12)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = float(np.std(np.asarray(y_true, dtype=np.float64)))
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) / max(scale, 1e-12))


def run(args: argparse.Namespace) -> None:
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

    forecasters = unified.make_forecasters(args)
    series_cache: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for spec in iter_specs(args):
        print(f"\n[spec] {spec.description}")
        for forecaster in forecasters:
            print(f"[model] {forecaster.name}")
            split_items = collect_predictions(args, unified, gafa, spec, forecaster, series_cache)
            train_items = split_items["train"]
            val_items = split_items["val"]
            test_items = split_items["test"]
            if not train_items or not val_items or not test_items:
                print("  [skip] not enough windows")
                continue

            y_train, _, _, _ = flatten(train_items, gafa)
            y_val, p_val, _, _ = flatten(val_items, gafa)
            y_test, p_test, _, _ = flatten(test_items, gafa)
            raw_val_metrics = gafa.metrics(y_val, p_val)
            raw_metrics = gafa.metrics(y_test, p_test)
            rows.append(
                {
                    "dataset": spec.dataset,
                    "target_col": spec.target_col,
                    "seq_len": spec.seq_len,
                    "pred_len": spec.pred_len,
                    "normalization": args.normalization,
                    "model": forecaster.name,
                    "mode": "foundation_raw",
                    "windows_train": len(train_items),
                    "windows_val": len(val_items),
                    "windows_test": len(test_items),
                    "calibration_points": len(y_train),
                    "selected_points": len(y_train),
                    "selected_ratio": 1.0,
                    "blend_ratio": 0.0,
                    "selected_strategy": "foundation_raw",
                    "val_mse": raw_val_metrics["mse"],
                    "val_r2": raw_val_metrics["r2"],
                    "tuning_config": "{}",
                    "mse_gain_vs_raw": 0.0,
                    "nrmse": nrmse(y_test, p_test),
                    **raw_metrics,
                }
            )

            best = gafa.tune_strategy(train_items, val_items, args)
            fit_items = gafa.calibration_items_from_splits(split_items, args.calibration_splits)
            adaptive_pred, adaptive_head = gafa.refit_and_predict(fit_items, test_items, best, args)
            adaptive_metrics = gafa.metrics(y_test, adaptive_pred)
            selected_points = len(y_train) if adaptive_head is None else adaptive_head["selected_points"]
            selected_ratio = 1.0 if adaptive_head is None else adaptive_head["selected_ratio"]
            rows.append(
                {
                    "dataset": spec.dataset,
                    "target_col": spec.target_col,
                    "seq_len": spec.seq_len,
                    "pred_len": spec.pred_len,
                    "normalization": args.normalization,
                    "model": forecaster.name,
                    "mode": "gafa",
                    "windows_train": len(train_items),
                    "windows_val": len(val_items),
                    "windows_test": len(test_items),
                    "calibration_points": len(gafa.flatten(fit_items)[0]),
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
                            "calibration_splits": args.calibration_splits,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    "mse_gain_vs_raw": relative_gain(raw_metrics["mse"], adaptive_metrics["mse"]),
                    "nrmse": nrmse(y_test, adaptive_pred),
                    **adaptive_metrics,
                }
            )
            print(
                "  "
                f"best={best['strategy']} blend={best['blend_ratio']:.2f} "
                f"gate={int(bool(best.get('use_gate', False)))} "
                f"val_mse={best['val_metrics']['mse']:.4f} "
                f"test_mse={adaptive_metrics['mse']:.4f} "
                f"gain={relative_gain(raw_metrics['mse'], adaptive_metrics['mse']):.2%}"
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
                                "dataset": spec.dataset,
                                "target_col": spec.target_col,
                                "seq_len": spec.seq_len,
                                "pred_len": spec.pred_len,
                                "normalization": args.normalization,
                                "model": forecaster.name,
                                "site": item["site"],
                                "window_index": item["window_index"],
                                "step": step,
                                "date": date,
                                "y_true": float(y),
                                "foundation_raw": float(raw_p),
                                "gafa": float(adaptive_p),
                                "uncertainty": float(unc),
                            }
                        )
                    offset += n

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "public_univariate_gafa_metrics.csv", index=False)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_root / "public_univariate_gafa_predictions.csv", index=False)
    write_summary(results, output_root / "public_univariate_gafa_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Public Univariate GAFA Results", ""]
    if df.empty:
        lines.append("No results.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "Target is the public benchmark's final `OT` column. Models receive only the univariate target history.",
            "If `normalization=context`, each input window is z-scored by its context mean/std and predictions are inverted before scoring.",
            "",
            "| Dataset | Horizon | Model | Mode | Strategy | Gate | Selected Ratio | MSE | MAE | RMSE | NRMSE | R2 | MSE Gain vs Raw |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    def truthy(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return bool(value)

    for row in df.sort_values(["dataset", "pred_len", "model", "mode"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.seq_len)}->{int(row.pred_len)} | {row.model} | {row.mode} | "
            f"{getattr(row, 'selected_strategy', '')} | {int(truthy(getattr(row, 'use_gate', False)))} | "
            f"{row.selected_ratio:.3f} | {row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{row.nrmse:.4f} | {row.r2:.4f} | {row.mse_gain_vs_raw:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public univariate raw foundation vs. GAFA evaluation.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(PUBLIC_DATASETS), default=sorted(PUBLIC_DATASETS))
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--models", nargs="+", choices=["timefm_v2", "chronos_base", "moirai2"], default=["timefm_v2", "chronos_base"])
    parser.add_argument("--output_root", default="carbon/public_univariate_gafa")
    parser.add_argument("--normalization", choices=["none", "context"], default="context")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
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
    parser.add_argument("--site_guard_grid", nargs="+", type=int, default=[0], help="Public datasets use one target series, so site guarding is off by default.")
    parser.add_argument("--min_relative_gain", type=float, default=0.03)
    parser.add_argument("--site_guard_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_dataset", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
