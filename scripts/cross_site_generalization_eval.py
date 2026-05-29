#!/usr/bin/env python3
"""Cross-site generalization evaluation for daily carbon forecasting.

Unlike the unified rolling-window benchmark, this script splits by site:
train, validation, and test sites are disjoint. This measures how well each
method transfers to unseen sites.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_EVAL = ROOT / "scripts" / "unified_foundation_eval.py"
SUPERVISED_EVAL = ROOT / "scripts" / "unified_supervised_eval.py"
SELECTIVE_EVAL = ROOT / "scripts" / "selective_residual_adaptation.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def split_sites(sites: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(sites), dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = max(1, int(round(n * train_ratio)))
    val_count = max(1, int(round(n * val_ratio)))
    if train_end + val_count >= n:
        train_end = max(1, n - 2)
        val_count = 1
    val_end = train_end + val_count
    return (
        sorted(str(x) for x in shuffled[:train_end]),
        sorted(str(x) for x in shuffled[train_end:val_end]),
        sorted(str(x) for x in shuffled[val_end:]),
    )


def collect_site_windows(
    unified: Any,
    supervised: Any,
    site_data: dict[str, pd.DataFrame],
    sites: list[str],
    spec: Any,
    stride: int | None,
    max_windows_per_site: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    x_marks: list[np.ndarray] = []
    dec_marks: list[np.ndarray] = []
    items: list[dict[str, Any]] = []
    for site in sites:
        windows = unified.build_windows(site_data[site], spec, stride or spec.pred_len, max_windows_per_site)
        for idx, window in enumerate(windows):
            x_mark, dec_mark = supervised.build_time_marks(window["dates"], spec, label_len=max(1, spec.seq_len // 7))
            xs.append(window["context"].astype(np.float32))
            ys.append(window["target"].astype(np.float32))
            x_marks.append(x_mark)
            dec_marks.append(dec_mark)
            items.append(
                {
                    "site": site,
                    "window_index": idx,
                    "dates": window["dates"],
                    "target": window["target"].astype(np.float64),
                    "context": window["context"].astype(np.float64),
                }
            )
    if not xs:
        raise RuntimeError(f"No windows for sites={sites} and spec={spec.description}")
    return np.stack(xs), np.stack(ys), np.stack(x_marks), np.stack(dec_marks), items


def collect_foundation_items(
    unified: Any,
    selective: Any,
    site_data: dict[str, pd.DataFrame],
    sites: list[str],
    spec: Any,
    forecaster: Any,
    stride: int | None,
    max_windows_per_site: int | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for site in sites:
        windows = unified.build_windows(site_data[site], spec, stride or spec.pred_len, max_windows_per_site)
        for idx, window in enumerate(windows):
            pred, unc = forecaster.predict(window["context"], spec)
            features = selective.build_features(window["context"], pred, unc, spec)
            items.append(
                {
                    "site": site,
                    "window_index": idx,
                    "dates": window["dates"],
                    "target": window["target"].astype(np.float64),
                    "pred": pred.astype(np.float64),
                    "uncertainty": unc.astype(np.float64),
                    "features": features,
                }
            )
    return items


def supervised_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        revin=args.revin,
        ia_layers=args.ia_layers,
        pd_layers=args.pd_layers,
        ca_layers=args.ca_layers,
        verbose=args.verbose,
    )


def foundation_args(args: argparse.Namespace) -> SimpleNamespace:
    timefm_path = Path(args.timefm_path)
    chronos_path = Path(args.chronos_path)
    moirai2_model_id = Path(args.moirai2_model_id) if args.moirai2_model_id else Path("ori/moirai")
    return SimpleNamespace(
        models=args.foundation_models,
        device=args.device,
        timefm_path=str(timefm_path if timefm_path.is_absolute() else ROOT / timefm_path),
        chronos_path=str(chronos_path if chronos_path.is_absolute() else ROOT / chronos_path),
        chronos_samples=args.chronos_samples,
        moirai2_model_id=str(moirai2_model_id if moirai2_model_id.is_absolute() else ROOT / moirai2_model_id),
        moirai2_batch_size=args.moirai2_batch_size,
    )


def tune_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ridge_alpha=args.ridge_alpha,
        alpha_grid=args.alpha_grid,
        uncertainty_keep_ratio=args.uncertainty_keep_ratio,
        residual_keep_ratio=args.residual_keep_ratio,
        uncertainty_grid=args.uncertainty_grid,
        residual_grid=args.residual_grid,
        blend_grid=args.blend_grid,
        use_gate_grid=args.use_gate_grid,
        site_guard_grid=[0],
        min_relative_gain=args.min_relative_gain,
        site_guard_min_relative_gain=0.0,
    )


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    unified = import_module(UNIFIED_EVAL, "unified_foundation_eval_cross_site")
    supervised = import_module(SUPERVISED_EVAL, "unified_supervised_eval_cross_site")
    selective = import_module(SELECTIVE_EVAL, "selective_residual_adaptation_cross_site")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    specs = unified.iter_specs("daily", args.custom_daily_specs)
    site_data = unified.load_clean(ROOT / "carbon/clean/all_sites_daily_clean.csv")
    train_sites, val_sites, test_sites = split_sites(sorted(site_data), args.train_site_ratio, args.val_site_ratio, args.seed)

    print("[site split]")
    print(f"  train_sites={len(train_sites)} {train_sites}")
    print(f"  val_sites={len(val_sites)} {val_sites}")
    print(f"  test_sites={len(test_sites)} {test_sites}")

    rows: list[dict[str, Any]] = []
    split_info = {
        "seed": args.seed,
        "train_site_ratio": args.train_site_ratio,
        "val_site_ratio": args.val_site_ratio,
        "train_sites": train_sites,
        "val_sites": val_sites,
        "test_sites": test_sites,
    }
    (output_root / "cross_site_split.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")

    sup_args = supervised_args(args)
    forecasters = unified.make_forecasters(foundation_args(args))
    sel_args = tune_args(args)

    for spec in specs:
        print(f"\n[spec] {spec.description}")
        train_x, train_y, train_x_mark, train_dec_mark, train_items = collect_site_windows(
            unified, supervised, site_data, train_sites, spec, args.stride, args.max_windows_per_site
        )
        val_x, val_y, val_x_mark, val_dec_mark, val_items = collect_site_windows(
            unified, supervised, site_data, val_sites, spec, args.stride, args.max_windows_per_site
        )
        test_x, test_y, test_x_mark, test_dec_mark, test_items = collect_site_windows(
            unified, supervised, site_data, test_sites, spec, args.stride, args.max_windows_per_site
        )
        print(f"  windows train={len(train_x)} val={len(val_x)} test={len(test_x)}")

        for model_name in args.supervised_models:
            print(f"  [supervised] {model_name}")
            model = supervised.make_model(model_name, spec, sup_args)
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
                sup_args,
            )
            pred = supervised.predict_model(model, test_x, test_y, test_x_mark, test_dec_mark, mean, std, sup_args)
            m = supervised.metrics(test_y, pred)
            rows.append(
                {
                    "task": spec.task,
                    "description": spec.description,
                    "family": "supervised",
                    "model": model_name,
                    "method": model_name,
                    "train_sites": len(train_sites),
                    "val_sites": len(val_sites),
                    "test_sites": len(test_sites),
                    "train_windows": len(train_x),
                    "val_windows": len(val_x),
                    "test_windows": len(test_x),
                    **m,
                }
            )
            print(f"    mse={m['mse']:.4f} mae={m['mae']:.4f} r2={m['r2']:.4f}")

        for forecaster in forecasters:
            print(f"  [foundation] {forecaster.name}")
            foundation_train = collect_foundation_items(
                unified, selective, site_data, train_sites, spec, forecaster, args.stride, args.max_windows_per_site
            )
            foundation_val = collect_foundation_items(
                unified, selective, site_data, val_sites, spec, forecaster, args.stride, args.max_windows_per_site
            )
            foundation_test = collect_foundation_items(
                unified, selective, site_data, test_sites, spec, forecaster, args.stride, args.max_windows_per_site
            )
            y_test, p_test, _, _ = selective.flatten(foundation_test)
            raw_metrics = selective.metrics(y_test, p_test)
            rows.append(
                {
                    "task": spec.task,
                    "description": spec.description,
                    "family": "foundation",
                    "model": forecaster.name,
                    "method": "foundation_raw",
                    "train_sites": len(train_sites),
                    "val_sites": len(val_sites),
                    "test_sites": len(test_sites),
                    "train_windows": len(foundation_train),
                    "val_windows": len(foundation_val),
                    "test_windows": len(foundation_test),
                    "selected_strategy": "foundation_raw",
                    "use_gate": False,
                    "blend_ratio": 0.0,
                    **raw_metrics,
                }
            )
            print(f"    raw mse={raw_metrics['mse']:.4f} mae={raw_metrics['mae']:.4f} r2={raw_metrics['r2']:.4f}")

            best = selective.tune_strategy(foundation_train, foundation_val, sel_args)
            adaptive_pred, adaptive_head = selective.refit_and_predict(
                selective.combine_items(foundation_train, foundation_val),
                foundation_test,
                best,
                sel_args,
            )
            adaptive_metrics = selective.metrics(y_test, adaptive_pred)
            rows.append(
                {
                    "task": spec.task,
                    "description": spec.description,
                    "family": "adaptive",
                    "model": forecaster.name,
                    "method": "adaptive_selective",
                    "train_sites": len(train_sites),
                    "val_sites": len(val_sites),
                    "test_sites": len(test_sites),
                    "train_windows": len(foundation_train),
                    "val_windows": len(foundation_val),
                    "test_windows": len(foundation_test),
                    "selected_strategy": best["strategy"],
                    "use_gate": bool(best.get("use_gate", False)),
                    "blend_ratio": float(best["blend_ratio"]),
                    "selected_ratio": 1.0 if adaptive_head is None else float(adaptive_head["selected_ratio"]),
                    "val_mse": float(best["val_metrics"]["mse"]),
                    **adaptive_metrics,
                }
            )
            print(
                "    adaptive "
                f"strategy={best['strategy']} gate={int(bool(best.get('use_gate', False)))} "
                f"blend={best['blend_ratio']:.2f} mse={adaptive_metrics['mse']:.4f} r2={adaptive_metrics['r2']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "cross_site_metrics.csv", index=False)
    write_summary(df, output_root / "cross_site_summary.md", split_info)
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path, split_info: dict[str, Any]) -> None:
    lines = [
        "# Cross-Site Generalization Evaluation",
        "",
        "Train, validation, and test sets are disjoint at the site level.",
        "",
        f"- Seed: `{split_info['seed']}`",
        f"- Train sites: {', '.join(split_info['train_sites'])}",
        f"- Validation sites: {', '.join(split_info['val_sites'])}",
        f"- Test sites: {', '.join(split_info['test_sites'])}",
        "",
        "| Description | Family | Model | Method | Strategy | Gate | MSE | MAE | RMSE | R2 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.sort_values(["description", "r2"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {row.description} | {row.family} | {row.model} | {row.method} | "
            f"{getattr(row, 'selected_strategy', '')} | {int(bool(getattr(row, 'use_gate', False)))} | "
            f"{row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-site generalization evaluation.")
    parser.add_argument("--custom_daily_specs", default="365:60", help="Comma-separated seq:pred pairs, e.g. 90:30,365:60")
    parser.add_argument("--output_root", default="carbon/exp_cross_site_generalization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_site_ratio", type=float, default=0.6)
    parser.add_argument("--val_site_ratio", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--supervised_models", nargs="+", default=["DLinear", "iTransformer", "TimeMixer", "Transformer"])
    parser.add_argument("--foundation_models", nargs="+", default=["timefm_v2", "chronos_base", "moirai2"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
    parser.add_argument("--moirai2_model_id", default="ori/moirai")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--uncertainty_keep_ratio", type=float, default=0.8)
    parser.add_argument("--residual_keep_ratio", type=float, default=0.9)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--uncertainty_grid", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--residual_grid", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--blend_grid", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--use_gate_grid", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--min_relative_gain", type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
