#!/usr/bin/env python3
"""Unified rolling-window evaluation for supervised time-series models.

This script uses the same cleaned data and window builder as
`unified_foundation_eval.py`, but trains supervised models on train windows,
uses validation windows for early stopping, and evaluates on held-out test
windows. It is intended for fair comparison with TimeFM/Chronos unified results.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_EVAL = ROOT / "scripts" / "unified_foundation_eval.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.timefeatures import time_features


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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "nse": r2}


def site_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, sites: list[str]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(sites) != len(y_true):
        raise ValueError("Number of site labels must match the number of windows.")

    site_rows = []
    for site in sorted(set(sites)):
        mask = np.asarray([x == site for x in sites], dtype=bool)
        site_rows.append(metrics(y_true[mask], y_pred[mask]))

    return {
        "site_macro_mse": float(np.mean([x["mse"] for x in site_rows])),
        "site_macro_mae": float(np.mean([x["mae"] for x in site_rows])),
        "site_macro_rmse": float(np.mean([x["rmse"] for x in site_rows])),
        "site_macro_r2": float(np.nanmean([x["r2"] for x in site_rows])),
        "site_median_mse": float(np.median([x["mse"] for x in site_rows])),
        "site_median_mae": float(np.median([x["mae"] for x in site_rows])),
    }


def split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if n >= 3:
        train_end = min(max(1, train_end), n - 2)
        val_end = min(max(train_end + 1, val_end), n - 1)
    else:
        train_end = max(1, n - 1)
        val_end = train_end
    return train_end, val_end


def collect_windows(
    args: argparse.Namespace,
    unified: Any,
    spec: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    site_data = unified.load_clean(spec.data_path)
    sites = sorted(site_data)
    if args.sites:
        wanted = set(args.sites)
        sites = [s for s in sites if s in wanted]
    if args.max_sites:
        sites = sites[: args.max_sites]

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    train_x_mark: list[np.ndarray] = []
    train_dec_mark: list[np.ndarray] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    val_x_mark: list[np.ndarray] = []
    val_dec_mark: list[np.ndarray] = []
    test_x: list[np.ndarray] = []
    test_y: list[np.ndarray] = []
    test_x_mark: list[np.ndarray] = []
    test_dec_mark: list[np.ndarray] = []

    for site in sites:
        windows = unified.build_windows(site_data[site], spec, args.stride or spec.pred_len, args.max_windows_per_site)
        if len(windows) < 3:
            continue
        train_end, val_end = split_counts(len(windows), args.train_ratio, args.val_ratio)
        for idx, window in enumerate(windows):
            x_mark, dec_mark = build_time_marks(window["dates"], spec, label_len=max(1, spec.seq_len // 7))
            if idx < train_end:
                target_x, target_y = train_x, train_y
                target_x_mark, target_dec_mark = train_x_mark, train_dec_mark
            elif idx < val_end:
                target_x, target_y = val_x, val_y
                target_x_mark, target_dec_mark = val_x_mark, val_dec_mark
            else:
                target_x, target_y = test_x, test_y
                target_x_mark, target_dec_mark = test_x_mark, test_dec_mark
            target_x.append(window["context"].astype(np.float32))
            target_y.append(window["target"].astype(np.float32))
            target_x_mark.append(x_mark)
            target_dec_mark.append(dec_mark)

    if not train_x or not val_x or not test_x:
        raise RuntimeError(f"Not enough windows for {spec.description}")

    return (
        np.stack(train_x),
        np.stack(train_y),
        np.stack(train_x_mark),
        np.stack(train_dec_mark),
        np.stack(val_x),
        np.stack(val_y),
        np.stack(val_x_mark),
        np.stack(val_dec_mark),
        np.stack(test_x),
        np.stack(test_y),
        np.stack(test_x_mark),
        np.stack(test_dec_mark),
    )


def collect_test_metadata(args: argparse.Namespace, unified: Any, spec: Any) -> list[dict[str, Any]]:
    site_data = unified.load_clean(spec.data_path)
    sites = sorted(site_data)
    if args.sites:
        wanted = set(args.sites)
        sites = [s for s in sites if s in wanted]
    if args.max_sites:
        sites = sites[: args.max_sites]

    metadata: list[dict[str, Any]] = []
    for site in sites:
        windows = unified.build_windows(site_data[site], spec, args.stride or spec.pred_len, args.max_windows_per_site)
        if len(windows) < 3:
            continue
        train_end, val_end = split_counts(len(windows), args.train_ratio, args.val_ratio)
        for idx, window in enumerate(windows):
            if idx < train_end or idx < val_end:
                continue
            metadata.append(
                {
                    "site": site,
                    "site_window_index": idx,
                    "segment_id": window["segment_id"],
                    "start": window["start"],
                }
            )
    return metadata


def build_time_marks(target_dates: np.ndarray, spec: Any, label_len: int) -> tuple[np.ndarray, np.ndarray]:
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates))
    context_end = target_index[0] - pd.tseries.frequencies.to_offset(spec.freq)
    context_index = pd.date_range(end=context_end, periods=spec.seq_len, freq=spec.freq)
    decoder_index = context_index[-label_len:].append(target_index)
    x_mark = time_features(context_index, freq=spec.freq).transpose(1, 0).astype(np.float32)
    dec_mark = time_features(decoder_index, freq=spec.freq).transpose(1, 0).astype(np.float32)
    return x_mark, dec_mark


def compatible_patch_len(seq_len: int, requested: int) -> int:
    seen: set[int] = set()
    for candidate in [requested, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1]:
        if candidate in seen:
            continue
        seen.add(candidate)
        if 1 <= candidate <= seq_len and seq_len % candidate == 0:
            return candidate
    return max(1, min(requested, seq_len))


def compatible_period(seq_len: int, requested: int) -> int:
    for candidate in range(min(requested, seq_len), 1, -1):
        if seq_len % candidate == 0:
            return candidate
    return 1


def base_configs(spec: Any, args: argparse.Namespace) -> SimpleNamespace:
    patch_len = compatible_patch_len(spec.seq_len, args.patch_len)
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="S",
        seq_len=spec.seq_len,
        label_len=max(1, spec.seq_len // 7),
        pred_len=spec.pred_len,
        enc_in=1,
        dec_in=1,
        c_out=1,
        num_class=1,
        moving_avg=25,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        factor=3,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        activation="gelu",
        embed="timeF",
        freq="h" if spec.freq == "30min" else "d",
        output_attention=False,
        patch_len=patch_len,
        revin=args.revin,
        period=compatible_period(spec.seq_len, max(1, int(spec.period))),
        num_p=None,
        ia_layers=args.ia_layers,
        pd_layers=args.pd_layers,
        ca_layers=args.ca_layers,
        stable_len=max(1, int(spec.period)),
        down_sampling_window=2,
        down_sampling_layers=0,
        down_sampling_method=None,
        channel_independence=0,
        decomp_method="moving_avg",
        top_k=5,
        use_norm=1,
    )


def make_model(model_name: str, spec: Any, args: argparse.Namespace) -> nn.Module:
    configs = base_configs(spec, args)
    if model_name == "DLinear":
        from models.DLinear import Model

        return Model(configs)
    if model_name == "PatchTST":
        from models.PatchTST import Model

        return Model(configs, patch_len=args.patch_len, stride=args.patch_stride)
    if model_name == "iTransformer":
        from models.iTransformer import Model

        return Model(configs)
    if model_name == "TSMixer":
        from models.TSMixer import Model

        return Model(configs)
    if model_name == "TimeMixer":
        from models.TimeMixer import Model

        return Model(configs)
    if model_name == "PatchMLP":
        from models.PatchMLP import Model

        return Model(configs)
    if model_name == "TimeBridge":
        from models.TimeBridge import Model

        return Model(configs)
    if model_name == "TimeXer":
        from models.TimeXer import Model

        return Model(configs)
    if model_name == "Transformer":
        from models.Transformer import Model

        return Model(configs)
    raise ValueError(f"Unsupported supervised model: {model_name}")


def build_decoder_input(x: torch.Tensor, y: torch.Tensor, dec_mark: torch.Tensor) -> torch.Tensor:
    label_len = dec_mark.shape[1] - y.shape[1]
    return torch.cat([x[:, -label_len:, :], torch.zeros_like(y)], dim=1)


def forward_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_mark: torch.Tensor,
    dec_mark: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    xb = x.to(device)
    yb = y.to(device)
    xmb = x_mark.to(device)
    dmb = dec_mark.to(device)
    if model.__class__.__module__ == "models.TimeBridge" and xmb.shape[-1] < 4:
        pad_width = 4 - xmb.shape[-1]
        xmb = torch.cat([xmb, torch.zeros((*xmb.shape[:2], pad_width), device=device, dtype=xmb.dtype)], dim=-1)
        if dmb.shape[-1] < 4:
            dec_pad = 4 - dmb.shape[-1]
            dmb = torch.cat([dmb, torch.zeros((*dmb.shape[:2], dec_pad), device=device, dtype=dmb.dtype)], dim=-1)
    x_dec = build_decoder_input(xb, yb, dmb)
    return model(xb, xmb, x_dec, dmb)


def evaluate_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_mark: torch.Tensor,
    dec_mark: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    device = torch.device(args.device)
    loader = DataLoader(TensorDataset(x, y, x_mark, dec_mark), batch_size=args.batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    losses = []
    model.eval()
    with torch.no_grad():
        for xb, yb, xmb, dmb in loader:
            pred = forward_batch(model, xb, yb, xmb, dmb, device)
            loss = loss_fn(pred, yb.to(device))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_x_mark: np.ndarray,
    train_dec_mark: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_x_mark: np.ndarray,
    val_dec_mark: np.ndarray,
    args: argparse.Namespace,
) -> tuple[nn.Module, float, float]:
    device = torch.device(args.device)
    model = model.to(device)
    mean = float(train_x.mean())
    std = float(train_x.std())
    if std == 0:
        std = 1.0

    x = torch.tensor((train_x - mean) / std, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor((train_y - mean) / std, dtype=torch.float32).unsqueeze(-1)
    xm = torch.tensor(train_x_mark, dtype=torch.float32)
    dm = torch.tensor(train_dec_mark, dtype=torch.float32)
    vx = torch.tensor((val_x - mean) / std, dtype=torch.float32).unsqueeze(-1)
    vy = torch.tensor((val_y - mean) / std, dtype=torch.float32).unsqueeze(-1)
    vxm = torch.tensor(val_x_mark, dtype=torch.float32)
    vdm = torch.tensor(val_dec_mark, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y, xm, dm), batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, xmb, dmb in loader:
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = forward_batch(model, xb, yb, xmb, dmb, device)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, vx, vy, vxm, vdm, args)
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if args.verbose:
            print(f"    epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std


def predict_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    x_mark: np.ndarray,
    dec_mark: np.ndarray,
    mean: float,
    std: float,
    args: argparse.Namespace,
) -> np.ndarray:
    device = torch.device(args.device)
    tensor = torch.tensor((x - mean) / std, dtype=torch.float32).unsqueeze(-1)
    target = torch.tensor((y - mean) / std, dtype=torch.float32).unsqueeze(-1)
    x_mark_tensor = torch.tensor(x_mark, dtype=torch.float32)
    dec_mark_tensor = torch.tensor(dec_mark, dtype=torch.float32)
    preds: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(tensor, target, x_mark_tensor, dec_mark_tensor), batch_size=args.batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for xb, yb, xmb, dmb in loader:
            out = forward_batch(model, xb, yb, xmb, dmb, device).detach().cpu().numpy().squeeze(-1)
            preds.append(out * std + mean)
    return np.concatenate(preds, axis=0)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    unified = load_unified_module()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

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
        ) = collect_windows(args, unified, spec)
        test_metadata = collect_test_metadata(args, unified, spec)
        if len(test_metadata) != len(test_y):
            raise RuntimeError(
                f"Metadata/window mismatch for {spec.description}: "
                f"{len(test_metadata)} metadata rows vs {len(test_y)} test windows"
            )
        test_sites = [str(x["site"]) for x in test_metadata]
        print(f"  windows train={len(train_x)} val={len(val_x)} test={len(test_x)}")

        for model_name in args.models:
            print(f"  [model] {model_name}")
            set_global_seed(args.seed)
            model = make_model(model_name, spec, args)
            model, mean, std = train_model(
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
            pred = predict_model(model, test_x, test_y, test_x_mark, test_dec_mark, mean, std, args)
            row = {
                "model": model_name,
                "task": spec.task,
                "description": spec.description,
                "train_windows": len(train_x),
                "val_windows": len(val_x),
                "test_windows": len(test_x),
                **metrics(test_y, pred),
                **site_macro_metrics(test_y, pred, test_sites),
            }
            rows.append(row)
            if args.save_predictions:
                for window_idx in range(len(test_y)):
                    for step in range(spec.pred_len):
                        meta = test_metadata[window_idx]
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "task": spec.task,
                                "description": spec.description,
                                "window_index": window_idx,
                                "site": meta["site"],
                                "site_window_index": meta["site_window_index"],
                                "segment_id": meta["segment_id"],
                                "start": meta["start"],
                                "step": step + 1,
                                "y_true": float(test_y[window_idx, step]),
                                "y_pred": float(pred[window_idx, step]),
                            }
                        )
            print(f"    mse={row['mse']:.4f} mae={row['mae']:.4f} rmse={row['rmse']:.4f} r2={row['r2']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "unified_supervised_metrics.csv", index=False)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_root / "unified_supervised_predictions.csv", index=False)
    write_summary(df, output_root / "unified_supervised_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Unified Supervised Forecast Evaluation",
        "",
        "Supervised models are trained and tested on the same cleaned rolling windows used by the foundation-model evaluation.",
        "",
        "| Task | Description | Model | Train Windows | Test Windows | MSE | MAE | RMSE | R2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.sort_values(["task", "description", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.description} | {row.model} | {row.train_windows} | {row.test_windows} | "
            f"{row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    if {"site_macro_mse", "site_macro_r2"}.issubset(df.columns):
        lines.extend(
            [
                "",
        "Site-macro metrics average each site's point-level metric first. Site-median metrics expose whether a result is dominated by a small number of hard sites.",
                "",
                "| Task | Description | Model | Site-Macro MSE | Site-Macro MAE | Site-Macro RMSE | Site-Macro R2 | Site-Median MSE | Site-Median MAE |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in df.sort_values(["task", "description", "model"]).itertuples(index=False):
            lines.append(
                f"| {row.task} | {row.description} | {row.model} | "
                f"{row.site_macro_mse:.4f} | {row.site_macro_mae:.4f} | "
                f"{row.site_macro_rmse:.4f} | {row.site_macro_r2:.4f} | "
                f"{row.site_median_mse:.4f} | {row.site_median_mae:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified supervised model evaluation.")
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="30min")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["DLinear", "PatchTST", "iTransformer", "TSMixer", "TimeMixer", "PatchMLP", "TimeBridge", "TimeXer", "Transformer"],
        default=["DLinear"],
    )
    parser.add_argument("--output_root", default="carbon/unified_supervised_30min")
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
    parser.add_argument("--custom_daily_specs", default=None, help="Comma-separated seq:pred pairs, e.g. 180:60,365:90")
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
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
