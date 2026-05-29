#!/usr/bin/env python3
"""Generate paper-ready visualizations from existing carbon forecasting results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "carbon" / "exp_allscales_gated" / "selective_residual_predictions.csv"
DEFAULT_METRICS = ROOT / "carbon" / "exp_allscales_gated" / "selective_residual_metrics.csv"
DEFAULT_SUPERVISED_PREDICTIONS = ROOT / "carbon" / "exp_daily_long_supervised" / "unified_supervised_predictions.csv"
DEFAULT_SITE_METRICS = ROOT / "docs" / "figures" / "site_level_365_60_metrics.csv"
DEFAULT_CLEAN_DATA = ROOT / "carbon" / "clean" / "all_sites_daily_clean.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures"


PAPER_STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def parse_lengths(description: str) -> tuple[int, int]:
    numbers = [int(x) for x in re.findall(r"\d+", str(description))]
    if len(numbers) < 2:
        raise ValueError(f"Cannot parse sequence/prediction lengths from {description!r}")
    return numbers[0], numbers[1]


def format_horizon(description: str) -> str:
    seq_len, pred_len = parse_lengths(description)
    return f"{seq_len}-day input -> {pred_len}-day forecast"


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    corr = float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def describe_window(group: pd.DataFrame) -> dict[str, float | str | int]:
    y_true = group["y_true"].to_numpy(float)
    raw = group["foundation_raw"].to_numpy(float)
    gafa = group["adaptive_selective"].to_numpy(float)
    raw_metrics = metric_values(y_true, raw)
    gafa_metrics = metric_values(y_true, gafa)
    amp = float(np.max(y_true) - np.min(y_true))
    norm_rmse = float(gafa_metrics["rmse"] / max(amp, 1e-8))
    gain_vs_raw = float(raw_metrics["mse"] - gafa_metrics["mse"])
    return {
        "site": str(group["site"].iloc[0]),
        "window_index": int(group["window_index"].iloc[0]),
        "start_date": str(group["date"].iloc[0]),
        "end_date": str(group["date"].iloc[-1]),
        "mse_raw": raw_metrics["mse"],
        "mse_gafa": gafa_metrics["mse"],
        "mae_gafa": gafa_metrics["mae"],
        "rmse_gafa": gafa_metrics["rmse"],
        "corr_gafa": gafa_metrics["corr"],
        "amp": amp,
        "norm_rmse": norm_rmse,
        "gain_vs_raw": gain_vs_raw,
    }


def select_representative_windows(
    pred: pd.DataFrame,
    description: str,
    model: str,
    top_k: int,
    prefer_positive_gain: bool = False,
) -> pd.DataFrame:
    subset = pred[(pred["description"] == description) & (pred["model"] == model)].copy()
    if subset.empty:
        raise ValueError(f"No predictions found for model={model!r}, description={description!r}")

    rows = [
        describe_window(group.sort_values("step"))
        for _, group in subset.groupby(["site", "window_index"], sort=False)
    ]
    windows = pd.DataFrame(rows)
    amp_floor = float(windows["amp"].quantile(0.35))
    candidates = windows[
        (windows["amp"] >= amp_floor)
        & (windows["corr_gafa"].fillna(-1.0) >= 0.55)
        & (windows["mse_gafa"] <= windows["mse_gafa"].quantile(0.75))
        & (windows["norm_rmse"] <= windows["norm_rmse"].quantile(0.85))
    ].copy()
    if prefer_positive_gain:
        gain_candidates = windows[
            (windows["gain_vs_raw"] > 0)
            & (windows["corr_gafa"].fillna(-1.0) >= 0.40)
            & (windows["norm_rmse"] <= windows["norm_rmse"].quantile(0.85))
        ].copy()
        if not gain_candidates.empty:
            supplemental = windows[
                (~windows.set_index(["site", "window_index"]).index.isin(
                    gain_candidates.set_index(["site", "window_index"]).index
                ))
                & (windows["corr_gafa"].fillna(-1.0) >= 0.60)
                & (windows["norm_rmse"] <= windows["norm_rmse"].quantile(0.85))
            ].copy()
            candidates = pd.concat([gain_candidates, supplemental], ignore_index=True)
    if len(candidates) < top_k:
        candidates = windows.copy()

    candidates["score"] = (
        candidates["corr_gafa"].fillna(-1.0)
        + 0.35 * candidates["amp"].rank(pct=True)
        + 0.55 * candidates["gain_vs_raw"].rank(pct=True)
        - 0.65 * candidates["norm_rmse"].rank(pct=True)
        - 0.55 * candidates["mse_gafa"].rank(pct=True)
    )
    return candidates.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def load_history(clean_data: pd.DataFrame, site: str, start_date: str, seq_len: int) -> pd.DataFrame:
    site_data = clean_data[clean_data["site"] == site].copy()
    site_data["date"] = pd.to_datetime(site_data["date"])
    start = pd.to_datetime(start_date)
    history = site_data[site_data["date"] < start].tail(seq_len)
    return history[["date", "NEE_clean"]]


def plot_prediction_case(
    pred: pd.DataFrame,
    clean_data: pd.DataFrame,
    description: str,
    model: str,
    window: pd.Series,
    output_path: Path,
) -> None:
    seq_len, pred_len = parse_lengths(description)
    case = pred[
        (pred["description"] == description)
        & (pred["model"] == model)
        & (pred["site"] == window["site"])
        & (pred["window_index"] == window["window_index"])
    ].sort_values("step")
    history = load_history(clean_data, str(window["site"]), str(window["start_date"]), seq_len)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    if not history.empty:
        ax.plot(history["date"], history["NEE_clean"], color="#b7b7b7", linewidth=1.2, label="Input context")
        ax.axvspan(history["date"].iloc[0], history["date"].iloc[-1], color="#f2f2f2", alpha=0.75, zorder=0)

    dates = pd.to_datetime(case["date"])
    ax.plot(
        dates,
        case["y_true"],
        color="#111111",
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        label="Ground truth",
    )
    ax.plot(
        dates,
        case["foundation_raw"],
        color="#4c78a8",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=2.8,
        label="Raw foundation",
    )
    ax.plot(
        dates,
        case["adaptive_selective"],
        color="#d62728",
        linewidth=1.9,
        marker="^",
        markersize=3.0,
        label="GAFA",
    )
    ax.axvline(dates.iloc[0], color="#777777", linestyle=":", linewidth=1.1)

    title = (
        f"{format_horizon(description)} | {model} | {window['site']} "
        f"({window['start_date']} to {window['end_date']})"
    )
    subtitle = (
        f"GAFA MSE={window['mse_gafa']:.4f}, MAE={window['mae_gafa']:.4f}, "
        f"corr={window['corr_gafa']:.3f}, amp={window['amp']:.3f}; daily predictions"
    )
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_ylabel("Normalized NEE")
    ax.set_xlabel("Date")
    ax.legend(ncol=4, loc="upper left", frameon=False)
    ax.grid(alpha=0.22)
    fig.autofmt_xdate()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cumulative_prediction_case(
    pred: pd.DataFrame,
    clean_data: pd.DataFrame,
    description: str,
    model: str,
    window: pd.Series,
    output_path: Path,
) -> None:
    seq_len, _ = parse_lengths(description)
    case = pred[
        (pred["description"] == description)
        & (pred["model"] == model)
        & (pred["site"] == window["site"])
        & (pred["window_index"] == window["window_index"])
    ].sort_values("step")
    history = load_history(clean_data, str(window["site"]), str(window["start_date"]), seq_len)

    dates = pd.to_datetime(case["date"])
    true_cum = np.cumsum(case["y_true"].to_numpy(float))
    raw_cum = np.cumsum(case["foundation_raw"].to_numpy(float))
    gafa_cum = np.cumsum(case["adaptive_selective"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    if not history.empty:
        hist_dates = pd.to_datetime(history["date"])
        hist_cum = np.cumsum(history["NEE_clean"].to_numpy(float))
        hist_cum = hist_cum - hist_cum[-1]
        ax.plot(hist_dates, hist_cum, color="#b7b7b7", linewidth=1.2, label="Input context cumulative")
        ax.axvspan(hist_dates.iloc[0], hist_dates.iloc[-1], color="#f2f2f2", alpha=0.75, zorder=0)

    ax.plot(dates, true_cum, color="#111111", linewidth=2.2, marker="o", markersize=3.0, label="Ground truth cumulative")
    ax.plot(dates, raw_cum, color="#4c78a8", linewidth=1.8, linestyle="--", marker="s", markersize=2.6, label="Raw cumulative")
    ax.plot(dates, gafa_cum, color="#d62728", linewidth=2.0, marker="^", markersize=2.8, label="GAFA cumulative")
    ax.axvline(dates.iloc[0], color="#777777", linestyle=":", linewidth=1.1)

    final_true = true_cum[-1]
    final_gafa = gafa_cum[-1]
    final_raw = raw_cum[-1]
    title = (
        f"Cumulative forecast: {format_horizon(description)} | {model} | {window['site']} "
        f"({window['start_date']} to {window['end_date']})"
    )
    subtitle = (
        f"Final true={final_true:.3f}, raw error={final_raw - final_true:+.3f}, "
        f"GAFA error={final_gafa - final_true:+.3f}"
    )
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_ylabel("Cumulative normalized NEE")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.grid(alpha=0.22)
    fig.autofmt_xdate()
    fig.savefig(output_path)
    plt.close(fig)


def scatter_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    limits: tuple[float, float] | None = None,
) -> None:
    metrics = metric_values(y_true, y_pred)
    sample = min(len(y_true), 5000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=sample, replace=False) if len(y_true) > sample else np.arange(len(y_true))
    if limits is None:
        low = float(min(np.min(y_true), np.min(y_pred)))
        high = float(max(np.max(y_true), np.max(y_pred)))
        clipped = 0
    else:
        low, high = limits
        clipped = int(np.sum((y_true < low) | (y_true > high) | (y_pred < low) | (y_pred > high)))
    ax.scatter(y_true[idx], y_pred[idx], s=9, alpha=0.24, color="#2f6f9f", edgecolors="none")
    ax.plot([low, high], [low, high], color="#222222", linewidth=1.2, linestyle="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    clip_note = f", clipped={clipped}" if clipped else ""
    ax.set_title(f"{title}\nR2={metrics['r2']:.3f}, MSE={metrics['mse']:.4f}{clip_note}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(alpha=0.22)


def plot_global_scatter(pred: pd.DataFrame, description: str, model: str, output_path: Path) -> None:
    subset = pred[(pred["description"] == description) & (pred["model"] == model)].copy()
    if subset.empty:
        raise ValueError(f"No scatter data for model={model!r}, description={description!r}")
    y_true = subset["y_true"].to_numpy(float)
    raw = subset["foundation_raw"].to_numpy(float)
    gafa = subset["adaptive_selective"].to_numpy(float)
    central_values = np.concatenate([y_true, raw, gafa])
    low, high = np.quantile(central_values, [0.01, 0.99])
    pad = 0.05 * max(high - low, 1e-8)
    limits = (float(low - pad), float(high + pad))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharex=True, sharey=True)
    scatter_panel(axes[0], y_true, raw, "Raw foundation", limits=limits)
    scatter_panel(axes[1], y_true, gafa, "GAFA", limits=limits)
    fig.suptitle(f"Central true-vs-predicted scatter: {format_horizon(description)} | {model}", y=1.03)
    fig.savefig(output_path)
    plt.close(fig)


def aggregate_window_table(pred: pd.DataFrame, description: str, model: str) -> pd.DataFrame:
    subset = pred[(pred["description"] == description) & (pred["model"] == model)].copy()
    rows = []
    for (site, window_index), group in subset.groupby(["site", "window_index"], sort=False):
        y_true = group["y_true"].to_numpy(float)
        raw = group["foundation_raw"].to_numpy(float)
        gafa = group["adaptive_selective"].to_numpy(float)
        rows.append(
            {
                "site": site,
                "window_index": window_index,
                "true_mean": float(np.mean(y_true)),
                "raw_mean": float(np.mean(raw)),
                "gafa_mean": float(np.mean(gafa)),
                "true_cumulative": float(np.sum(y_true)),
                "raw_cumulative": float(np.sum(raw)),
                "gafa_cumulative": float(np.sum(gafa)),
                "daily_mse_raw": metric_values(y_true, raw)["mse"],
                "daily_mse_gafa": metric_values(y_true, gafa)["mse"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, raw_col, gafa_col, label in [
        ("true_mean", "raw_mean", "gafa_mean", "window_mean"),
        ("true_cumulative", "raw_cumulative", "gafa_cumulative", "window_cumulative"),
    ]:
        for method, pred_col in [("Raw foundation", raw_col), ("GAFA", gafa_col)]:
            metrics = metric_values(aggregate[target].to_numpy(float), aggregate[pred_col].to_numpy(float))
            rows.append(
                {
                    "aggregation": label,
                    "method": method,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "corr": metrics["corr"],
                }
            )
    return pd.DataFrame(rows)


def plot_aggregate_scatter(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    panels = [
        ("true_mean", "raw_mean", "gafa_mean", "Window mean NEE"),
        ("true_cumulative", "raw_cumulative", "gafa_cumulative", "Cumulative 60-day NEE"),
    ]
    colors = {"Raw foundation": "#4c78a8", "GAFA": "#d62728"}
    for ax, (target_col, raw_col, gafa_col, title) in zip(axes, panels):
        y_true = aggregate[target_col].to_numpy(float)
        values = np.concatenate([y_true, aggregate[raw_col].to_numpy(float), aggregate[gafa_col].to_numpy(float)])
        low, high = float(np.min(values)), float(np.max(values))
        pad = 0.06 * max(high - low, 1e-8)
        low -= pad
        high += pad
        for method, pred_col in [("Raw foundation", raw_col), ("GAFA", gafa_col)]:
            y_pred = aggregate[pred_col].to_numpy(float)
            metrics = metric_values(y_true, y_pred)
            ax.scatter(
                y_true,
                y_pred,
                s=28,
                alpha=0.72,
                color=colors[method],
                edgecolors="white",
                linewidth=0.4,
                label=f"{method} R2={metrics['r2']:.3f}",
            )
        ax.plot([low, high], [low, high], color="#222222", linewidth=1.2, linestyle="--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(title)
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Window-level aggregate prediction quality for 365-day input -> 60-day forecast", y=1.03)
    fig.savefig(output_path)
    plt.close(fig)


def window_mse_table(pred: pd.DataFrame, description: str, model: str) -> pd.DataFrame:
    subset = pred[(pred["description"] == description) & (pred["model"] == model)].copy()
    rows = []
    for (site, window_index), group in subset.groupby(["site", "window_index"], sort=False):
        y_true = group["y_true"].to_numpy(float)
        rows.append(
            {
                "site": site,
                "window_index": window_index,
                "method": "Raw foundation",
                "mse": metric_values(y_true, group["foundation_raw"].to_numpy(float))["mse"],
            }
        )
        rows.append(
            {
                "site": site,
                "window_index": window_index,
                "method": "GAFA",
                "mse": metric_values(y_true, group["adaptive_selective"].to_numpy(float))["mse"],
            }
        )
    return pd.DataFrame(rows)


def supervised_window_mse(supervised_pred: pd.DataFrame, description: str, model: str) -> pd.DataFrame:
    subset = supervised_pred[
        (supervised_pred["description"] == description) & (supervised_pred["model"] == model)
    ].copy()
    rows = []
    for window_index, group in subset.groupby("window_index", sort=False):
        rows.append(
            {
                "site": "pooled",
                "window_index": window_index,
                "method": model,
                "mse": metric_values(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float))["mse"],
            }
        )
    return pd.DataFrame(rows)


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


def build_site_windows(site_df: pd.DataFrame, seq_len: int, pred_len: int, max_windows: int) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    need = seq_len + pred_len
    valid = site_df[site_df["segment_id"] >= 0].copy()
    for segment_id, seg in valid.groupby("segment_id", sort=True):
        seg = seg.sort_values("date").reset_index(drop=True)
        values = seg["NEE_clean"].to_numpy(dtype=np.float32)
        if len(values) < need:
            continue
        starts = list(range(0, len(values) - need + 1, pred_len))
        if max_windows:
            starts = starts[-max_windows:]
        for start in starts:
            chunk = values[start : start + need]
            if np.isnan(chunk).any():
                continue
            windows.append(
                {
                    "segment_id": int(segment_id),
                    "start": int(start),
                    "target_dates": seg["date"].iloc[start + seq_len : start + need].to_numpy(),
                }
            )
    return windows


def supervised_test_window_sites(
    clean_data: pd.DataFrame,
    description: str,
    max_windows_per_site: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> pd.DataFrame:
    seq_len, pred_len = parse_lengths(description)
    df = clean_data.copy()
    if "segment_id" not in df.columns:
        df["segment_id"] = 0
    df["NEE_clean"] = pd.to_numeric(df["NEE_clean"], errors="coerce")
    rows = []
    global_idx = 0
    for site, site_df in df.groupby("site", sort=True):
        windows = build_site_windows(site_df, seq_len, pred_len, max_windows_per_site)
        if len(windows) < 3:
            continue
        train_end, val_end = split_counts(len(windows), train_ratio, val_ratio)
        for local_idx, window in enumerate(windows):
            if local_idx < val_end:
                continue
            dates = pd.to_datetime(window["target_dates"])
            rows.append(
                {
                    "window_index": global_idx,
                    "site": site,
                    "local_window_index": local_idx,
                    "start_date": str(dates[0].date()),
                    "end_date": str(dates[-1].date()),
                }
            )
            global_idx += 1
    return pd.DataFrame(rows)


def compute_site_metrics(
    pred: pd.DataFrame,
    supervised_pred: pd.DataFrame,
    clean_data: pd.DataFrame,
    description: str,
    foundation_model: str,
    supervised_model: str,
) -> pd.DataFrame:
    foundation = pred[(pred["description"] == description) & (pred["model"] == foundation_model)].copy()
    supervised = supervised_pred[
        (supervised_pred["description"] == description) & (supervised_pred["model"] == supervised_model)
    ].copy()
    site_map = supervised_test_window_sites(clean_data, description)
    if "site" not in supervised.columns:
        supervised = supervised.merge(site_map[["window_index", "site"]], on="window_index", how="left")

    rows = []
    all_sites = sorted(set(foundation["site"].dropna()) | set(supervised["site"].dropna()))
    for site in all_sites:
        f_site = foundation[foundation["site"] == site]
        s_site = supervised[supervised["site"] == site]
        if f_site.empty:
            continue
        y_true = f_site["y_true"].to_numpy(float)
        raw_metrics = metric_values(y_true, f_site["foundation_raw"].to_numpy(float))
        gafa_metrics = metric_values(y_true, f_site["adaptive_selective"].to_numpy(float))
        if s_site.empty:
            supervised_metrics = {"mse": np.nan, "mae": np.nan, "r2": np.nan}
        else:
            supervised_metrics = metric_values(s_site["y_true"].to_numpy(float), s_site["y_pred"].to_numpy(float))
        rows.append(
            {
                "site": site,
                "split": "test",
                "points": int(len(f_site)),
                "windows": int(f_site[["site", "window_index"]].drop_duplicates().shape[0]),
                f"{supervised_model}_mse": supervised_metrics["mse"],
                f"{supervised_model}_mae": supervised_metrics["mae"],
                f"{supervised_model}_r2": supervised_metrics["r2"],
                "TimeFM raw_mse": raw_metrics["mse"],
                "TimeFM raw_mae": raw_metrics["mae"],
                "TimeFM raw_r2": raw_metrics["r2"],
                "TimeFM + GAFA_mse": gafa_metrics["mse"],
                "TimeFM + GAFA_mae": gafa_metrics["mae"],
                "TimeFM + GAFA_r2": gafa_metrics["r2"],
                "gafa_gain_vs_timemixer_mse": supervised_metrics["mse"] - gafa_metrics["mse"],
                "gafa_gain_vs_raw_mse": raw_metrics["mse"] - gafa_metrics["mse"],
            }
        )
    return pd.DataFrame(rows)


def plot_window_mse_distribution(
    pred: pd.DataFrame,
    supervised_pred: pd.DataFrame,
    description: str,
    model: str,
    supervised_model: str,
    output_path: Path,
) -> pd.DataFrame:
    foundation_mse = window_mse_table(pred, description, model)
    supervised_mse = supervised_window_mse(supervised_pred, description, supervised_model)
    combined = pd.concat([supervised_mse, foundation_mse], ignore_index=True)
    order = [supervised_model, "Raw foundation", "GAFA"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = [combined.loc[combined["method"] == method, "mse"].to_numpy(float) for method in order]
    box = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    colors = ["#bab0ab", "#4c78a8", "#d62728"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    rng = np.random.default_rng(7)
    for i, values in enumerate(data, start=1):
        jitter = rng.normal(i, 0.035, size=len(values))
        ax.scatter(jitter, values, s=16, alpha=0.45, color=colors[i - 1], edgecolors="none")
    ax.set_title(f"Window-level MSE distribution: {format_horizon(description)}")
    ax.set_ylabel("MSE per forecast window")
    positive = combined.loc[combined["mse"] > 0, "mse"]
    if not positive.empty:
        ax.set_ylim(float(positive.min() * 0.55), float(positive.quantile(0.98) * 1.8))
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)
    return combined


def plot_site_gain(site_metrics: pd.DataFrame, output_path: Path) -> None:
    test = site_metrics[site_metrics["split"] == "test"].copy()
    if test.empty:
        test = site_metrics.copy()
    gain_col = "gafa_gain_vs_timemixer_mse"
    if gain_col not in test.columns:
        gain_col = "gafa_gain_vs_raw_mse"
    test = test.sort_values(gain_col, ascending=True)
    colors = np.where(test[gain_col] >= 0, "#d62728", "#4c78a8")

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    ax.barh(test["site"], test[gain_col], color=colors, alpha=0.82)
    ax.axvline(0, color="#222222", linewidth=1.0)
    baseline = "TimeMixer" if gain_col == "gafa_gain_vs_timemixer_mse" else "raw TimeFM v2"
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_title(f"Site-level MSE reduction of TimeFM v2 + GAFA over {baseline} (365 -> 60)")
    ax.set_xlabel("MSE reduction (positive means GAFA is better)")
    ax.set_ylabel("Site")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)


def plot_site_mse(site_metrics: pd.DataFrame, output_path: Path) -> None:
    test = site_metrics[site_metrics["split"] == "test"].copy()
    if test.empty:
        test = site_metrics.copy()
    required = ["TimeMixer_mse", "TimeFM raw_mse", "TimeFM + GAFA_mse"]
    missing = [col for col in required if col not in test.columns]
    if missing:
        return

    test = test.sort_values("TimeFM + GAFA_mse", ascending=True)
    x = np.arange(len(test))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.bar(x - width, test["TimeMixer_mse"], width, label="TimeMixer", color="#bab0ab", alpha=0.82)
    ax.bar(x, test["TimeFM raw_mse"], width, label="TimeFM v2 raw", color="#4c78a8", alpha=0.82)
    ax.bar(x + width, test["TimeFM + GAFA_mse"], width, label="TimeFM v2 + GAFA", color="#d62728", alpha=0.82)
    ax.set_yscale("log")
    ax.set_ylabel("Site-level MSE (log scale)")
    ax.set_xlabel("Test site")
    ax.set_title("Site-level MSE comparison for 365-day input -> 60-day forecast")
    ax.set_xticks(x)
    ax.set_xticklabels(test["site"], rotation=45, ha="right")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--supervised_predictions", type=Path, default=DEFAULT_SUPERVISED_PREDICTIONS)
    parser.add_argument("--site_metrics", type=Path, default=DEFAULT_SITE_METRICS)
    parser.add_argument("--clean_data", type=Path, default=DEFAULT_CLEAN_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    plt.rcParams.update(PAPER_STYLE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions)
    supervised_pred = pd.read_csv(args.supervised_predictions)
    clean_data = pd.read_csv(args.clean_data)

    flagship_description = "365天输入 -> 60天预测"
    flagship_model = "timefm_v2"
    flagship_supervised_model = "TimeMixer"
    short_description = "60天输入 -> 14天预测"
    short_model = "chronos_base"
    mid_description = "90天输入 -> 30天预测"
    mid_model = "timefm_v2"

    all_cases = []
    case_specs = [
        (short_description, short_model, "60_14_chronos_gafa", False),
        (mid_description, mid_model, "90_30_timefm_gafa", False),
        (flagship_description, flagship_model, "365_60_timefm_gafa", True),
        (flagship_description, flagship_model, "365_60_timefm_bestfit", False),
    ]
    for description, model, slug, prefer_positive_gain in case_specs:
        selected = select_representative_windows(
            pred,
            description,
            model,
            args.top_k,
            prefer_positive_gain=prefer_positive_gain,
        )
        selected.insert(0, "description", description)
        selected.insert(1, "model", model)
        selected.insert(2, "slug", slug)
        all_cases.append(selected)
        for idx, row in selected.iterrows():
            output = args.output_dir / f"paper_prediction_curve_{slug}_case{idx + 1}_{row['site']}.png"
            plot_prediction_case(pred, clean_data, description, model, row, output)
            if description == flagship_description:
                cumulative_output = args.output_dir / f"paper_cumulative_curve_{slug}_case{idx + 1}_{row['site']}.png"
                plot_cumulative_prediction_case(pred, clean_data, description, model, row, cumulative_output)

    cases = pd.concat(all_cases, ignore_index=True)
    cases.to_csv(args.output_dir / "paper_prediction_curve_selected_cases.csv", index=False)

    plot_global_scatter(
        pred,
        flagship_description,
        flagship_model,
        args.output_dir / "paper_scatter_365_60_timefm_raw_vs_gafa.png",
    )
    aggregate = aggregate_window_table(pred, flagship_description, flagship_model)
    aggregate.to_csv(args.output_dir / "paper_aggregate_window_365_60.csv", index=False)
    aggregate_metrics = aggregate_metrics_table(aggregate)
    aggregate_metrics.to_csv(args.output_dir / "paper_aggregate_metrics_365_60.csv", index=False)
    plot_aggregate_scatter(aggregate, args.output_dir / "paper_aggregate_scatter_365_60.png")
    window_mse = plot_window_mse_distribution(
        pred,
        supervised_pred,
        flagship_description,
        flagship_model,
        flagship_supervised_model,
        args.output_dir / "paper_window_mse_distribution_365_60.png",
    )
    window_mse.to_csv(args.output_dir / "paper_window_mse_distribution_365_60.csv", index=False)
    if args.site_metrics.exists():
        site_metrics = pd.read_csv(args.site_metrics)
    else:
        site_metrics = compute_site_metrics(
            pred,
            supervised_pred,
            clean_data,
            flagship_description,
            flagship_model,
            flagship_supervised_model,
        )
        site_metrics.to_csv(args.site_metrics, index=False)
    plot_site_gain(site_metrics, args.output_dir / "paper_site_level_gafa_gain_365_60.png")
    plot_site_mse(site_metrics, args.output_dir / "paper_site_level_mse_365_60.png")

    print("Generated paper visualizations in", args.output_dir)
    print(cases[["description", "model", "site", "window_index", "mse_gafa", "corr_gafa", "amp", "score"]])


if __name__ == "__main__":
    main()
