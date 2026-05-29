#!/usr/bin/env python3
"""Evaluate simple ensembles over saved foundation-model prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["task", "description", "site", "window_index", "step", "date"]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "nse": r2}


def load_prediction(path: Path, pred_column: str, index: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in [*KEY_COLUMNS, "y_true", pred_column] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[[*KEY_COLUMNS, "y_true", pred_column]].rename(columns={pred_column: f"pred_{index}"})


def run(args: argparse.Namespace) -> None:
    paths = [Path(x) for x in args.predictions]
    if len(paths) < 2:
        raise ValueError("At least two prediction files are required for an ensemble.")

    merged = load_prediction(paths[0], args.pred_column, 0)
    for index, path in enumerate(paths[1:], start=1):
        merged = merged.merge(load_prediction(path, args.pred_column, index), on=[*KEY_COLUMNS, "y_true"], how="inner")

    pred_columns = [col for col in merged.columns if col.startswith("pred_")]
    pred_matrix = merged[pred_columns].to_numpy(dtype=np.float64)
    y_true = merged["y_true"].to_numpy(dtype=np.float64)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, values in {
        "ensemble_mean": pred_matrix.mean(axis=1),
        "ensemble_median": np.median(pred_matrix, axis=1),
    }.items():
        rows.append({"method": method, "members": len(pred_columns), **metric_values(y_true, values)})
        out = merged[[*KEY_COLUMNS, "y_true"]].copy()
        out["y_pred"] = values
        out.to_csv(output_root / f"{method}_predictions.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_root / "ensemble_metrics.csv", index=False)

    lines = [
        "# Foundation Prediction Ensemble",
        "",
        f"Prediction files: {len(paths)}",
        "",
        "| Method | Members | MSE | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.members} | {row.mse:.6f} | {row.mae:.6f} | "
            f"{row.rmse:.6f} | {row.r2:.6f} |"
        )
    (output_root / "ensemble_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mean/median ensembles from prediction CSV files.")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--pred_column", default="y_pred")
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
