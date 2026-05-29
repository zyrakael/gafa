#!/usr/bin/env python3
"""Site-paired bootstrap confidence intervals for GAFA gains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RUNS = {
    "30->7": {
        "setting": "30天输入 -> 7天预测",
        "path": "carbon/exp_allscales_gated/selective_residual_predictions.csv",
    },
    "60->14": {
        "setting": "60天输入 -> 14天预测",
        "path": "carbon/exp_allscales_gated/selective_residual_predictions.csv",
    },
    "90->30": {
        "setting": "90天输入 -> 30天预测",
        "path": "carbon/exp_allscales_nogate/selective_residual_predictions.csv",
    },
}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def paired_site_groups(df: pd.DataFrame, model: str, setting: str) -> list[dict[str, np.ndarray | str | float]]:
    subset = df[(df["model"] == model) & (df["description"] == setting)].copy()
    if subset.empty:
        raise ValueError(f"No predictions found for model={model} setting={setting}")

    rows = []
    for site, group in subset.groupby("site", sort=True):
        y = group["y_true"].to_numpy(dtype=np.float64)
        raw = group["foundation_raw"].to_numpy(dtype=np.float64)
        aligned = group["adaptive_selective"].to_numpy(dtype=np.float64)
        raw_r2 = r2_score(y, raw)
        aligned_r2 = r2_score(y, aligned)
        raw_mse = float(np.mean((y - raw) ** 2))
        aligned_mse = float(np.mean((y - aligned) ** 2))
        rows.append(
            {
                "site": site,
                "y_true": y,
                "raw_pred": raw,
                "aligned_pred": aligned,
                "raw_r2": raw_r2,
                "aligned_r2": aligned_r2,
                "r2_gain": aligned_r2 - raw_r2,
                "raw_mse": raw_mse,
                "aligned_mse": aligned_mse,
                "mse_reduction": raw_mse - aligned_mse,
            }
        )
    return rows


def pooled_metrics(groups: list[dict[str, np.ndarray | str | float]]) -> dict[str, float]:
    y = np.concatenate([np.asarray(group["y_true"], dtype=np.float64) for group in groups])
    raw = np.concatenate([np.asarray(group["raw_pred"], dtype=np.float64) for group in groups])
    aligned = np.concatenate([np.asarray(group["aligned_pred"], dtype=np.float64) for group in groups])
    raw_mse = float(np.mean((y - raw) ** 2))
    aligned_mse = float(np.mean((y - aligned) ** 2))
    raw_r2 = r2_score(y, raw)
    aligned_r2 = r2_score(y, aligned)
    return {
        "raw_r2": raw_r2,
        "aligned_r2": aligned_r2,
        "r2_gain": aligned_r2 - raw_r2,
        "raw_mse": raw_mse,
        "aligned_mse": aligned_mse,
        "mse_reduction": raw_mse - aligned_mse,
    }


def bootstrap_gain(groups: list[dict[str, np.ndarray | str | float]], repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(groups)
    gains = np.empty(repeats, dtype=np.float64)
    mse_gains = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        sample = [groups[j] for j in idx]
        stats = pooled_metrics(sample)
        gains[i] = stats["r2_gain"]
        mse_gains[i] = stats["mse_reduction"]
    observed = pooled_metrics(groups)
    p_nonpositive = float(np.mean(gains <= 0.0))
    return {
        "raw_pooled_r2": observed["raw_r2"],
        "aligned_pooled_r2": observed["aligned_r2"],
        "r2_gain": observed["r2_gain"],
        "r2_gain_ci_low": float(np.quantile(gains, 0.025)),
        "r2_gain_ci_high": float(np.quantile(gains, 0.975)),
        "bootstrap_p_gain_leq_0": p_nonpositive,
        "raw_mse": observed["raw_mse"],
        "aligned_mse": observed["aligned_mse"],
        "mse_reduction": observed["mse_reduction"],
        "mse_reduction_ci_low": float(np.quantile(mse_gains, 0.025)),
        "mse_reduction_ci_high": float(np.quantile(mse_gains, 0.975)),
    }


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    site_tables = []
    for offset, (label, config) in enumerate(DEFAULT_RUNS.items()):
        df = pd.read_csv(ROOT / config["path"])
        groups = paired_site_groups(df, args.model, config["setting"])
        stats = bootstrap_gain(groups, args.repeats, args.seed + offset)
        rows.append(
            {
                "setting": label,
                "source": config["path"],
                "sites": len(groups),
                **stats,
            }
        )
        site_rows = []
        for group in groups:
            site_rows.append(
                {
                    "setting": label,
                    "site": group["site"],
                    "raw_r2": group["raw_r2"],
                    "aligned_r2": group["aligned_r2"],
                    "r2_gain": group["r2_gain"],
                    "raw_mse": group["raw_mse"],
                    "aligned_mse": group["aligned_mse"],
                    "mse_reduction": group["mse_reduction"],
                }
            )
        site_tables.append(pd.DataFrame(site_rows))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "paired_bootstrap_ci.csv", index=False)
    pd.concat(site_tables, ignore_index=True).to_csv(output_root / "paired_bootstrap_site_scores.csv", index=False)
    write_summary(summary, output_root / "paired_bootstrap_ci.md")
    print(summary.to_string(index=False))
    print(f"\n[done] wrote paired bootstrap results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Paired Bootstrap CI",
        "",
        "Bootstrap unit: site. Each replicate resamples sites with replacement, pools the sampled forecast points, and recomputes the aligned-minus-raw R2 gain.",
        "",
        "| Setting | Sites | Raw pooled R2 | Aligned pooled R2 | Gain | 95% CI | P(gain <= 0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.sites} | {row.raw_pooled_r2:.4f} | {row.aligned_pooled_r2:.4f} | "
            f"{row.r2_gain:.4f} | [{row.r2_gain_ci_low:.4f}, {row.r2_gain_ci_high:.4f}] | "
            f"{row.bootstrap_p_gain_leq_0:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute site-paired bootstrap CI for GAFA gains.")
    parser.add_argument("--model", default="chronos_base")
    parser.add_argument("--output_root", default="carbon/paired_bootstrap_ci_cikm")
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
