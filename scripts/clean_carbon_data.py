#!/usr/bin/env python3
"""Clean carbon NEE datasets and keep explicit quality masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower().strip(): c for c in df.columns}
    required = {"site", "date", "nee"}
    if not required.issubset(col_map):
        raise ValueError("CSV must contain site/date/NEE columns.")
    out = df.rename(columns={col_map["site"]: "site", col_map["date"]: "date", col_map["nee"]: "NEE"})
    out = out[["site", "date", "NEE"]].copy()
    out["site"] = out["site"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["NEE"] = pd.to_numeric(out["NEE"], errors="coerce")
    return out.dropna(subset=["site", "date", "NEE"])


def robust_outlier_mask(values: pd.Series, z_threshold: float) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=values.index)
    robust_z = 0.6745 * (values - median) / mad
    return robust_z.abs() > z_threshold


def contiguous_missing_run(mask: pd.Series) -> pd.Series:
    run = np.zeros(len(mask), dtype=int)
    current = 0
    for i, is_missing in enumerate(mask.to_numpy(dtype=bool)):
        current = current + 1 if is_missing else 0
        run[i] = current
    # propagate final run length to every point inside each missing block
    out = run.copy()
    i = len(out) - 1
    while i >= 0:
        if out[i] == 0:
            i -= 1
            continue
        length = out[i]
        out[i - length + 1 : i + 1] = length
        i -= length
    return pd.Series(out, index=mask.index)


def clean_site(
    site: str,
    site_df: pd.DataFrame,
    freq: str,
    max_interp_steps: int,
    z_threshold: float,
    winsor_q: float,
) -> pd.DataFrame:
    grouped = (
        site_df.groupby("date", as_index=False)
        .agg(NEE_raw=("NEE", "mean"), duplicate_count=("NEE", "size"))
        .sort_values("date")
    )
    full_index = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq=freq)
    out = grouped.set_index("date").reindex(full_index)
    out.index.name = "date"
    out["site"] = site
    out["is_observed"] = out["NEE_raw"].notna()
    out["duplicate_count"] = out["duplicate_count"].fillna(0).astype(int)

    missing_run = contiguous_missing_run(~out["is_observed"])
    out["missing_run_length"] = missing_run.astype(int)
    out["is_long_gap"] = (~out["is_observed"]) & (out["missing_run_length"] > max_interp_steps)

    interp = out["NEE_raw"].interpolate("time", limit=max_interp_steps, limit_direction="both")
    interp[out["is_long_gap"]] = np.nan
    out["is_interpolated"] = (~out["is_observed"]) & interp.notna()

    observed_values = out.loc[out["is_observed"], "NEE_raw"]
    lower = observed_values.quantile(winsor_q)
    upper = observed_values.quantile(1 - winsor_q)
    outlier_mask = robust_outlier_mask(out["NEE_raw"], z_threshold).fillna(False)
    out["is_outlier"] = outlier_mask
    out["NEE_clean"] = interp.clip(lower=lower, upper=upper)

    # Segment ids break at long gaps so downstream windows never cross artificial discontinuities.
    segment_break = out["is_long_gap"] & ~out["is_long_gap"].shift(fill_value=False)
    out["segment_id"] = segment_break.cumsum().astype(int)
    out.loc[out["is_long_gap"], "segment_id"] = -1
    return out.reset_index()[["site", "date", "NEE_raw", "NEE_clean", "is_observed", "is_interpolated", "is_long_gap", "is_outlier", "duplicate_count", "missing_run_length", "segment_id"]]


def clean_file(
    input_path: Path,
    output_path: Path,
    freq: str,
    max_interp_steps: int,
    z_threshold: float,
    winsor_q: float,
) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(input_path))
    cleaned = [
        clean_site(site, site_df, freq, max_interp_steps, z_threshold, winsor_q)
        for site, site_df in df.groupby("site", sort=True)
    ]
    out = pd.concat(cleaned, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def write_summary(clean_30min: pd.DataFrame, clean_daily: pd.DataFrame, output_dir: Path) -> None:
    def summarize(df: pd.DataFrame, name: str) -> list[str]:
        return [
            f"### {name}",
            "",
            f"- rows after reindex: {len(df):,}",
            f"- sites: {df['site'].nunique()}",
            f"- observed rows: {int(df['is_observed'].sum()):,}",
            f"- interpolated rows: {int(df['is_interpolated'].sum()):,}",
            f"- long-gap rows left as missing: {int(df['is_long_gap'].sum()):,}",
            f"- duplicated original timestamps: {int((df['duplicate_count'] > 1).sum()):,}",
            f"- outlier-flagged observed rows: {int(df['is_outlier'].sum()):,}",
            "",
        ]

    lines = ["# Clean Carbon Data Summary", ""]
    lines.extend(summarize(clean_30min, "30-minute"))
    lines.extend(summarize(clean_daily, "Daily"))
    lines.extend(
        [
            "## Outputs",
            "",
            "- `all_sites_30min_clean.csv`: full 30-minute grid with quality masks.",
            "- `all_sites_daily_clean.csv`: full daily grid with quality masks.",
            "",
            "Long gaps are intentionally not imputed. Downstream rolling-window evaluation should avoid windows crossing `segment_id = -1` rows.",
        ]
    )
    (output_dir / "cleaning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean carbon NEE data and write compact cleaned files.")
    parser.add_argument("--input_30min", default="carbon/all_sites_30min.csv")
    parser.add_argument("--input_daily", default="carbon/all_sites_daily.csv")
    parser.add_argument("--output_dir", default="carbon/clean")
    parser.add_argument("--max_interp_30min", type=int, default=4, help="Interpolate gaps up to this many 30-min steps.")
    parser.add_argument("--max_interp_daily", type=int, default=2, help="Interpolate gaps up to this many daily steps.")
    parser.add_argument("--outlier_z", type=float, default=12.0)
    parser.add_argument("--winsor_q", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    clean_30min = clean_file(
        Path(args.input_30min),
        output_dir / "all_sites_30min_clean.csv",
        "30min",
        args.max_interp_30min,
        args.outlier_z,
        args.winsor_q,
    )
    clean_daily = clean_file(
        Path(args.input_daily),
        output_dir / "all_sites_daily_clean.csv",
        "D",
        args.max_interp_daily,
        args.outlier_z,
        args.winsor_q,
    )
    write_summary(clean_30min, clean_daily, output_dir)
    print(f"Wrote cleaned data to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
