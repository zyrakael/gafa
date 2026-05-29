#!/usr/bin/env python3
"""Zero-shot carbon NEE forecasting with local TimeFM v2 and Chronos Base models."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ForecastSpec:
    task: str
    data_path: Path
    freq: str
    seq_len: int
    pred_len: int
    description: str


DEFAULT_30MIN_SPECS = (
    ForecastSpec("30min_longterm", Path("carbon/all_sites_30min.csv"), "30min", 336, 96, "7天输入 -> 2天预测"),
    ForecastSpec("30min_longterm", Path("carbon/all_sites_30min.csv"), "30min", 336, 192, "7天输入 -> 4天预测"),
    ForecastSpec("30min_longterm", Path("carbon/all_sites_30min.csv"), "30min", 336, 336, "7天输入 -> 7天预测"),
    ForecastSpec("30min_longterm", Path("carbon/all_sites_30min.csv"), "30min", 672, 672, "14天输入 -> 14天预测"),
    ForecastSpec("30min_longterm", Path("carbon/all_sites_30min.csv"), "30min", 672, 1440, "14天输入 -> 30天预测"),
)

DEFAULT_DAILY_SPECS = (
    ForecastSpec("daily_shortterm", Path("carbon/all_sites_daily.csv"), "D", 30, 7, "30天输入 -> 7天预测"),
    ForecastSpec("daily_shortterm", Path("carbon/all_sites_daily.csv"), "D", 60, 14, "60天输入 -> 14天预测"),
    ForecastSpec("daily_shortterm", Path("carbon/all_sites_daily.csv"), "D", 90, 30, "90天输入 -> 30天预测"),
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower().strip(): c for c in df.columns}
    required = {"site", "date", "nee"}
    if not required.issubset(col_map):
        raise ValueError("Input CSV must contain site/date/NEE columns.")
    out = df.rename(columns={col_map["site"]: "site", col_map["date"]: "date", col_map["nee"]: "NEE"})
    out = out[["site", "date", "NEE"]].copy()
    out["site"] = out["site"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["NEE"] = pd.to_numeric(out["NEE"], errors="coerce")
    return out.dropna(subset=["site", "date", "NEE"])


def reindex_site(site_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    site_df = site_df.sort_values("date").drop_duplicates("date", keep="last")
    full_index = pd.date_range(site_df["date"].min(), site_df["date"].max(), freq=freq)
    out = site_df.set_index("date").reindex(full_index)
    out.index.name = "date"
    out["site"] = site_df["site"].iloc[0]
    out["NEE"] = out["NEE"].interpolate("time").ffill().bfill()
    return out.reset_index()


def gluonts_freq(freq: str) -> str:
    if freq == "30min":
        return "30min"
    if freq == "D":
        return "D"
    return freq


def forecast_to_1d(values: object, pred_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim >= 2:
        arr = np.squeeze(arr, axis=tuple(i for i, size in enumerate(arr.shape[:-1]) if size == 1))
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr[:pred_len]


def load_series(data_path: Path, freq: str, sites: set[str] | None, reindex: bool) -> dict[str, pd.DataFrame]:
    df = normalize_columns(pd.read_csv(data_path))
    if sites:
        df = df[df["site"].isin(sites)].copy()
    if df.empty:
        raise ValueError(f"No usable rows found in {data_path}")

    series = {}
    for site, site_df in df.groupby("site", sort=True):
        prepared = reindex_site(site_df, freq) if reindex else site_df.sort_values("date").reset_index(drop=True)
        series[site] = prepared
    return series


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    score = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": score, "nse": score}


class TimeFmV2Forecaster:
    name = "timefm_v2"

    def __init__(self, model_path: Path, device: str):
        from transformers import TimesFmModelForPrediction

        self.device = torch.device(device)
        self.model = TimesFmModelForPrediction.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        self.chunk_len = int(getattr(self.model.config, "horizon_length", 128))

    def predict(self, context: np.ndarray, pred_len: int, freq: str) -> np.ndarray:
        history = np.asarray(context, dtype=np.float32)
        preds: list[np.ndarray] = []
        freq_id = 0 if freq in {"30min", "D"} else 1

        while sum(len(p) for p in preds) < pred_len:
            model_history = self._pad_to_patch_multiple(history)
            with torch.no_grad():
                out = self.model(
                    past_values=[torch.tensor(model_history, dtype=torch.float32, device=self.device)],
                    freq=[freq_id],
                    forecast_context_len=min(len(model_history), int(self.model.config.context_length)),
                )
            chunk = out.mean_predictions[0].detach().cpu().numpy()
            needed = pred_len - sum(len(p) for p in preds)
            chunk = chunk[: min(needed, self.chunk_len)]
            preds.append(chunk)
            history = np.concatenate([history, chunk])

        return np.concatenate(preds)[:pred_len]

    def _pad_to_patch_multiple(self, values: np.ndarray) -> np.ndarray:
        patch_len = int(self.model.config.patch_length)
        remainder = len(values) % patch_len
        if remainder == 0:
            return values
        pad_len = patch_len - remainder
        pad_value = values[0] if len(values) else 0.0
        return np.concatenate([np.full(pad_len, pad_value, dtype=np.float32), values])


class ChronosBaseForecaster:
    name = "chronos_base"

    def __init__(self, model_path: Path, device: str, num_samples: int):
        from chronos import ChronosPipeline

        self.device = device
        self.num_samples = num_samples
        self.pipeline = ChronosPipeline.from_pretrained(
            str(model_path),
            device_map=device,
            torch_dtype=torch.float32,
        )

    def predict(self, context: np.ndarray, pred_len: int, freq: str) -> np.ndarray:
        del freq
        tensor = torch.tensor(context, dtype=torch.float32)
        with torch.no_grad():
            samples = self.pipeline.predict(
                tensor,
                prediction_length=pred_len,
                num_samples=self.num_samples,
                limit_prediction_length=False,
            )
        return torch.median(samples[0], dim=0).values.detach().cpu().numpy()


class Moirai2Forecaster:
    name = "moirai2"

    def __init__(self, model_id: str, device: str, batch_size: int):
        try:
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "Moirai 2.0 requires the `uni2ts` and `gluonts` packages to be installed."
            ) from exc

        self.device = device
        self.batch_size = batch_size
        self._list_dataset_cls = ListDataset
        try:
            module = Moirai2Module.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Moirai 2.0 weights. If this machine cannot access Hugging Face, "
                "download `Salesforce/moirai-2.0-R-small` to a local directory first and pass that "
                "directory via `--moirai2_model_id /path/to/moirai-2.0-R-small`."
            ) from exc

        self._forecast = Moirai2Forecast(
            module=module,
            prediction_length=1,
            context_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def predict(self, context: np.ndarray, pred_len: int, freq: str) -> np.ndarray:
        freq = gluonts_freq(freq)
        dataset = self._list_dataset_cls(
            [
                {
                    "start": pd.Period("2000-01-01", freq=freq),
                    "target": np.asarray(context, dtype=np.float32),
                }
            ],
            freq=freq,
            one_dim_target=True,
        )
        with self._forecast.hparams_context(
            prediction_length=pred_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=len(context),
        ):
            predictor = self._forecast.create_predictor(batch_size=self.batch_size, device=self.device)
            forecast = next(iter(predictor.predict(dataset)))
        return forecast_to_1d(forecast.quantile(0.5), pred_len)


def build_forecasters(args: argparse.Namespace) -> list[object]:
    requested = set(args.models)
    forecasters: list[object] = []
    if "timefm_v2" in requested:
        forecasters.append(TimeFmV2Forecaster(Path(args.timefm_path), args.device))
    if "chronos_base" in requested:
        forecasters.append(ChronosBaseForecaster(Path(args.chronos_path), args.device, args.chronos_samples))
    if "moirai2" in requested:
        forecasters.append(Moirai2Forecaster(args.moirai2_model_id, args.device, args.moirai2_batch_size))
    return forecasters


def iter_specs(args: argparse.Namespace) -> Iterable[ForecastSpec]:
    if args.task in {"30min", "both"}:
        yield from DEFAULT_30MIN_SPECS
    if args.task in {"daily", "both"}:
        yield from DEFAULT_DAILY_SPECS


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sites = set(args.sites) if args.sites else None
    forecasters = build_forecasters(args)
    series_cache: dict[tuple[Path, str], dict[str, pd.DataFrame]] = {}
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for spec in iter_specs(args):
        data_path = (project_root / spec.data_path).resolve()
        cache_key = (data_path, spec.freq)
        if cache_key not in series_cache:
            series_cache[cache_key] = load_series(data_path, spec.freq, sites, args.reindex)

        site_items = sorted(series_cache[cache_key].items())
        if args.max_sites:
            site_items = site_items[: args.max_sites]

        print(f"\n[spec] {spec.task} | {spec.description} | sites={len(site_items)}")
        for forecaster in forecasters:
            print(f"[model] {forecaster.name}")
            for site, site_df in site_items:
                need_len = spec.seq_len + spec.pred_len
                if len(site_df) < need_len:
                    print(f"  [skip] {site}: only {len(site_df)} rows, need {need_len}")
                    continue

                tail = site_df.iloc[-need_len:].reset_index(drop=True)
                context = tail["NEE"].iloc[: spec.seq_len].to_numpy(dtype=np.float32)
                y_true = tail["NEE"].iloc[spec.seq_len :].to_numpy(dtype=np.float32)
                dates = tail["date"].iloc[spec.seq_len :].to_numpy()
                y_pred = forecaster.predict(context, spec.pred_len, spec.freq)
                score = metrics(y_true, y_pred)

                metric_rows.append(
                    {
                        "model": forecaster.name,
                        "task": spec.task,
                        "site": site,
                        "seq_len": spec.seq_len,
                        "pred_len": spec.pred_len,
                        "description": spec.description,
                        **score,
                    }
                )
                for step, (date, true_value, pred_value) in enumerate(zip(dates, y_true, y_pred), start=1):
                    prediction_rows.append(
                        {
                            "model": forecaster.name,
                            "task": spec.task,
                            "site": site,
                            "seq_len": spec.seq_len,
                            "pred_len": spec.pred_len,
                            "step": step,
                            "date": date,
                            "y_true": float(true_value),
                            "y_pred": float(pred_value),
                        }
                    )
                print(f"  {site}: mse={score['mse']:.4f}, mae={score['mae']:.4f}, r2={score['r2']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(output_root / "foundation_metrics_by_site.csv", index=False)
    predictions_df.to_csv(output_root / "foundation_predictions.csv", index=False)
    write_summary(metrics_df, output_root / "foundation_results_summary.md")
    return metrics_df, predictions_df


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    return (
        metrics_df.groupby(["model", "task", "seq_len", "pred_len", "description"], as_index=False)
        .agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            nse=("nse", "mean"),
            sites=("site", "nunique"),
        )
        .sort_values(["task", "pred_len", "mse", "model"])
    )


def write_summary(metrics_df: pd.DataFrame, path: Path) -> None:
    agg = aggregate_metrics(metrics_df)
    lines = ["# Foundation Model Carbon Forecast Results", ""]
    if agg.empty:
        lines.append("No successful forecasts were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "本报告由 `run_foundation_carbon.py` 生成，指标为各站点最后一个预测窗口的平均结果。",
            "",
            "## Aggregate Metrics",
            "",
            "| Model | Task | Description | Sites | MSE | MAE | RMSE | R2 | NSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in agg.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.task} | {row.description} | {row.sites} | "
            f"{row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} | {row.nse:.4f} |"
        )

    lines.extend(["", "## Best Model by Horizon", "", "| Task | Description | Best Model | MSE | R2 |", "| --- | --- | --- | --- | --- |"])
    for _, group in agg.groupby(["task", "seq_len", "pred_len", "description"], sort=False):
        best = group.sort_values("mse").iloc[0]
        lines.append(f"| {best['task']} | {best['description']} | {best['model']} | {best['mse']:.4f} | {best['r2']:.4f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot carbon forecasting with local foundation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=["timefm_v2", "chronos_base"])
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
    parser.add_argument("--moirai2_model_id", default="Salesforce/moirai-2.0-R-small")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--output_root", default="carbon/foundation_results")
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reindex", dest="reindex", action="store_true", help="Reindex each site to a complete time grid.")
    parser.add_argument("--no_reindex", dest="reindex", action="store_false", help="Use raw timestamps without reindexing.")
    parser.set_defaults(reindex=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
