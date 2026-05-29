#!/usr/bin/env python3
"""Unified rolling-window evaluation for baselines and time-series foundation models.

This script evaluates every model on the same cleaned windows and adds a
selective-learning-inspired reliability view: foundation model timesteps with
high predictive uncertainty can be masked out, reporting both coverage and error.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Spec:
    task: str
    data_path: Path
    freq: str
    seq_len: int
    pred_len: int
    period: int
    description: str


SPECS_30MIN = (
    Spec("30min_longterm", Path("carbon/clean/all_sites_30min_clean.csv"), "30min", 336, 96, 48, "7天输入 -> 2天预测"),
    Spec("30min_longterm", Path("carbon/clean/all_sites_30min_clean.csv"), "30min", 336, 192, 48, "7天输入 -> 4天预测"),
    Spec("30min_longterm", Path("carbon/clean/all_sites_30min_clean.csv"), "30min", 336, 336, 48, "7天输入 -> 7天预测"),
    Spec("30min_longterm", Path("carbon/clean/all_sites_30min_clean.csv"), "30min", 672, 672, 48, "14天输入 -> 14天预测"),
    Spec("30min_longterm", Path("carbon/clean/all_sites_30min_clean.csv"), "30min", 672, 1440, 48, "14天输入 -> 30天预测"),
)

SPECS_DAILY = (
    Spec("daily_shortterm", Path("carbon/clean/all_sites_daily_clean.csv"), "D", 30, 7, 7, "30天输入 -> 7天预测"),
    Spec("daily_shortterm", Path("carbon/clean/all_sites_daily_clean.csv"), "D", 60, 14, 7, "60天输入 -> 14天预测"),
    Spec("daily_shortterm", Path("carbon/clean/all_sites_daily_clean.csv"), "D", 90, 30, 7, "90天输入 -> 30天预测"),
)


def parse_custom_daily_specs(spec_text: str | None) -> tuple[Spec, ...]:
    if not spec_text:
        return ()
    specs: list[Spec] = []
    for chunk in spec_text.split(","):
        item = chunk.strip()
        if not item:
            continue
        seq_text, pred_text = item.split(":")
        seq_len = int(seq_text)
        pred_len = int(pred_text)
        specs.append(
            Spec(
                "daily_longhorizon",
                Path("carbon/clean/all_sites_daily_clean.csv"),
                "D",
                seq_len,
                pred_len,
                7,
                f"{seq_len}天输入 -> {pred_len}天预测",
            )
        )
    return tuple(specs)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.sum() == 0:
        return {"mse": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "nse": np.nan, "coverage": 0.0}
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2)) / denom
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "nse": r2, "coverage": float(mask.mean())}


def load_clean(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    df["site"] = df["site"].astype(str)
    df["NEE_clean"] = pd.to_numeric(df["NEE_clean"], errors="coerce")
    if "segment_id" not in df.columns:
        df["segment_id"] = 0
    return {site: g.sort_values("date").reset_index(drop=True) for site, g in df.groupby("site", sort=True)}


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


def build_windows(
    site_df: pd.DataFrame,
    spec: Spec,
    stride: int,
    max_windows: int | None,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    need = spec.seq_len + spec.pred_len
    valid = site_df[site_df["segment_id"] >= 0].copy()
    for segment_id, seg in valid.groupby("segment_id", sort=True):
        seg = seg.reset_index(drop=True)
        values = seg["NEE_clean"].to_numpy(dtype=np.float32)
        if len(values) < need:
            continue
        starts = list(range(0, len(values) - need + 1, stride))
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
                    "dates": seg["date"].iloc[start + spec.seq_len : start + need].to_numpy(),
                    "context": chunk[: spec.seq_len].copy(),
                    "target": chunk[spec.seq_len :].copy(),
                }
            )
    return windows


class Forecaster:
    name: str

    def predict(self, context: np.ndarray, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TimeFmForecaster(Forecaster):
    name = "timefm_v2"

    def __init__(self, model_path: Path, device: str):
        from transformers import TimesFmModelForPrediction

        self.device = torch.device(device)
        self.model = TimesFmModelForPrediction.from_pretrained(str(model_path), local_files_only=True, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.chunk_len = int(self.model.config.horizon_length)
        self.patch_len = int(self.model.config.patch_length)

    def _pad(self, values: np.ndarray) -> np.ndarray:
        remainder = len(values) % self.patch_len
        if remainder == 0:
            return values
        pad_len = self.patch_len - remainder
        return np.concatenate([np.full(pad_len, values[0], dtype=np.float32), values])

    def predict(self, context: np.ndarray, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
        history = context.astype(np.float32)
        preds: list[np.ndarray] = []
        uncs: list[np.ndarray] = []
        freq_id = 0 if spec.freq in {"30min", "D"} else 1
        while sum(len(x) for x in preds) < spec.pred_len:
            model_history = self._pad(history)
            with torch.no_grad():
                out = self.model(
                    past_values=[torch.tensor(model_history, dtype=torch.float32, device=self.device)],
                    freq=[freq_id],
                    forecast_context_len=min(len(model_history), int(self.model.config.context_length)),
                )
            mean = out.mean_predictions[0].detach().cpu().numpy()
            full = out.full_predictions[0].detach().cpu().numpy()
            unc = np.nanstd(full, axis=1)
            need = spec.pred_len - sum(len(x) for x in preds)
            preds.append(mean[:need])
            uncs.append(unc[:need])
            history = np.concatenate([history, mean[:need]])
        return np.concatenate(preds)[: spec.pred_len], np.concatenate(uncs)[: spec.pred_len]


class ChronosForecaster(Forecaster):
    name = "chronos_base"

    def __init__(self, model_path: Path, device: str, num_samples: int):
        from chronos import ChronosPipeline

        self.pipeline = ChronosPipeline.from_pretrained(str(model_path), device_map=device, torch_dtype=torch.float32)
        self.num_samples = num_samples

    def predict(self, context: np.ndarray, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            samples = self.pipeline.predict(
                torch.tensor(context, dtype=torch.float32),
                prediction_length=spec.pred_len,
                num_samples=self.num_samples,
                limit_prediction_length=False,
            )[0]
        pred = torch.median(samples, dim=0).values.detach().cpu().numpy()
        unc = torch.std(samples, dim=0).detach().cpu().numpy()
        return pred, unc


class Moirai2Forecaster(Forecaster):
    name = "moirai2"

    def __init__(self, model_id: str, device: str, batch_size: int):
        try:
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "Moirai 2.0 requires the `uni2ts` and `gluonts` packages. "
                "Install them first, e.g. `python -m pip install uni2ts --no-deps` "
                "plus the missing lightweight runtime dependencies in your environment."
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

    def predict(self, context: np.ndarray, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
        freq = gluonts_freq(spec.freq)
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
            prediction_length=spec.pred_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=len(context),
        ):
            predictor = self._forecast.create_predictor(batch_size=self.batch_size, device=self.device)
            forecast = next(iter(predictor.predict(dataset)))

        pred = forecast_to_1d(forecast.quantile(0.5), spec.pred_len)
        q10 = forecast_to_1d(forecast.quantile(0.1), spec.pred_len)
        q90 = forecast_to_1d(forecast.quantile(0.9), spec.pred_len)
        unc = np.maximum(0.0, q90 - q10).astype(np.float32)
        return pred, unc


def make_forecasters(args: argparse.Namespace) -> list[Forecaster]:
    models = set(args.models)
    out: list[Forecaster] = []
    if "timefm_v2" in models:
        out.append(TimeFmForecaster(Path(args.timefm_path), args.device))
    if "chronos_base" in models:
        out.append(ChronosForecaster(Path(args.chronos_path), args.device, args.chronos_samples))
    if "moirai2" in models:
        out.append(Moirai2Forecaster(args.moirai2_model_id, args.device, args.moirai2_batch_size))
    return out


def iter_specs(task: str, custom_daily_specs: str | None = None) -> tuple[Spec, ...]:
    custom_specs = parse_custom_daily_specs(custom_daily_specs)
    if custom_specs:
        if task == "daily":
            return custom_specs
        if task == "both":
            return SPECS_30MIN + custom_specs
    if task == "30min":
        return SPECS_30MIN
    if task == "daily":
        return SPECS_DAILY
    return SPECS_30MIN + SPECS_DAILY


def selection_threshold(calibration_uncertainty: np.ndarray, select_ratio: float) -> float:
    finite = calibration_uncertainty[np.isfinite(calibration_uncertainty)]
    if len(finite) == 0:
        return float("inf")
    return float(np.quantile(finite, select_ratio))


def split_name_for_index(idx: int, n: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if n >= 3:
        train_end = min(max(1, train_end), n - 2)
        val_end = min(max(train_end + 1, val_end), n - 1)
    else:
        train_end = max(1, n - 1)
        val_end = train_end
    if idx < train_end:
        return "train"
    if idx < val_end:
        return "val"
    return "test"


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    forecasters = make_forecasters(args)
    rows = []
    pred_rows = []

    for spec in iter_specs(args.task, args.custom_daily_specs):
        site_data = load_clean(spec.data_path)
        sites = sorted(site_data)
        if args.sites:
            wanted = set(args.sites)
            sites = [s for s in sites if s in wanted]
        if args.max_sites:
            sites = sites[: args.max_sites]
        print(f"\n[spec] {spec.task} | {spec.description} | sites={len(sites)}")

        all_windows = []
        for site in sites:
            windows = build_windows(site_data[site], spec, args.stride or spec.pred_len, args.max_windows_per_site)
            for idx, window in enumerate(windows):
                split_name = split_name_for_index(idx, len(windows), args.train_ratio, args.val_ratio)
                all_windows.append((site, split_name, idx, window))
        print(f"[windows] {len(all_windows)} total")

        for model in forecasters:
            print(f"[model] {model.name}")
            model_cache = []
            for site, split_name, idx, window in all_windows:
                pred, unc = model.predict(window["context"], spec)
                target = window["target"]
                model_cache.append((site, split_name, idx, target, pred, unc, window["dates"]))

            cal_unc = np.concatenate([x[5] for x in model_cache if x[1] == "val"], axis=0) if any(x[1] == "val" for x in model_cache) else np.array([])
            threshold = selection_threshold(cal_unc, args.select_ratio)

            for split_name in ["train", "val", "test"]:
                subset = [x for x in model_cache if x[1] == split_name]
                if not subset:
                    continue
                y_true = np.concatenate([x[3] for x in subset])
                y_pred = np.concatenate([x[4] for x in subset])
                unc = np.concatenate([x[5] for x in subset])
                base = metrics(y_true, y_pred)
                selected = metrics(y_true, y_pred, unc <= threshold)
                rows.append({"model": model.name, "task": spec.task, "description": spec.description, "split": split_name, "mode": "all", **base, "uncertainty_threshold": threshold})
                rows.append({"model": model.name, "task": spec.task, "description": spec.description, "split": split_name, "mode": "selective", **selected, "uncertainty_threshold": threshold})

            for site, split_name, idx, target, pred, unc, dates in model_cache:
                if split_name != "test" or not args.save_predictions:
                    continue
                for step, (date, yt, yp, u) in enumerate(zip(dates, target, pred, unc), start=1):
                    pred_rows.append(
                        {
                            "model": model.name,
                            "task": spec.task,
                            "description": spec.description,
                            "site": site,
                            "window_index": idx,
                            "step": step,
                            "date": date,
                            "y_true": float(yt),
                            "y_pred": float(yp),
                            "uncertainty": float(u),
                            "selected": bool(u <= threshold),
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_root / "unified_metrics.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(output_root / "unified_predictions.csv", index=False)
    write_summary(metrics_df, output_root / "unified_summary.md")
    print(f"\n[done] wrote results to {output_root}")


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Unified Carbon Forecast Evaluation", ""]
    if df.empty:
        lines.append("No results.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    test = df[df["split"] == "test"].copy()
    lines.extend(
        [
            "同一清洗数据、同一滚动窗口下的统一评估。`selective` 是选择学习启发的结果：只统计预测不确定性低于校准阈值的时间点。",
            "",
            "| Task | Description | Model | Mode | Coverage | MSE | MAE | RMSE | R2 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in test.sort_values(["task", "description", "mode", "mse", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.description} | {row.model} | {row.mode} | "
            f"{row.coverage:.3f} | {row.mse:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation with selective-learning-inspired masking.")
    parser.add_argument("--task", choices=["30min", "daily", "both"], default="daily")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["timefm_v2", "chronos_base", "moirai2"],
        default=["timefm_v2", "chronos_base"],
    )
    parser.add_argument("--output_root", default="carbon/unified_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timefm_path", default="ori/timefm_v2")
    parser.add_argument("--chronos_path", default="ori/chronos-base")
    parser.add_argument("--chronos_samples", type=int, default=5)
    parser.add_argument("--moirai2_model_id", default="Salesforce/moirai-2.0-R-small")
    parser.add_argument("--moirai2_batch_size", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--select_ratio", type=float, default=0.8, help="Coverage target for uncertainty-based selection.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling-based foundation models.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_windows_per_site", type=int, default=8)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--custom_daily_specs", default=None, help="Comma-separated seq:pred pairs, e.g. 180:60,365:90")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
