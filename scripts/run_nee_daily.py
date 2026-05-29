#!/usr/bin/env python3
"""Short-horizon carbon NEE forecasting from the daily all-sites file."""

from __future__ import annotations

import argparse
import itertools
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    seq_len: int
    label_len: int
    pred_len: int


# 日数据做短期预测：1 周、2 周、1 个月。
DEFAULT_RUN_SPECS = (
    RunSpec(seq_len=30, label_len=7, pred_len=7),
    RunSpec(seq_len=60, label_len=14, pred_len=14),
    RunSpec(seq_len=90, label_len=30, pred_len=30),
)

DEFAULT_MODELS = (
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimeMixer",
)


def iter_specs() -> Iterable[RunSpec]:
    yield from DEFAULT_RUN_SPECS


def base_cmd_for(model: str, args: argparse.Namespace, data_path: pathlib.Path) -> list[str]:
    checkpoints = pathlib.Path(args.output_root) / "checkpoints" / "daily_shortterm"
    return [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(data_path.parent),
        "--data_path",
        data_path.name,
        "--data",
        "custom",
        "--features",
        "S",
        "--target",
        "NEE",
        "--freq",
        "d",
        "--checkpoints",
        str(checkpoints),
        "--enc_in",
        "1",
        "--dec_in",
        "1",
        "--c_out",
        "1",
        "--itr",
        "1",
        "--model",
        model,
        "--model_id",
        "Carbon_daily_shortterm",
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--n_heads",
        str(args.n_heads),
        "--e_layers",
        str(args.e_layers),
        "--dropout",
        str(args.dropout),
        "--train_epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--var_rank",
        "4",
        "--ctx_len",
        "8",
        "--des",
        "Carbon_daily_shortterm",
    ]


def build_run_cmd(
    *,
    spec: RunSpec,
    patch_len: int,
    model: str,
    args: argparse.Namespace,
    data_path: pathlib.Path,
) -> list[str]:
    cmd = list(base_cmd_for(model=model, args=args, data_path=data_path))
    cmd.extend(
        [
            "--seq_len",
            str(spec.seq_len),
            "--label_len",
            str(spec.label_len),
            "--pred_len",
            str(spec.pred_len),
            "--patch_len",
            str(patch_len),
        ]
    )
    return cmd


def run_one(
    *,
    cmd: list[str],
    log_path: pathlib.Path,
    project_root: pathlib.Path,
    dry_run: bool,
) -> int:
    pretty = shlex.join(cmd)
    if dry_run:
        print(pretty)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(pretty + "\n\n")
        fh.flush()
        completed = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(completed.returncode)


def run_experiments(data_path: pathlib.Path, args: argparse.Namespace) -> int:
    exit_code = 0
    project_root = pathlib.Path(__file__).resolve().parent
    logs_root = pathlib.Path(args.output_root) / "logs_daily_shortterm"

    for model in args.models:
        for spec, patch_len in itertools.product(iter_specs(), tuple(args.patch_lens)):
            log_name = f"daily_{model}_sl{spec.seq_len}_pl{spec.pred_len}_patch{patch_len}.log"
            log_path = logs_root / model / log_name
            print(f"[run] {model} | seq_len={spec.seq_len} | pred_len={spec.pred_len} | patch={patch_len}")
            code = run_one(
                cmd=build_run_cmd(
                    spec=spec,
                    patch_len=patch_len,
                    model=model,
                    args=args,
                    data_path=data_path,
                ),
                log_path=log_path,
                project_root=project_root,
                dry_run=args.dry_run,
            )
            if code != 0:
                exit_code = code
                print(f"[fail] exit_code={code} | {log_path}")
                if args.fail_fast:
                    return exit_code
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short-horizon daily carbon NEE forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        default="./carbon/all_sites_daily.csv",
        help="Path to daily carbon CSV (site/date/NEE).",
    )
    parser.add_argument(
        "--output_root",
        default="./carbon",
        help="Directory to store logs and results.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Models to run.")
    parser.add_argument("--patch_lens", nargs="+", type=int, default=[7], help="Patch lengths to run.")
    parser.add_argument("--train_epochs", type=int, default=50, help="Training epochs for run.py.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--e_layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of data loading workers.")
    parser.add_argument("--dry_run", action="store_true", help="Only print generated commands.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = pathlib.Path(args.input_path).resolve()
    output_root = pathlib.Path(args.output_root).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    grouped_df = pd.read_csv(data_path)
    print(f"[main] Reading daily data from: {data_path}")
    print(f"[main] Output root: {output_root}")
    print(f"[main] Total sites: {grouped_df['site'].nunique()}")
    print(f"[main] Total records: {len(grouped_df)}")
    print(f"[main] Date range: {grouped_df['date'].min()} to {grouped_df['date'].max()}")
    print(f"[main] NEE stats: mean={grouped_df['NEE'].mean():.4f}, std={grouped_df['NEE'].std():.4f}")
    print("[main] Short-term specs:")
    for spec in iter_specs():
        print(f"  seq_len={spec.seq_len} days -> pred_len={spec.pred_len} days")

    return run_experiments(data_path=data_path, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
