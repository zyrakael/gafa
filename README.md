# GAFA Reproduction Guide

This README lists the steps needed to reproduce the experiments.

## 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

Put the downloaded datasets in the location expected by the loader, or add an adapter in `data_provider/`.

### Data Sources

The datasets used in this work were obtained from:

- [Science Data Bank](https://www.scidb.cn/)
- [National Earth System Data Center](https://www.nesdc.org.cn/)

Please register or log in on each site, then download the datasets according to the platform instructions. After downloading, place the raw files in a project data folder such as `data/`, or implement a dataset adapter in `data_provider/`.

If a dataset needs extra preprocessing, keep those steps in a script under `scripts/` so the process stays reproducible.

## 3. Run Experiments

Example commands:

```bash
python run_nee_daily.py
```

```bash
python run_nee_daily_improved.py
```

```bash
python run_foundation_carbon.py
```

For a specific run, you can also pass arguments such as `--device` and `--seed` if the script supports them.

Key Scripts：
| Script | Purpose |
|---|---|
| `clean_carbon_data.py` | Preprocess raw carbon flux data with robust outlier filtering and quality masks |
| `unified_foundation_eval.py` | Rolling-window evaluation of frozen foundation models (Chronos, TimesFM, Moirai/Moirai2) with selective-reliability metrics |
| `unified_supervised_eval.py` | Rolling-window evaluation of supervised baselines|
| `selective_residual_adaptation.py` | Core GAFA implementation: ridge residual head with learned gating on frozen foundation outputs |
| `gafa_ablation_study.py` | Ablation comparing gated vs. ungated residual adaptation and input features |
| `gafa_overhead_eval.py` | Output-space overhead evaluation: training time, inference latency, and parameter counts |
| `gafa_overhead_microbenchmark.py` | Microbenchmark of the 32-parameter GAFA output layer |
| `supervised_output_calibration.py` | Output-space calibration control for target-trained supervised models |
| `cross_site_generalization_eval.py` | Cross-site generalization: train on one site, evaluate on held-out sites |
| `public_univariate_gafa_eval.py` | GAFA evaluation on public univariate benchmarks |
| `chronos_simple_calibration_eval.py` | Conservative calibration experiments with Chronos on the carbon benchmark |
| `ensemble_foundation_predictions.py` | Simple ensembles over saved foundation-model prediction outputs |
| `paired_bootstrap_ci.py` | Site-paired bootstrap confidence intervals for GAFA vs. baselines |

## 4. Notes

- If you install `torch` with CUDA tags such as `+cu121`, follow the official PyTorch wheel instructions for your CUDA version.
- Some scripts download pretrained models from the Hugging Face Model Hub at runtime, such as TimesFM, Chronos, and some `uni2ts` models.
- If you need private models, run `huggingface-cli login` or set `HF_TOKEN` / `HUGGINGFACE_TOKEN` before running the scripts.
