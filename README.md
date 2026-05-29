# GAFA Reproduction Guide

This README only lists the steps needed to reproduce the experiments.

## 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

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


## 4. Notes

- If you install `torch` with CUDA tags such as `+cu121`, follow the official PyTorch wheel instructions for your CUDA version.
- Some scripts download pretrained models from the Hugging Face Model Hub at runtime, such as TimesFM, Chronos, and some `uni2ts` models.
- If you need private models, run `huggingface-cli login` or set `HF_TOKEN` / `HUGGINGFACE_TOKEN` before running the scripts.
- You can also point a script to a local model directory if it supports a path argument such as `--chronos_path`.
