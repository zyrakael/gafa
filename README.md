# GAFA Reproduction Guide

This README lists the steps needed to reproduce the experiments.

## 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| Operating system | Ubuntu 22.04 LTS |
| CPU | Intel Xeon Platinum 8383C @ 2.70GHz |
| Memory | 512 GB RAM |
| GPU | 8× NVIDIA GeForce RTX 4090 (24 GB each) |
| CUDA | CUDA 12.1 |

## 3. Data Preparation

Put the downloaded datasets in the location expected by the loader, or add an adapter in `data_provider/`.
**Copyright notice:** The carbon sink data is subject to copyright. Users must obtain the data themselves. We provide download links and a processing script for reference only.
- [Dataset 1 (SciDB)](https://www.scidb.cn/detail?dataSetId=824941006418870272&version=V1)
- [Dataset 2 (NESDC)](https://www.nesdc.org.cn/sdo/detail?id=64e6c4f07e2817429fbc7afa)
- [Dataset 3 (SciDB)](https://www.scidb.cn/detail?dataSetId=720626422036561920&version=V1)
- [Dataset 4 (SciDB)](https://www.scidb.cn/detail?dataSetId=4935daa458a34c3dae22a36cb317826c&version=V1)
- [Dataset 5 (SciDB File)](https://www.scidb.cn/file?fid=60504df8124e3600e55445d5&mode=front)
- [Dataset 6 (SciDB)](https://www.scidb.cn/detail?dataSetId=9b649cdd9cb143cc9b3188d7a6a38a31&version=V3)
- [Dataset 7 (NESDC)](https://www.nesdc.org.cn/sdo/detail?id=64e6c14f7e2817429fbc7af7)
- [Dataset 8 (SciDB)](https://www.scidb.cn/detail?dataSetId=755472332243337216&version=V2)
- [Dataset 9 (SciDB)](https://www.scidb.cn/detail?dataSetId=be0acc7ca1804710b363fab019ce8336&version=V4)
- [Dataset 10 (SciDB)](https://www.scidb.cn/detail?dataSetId=c800dd446426478abba3b6ec24757ade&version=V1)

### Data Sources

The datasets used in this work were obtained from:

- [Science Data Bank](https://www.scidb.cn/)
- [National Earth System Data Center](https://www.nesdc.org.cn/)

Please register or log in on each site, then download the datasets according to the platform instructions. After downloading, place the raw files in a project data folder such as `data/`, or implement a dataset adapter in `data_provider/`.

If a dataset needs extra preprocessing, keep those steps in a script under `scripts/` so the process stays reproducible.

## 4. Run Experiments

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

## 5. Notes

- Some scripts download pretrained models from the Hugging Face Model Hub at runtime, such as TimesFM, Chronos, and some `uni2ts` models.
- If you need private models, run `huggingface-cli login` or set `HF_TOKEN` / `HUGGINGFACE_TOKEN` before running the scripts.
