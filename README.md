GAFA — Reproduction Instructions
======================================

Quick steps to reproduce experiments:

1) Create and activate a Python virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\activate
```

2) Install dependencies (listed below)

```bash
pip install -r requirements.txt
```

3) Prepare data

- Put datasets where the loaders expect them or add an adapter in `data_provider/`.

Data sources

- The datasets used for experiments were obtained from the following platforms:

	- https://www.scidb.cn/ (Science Data Bank)
	- https://www.nesdc.org.cn/ (National Earth System Data Center)

- Please register or log in on each site and download datasets following the site instructions. After downloading, place raw files under a project data folder (for example `data/`) or implement an adapter in `data_provider/` to match the downloaded layout.

- If the dataset requires specific preprocessing (time-zone alignment, resampling, missing-value handling), include those steps in a script under `scripts/` or within a `data_provider` adapter so that experiments are reproducible.

4) Run example scripts

```bash
python run_nee_daily.py
python run_nee_daily_improved.py
python run_foundation_carbon.py
```

For GPU runs or specific experiments, pass `--device` and `--seed` or use the experiment's config file.

Requirements
------------
absl-py==2.4.0
accelerate==1.13.0
aiohttp==3.13.5
aiohappyeyeballs==2.6.1
attrs==26.1.0
einops==0.8.2
fsspec==2025.12.0
gitpython==3.1.46
gluonts==0.14.4
hydra-core==1.3.0
matplotlib==3.10.8
numpy==1.26.4
pandas==2.1.4
Pillow==12.0.0
protobuf==6.33.5
pytorch-lightning==2.6.1
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
torchmetrics==1.9.0
transformers==4.57.6
tokenizers==0.22.2
timesfm==1.3.0
uni2ts==2.0.0
utilsforecast==0.2.15
wandb==0.25.1
omegaconf==2.3.0
PyYAML==6.0.3
requests==2.32.5
tqdm==4.67.3

Notes
-----
- If you install `torch` with CUDA tags (`+cu121`), follow PyTorch's wheel instructions for your CUDA version.
- If you prefer a single-file reference, `requirements.txt` contains the same list.

Hugging Face model downloads
----------------------------
- Some scripts download pretrained models or pipelines from the Hugging Face Model Hub at runtime (examples: TimesFM via `TimesFmModelForPrediction.from_pretrained`, Chronos pipelines via `ChronosPipeline.from_pretrained`, and some `uni2ts` models). These calls will fetch model weights into your local HF cache.
- If you need to access private models, run `huggingface-cli login` or set `HF_TOKEN`/`HUGGINGFACE_TOKEN` in your environment before running the scripts.
- You can also point scripts to a local model directory (some scripts accept paths such as `--chronos_path`) to avoid network downloads.
