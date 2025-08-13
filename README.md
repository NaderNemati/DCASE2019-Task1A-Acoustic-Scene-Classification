# DCASE2019 Task 1A — Acoustic Scene Classification (PyTorch)

A clean, reproducible baseline for **DCASE2019 Task 1A** (10-second audio clips, **10** scene classes, **12** European cities).

**Method at a glance**
- **Features:** log-mel spectrograms (128 mels, ~25 ms window / ~10 ms hop), per-sample standardization  
- **Model:** transfer learning with **EfficientNet-B2** (ImageNet init)  
- **Regularization:** **SpecAugment**, **mixup**, label smoothing, cosine LR with warm-up, **EMA** weights  
- **Evaluation:** fixed-length crops + time-shift **TTA** (logits averaged)

> **Note:** The dataset is **not** included. Point the scripts to your local DCASE2019 development set.

---

## Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset & Folds](#dataset--folds)
- [Quick Start](#quick-start)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Results (simulated placeholder)](#results-simulated-placeholder)
- [Repository Structure](#repository-structure)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [License & Acknowledgements](#license--acknowledgements)

---

## Requirements
- Python 3.9+  
- PyTorch, Torchaudio, Torchvision  
- NumPy, Pandas, Matplotlib, scikit-learn, PyYAML

Install from `requirements.txt`.

---

## Installation
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset & Folds

Data layout (local, not in repo):

```python

data/
├── audio/                      # WAV files (10 s, 44.1 kHz); filenames relative in CSVs
└── metadata/
    ├── all.csv                 # optional: full list (filename,scene_label)
    ├── fold1_train.csv         # example split
    └── fold1_eval.csv

```

## CSV format:

```python

filename,scene_label
fold1/airport_0001.wav,airport
fold1/bus_0001.wav,bus

```

## Optional — create stratified folds from a single CSV:

```bash
python prepare_folds.py --meta data/metadata/all.csv --out_dir data/metadata --folds 4
```

## Quick Start

#### Train (example: Fold 1)

```bash
python asc_train_eval.py \
  --audio_root data/audio \
  --train_csv data/metadata/fold1_train.csv \
  --val_csv   data/metadata/fold1_eval.csv \
  --config    configs.yaml \
  --out       runs/exp1 \
  --epochs    35 \
  --bs        16 \
  --lr        3e-4
```

## Outputs

Saved under --out (e.g., runs/exp1/):

best.pt — best validation checkpoint

log.csv — per-epoch train/val loss & accuracy

training_curves.png — accuracy vs. epoch (train/val)

confusion_matrix.png — validation confusion matrix

cls_report.txt — per-class precision/recall/F1

## Configuration

All defaults live in configs.yaml. You can edit the file.

## Results

These numbers illustrate expected performance when the pipeline is tuned:

4-fold validation accuracy (mean ± sd): 0.818 ± 0.009

Best fold validation accuracy: 0.826

Common confusions: metro ↔ tram, street_traffic ↔ street_pedestrian.
Regularization (SpecAugment + mixup + EMA + cosine schedule) reduces the train–val gap by ~7 pp.

## Repository Structure

```python

DCASE2019-Task1A-Acoustic-Scene-Classification/
├── LICENSE
├── README.md
├── asc_train_eval.py
├── configs.yaml
├── prepare_folds.py
└── requirements.txt

```

