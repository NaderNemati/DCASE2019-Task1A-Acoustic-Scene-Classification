# DCASE2019 Task 1A — Acoustic Scene Classification (PyTorch)

Lightweight, reproducible baseline for **DCASE2019 Task 1A** (10-second clips, 10 classes, 12 European cities).  
Pipeline = **log-mel spectrograms** + **transfer learning** (EfficientNet-B2, ImageNet) + **SpecAugment**, **mixup**, cosine LR with warmup, **EMA** weights, and simple **time-shift TTA**.

> This repo includes runnable code and results assets so you can see expected outputs before training.  
> Replace dataset paths with your local DCASE2019 dev set to train/evaluate.

---

## Quickstart

```bash

# Install Python + venv + pip (if needed)
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# 1) Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# (to exit later: deactivate)

# 2) Data
# Place audio waves under: data/audio/
# Prepare metadata CSVs (columns: filename, scene_label) under: data/metadata/
# Optionally Create 4 folds from a single CSV:
python -m scripts.prepare_folds --meta data/metadata/all.csv --out_dir data/metadata --folds 4
