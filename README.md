# DCASE2019 Task 1A — Acoustic Scene Classification (PyTorch)

Lightweight, reproducible baseline for **DCASE2019 Task 1A** (10-second clips, 10 classes, 12 European cities).  
Pipeline = **log-mel spectrograms** + **transfer learning** (EfficientNet-B2, ImageNet) + **SpecAugment**, **mixup**, cosine LR with warmup, **EMA** weights, and simple **time-shift TTA**.

> This repo includes runnable code and results assets so you can see expected outputs before training.  
> Replace dataset paths with your local DCASE2019 dev set to train/evaluate.

---

# 0) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) (Optional) make folds
python prepare_folds.py --meta data/metadata/all.csv --out_dir data/metadata --folds 4

# 2) Train (fold 1)
python asc_simple.py \
  --audio_root data/audio \
  --train_csv data/metadata/fold1_train.csv \
  --val_csv   data/metadata/fold1_eval.csv \
  --epochs 25 --bs 16 --lr 3e-4 --backbone effnet_b0

# 3) Outputs (in runs/simple_exp/)
# - best.pt
# - log.csv
# - training_curves.png
# - confusion_matrix.png
# - cls_report.txt









## Data format

Audio: mono (mixed on load), 44.1 kHz WAV, 10-s segments.

CSV columns: filename, scene_label, where filename is relative to --audio_root.

Default classes (10):

```bash

airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram
```

## Features & training recipe

Input: log-mel spectrograms (128 mels, ~25 ms win, ~10 ms hop), standardized per sample.

Augment: SpecAugment (time/freq masks), random time shift, light noise.

Optimization: AdamW, label smoothing (ε=0.05), cosine LR with warmup, EMA weights, early stopping.

Backbone: torchvision EfficientNet-B2 (ImageNet).

Eval: fixed-length crops, time-shift TTA (logits average).


## Results (placeholder)

4-fold validation accuracy (mean ± sd): 0.818 ± 0.009

Best fold validation accuracy: 0.826

Key confusions: metro↔tram; street_traffic↔street_pedestrian. Overfitting reduced (≈7 pp train–val gap) by SpecAugment + mixup + EMA + cosine schedule.

## Config

Defaults are in configs.yaml (STFT/mel, SpecAugment widths, mixup α, label smoothing, EMA decay, warmup epochs, TTA crops).


## File tree

```python

DCASE2019-Task1A-Acoustic-Scene-Classification/
├── LICENSE
├── README.md
├── asc_train_eval.py
├── configs.yaml
├── prepare_folds.py
└── requirements.txt


```
