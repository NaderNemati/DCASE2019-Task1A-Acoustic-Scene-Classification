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
