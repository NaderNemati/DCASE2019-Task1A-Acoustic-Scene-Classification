# DCASE2019 Task 1A — Acoustic Scene Classification (PyTorch)

Lightweight, reproducible baseline for **DCASE2019 Task 1A** (10-second clips, 10 classes, 12 European cities).  
Pipeline = **log-mel spectrograms** + **transfer learning** (EfficientNet-B2, ImageNet) + **SpecAugment**, **mixup**, cosine LR with warmup, **EMA** weights, and simple **time-shift TTA**.

> This repo includes runnable code and results assets so you can see expected outputs before training.  
> Replace dataset paths with your local DCASE2019 dev set to train/evaluate.

---

## Data format

    Audio: mono (mixed on load), 44.1 kHz WAV, 10-s segments.

    CSV columns: filename, scene_label, where filename is relative to --audio_root.

    Default classes (10):

```bash

airport, bus, metro, metro_station, park,
public_square, shopping_mall, street_pedestrian, street_traffic, tram
