Lightweight, reproducible baseline for **DCASE2019 Task 1A** (10-second clips, 10 classes, 12 European cities).  
Pipeline = **log-mel spectrograms** + **transfer learning** (EfficientNet-B2, ImageNet) + **SpecAugment**, **mixup**, cosine LR with warmup, **EMA** weights, and simple **time-shift TTA**.
