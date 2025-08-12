#!/usr/bin/env python3
"""
DCASE2019 Task 1A — Simple Baseline (single file)
- Log-mel spectrograms
- EfficientNet-B0 (ImageNet)
- Optional SpecAugment + mixup
- Train/Eval
"""

import os, argparse, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ---------------------------
# Config / constants
# ---------------------------
CLASSES = [
    'airport','bus','metro','metro_station','park','public_square',
    'shopping_mall','street_pedestrian','street_traffic','tram'
]
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}

# ---------------------------
# Data
# ---------------------------
class LogMel:
    def __init__(self, sr=44100, n_fft=1024, hop_length=441, n_mels=128, f_min=0, f_max=None):
        self.sr = sr
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max or sr//2
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, wav):
        S = self.to_db(self.mel(wav))             # [1, n_mels, T]
        S = (S - S.mean()) / (S.std() + 1e-6)     # per-sample standardize
        return S

def spec_augment(spec, time_masks=2, time_width=64, freq_masks=2, freq_width=16):
    # spec: [1, n_mels, T]
    x = spec.clone()
    n_mels, T = x.shape[-2], x.shape[-1]
    for _ in range(freq_masks):
        start = torch.randint(0, max(1, n_mels - freq_width), (1,)).item()
        width = torch.randint(0, freq_width + 1, (1,)).item()
        x[:, start:start+width, :] = 0
    for _ in range(time_masks):
        start = torch.randint(0, max(1, T - time_width), (1,)).item()
        width = torch.randint(0, time_width + 1, (1,)).item()
        x[:, :, start:start+width] = 0
    return x

def pad_or_trim_time(spec, T_target=1024):
    T = spec.shape[-1]
    if T == T_target: return spec
    if T > T_target:
        s = (T - T_target)//2
        return spec[..., s:s+T_target]
    pad_l = (T_target - T)//2
    pad_r = T_target - T - pad_l
    return F.pad(spec, (pad_l, pad_r))

def time_shift(wav, max_shift=0.1):
    shift = int((torch.rand(1).item()*2 - 1) * max_shift * wav.shape[-1])
    return torch.roll(wav, shifts=shift, dims=-1)

def add_noise_gain(wav, noise_snr=(20,35), gain_db=(-3,3)):
    g = gain_db[0] + torch.rand(1).item()*(gain_db[1]-gain_db[0])
    wav = wav * (10**(g/20))
    snr = noise_snr[0] + torch.rand(1).item()*(noise_snr[1]-noise_snr[0])
    noise = torch.randn_like(wav) * (wav.std() / (10**(snr/20) + 1e-6))
    return wav + noise

class DCASEDataset(Dataset):
    def __init__(self, audio_root, csv_path, sr=44100, n_fft=1024, hop_length=441,
                 n_mels=128, time_crop_frames=1024, train=True, use_aug=True):
        self.audio_root = audio_root
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.train = train
        self.use_aug = use_aug and train
        self.time_crop_frames = time_crop_frames
        self.lm = LogMel(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.audio_root, row["filename"])
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono

        if self.use_aug:
            wav = time_shift(wav, max_shift=0.1)
            wav = add_noise_gain(wav)

        spec = self.lm(wav)                        # [1, n_mels, T]
        spec = pad_or_trim_time(spec, self.time_crop_frames)
        if self.use_aug:
            spec = spec_augment(spec, time_masks=2, time_width=64, freq_masks=2, freq_width=16)

        spec3 = spec.repeat(3,1,1)                 # 3-ch for ImageNet nets
        y = torch.tensor(CLASS2IDX[row["scene_label"]], dtype=torch.long)
        return spec3, y

def mixup_collate(batch, alpha=0.2, num_classes=10, label_smoothing=0.0):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys)
    if alpha <= 0:
        return x, y, False
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    x = lam * x + (1 - lam) * x[idx, :]

    y1 = torch.zeros(x.size(0), num_classes, dtype=torch.float32)
    y1.scatter_(1, y.view(-1,1), 1.0)
    y2 = torch.zeros_like(y1)
    y2.scatter_(1, y[idx].view(-1,1), 1.0)
    if label_smoothing > 0:
        y1 = (1 - label_smoothing) * y1 + label_smoothing / num_classes
        y2 = (1 - label_smoothing) * y2 + label_smoothing / num_classes
    y_mix = lam * y1 + (1 - lam) * y2
    return x, y_mix, True

# ---------------------------
# Models
# ---------------------------
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

def build_model(backbone="effnet_b0", pretrained=True, n_classes=10):
    if backbone == "cnn":
        return CNN(n_classes)
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_classes)
    return m

# ---------------------------
# Train / Eval
# ---------------------------
def accuracy_from_logits(logits, target):
    preds = logits.argmax(1)
    if target.ndim == 2:  # one-hot
        target = target.argmax(1)
    return (preds == target).float().mean().item()

def train_one_epoch(model, loader, opt, device, label_smoothing=0.05):
    model.train()
    losses, accs = [], []
    for batch in loader:
        x, y, mixed = batch
        x = x.to(device)
        opt.zero_grad()
        logits = model(x)
        if mixed:
            y = y.to(device)
            loss = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            acc = accuracy_from_logits(logits, y)
        else:
            y = y.to(device)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            acc = (logits.argmax(1) == y).float().mean().item()
        loss.backward(); opt.step()
        losses.append(loss.item()); accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, accs = [], []
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        losses.append(loss.item()); accs.append(acc)
        all_y.extend(y.tolist()); all_p.extend(logits.argmax(1).tolist())
    return float(np.mean(losses)), float(np.mean(accs)), np.array(all_y), np.array(all_p)

def save_curves(log, out_dir):
    ep = [r["epoch"] for r in log]
    tr = [r["train_acc"] for r in log]
    va = [r["val_acc"] for r in log]
    plt.figure(figsize=(6.5,4.5))
    plt.plot(ep, tr, label="Train acc")
    plt.plot(ep, va, label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=160)
    plt.close()

def plot_confmat(cm, out_path):
    plt.figure(figsize=(7.5,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (val)")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45, ha="right")
    plt.yticks(ticks, CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out", default="runs/simple_exp")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", choices=["effnet_b0","cnn"], default="effnet_b0")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = DCASEDataset(args.audio_root, args.train_csv, train=True, use_aug=not args.no_aug)
    val_ds   = DCASEDataset(args.audio_root, args.val_csv,   train=False, use_aug=False)

    collate = (lambda b: mixup_collate(b, alpha=args.mixup, num_classes=len(CLASSES), label_smoothing=0.05))
    if args.mixup <= 0:  # plain batch
        def collate(b):
            xs, ys = zip(*b)
            return torch.stack(xs,0), torch.tensor(ys), False

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2, collate_fn=collate)
    val_ld   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=lambda b: (torch.stack([x for x,_ in b],0), torch.tensor([y for _,y in b])))

    model = build_model(args.backbone, pretrained=(not args.no_pretrained), n_classes=len(CLASSES)).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, bad, patience = 0.0, 0, 8
    log_rows = []
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, opt, device, label_smoothing=0.05)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_ld, device)
        sched.step()
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:02d} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f} (best={best:.3f})")

        if va_acc > best:
            best, bad = va_acc, 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            # also save confusion matrix for best
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
            plot_confmat(cm, os.path.join(args.out, "confusion_matrix.png"))
            with open(os.path.join(args.out,"cls_report.txt"), "w") as f:
                f.write(classification_report(y_true, y_pred, target_names=CLASSES))
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping."); break

    # save logs and curves
    pd.DataFrame(log_rows).to_csv(os.path.join(args.out, "log.csv"), index=False)
    save_curves(log_rows, args.out)
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(CLASSES, f)

if __name__ == "__main__":
    main()

