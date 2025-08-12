#!/usr/bin/env python3
# Create stratified folds
import argparse, os, pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.meta)
    y = df["scene_label"].values
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for i, (tr, va) in enumerate(skf.split(df, y), 1):
        df.iloc[tr].to_csv(os.path.join(args.out_dir, f"fold{i}_train.csv"), index=False)
        df.iloc[va].to_csv(os.path.join(args.out_dir, f"fold{i}_eval.csv"), index=False)
        print(f"Wrote fold {i}")

if __name__ == "__main__":
    main()

