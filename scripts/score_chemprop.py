"""Score Chemprop ensemble predictions on the scaffold test split.

Uses the repo's own `compute_auprc` (AUPRC + 100x bootstrap 95% CI) so the
number is directly comparable to the GPS/LeJEPA model's reported metrics.

Usage:
    uv run python scripts/score_chemprop.py \
        --labels splits_out/scaffold_split/test.csv \
        --preds  checkpoints/chemprop_scaffold/test_preds.csv \
        --target antibiotic
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from src.evaluate import compute_auprc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="splits_out/scaffold_split/test.csv")
    ap.add_argument("--preds", default="checkpoints/chemprop_scaffold/test_preds.csv")
    ap.add_argument("--target", default="antibiotic")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    pred = pd.read_csv(args.preds)

    # chemprop_predict keeps the smiles column and the target name; align on
    # smiles to be safe against any dropped/reordered rows.
    merged = lab.merge(pred, on="smiles", suffixes=("_true", "_pred"))
    y_col = f"{args.target}_true" if f"{args.target}_true" in merged else args.target
    p_col = f"{args.target}_pred" if f"{args.target}_pred" in merged else args.target

    y = torch.tensor(merged[y_col].to_numpy(), dtype=torch.float).unsqueeze(1)
    p = torch.tensor(merged[p_col].to_numpy(), dtype=torch.float).unsqueeze(1)

    n, n_pos = len(y), int(y.sum().item())
    print(f"target        : {args.target}")
    print(f"test compounds: {n}  (positives={n_pos}, {100*n_pos/n:.2f}%)")
    print(f"baseline AUPRC: {n_pos/n:.4f}  (random)")
    print("Chemprop v1 ensemble (repo compute_auprc):")
    compute_auprc(p, y, task_names=[args.target])  # prints AUPRC + 95% CI


if __name__ == "__main__":
    main()