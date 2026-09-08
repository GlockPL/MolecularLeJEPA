"""Marginal (single-model) bootstrap CI of AUPRC, pooled over split CSVs.

Companion to ``paired_bootstrap.py``: that script reports the *difference*
ΔAUPRC between two models on a shared test set; this one reports an individual
model's pooled AUPRC with a 95% bootstrap confidence interval, using the
IDENTICAL resampling procedure (pool all aligned rows across K splits, then draw
``n_bootstrap`` uniform resamples of size n with ``default_rng(seed)``, AUPRC per
replicate, skip degenerate no-positive resamples, take the 2.5/97.5 percentiles).

This is what gives every row of the antibiotic random-split table a comparable
95% CI: the graph rows came from ``paired_bootstrap.py`` (which prints each
model's marginal CI as a side product); the descriptor rows come from here, run
on the ``descriptor_baseline.py --save-probs`` CSVs. Same pooling, same seed,
same replicate count → directly comparable intervals.

Usage (all three descriptor models, pooled over the five random splits):
    uv run python scripts/marginal_bootstrap.py \\
        --csv checkpoints/descriptor_probs/descriptor_probs_antibiotic_seed*.csv \\
        --label-col label --prob-cols LogReg_prob HistGBM_prob XGB_prob
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Marginal bootstrap CI of AUPRC, pooled over CSVs")
    ap.add_argument("--csv", required=True, nargs="+",
                    help="One or more per-split prediction CSVs (globs allowed).")
    ap.add_argument("--label-col", default="label",
                    help="Binary label column (default 'label').")
    ap.add_argument("--prob-cols", required=True, nargs="+",
                    help="Probability column(s) to evaluate, e.g. XGB_prob HistGBM_prob.")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Expand any globs the shell did not (e.g. quoted patterns).
    files: list[str] = []
    for c in args.csv:
        files.extend(sorted(glob.glob(c)) or [c])

    parts = [pd.read_csv(f) for f in files]
    for f, part in zip(files, parts):
        print(f"  loaded {len(part):5d} rows, {int(part[args.label_col].sum()):4d} pos   ({f})")
    df = pd.concat(parts, ignore_index=True)

    y = df[args.label_col].to_numpy().astype(int)
    n = len(df)
    n_pos = int(y.sum())
    print(f"\nPOOLED over {len(files)} split(s): {n} compounds  "
          f"positives={n_pos} ({100 * n_pos / max(n, 1):.2f}%)\n")

    # One shared resample-index matrix so every model's CI is built on the
    # EXACT same replicates (matches paired_bootstrap's single-rng pooling).
    rng = np.random.default_rng(args.seed)
    for col in args.prob_cols:
        p = df[col].to_numpy()
        point = _ap(y, p)
        r = np.random.default_rng(args.seed)   # reset per col → identical replicates
        reps, skipped = [], 0
        for _ in range(args.n_bootstrap):
            idx = r.integers(0, n, size=n)
            if y[idx].sum() == 0:
                skipped += 1
                continue
            reps.append(_ap(y[idx], p[idx]))
        reps = np.asarray(reps)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        print(f"  {col:14s} point AUPRC = {point:.4f}   "
              f"bootstrap mean {reps.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]   "
              f"({len(reps)} reps, {skipped} skipped)")
    _ = rng  # keep a top-level rng around for clarity; per-col rng used above


if __name__ == "__main__":
    main()