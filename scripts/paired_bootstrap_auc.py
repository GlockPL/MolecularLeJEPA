"""Paired bootstrap of the ROC-AUC difference between two feature sets on molhiv.

The ROC-AUC analog of ``paired_bootstrap.py`` (which does AUPRC and joins by SMILES
from ensemble_eval CSVs). Inputs here are the ``.npz`` files written by
``xgb_molhiv.py --save-probs`` / ``xgb_molhiv_embed.py --save-probs``: ensemble test
probabilities for the SAME test set in the SAME row order, so the pairing is exact
and no join is needed (the label vectors are asserted identical).

Why paired: molhiv's scaffold test split has only 130 actives, and the per-seed
RF std is ~0.004 — so a 0.005 AUC gap between two feature sets is unreadable from
the marginal numbers. Most of the resample-to-resample variance is COMMON to both
models (same molecules); resampling once per replicate and scoring both on that
resample removes it and exposes the difference.

Reports each model's point AUC, mean ΔAUC = ours − other with a 95% CI, and a
one-sided bootstrap p-value (fraction of replicates with ours ≤ other).

Usage:
    uv run python scripts/paired_bootstrap_auc.py \\
        --ours  logs/molhiv_concat_<STAMP>/morgan_desc_pooled.npz \\
        --other logs/molhiv_concat_<STAMP>/morgan.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["probs"].astype(np.float64), d["y"].astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help=".npz from --save-probs (the new feature set)")
    ap.add_argument("--other", required=True, help=".npz from --save-probs (the reference)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p_ours, y_ours = load(args.ours)
    p_other, y_other = load(args.other)
    if len(p_ours) != len(p_other):
        raise ValueError(f"row counts differ ({len(p_ours)} vs {len(p_other)}) — the two runs "
                         "did not score the same test split")
    if not np.array_equal(y_ours, y_other):
        raise ValueError("label vectors differ — the .npz files are not row-aligned "
                         "(same split, same order) so the pairing would be invalid")
    y = y_ours

    a_ours, a_other = roc_auc_score(y, p_ours), roc_auc_score(y, p_other)
    print(f"  ours  {args.ours}\n    ROC-AUC = {a_ours:.4f}")
    print(f"  other {args.other}\n    ROC-AUC = {a_other:.4f}")
    print(f"  point Delta = {a_ours - a_other:+.4f}   ({len(y)} compounds, {int(y.sum())} active)\n")

    rng = np.random.default_rng(args.seed)
    n = len(y)
    deltas, skipped = [], 0
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():        # degenerate resample (one class) — AUC undefined
            skipped += 1
            continue
        deltas.append(roc_auc_score(yb, p_ours[idx]) - roc_auc_score(yb, p_other[idx]))
    deltas = np.asarray(deltas)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    pval = float((deltas <= 0).mean())
    print(f"  paired bootstrap ({len(deltas)} replicates"
          + (f", {skipped} degenerate skipped" if skipped else "") + ")")
    print(f"    mean Delta AUC = {deltas.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    one-sided p(ours <= other) = {pval:.4f}")
    verdict = ("SIGNIFICANT — CI excludes 0" if lo > 0 or hi < 0
               else "NOT significant — CI spans 0")
    print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
