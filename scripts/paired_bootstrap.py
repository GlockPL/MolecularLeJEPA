"""Paired bootstrap of AUPRC difference between two models on the SAME test set.

Comparing two marginal bootstrap CIs is underpowered when the test set is small
and shared (102 antibiotic positives) — the CIs can overlap even when one model
is reliably better, because most of the per-resample variance is *common* to both
(it's the same molecules). A PAIRED bootstrap removes that shared variance:
resample the test compounds once per replicate, score BOTH models on that same
resample, and look at the distribution of the *difference* ΔAUPRC = ours − other.

Reports: each model's point AUPRC, mean ΔAUPRC with a 95% CI, and a one-sided
bootstrap p-value (fraction of replicates where ours ≤ other). If the ΔAUPRC CI
excludes 0, the win is significant despite overlapping marginal CIs.

Inputs are aligned by SMILES (inner join), so a dropped/unparseable compound on
either side is handled. Labels are taken from the --ours CSV (written by
ensemble_eval.py --save-probs).

Usage (single split):
    uv run python scripts/paired_bootstrap.py \\
        --ours checkpoints/ens_random_pretrained/ensemble_probs_test.csv \\
        --other checkpoints/chemprop_random_antibiotic/test_preds.csv \\
        --target antibiotic

Usage (POOL several split seeds → ~K× positives → tighter CI; pass parallel lists):
    uv run python scripts/paired_bootstrap.py \\
        --ours   checkpoints/ens_random_antibiotic_pretrained_seed*/ensemble_probs_test.csv \\
        --other  checkpoints/chemprop_random_antibiotic_seed*/test_preds.csv \\
        --target antibiotic
    (ensure the two globs expand in the SAME seed order — they do for matching names)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _align_pair(ours_csv: str, other_csv: str, t: str,
                ours_prob_col: str, ours_label_col: str,
                other_prob_col: str) -> pd.DataFrame:
    """Load+align one (ours, other) prediction pair on SMILES, deduped.

    Column names are configurable so this works with both ensemble_eval's
    `<target>_prob`/`<target>_label` format and descriptor_baseline's
    `XGB_prob`/`label` format. Returns a frame with columns y, p_ours, p_other.
    """
    ours = pd.read_csv(ours_csv)
    other = pd.read_csv(other_csv)

    for c in (ours_prob_col, ours_label_col):
        if c not in ours.columns:
            raise SystemExit(f"{ours_csv}: missing column {c!r} (have {list(ours.columns)})")
    if other_prob_col not in other.columns:
        raise SystemExit(f"{other_csv}: missing prediction column {other_prob_col!r} "
                         f"(have {list(other.columns)})")

    ours = ours[["smiles", ours_prob_col, ours_label_col]].rename(
        columns={ours_prob_col: "p_ours", ours_label_col: "y"})
    other = other[["smiles", other_prob_col]].rename(columns={other_prob_col: "p_other"})
    # Drop rows where the other model failed to predict (chemprop emits blanks for
    # invalid SMILES).
    other = other.dropna(subset=["p_other"])
    # Deduplicate on SMILES BEFORE merging — the Wong set has a few repeated
    # structures; without this an inner join cross-products them (a duplicate
    # SMILES with k copies on each side yields k*k rows), inflating and
    # contaminating the pairing. Keep one row per unique molecule.
    ours = ours.drop_duplicates("smiles", keep="first")
    other = other.drop_duplicates("smiles", keep="first")
    df = ours.merge(other, on="smiles", how="inner")
    df = df.dropna(subset=["p_ours", "p_other", "y"])
    return df[["y", "p_ours", "p_other"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap of AUPRC difference")
    ap.add_argument("--ours", required=True, nargs="+",
                    help="One or more CSVs from ensemble_eval.py --save-probs "
                         "(smiles, <target>_prob, <target>_label). Pass several "
                         "(one per split seed) to POOL them — multiplies positives.")
    ap.add_argument("--other", required=True, nargs="+",
                    help="Other model's prediction CSV(s) (smiles, <target>=prob), "
                         "e.g. chemprop test_preds.csv. Must be parallel to --ours "
                         "(same count, same order = same split seed).")
    ap.add_argument("--target", default="antibiotic",
                    help="Task name; default column names derive from it.")
    ap.add_argument("--ours-prob-col", default=None,
                    help="Probability column in --ours CSVs (default '<target>_prob'; "
                         "use e.g. 'XGB_prob' for descriptor_baseline --save-probs output).")
    ap.add_argument("--ours-label-col", default=None,
                    help="Label column in --ours CSVs (default '<target>_label'; "
                         "use 'label' for descriptor_baseline --save-probs output).")
    ap.add_argument("--other-prob-col", default=None,
                    help="Prediction column in --other CSVs (default '<target>', "
                         "e.g. chemprop's test_preds.csv uses the task name).")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ours_prob_col = args.ours_prob_col or f"{args.target}_prob"
    ours_label_col = args.ours_label_col or f"{args.target}_label"
    other_prob_col = args.other_prob_col or args.target

    if len(args.ours) != len(args.other):
        raise SystemExit(f"--ours ({len(args.ours)}) and --other ({len(args.other)}) "
                         f"must have the same number of files (parallel pairs).")

    t = args.target
    # Align each seed's pair, then POOL all aligned rows into one set. Pooling
    # across K random splits ~K-multiplies the positive count (each molecule lands
    # in test ~test_size of the time per split) → tighter ΔAUPRC CI. A molecule may
    # recur across splits; that is the standard repeated-subsampling pooled eval.
    parts = []
    for i, (o, c) in enumerate(zip(args.ours, args.other)):
        part = _align_pair(o, c, t, ours_prob_col, ours_label_col, other_prob_col)
        parts.append(part)
        print(f"  pair {i}: {len(part):5d} aligned, {int(part['y'].sum()):4d} pos"
              f"   ({o}  vs  {c})")
    df = pd.concat(parts, ignore_index=True)

    n = len(df)
    n_pos = int(df["y"].sum())
    print(f"\nPOOLED over {len(parts)} split(s): {n} compounds  "
          f"positives={n_pos} ({100*n_pos/max(n,1):.2f}%)")
    if n_pos == 0:
        raise SystemExit("No positives after alignment — check --target / inputs.")

    y = df["y"].to_numpy().astype(int)
    p_ours = df["p_ours"].to_numpy()
    p_other = df["p_other"].to_numpy()

    ap_ours = _ap(y, p_ours)
    ap_other = _ap(y, p_other)
    print(f"\nPoint AUPRC  ours = {ap_ours:.4f}   other = {ap_other:.4f}   "
          f"Δ = {ap_ours - ap_other:+.4f}")

    rng = np.random.default_rng(args.seed)
    deltas, ours_b, other_b = [], [], []
    skipped = 0
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() == 0:           # degenerate resample (no positives)
            skipped += 1
            continue
        a = _ap(y[idx], p_ours[idx])
        b = _ap(y[idx], p_other[idx])
        deltas.append(a - b)
        ours_b.append(a)
        other_b.append(b)

    deltas = np.asarray(deltas)
    ours_b = np.asarray(ours_b)
    other_b = np.asarray(other_b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float((deltas <= 0).mean())   # one-sided: P(ours not better)

    print(f"\nPaired bootstrap ({len(deltas)} valid replicates, {skipped} skipped):")
    print(f"  ours   AUPRC  : {ours_b.mean():.4f}  95% CI [{np.percentile(ours_b,2.5):.4f}, {np.percentile(ours_b,97.5):.4f}]")
    print(f"  other  AUPRC  : {other_b.mean():.4f}  95% CI [{np.percentile(other_b,2.5):.4f}, {np.percentile(other_b,97.5):.4f}]")
    print(f"  Δ (ours−other): {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  one-sided bootstrap p (ours ≤ other): {p_value:.4f}")
    verdict = ("SIGNIFICANT win (Δ CI excludes 0)" if lo > 0 else
               "SIGNIFICANT loss (Δ CI excludes 0)" if hi < 0 else
               "NOT significant (Δ CI spans 0)")
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()