"""Is the 0.788-vs-0.809 gap about REPRESENTATION QUALITY, or just WIDTH?

Every frozen probe in this project compares a 128-dim dense embedding against a
2048-bit sparse fingerprint, under a RandomForest — a head that thrives on many
weak, decorrelated features (max_features='sqrt' samples ~45 of 2048 columns per
split, vs ~11 of 128). The comparison has never been controlled for width, so we
cannot currently tell "Morgan is a better molecular representation" apart from
"Morgan has 16x more columns for the trees to work with".

Two sweeps, same RF head, same split:

  1. MORGAN vs n_bits (64 .. 4096). If Morgan at 128 bits falls to ~0.75, then most
     of its apparent advantage is dimensionality and the actionable move is to
     pretrain a WIDER backbone (hidden_dim/embed_dim 128 -> 512), not to redesign
     the objective. If Morgan at 128 bits still scores ~0.80, width is irrelevant
     and the gap is genuinely about what the features encode.

  2. EMBEDDING vs retained dimensions (8 .. 128). If AUC is still climbing at 128,
     the representation is width-limited and a wider model should help. If it
     saturates by 64, width is NOT the bottleneck and a wider model is wasted GPU.
     Reported two ways: PCA top-k (best k-dim linear summary) and random raw-column
     subsets (no rotation). The PCA rows must be read against PCA-128, not against
     raw-128 — a PCA rotation alone costs ~0.027 on this probe.

Usage:
    uv run python scripts/width_probe_molhiv.py --n-models 5 --n-trees 500
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.molhiv_dataset import MolHIVDataset
from src.finetune import load_backbone
from scripts.xgb_molhiv import featurize_morgan
from scripts.xgb_molhiv_embed import embed_split
from scripts.spectrum import Spectrum


def rf_auc(Xtr, ytr, Xte, yte, n_models: int, n_trees: int,
           Xva=None, yva=None) -> tuple[float, float, float]:
    """Returns (test mean, test std, VALID mean).

    Valid is reported because a curve read off TEST is a curve selected on test:
    the shape of these sweeps is what motivates the truncation dimension used
    downstream, so the shape has to be legible on a split we are allowed to look
    at. Test is kept as the held-out report for the point valid selects.
    """
    aucs, vaucs = [], []
    for seed in range(n_models):
        clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        if Xva is not None:
            vaucs.append(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))
    return (float(np.mean(aucs)), float(np.std(aucs)),
            float(np.mean(vaucs)) if vaucs else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/pretrain_chembl/final.pt")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--n-trees", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="logs/width_probe_molhiv.csv")
    ap.add_argument("--skip-morgan", action="store_true",
                    help="only sweep the embedding's retained dimensions. Use when probing "
                         "a NEW checkpoint — the Morgan curve is a fixed property of the "
                         "dataset and does not need re-measuring per backbone.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = cfg.data.get("featurizer", "ours")
    tr = MolHIVDataset(split="train", root=cfg.data.root, featurizer=feat)
    va = MolHIVDataset(split="valid", root=cfg.data.root, featurizer=feat,
                       desc_stats=tr.desc_stats)
    te = MolHIVDataset(split="test", root=cfg.data.root, featurizer=feat,
                       desc_stats=tr.desc_stats)
    ytr = np.array([int(d.y[0, 0]) for d in tr._data_list])
    yva = np.array([int(d.y[0, 0]) for d in va._data_list])
    yte = np.array([int(d.y[0, 0]) for d in te._data_list])

    rows = []
    print(f"\nWidth probe | RF {args.n_trees} trees x {args.n_models} seeds"
          f"\n{'='*72}\n  MORGAN vs n_bits", flush=True)
    for nb in ([] if args.skip_morgan else [64, 128, 256, 512, 1024, 2048, 4096]):
        m, s, v = rf_auc(featurize_morgan(tr.smiles, 2, nb), ytr,
                         featurize_morgan(te.smiles, 2, nb), yte,
                         args.n_models, args.n_trees,
                         featurize_morgan(va.smiles, 2, nb), yva)
        print(f"    morgan {nb:5d} bits   VALID {v:.4f}   test {m:.4f} +/- {s:.4f}", flush=True)
        rows.append(("morgan", nb, m, s, v))

    bb = load_backbone(args.checkpoint, cfg, device); bb.eval()
    Xtr, _ = embed_split(bb, tr, device, args.batch_size, "pooled")
    Xva, _ = embed_split(bb, va, device, args.batch_size, "pooled")
    Xte, _ = embed_split(bb, te, device, args.batch_size, "pooled")
    D = Xtr.shape[1]

    print(f"\n  EMBEDDING vs retained dims (PCA top-k; compare against PCA-{D})", flush=True)
    sp = Spectrum(Xtr)
    Ctr, Cva, Cte = (sp.apply(Xtr, "center"), sp.apply(Xva, "center"),
                     sp.apply(Xte, "center"))
    for k in [8, 16, 32, 64, 96, D]:
        m, s, v = rf_auc(Ctr[:, :k], ytr, Cte[:, :k], yte, args.n_models,
                         args.n_trees, Cva[:, :k], yva)
        print(f"    embed PCA top-{k:4d}    VALID {v:.4f}   test {m:.4f} +/- {s:.4f}", flush=True)
        rows.append(("embed_pca", k, m, s, v))

    print(f"\n  EMBEDDING vs retained dims (random raw columns, no rotation)", flush=True)
    rng = np.random.default_rng(0)
    for k in [8, 16, 32, 64, 96, D]:
        cols = np.sort(rng.choice(D, size=k, replace=False)) if k < D else np.arange(D)
        m, s, v = rf_auc(Xtr[:, cols], ytr, Xte[:, cols], yte, args.n_models,
                         args.n_trees, Xva[:, cols], yva)
        print(f"    embed raw  {k:4d} cols  VALID {v:.4f}   test {m:.4f} +/- {s:.4f}", flush=True)
        rows.append(("embed_raw", k, m, s, v))

    import csv
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "width", "test_mean", "test_std", "valid_mean"])
        w.writerows([(a, b, f"{c:.4f}", f"{d:.4f}", f"{e:.4f}") for a, b, c, d, e in rows])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
