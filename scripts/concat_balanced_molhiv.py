"""Redesigned Morgan+embedding concat — each representation at its OWN measured optimum.

The first concat test (run_molhiv_concat.sh) was null: morgan+desc+pooled 0.8073 vs
morgan 0.8086, p=0.53. But `width_probe_molhiv.py` later showed that test could not have
detected complementarity even if it existed:

  * the design matrix was morgan(2048) + desc(217) + pooled(128) = 2393 columns;
  * RF with max_features='sqrt' samples ~49 columns per split, so ~2.6 are embedding dims;
  * and only ~32 of those 128 embedding dims carry signal (the rest are flat-to-harmful),
    leaving well under ONE informative embedding column per split.

So the embedding was effectively invisible, while its ~96 uninformative columns added
noise the forest could overfit — an effect we measured directly elsewhere (untrained
columns cost 0.020).

This version balances the blocks by using each at its own optimum from the width sweep:
Morgan at its peak (~512-1024 bits; it DECLINES by 4096) and the embedding truncated to
its informative subspace (~32 dims). ~1056 columns instead of 2393, with the embedding
now ~3% of the matrix rather than 5% of which most is noise. ``--max-features`` is also
swept, since that directly controls how often the embedding block is even sampled.

Truncation is by PCA top-k fitted on TRAIN (the best k-dim linear summary). PCA rotation
alone costs ~0.027 on this probe, so embedding-only rows are reported against a PCA
baseline as well as the raw one — but in a CONCAT the rotation is applied to the embedding
block only, and what matters is whether the pair beats Morgan alone.

Usage:
    uv run python scripts/concat_balanced_molhiv.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.molhiv_dataset import MolHIVDataset
from src.finetune import load_backbone
from src.models.gps_transformer import GPSTransformer
from scripts.xgb_molhiv import featurize_morgan, _parse_max_features
from scripts.xgb_molhiv_embed import embed_split
from scripts.spectrum import Spectrum


def rf(Xtr, ytr, Xte, yte, n_models, n_trees, max_features, Xva=None, yva=None,
       save_probs: str = ""):
    """Returns (test mean, test std, test ensemble, valid mean). VALID is what the
    OGB protocol selects on — reporting only test would let us pick the winner by
    peeking, which is exactly the failure mode this project keeps finding."""
    probs, aucs, vaucs = [], [], []
    for seed in range(n_models):
        c = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2,
                                   max_features=_parse_max_features(max_features),
                                   n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        p = c.predict_proba(Xte)[:, 1]
        probs.append(p); aucs.append(roc_auc_score(yte, p))
        if Xva is not None:
            vaucs.append(roc_auc_score(yva, c.predict_proba(Xva)[:, 1]))
    if save_probs:
        np.savez(save_probs, probs=np.mean(probs, 0), y=yte)
    return (float(np.mean(aucs)), float(np.std(aucs)),
            roc_auc_score(yte, np.mean(probs, 0)),
            float(np.mean(vaucs)) if vaucs else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/pretrain_chembl/final.pt")
    ap.add_argument("--morgan-bits", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--embed-dims", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--max-features", nargs="+", default=["sqrt", "0.1"])
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--n-trees", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--random-init", action="store_true",
                    help="CONTROL: use an UNTRAINED backbone of the same architecture "
                         "instead of the checkpoint. Answers the referee question the "
                         "pretrained-vs-Morgan comparison cannot: is the gain from what "
                         "pretraining learned, or would any 32 dense graph-derived "
                         "columns help a fingerprint equally? Run with the same flags as "
                         "the main sweep and compare the delta column cell for cell.")
    ap.add_argument("--init-seed", type=int, default=100)
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

    if args.random_init:
        torch.manual_seed(args.init_seed)
        bb = GPSTransformer(
            hidden_dim=cfg.model.hidden_dim, embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
            walk_length=cfg.model.get("walk_length", 20),
            dropout=cfg.model.get("dropout", 0.1),
            n_descriptors=cfg.model.get("n_descriptors", 0),
            readout=cfg.model.get("readout", "mean")).to(device)
        print(f"CONTROL: UNTRAINED backbone (seed {args.init_seed}), checkpoint ignored")
    else:
        bb = load_backbone(args.checkpoint, cfg, device)
    bb.eval()
    Etr, _ = embed_split(bb, tr, device, args.batch_size, "pooled")
    Eva, _ = embed_split(bb, va, device, args.batch_size, "pooled")
    Ete, _ = embed_split(bb, te, device, args.batch_size, "pooled")
    sp = Spectrum(Etr)
    Ptr, Pva, Pte = (sp.apply(Etr, "center"), sp.apply(Eva, "center"),
                     sp.apply(Ete, "center"))   # PCA coords, TRAIN-fitted

    morgan = {nb: (featurize_morgan(tr.smiles, 2, nb), featurize_morgan(va.smiles, 2, nb),
                   featurize_morgan(te.smiles, 2, nb)) for nb in args.morgan_bits}

    print(f"\nBalanced concat | RF {args.n_trees} trees x {args.n_models} seeds"
          f"\n{'='*84}\n  {'features':38s} {'dim':>5s} {'test AUC':>16s} {'ens':>8s}"
          f" {'VALID':>8s}", flush=True)

    results = {}

    pfx = "randinit_" if args.random_init else ""      # never overwrite the pretrained npz

    def row(name, Xtr, Xva_, Xte, mf, tag=""):
        m, sd, ens, vm = rf(Xtr, ytr, Xte, yte, args.n_models, args.n_trees, mf,
                            Xva_, yva,
                            save_probs=(f"logs/concat_{pfx}{tag}.npz" if tag else ""))
        print(f"  {name:38s} {Xtr.shape[1]:5d} {m:.4f} +/- {sd:.4f} {ens:8.4f} {vm:8.4f}",
              flush=True)
        results[name] = (m, ens, vm)
        return m

    base = {}
    for nb in args.morgan_bits:
        for mf in args.max_features:
            base[(nb, mf)] = row(f"morgan {nb} [mf={mf}]", morgan[nb][0], morgan[nb][1],
                                 morgan[nb][2], mf, tag=f"morgan{nb}_{mf}")
    for k in args.embed_dims:
        row(f"embed PCA-{k} alone", Ptr[:, :k], Pva[:, :k], Pte[:, :k], "sqrt")

    print(f"  {'-'*80}", flush=True)
    cells = {}
    for nb, k, mf in itertools.product(args.morgan_bits, args.embed_dims, args.max_features):
        Xtr = np.concatenate([morgan[nb][0], Ptr[:, :k]], axis=1)
        Xva_ = np.concatenate([morgan[nb][1], Pva[:, :k]], axis=1)
        Xte = np.concatenate([morgan[nb][2], Pte[:, :k]], axis=1)
        nm = f"morgan {nb} + embed PCA-{k} [mf={mf}]"
        m = row(nm, Xtr, Xva_, Xte, mf, tag=f"c{nb}_{k}_{mf}")
        d = m - base[(nb, mf)]
        cells[nm] = (nb, k, mf)
        print(f"  {'':38s} {'':5s} delta vs morgan alone: {d:+.4f}", flush=True)

    # SELECT ON VALID, report that cell's TEST number. Picking the best test cell
    # would be exactly the peeking this project keeps catching elsewhere.
    best = max(cells, key=lambda n: results[n][2])
    bm, bens, bvm = results[best]
    nb, k, mf = cells[best]
    bl = f"morgan {nb} [mf={mf}]"
    print(f"\n  VALID-SELECTED cell: {best}")
    print(f"    valid {bvm:.4f}  ->  TEST {bm:.4f} (ens {bens:.4f})")
    print(f"    its morgan-alone baseline: test {results[bl][0]:.4f} (valid {results[bl][2]:.4f})")
    print(f"    honest delta = {bm - results[bl][0]:+.4f}")
    print(f"\n  paired bootstrap:  uv run python scripts/paired_bootstrap_auc.py \\\n"
          f"      --ours logs/concat_{pfx}c{nb}_{k}_{mf}.npz"
          f" --other logs/concat_{pfx}morgan{nb}_{mf}.npz")


if __name__ == "__main__":
    main()
