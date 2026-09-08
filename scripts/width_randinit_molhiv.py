"""Does WIDTH alone lift the frozen probe? Untrained backbones at increasing hidden_dim.

The cheap half of the width question. `width_probe_molhiv.py` showed Morgan loses ~0.05
when narrowed from 2048 to 128 bits, which suggests much of the fingerprint's edge is
column count rather than feature quality. Before spending a multi-day wide pretrain to
test that, ask whether width helps AT ALL on this probe — using RANDOM-INIT backbones,
which cost nothing to make.

The probe reads the POOLED representation (embed_split's 'pooled' bypasses proj_head),
so `hidden_dim` — not `embed_dim` — sets the probe's feature count. Each row is a fresh
untrained GPS at that width, averaged over several inits since an untrained network is
a single random draw.

Reading it:
  * random-init AUC rising steeply with width  -> width is mechanically valuable to the
    RF head, and a wider PRETRAINED model is well motivated.
  * random-init AUC flat                       -> width alone buys nothing; any gain
    would have to come from the objective filling the extra dimensions with real
    information, which is a much weaker bet.

Usage:
    uv run python scripts/width_randinit_molhiv.py --widths 128 256 512 1024
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.molhiv_dataset import MolHIVDataset
from src.models.gps_transformer import GPSTransformer
from scripts.xgb_molhiv_embed import embed_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--readouts", nargs="+", default=["mean", "max"])
    ap.add_argument("--inits", type=int, default=2, help="random inits to average")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--n-trees", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = cfg.data.get("featurizer", "ours")
    tr = MolHIVDataset(split="train", root=cfg.data.root, featurizer=feat)
    te = MolHIVDataset(split="test", root=cfg.data.root, featurizer=feat,
                       desc_stats=tr.desc_stats)
    ytr = np.array([int(d.y[0, 0]) for d in tr._data_list])
    yte = np.array([int(d.y[0, 0]) for d in te._data_list])

    print(f"\nUntrained-backbone width sweep | RF {args.n_trees} trees x {args.n_models} "
          f"seeds, {args.inits} inits\n{'='*66}", flush=True)
    for ro in args.readouts:
        for w in args.widths:
            per_init = []
            for i in range(args.inits):
                torch.manual_seed(100 + i)
                bb = GPSTransformer(
                    hidden_dim=w, embed_dim=min(w, cfg.model.embed_dim),
                    num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
                    walk_length=cfg.model.get("walk_length", 20),
                    dropout=cfg.model.get("dropout", 0.1),
                    n_descriptors=cfg.model.get("n_descriptors", 0),
                    readout=ro).to(device)
                bb.eval()
                Xtr, _ = embed_split(bb, tr, device, args.batch_size, "pooled")
                Xte, _ = embed_split(bb, te, device, args.batch_size, "pooled")
                aucs = [roc_auc_score(yte, RandomForestClassifier(
                            n_estimators=args.n_trees, min_samples_leaf=2, n_jobs=-1,
                            random_state=s).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
                        for s in range(args.n_models)]
                per_init.append(float(np.mean(aucs)))
                del bb, Xtr, Xte
                torch.cuda.empty_cache()
            print(f"  random-init {ro:4s} hidden_dim {w:5d}  test AUC "
                  f"{np.mean(per_init):.4f} +/- {np.std(per_init):.4f}  (over inits)",
                  flush=True)


if __name__ == "__main__":
    main()
