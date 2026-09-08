"""Frozen-backbone "neural fingerprint" + XGBoost on ogbg-molhiv.

The cleanest isolation of representation quality: take a pretrained LeJEPA
backbone, FREEZE it, pool every molecule into a single embedding vector, and feed
those vectors to the exact same XGBoost ensemble used in ``xgb_molhiv.py`` for
Morgan/descriptors. Same split, same head, same protocol — the only thing that
changes is the 2048 ECFP bits -> the learned GNN embedding. So the comparison is
"is the learned neural fingerprint a better molecule representation than a fixed
circular fingerprint?", with the finetuning head removed as a confound.

``--embed-mode`` is a ``+``-joined list of feature blocks, concatenated
left-to-right into one design matrix:
  * ``pooled`` — the PURE pooled GNN representation (hidden_dim, pre-projection,
                 pre-descriptor). This is the SSL-convention backbone embedding and
                 the apples-to-apples analog of structure-only Morgan bits. DEFAULT.
  * ``proj``   — the post-proj_head pretraining embedding ``z`` (embed_dim); this
                 folds in the 217 RDKit descriptors the proj_head consumes, so it
                 is "neural fingerprint + descriptors", not graph-only.
  * ``desc``   — the 217 z-scored RDKit descriptors (global physicochemistry).
  * ``morgan`` — Morgan/ECFP bits (``--radius`` / ``--n-bits``), recomputed from the
                 split's SMILES so the rows stay aligned with the embeddings.
At most one of ``pooled``/``proj``; ``desc``/``morgan`` alone skip the backbone.

Usage:
    uv run python scripts/xgb_molhiv_embed.py \
        --checkpoint checkpoints/pretrain_chembl/final.pt --n-models 10
    # graph-only embedding vs descriptors-augmented:
    uv run python scripts/xgb_molhiv_embed.py --embed-mode pooled+desc \
        --checkpoint checkpoints/pretrain_chembl/final.pt --n-models 10
    # everything concatenated, against the Morgan+RF leaderboard head:
    uv run python scripts/xgb_molhiv_embed.py --embed-mode morgan+desc+pooled \
        --model rf --n-estimators 500 \
        --checkpoint checkpoints/pretrain_chembl/final.pt --n-models 5
    # untrained backbone control (random init) — isolates the pretraining lift:
    uv run python scripts/xgb_molhiv_embed.py --random-init --n-models 10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.molhiv_dataset import MolHIVDataset
from src.finetune import load_backbone
from scripts.xgb_molhiv import run_ensemble, add_model_args, featurize_morgan

BLOCKS = ("pooled", "proj", "desc", "morgan")


@torch.no_grad()
def embed_split(backbone, dataset, device, batch_size, mode: str,
                morgan_radius: int = 2, morgan_bits: int = 2048,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a split -> (N, D) design matrix and (N,) labels.

    ``mode`` is a '+'-joined list of blocks from ``BLOCKS``, concatenated in the
    order written:
      'pooled' temporarily neutralizes the proj_head + descriptor concat so the
      backbone returns the raw pooled GNN representation (hidden_dim);
      'proj'   is the post-proj_head embedding z (folds in the descriptors);
      'desc'   is the 217 z-scored RDKit descriptors carried on ``global_feat``;
      'morgan' is ECFP bits recomputed from ``dataset.smiles``, which the dataset
               must expose index-aligned with its items (unparseable rows are dropped
               at load time, so the raw split order is NOT safe to reuse).
    At most one of 'pooled'/'proj'. A mode with neither skips the backbone entirely,
    so 'desc' / 'morgan' / 'morgan+desc' are cheap references that still go through
    the exact same head as the embedding modes.
    """
    blocks = [b for b in mode.split("+") if b]
    unknown = [b for b in blocks if b not in BLOCKS]
    if unknown:
        raise ValueError(f"unknown feature block(s) {unknown} in mode {mode!r}; "
                         f"valid blocks are {list(BLOCKS)} joined by '+'")
    gnn = [b for b in blocks if b in ("pooled", "proj")]
    if len(gnn) > 1:
        raise ValueError(f"mode {mode!r} asks for both 'pooled' and 'proj' — pick one")
    base = gnn[0] if gnn else None

    if base == "pooled":
        # Bypass the projection head and the internal descriptor concat so we read
        # the pure pooled GNN embedding (the SSL backbone representation).
        saved_proj, saved_ndesc = backbone.proj_head, backbone.n_descriptors
        backbone.proj_head = nn.Identity()
        backbone.n_descriptors = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=Batch.from_data_list)
    embs, descs, ys = [], [], []
    try:
        for batch in loader:
            ys.append(batch.y.view(-1).cpu().numpy())
            if "desc" in blocks:
                descs.append(batch.global_feat.cpu().numpy())
            if base is not None:
                embs.append(backbone(batch.to(device)).cpu().numpy())   # (B, D)
    finally:
        if base == "pooled":
            backbone.proj_head, backbone.n_descriptors = saved_proj, saved_ndesc

    y = np.concatenate(ys, axis=0).astype(np.int32)
    parts: dict[str, np.ndarray] = {}
    if base is not None:
        parts[base] = np.concatenate(embs, axis=0)
    if "desc" in blocks:
        parts["desc"] = np.concatenate(descs, axis=0)
    if "morgan" in blocks:
        smiles = getattr(dataset, "smiles", None)
        if smiles is None:
            raise AttributeError(
                f"block 'morgan' needs {type(dataset).__name__} to expose a `.smiles` "
                "list aligned with its items")
        if len(smiles) != len(y):
            raise ValueError(f"`.smiles` has {len(smiles)} rows but the split yielded "
                             f"{len(y)} labels — fingerprints would not line up")
        parts["morgan"] = featurize_morgan(list(smiles), morgan_radius, morgan_bits)

    print("  blocks: " + " + ".join(f"{b}({parts[b].shape[1]})" for b in blocks), flush=True)
    return np.concatenate([parts[b] for b in blocks], axis=1).astype(np.float32), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/pretrain_chembl/final.pt",
                    help="pretrained backbone; ignored if --random-init")
    ap.add_argument("--random-init", action="store_true",
                    help="use an untrained backbone (control for the pretraining lift)")
    ap.add_argument("--embed-mode", default="pooled",
                    help="'+'-joined feature blocks from {pooled, proj, desc, morgan}, "
                         "concatenated in the order written (e.g. 'morgan+desc+pooled'); "
                         "at most one of pooled/proj")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (2 = ECFP4)")
    ap.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint length")
    ap.add_argument("--batch-size", type=int, default=256)
    # Tabular-head knobs (consumed by run_ensemble): --model/--n-models/--n-estimators/...
    add_model_args(ap)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = "" if args.random_init else args.checkpoint
    backbone = load_backbone(ckpt, cfg, device)
    backbone.eval()
    tag = "RANDOM-INIT (untrained)" if args.random_init else args.checkpoint
    print(f"Backbone: {tag}  |  embed-mode: {args.embed_mode}", flush=True)

    print("Featurizing molhiv splits (graph + descriptors) ...", flush=True)
    feat = cfg.data.get("featurizer", "ours")
    train_ds = MolHIVDataset(split="train", root=cfg.data.root, featurizer=feat)
    valid_ds = MolHIVDataset(split="valid", root=cfg.data.root, featurizer=feat,
                             desc_stats=train_ds.desc_stats)
    test_ds = MolHIVDataset(split="test", root=cfg.data.root, featurizer=feat,
                            desc_stats=train_ds.desc_stats)

    print("Building the feature blocks ...", flush=True)
    kw = dict(morgan_radius=args.radius, morgan_bits=args.n_bits)
    Xtr, ytr = embed_split(backbone, train_ds, device, args.batch_size, args.embed_mode, **kw)
    Xva, yva = embed_split(backbone, valid_ds, device, args.batch_size, args.embed_mode, **kw)
    Xte, yte = embed_split(backbone, test_ds, device, args.batch_size, args.embed_mode, **kw)

    run_ensemble(Xtr, ytr, Xva, yva, Xte, yte, args)


if __name__ == "__main__":
    main()
