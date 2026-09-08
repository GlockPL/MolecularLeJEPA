"""Finetune a (LeJEPA-pretrained or scratch) backbone on ogbg-molhiv.

Follows the OGB benchmark protocol: canonical scaffold split, single binary task,
ROC-AUC, **select on VALID, report TEST**, full end-to-end finetuning, multiple
seeds → mean±std. This is the apples-to-apples scratch-vs-pretrained test the
antibiotic task couldn't give us (Phase 11). Compare to the GNN-SSL band
(~0.75-0.79: GIN scratch ~0.757, MolCLR ~0.77, Mole-BERT ~0.787) — NOT the
fingerprint-ensemble leaderboard #1.

Metric note: OGB's Evaluator 'rocauc' for ogbg-molhiv = mean over tasks of
sklearn ``roc_auc_score``; with the single HIV task it is IDENTICAL to
``roc_auc_score`` used here. Pass --use-ogb-evaluator to use the official
Evaluator instead (requires the ``ogb`` package).

Usage:
    # pretrained:
    uv run python scripts/finetune_molhiv.py --config configs/finetune_molhiv.yaml \
        --checkpoint checkpoints/pretrain_chembl/final.pt --seeds 0,1,2,3,4
    # scratch control:
    uv run python scripts/finetune_molhiv.py --config configs/finetune_molhiv.yaml \
        --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.molhiv_dataset import MolHIVDataset
from src.finetune import load_backbone
from src.logutil import RunLogger
from src.models.heads import MLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rocauc(y_true: np.ndarray, y_score: np.ndarray, evaluator=None) -> float:
    if evaluator is not None:
        return evaluator.eval({"y_true": y_true.reshape(-1, 1),
                               "y_pred": y_score.reshape(-1, 1)})["rocauc"]
    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def _eval(backbone, head, loader, device, evaluator=None) -> tuple[float, np.ndarray, np.ndarray]:
    backbone.eval(); head.eval()
    probs, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        z = backbone(batch)
        if hasattr(batch, "global_feat"):
            z = torch.cat([z, batch.global_feat], dim=1)
        probs.append(torch.sigmoid(head(z).squeeze(1)).cpu())
        targets.append(batch.y.squeeze(1).cpu())
    p = torch.cat(probs).numpy()
    t = torch.cat(targets).numpy()
    return _rocauc(t, p, evaluator), p, t


def _make_head(cfg, device):
    in_dim = cfg.model.embed_dim + cfg.model.get("n_descriptors", 0)
    return MLP(in_dim, cfg.model.get("head_hidden_dim", 256), 1,
               dropout=cfg.model.get("dropout", 0.1),
               num_layers=cfg.model.get("head_num_layers", 1)).to(device)


def run_seed(seed, cfg, checkpoint, loaders, pos_weight, device, out_dir, evaluator):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    backbone = load_backbone(checkpoint, cfg, device)
    head = _make_head(cfg, device)
    pw = torch.tensor([pos_weight], device=device) if cfg.training.get("pos_weight", False) else None
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    probe_epochs = int(cfg.training.get("probe_epochs", 0))
    probe_lr = float(cfg.training.get("probe_lr", cfg.training.lr))
    wd = cfg.training.get("weight_decay", 0.0)

    seed_dir = Path(out_dir) / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Best is tracked across BOTH stages (probe + finetune) and selected by val.
    best = {"val": -1.0, "test": None, "epoch": -1, "stage": None}

    def _train_stage(optimizer, n_epochs, stage, ep_offset, freeze_backbone):
        """Run n_epochs; eval val+test each; track/save global best-by-val."""
        params = list(head.parameters()) if freeze_backbone \
            else list(backbone.parameters()) + list(head.parameters())
        for e in range(1, n_epochs + 1):
            # Frozen probe: backbone in eval() (no dropout/BN updates), no grad
            # through it; only the head trains. Finetune: everything trains.
            head.train()
            backbone.eval() if freeze_backbone else backbone.train()
            tot, nb = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                if freeze_backbone:
                    with torch.no_grad():
                        z = backbone(batch)
                    z = z.detach()
                else:
                    z = backbone(batch)
                if hasattr(batch, "global_feat"):
                    z = torch.cat([z, batch.global_feat], dim=1)
                loss = bce(head(z).squeeze(1), batch.y.squeeze(1))
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                tot += loss.item(); nb += 1
            val_auc, _, _ = _eval(backbone, head, val_loader, device, evaluator)
            test_auc, _, _ = _eval(backbone, head, test_loader, device, evaluator)
            gep = ep_offset + e
            star = ""
            if val_auc > best["val"]:
                best.update(val=val_auc, test=test_auc, epoch=gep, stage=stage)
                star = "  *"
                torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                            "epoch": gep, "stage": stage, "val_auc": val_auc,
                            "test_auc": test_auc, "seed": seed}, seed_dir / "best.pt")
            print(f"  {stage:5} {gep:>3}  {tot/max(nb,1):>8.4f}  {val_auc:>8.4f}  {test_auc:>9.4f}{star}")

    print(f"\n  seed {seed}: probe {probe_epochs} ep (lr={probe_lr:g}) → "
          f"finetune {cfg.training.epochs} ep (lr={cfg.training.lr:g})")
    print(f"  {'stage':5} {'ep':>3}  {'loss':>8}  {'val_auc':>8}  {'test_auc':>9}")

    # Stage 1 — frozen-backbone linear/MLP probe (optional; probe_epochs=0 skips).
    if probe_epochs > 0:
        for p in backbone.parameters():
            p.requires_grad_(False)
        opt_probe = torch.optim.Adam(head.parameters(), lr=probe_lr, weight_decay=wd)
        _train_stage(opt_probe, probe_epochs, "probe", 0, freeze_backbone=True)
        for p in backbone.parameters():
            p.requires_grad_(True)

    # Stage 2 — full end-to-end finetune.
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                           lr=cfg.training.lr, weight_decay=wd)
    _train_stage(opt, cfg.training.epochs, "ft", probe_epochs, freeze_backbone=False)

    print(f"  → seed {seed}: best val {best['val']:.4f} @{best['stage']}-ep{best['epoch']}  "
          f"TEST ROC-AUC = {best['test']:.4f}")
    return best["test"], best["val"], best["epoch"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Finetune on ogbg-molhiv (OGB protocol)")
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--checkpoint", default=None, help="Pretrained backbone (omit = scratch)")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated model seeds")
    ap.add_argument("--out", default=None, help="Output root (default checkpoint_dir from config)")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override training.lr. OGB does NOT prescribe lr — sweep it and "
                         "select by VALIDATION ROC-AUC (never test). Tune scratch and "
                         "pretrained independently: a high lr can wash out pretrained "
                         "features, so forcing both to one lr can unfairly handicap the "
                         "pretrained model.")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="Few-shot: keep this fraction of the TRAIN set (stratified by "
                         "label; val/test fixed). <1.0 probes the low-label regime where "
                         "pretraining should help most. The subsample is fixed across "
                         "model seeds (--fewshot-seed); seeds vary only init/training.")
    ap.add_argument("--fewshot-seed", type=int, default=42,
                    help="Seed for the few-shot train subsample (reproducible).")
    ap.add_argument("--probe-epochs", type=int, default=None,
                    help="Frozen-backbone probe epochs BEFORE finetuning (0 = pure "
                         "end-to-end). For a pretrained backbone a short probe (e.g. "
                         "10-20) lets the head adapt first so the finetune lr can't wreck "
                         "the pretrained features in the first steps — often the single "
                         "biggest lever besides lr. Ignored/wasteful for scratch.")
    ap.add_argument("--probe-lr", type=float, default=None,
                    help="LR for the frozen probe (default = training.lr)")
    ap.add_argument("--use-ogb-evaluator", action="store_true",
                    help="Use the official ogb Evaluator (requires `ogb`); else sklearn "
                         "roc_auc_score (identical for single-task molhiv).")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    cfg = OmegaConf.load(args.config)
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.probe_epochs is not None:
        cfg.training.probe_epochs = args.probe_epochs
    if args.probe_lr is not None:
        cfg.training.probe_lr = args.probe_lr
    # Tag the output/log dir by lr (+ probe + few-shot frac) so a sweep doesn't overwrite itself.
    pe = int(cfg.training.get("probe_epochs", 0))
    lr_tag = (f"_lr{cfg.training.lr:g}" + (f"_probe{pe}" if pe else "")
              + (f"_frac{args.train_frac:g}" if args.train_frac < 1.0 else ""))
    out_dir = args.out or (cfg.get("checkpoint_dir", "checkpoints/finetune_molhiv") + lr_tag)
    tag = ("pretrained" if args.checkpoint else "scratch") + lr_tag

    log = RunLogger(
        f"finetune_molhiv_{tag}",
        title="ogbg-molhiv finetune (OGB protocol: scaffold split, ROC-AUC, val-select)",
        checkpoint=args.checkpoint or "(none — scratch)",
        seeds=seeds,
        out_dir=out_dir,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        probe_epochs=cfg.training.get("probe_epochs", 0),
        probe_lr=cfg.training.get("probe_lr", cfg.training.lr),
        backbone=cfg.model.get("backbone", "gps"),
        n_descriptors=cfg.model.get("n_descriptors", 0),
        train_frac=args.train_frac,
    ).start()
    print(OmegaConf.to_yaml(cfg).rstrip() + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = None
    if args.use_ogb_evaluator:
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name="ogbg-molhiv")
        print("Using official OGB Evaluator (ogbg-molhiv).")

    # Datasets built ONCE (the scaffold split is fixed; seeds only vary model
    # init + training stochasticity). valid/test share TRAIN descriptor stats.
    feat = cfg.data.get("featurizer", "ours")
    root = cfg.data.get("root", "data/ogbg_molhiv")
    print("\n[datasets]")
    train_ds = MolHIVDataset("train", root=root, featurizer=feat)
    val_ds = MolHIVDataset("valid", root=root, featurizer=feat, desc_stats=train_ds.desc_stats)
    test_ds = MolHIVDataset("test", root=root, featurizer=feat, desc_stats=train_ds.desc_stats)

    # Few-shot: stratified subsample of the TRAIN set (val/test untouched, descriptor
    # stats kept from the FULL train so normalization is consistent across budgets).
    if args.train_frac < 1.0:
        from sklearn.model_selection import train_test_split
        ys = np.array([int(d.y.item()) for d in train_ds._data_list])
        keep, _ = train_test_split(
            np.arange(len(ys)), train_size=args.train_frac, stratify=ys,
            random_state=args.fewshot_seed,
        )
        train_ds._data_list = [train_ds._data_list[i] for i in keep]
        print(f"  few-shot: train subsampled to {len(keep)} ({args.train_frac:.1%}), "
              f"{int(ys[keep].sum())} HIV-active (seed {args.fewshot_seed})")

    pos_weight = train_ds.pos_weight()
    print(f"  pos_weight (N_neg/N_pos) = {pos_weight:.1f}  "
          f"(applied: {bool(cfg.training.get('pos_weight', False))})")

    bs = cfg.training.batch_size
    loaders = (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=Batch.from_data_list),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=Batch.from_data_list),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=Batch.from_data_list),
    )

    results = []
    for seed in seeds:
        test_auc, best_val, best_ep = run_seed(
            seed, cfg, args.checkpoint, loaders, pos_weight, device, out_dir, evaluator)
        results.append(test_auc)

    arr = np.array(results)
    print("\n" + "=" * 60)
    print(f"ogbg-molhiv TEST ROC-AUC ({tag}) over {len(seeds)} seeds:")
    for s, r in zip(seeds, results):
        print(f"  seed {s}: {r:.4f}")
    print("-" * 60)
    print(f"  mean ± std = {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"  (GNN-SSL band ~0.75-0.79; GIN-scratch ~0.757, Mole-BERT ~0.787)")
    print("=" * 60)
    log.stop()


if __name__ == "__main__":
    main()