"""Frozen linear-probe diagnostic — does LeJEPA learn task-relevant structure?

The decisive "did the SSL representation learn anything useful" test (the
GraphJEPA/field-standard protocol): freeze the backbone, extract embeddings, fit
a LINEAR model on them, and compare conditions. No backbone training, so it
isolates representation quality from finetuning dynamics.

For one split it reports test AUPRC of an L2 logistic regression on:
  - desc-only                : the 217 RDKit descriptors alone (the ceiling we
                               already know dominates).
  - random-bb                : frozen RANDOM-init backbone embedding (128) alone.
  - pretrained-bb            : frozen PRETRAINED backbone embedding (128) alone.
  - random-bb  ⊕ desc        : random embedding + descriptors.
  - pretrained-bb ⊕ desc     : pretrained embedding + descriptors.

The two comparisons that answer the thesis question:
  (A) pretrained-bb  vs  random-bb        → does the LEARNED graph rep carry task
                                            signal the random init doesn't?
  (B) pretrained-bb⊕desc  vs  desc-only   → does the learned rep ADD anything over
                                            descriptors (which the head already has)?
If pretrained-bb ≈ random-bb and (bb⊕desc) ≈ desc-only, the embedding is "pretty"
(organizes by descriptors) but adds no task-relevant structure — exactly the
suspected explanation for the downstream nulls.

Usage:
    uv run python scripts/linear_probe.py --config configs/finetune_desc_scaffold.yaml \\
        --checkpoint checkpoints/pretrain_gps_p7_cover025/epoch_0011.pt --task antibiotic
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.antibiotic_dataset import AntibioticDataset, LABEL_COLS
from src.finetune import load_backbone
from src.logutil import RunLogger


@torch.no_grad()
def _extract(backbone, loader, device, task_col):
    """Return (Z_backbone (N,K), Desc (N,217), y (N,)) for a split, frozen."""
    backbone.eval()
    zs, descs, ys = [], [], []
    for batch in loader:
        batch = batch.to(device)
        zs.append(backbone(batch).cpu())
        descs.append(batch.global_feat.cpu())
        ys.append(batch.y[:, task_col].cpu())
    return torch.cat(zs).numpy(), torch.cat(descs).numpy(), torch.cat(ys).numpy()


def _probe(Xtr, ytr, Xte, yte, n_boot=2000, seed=42):
    """Fit L2 logistic regression (balanced) on frozen features; test AUPRC + CI."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    ap = average_precision_score(yte, s)
    rng = np.random.default_rng(seed)
    n = len(yte)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if yte[idx].sum() == 0:
            continue
        boots.append(average_precision_score(yte[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return ap, lo, hi


def main():
    ap = argparse.ArgumentParser(description="Frozen linear-probe diagnostic")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="Pretrained backbone to probe")
    ap.add_argument("--task", default="antibiotic", choices=list(LABEL_COLS))
    ap.add_argument("--seed", type=int, default=42, help="random-init backbone seed")
    ap.add_argument("--train-fracs", default="1.0",
                    help="Comma list of train fractions for the few-shot probe curve "
                         "(e.g. 0.05,0.1,0.25,0.5,1.0). Subsample is stratified + fixed.")
    ap.add_argument("--fewshot-seed", type=int, default=42,
                    help="Seed for the few-shot train subsample.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_col = LABEL_COLS.index(args.task)
    split_method = cfg.data.get("split_method", "scaffold")

    log = RunLogger(
        f"linear_probe_{args.task}_{split_method}",
        title="Frozen linear-probe diagnostic (does LeJEPA learn task-relevant structure?)",
        checkpoint=args.checkpoint, config=args.config, task=args.task,
        split=split_method, seed=args.seed,
    ).start()

    common = dict(
        test_size=cfg.data.get("test_size", 0.20),
        val_size=cfg.data.get("val_size", 0.10),
        seed=cfg.data.get("seed", 42),
        split_method=split_method,
        tasks=[args.task],
        featurizer=cfg.data.get("featurizer", "ours"),
    )
    xlsx = cfg.data.xlsx_path
    train_ds = AntibioticDataset(xlsx, split="train", **common)
    test_ds = AntibioticDataset(xlsx, split="test", desc_stats=train_ds.desc_stats, **common)
    bs = 256

    def loaders():
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=False, collate_fn=Batch.from_data_list),
            DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=Batch.from_data_list),
        )

    # Pretrained and random-init backbones (same architecture).
    torch.manual_seed(args.seed)
    rand_bb = load_backbone(None, cfg, device)
    pre_bb = load_backbone(args.checkpoint, cfg, device)

    tr_loader, te_loader = loaders()
    Zp_tr, D_tr, y_tr = _extract(pre_bb, tr_loader, device, task_col)
    Zp_te, D_te, y_te = _extract(pre_bb, te_loader, device, task_col)
    tr_loader, te_loader = loaders()
    Zr_tr, _, _ = _extract(rand_bb, tr_loader, device, task_col)
    Zr_te, _, _ = _extract(rand_bb, te_loader, device, task_col)

    cat = lambda a, b: np.concatenate([a, b], axis=1)
    # Condition -> (train_features, test_features) over the FULL train (subsampled per frac).
    cond = {
        "desc-only (217)":            (D_tr, D_te),
        "random-bb (128)":            (Zr_tr, Zr_te),
        "pretrained-bb (128)":        (Zp_tr, Zp_te),
        "random-bb ⊕ desc (345)":     (cat(Zr_tr, D_tr), cat(Zr_te, D_te)),
        "pretrained-bb ⊕ desc (345)": (cat(Zp_tr, D_tr), cat(Zp_te, D_te)),
    }
    names = list(cond)

    # Few-shot frozen-probe curve: subsample the TRAIN rows (stratified, fixed seed)
    # at each fraction, refit each condition, evaluate on the FULL test set. Tests
    # the reframed thesis: the pretrained rep should beat random/desc by MORE as
    # labels get scarce (a better representation = sample-efficient).
    from sklearn.model_selection import train_test_split
    fracs = [float(x) for x in args.train_fracs.split(",") if x.strip()]

    print(f"\n{'='*88}")
    print(f"FROZEN LINEAR-PROBE FEW-SHOT — task={args.task}, {split_method} split")
    print(f"full train {len(y_tr)} ({int(y_tr.sum())} pos)  test {len(y_te)} ({int(y_te.sum())} pos)")
    print(f"{'='*88}")
    header = f"{'frac':>6}{'n_tr':>7}{'pos':>5}  " + "".join(f"{n.split(' ')[0]:>16}" for n in names)
    print(header)
    print("-" * len(header))
    table = {}  # frac -> {name: ap}
    for frac in fracs:
        if frac < 1.0:
            keep, _ = train_test_split(np.arange(len(y_tr)), train_size=frac,
                                       stratify=y_tr, random_state=args.fewshot_seed)
        else:
            keep = np.arange(len(y_tr))
        yk = y_tr[keep]
        row = {}
        for name in names:
            Xtr, Xte = cond[name]
            ap_, _, _ = _probe(Xtr[keep], yk, Xte, y_te)
            row[name] = ap_
        table[frac] = row
        cells = "".join(f"{row[n]:>16.4f}" for n in names)
        print(f"{frac:>6.2f}{len(keep):>7}{int(yk.sum()):>5}  {cells}")
    print("-" * len(header))
    print("Deltas vs #labels (the thesis question):")
    print(f"{'frac':>6}{'pre-bb − rand-bb':>20}{'(pre⊕desc) − desc':>20}")
    for frac in fracs:
        r = table[frac]
        dA = r["pretrained-bb (128)"] - r["random-bb (128)"]
        dB = r["pretrained-bb ⊕ desc (345)"] - r["desc-only (217)"]
        print(f"{frac:>6.2f}{dA:>+20.4f}{dB:>+20.4f}")
    print("  Reframed-thesis prediction: pre-bb − rand-bb should be LARGE/positive and")
    print("  LARGEST at small frac (a better rep is more sample-efficient). Flat/zero ⇒")
    print("  the rep advantage doesn't translate to label efficiency either.")
    print(f"{'='*88}")
    log.stop()


if __name__ == "__main__":
    main()