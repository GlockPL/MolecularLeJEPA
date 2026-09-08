"""Frozen-probe learning curve: molhiv RF test ROC-AUC vs pretraining epoch.

Sweeps the per-epoch checkpoints of a LeJEPA pretraining run, runs the EXACT
frozen-probe protocol from ``xgb_molhiv_embed.py`` (freeze backbone -> pooled
embedding -> RandomForest, OGB MorganFP+RF leaderboard recipe) at each epoch, and
plots the resulting representation-quality curve. This visualizes the central
molhiv claim: the frozen representation gets monotonically more downstream-useful
with self-supervised pretraining, with NO finetuning in the loop.

Efficiency: molhiv is RDKit-featurized ONCE (the expensive step is identical for
every checkpoint); each epoch only re-embeds through the frozen backbone and
re-fits the RF. ~30 s/epoch instead of re-featurizing 40x.

Usage:
    uv run python scripts/probe_curve_molhiv.py \
        --ckpt-dir checkpoints/pretrain_chembl \
        --config configs/finetune_molhiv.yaml \
        --label "GPS + descriptors" \
        --out figures/molhiv_probe_curve

Writes <out>.csv (epoch, valid_mean, valid_std, test_mean, test_std, ensemble)
and <out>.png. Re-running reuses the CSV unless --force, so the plot can be
restyled without re-probing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.molhiv_dataset import MolHIVDataset
from src.finetune import load_backbone
from scripts.xgb_molhiv_embed import embed_split

# Reference lines for the plot (per-model test ROC-AUC, same recipe).
RANDOM_FLOOR = 0.665   # random-init GPS pooled + RF (architecture floor)
MORGAN_RF = 0.807      # Morgan ECFP4 2048-bit + RF (OGB leaderboard recipe)
SSL_BAND = (0.75, 0.79)  # GIN-scratch .757 / MolCLR .77 / Mole-BERT .787


def _epoch_of(p: Path) -> int:
    # epoch_0013.pt -> 13 ; final.pt sorts last (handled by caller).
    return int(p.stem.split("_")[1])


def run_rf_probe(Xtr, ytr, Xva, yva, Xte, yte, n_models: int, n_estimators: int):
    """OGB MorganFP+RF leaderboard recipe, n_models seed ensemble.
    Returns (valid_mean, valid_std, test_mean, test_std, ensemble_test)."""
    test_probs, val_aucs, test_aucs = [], [], []
    for seed in range(n_models):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        )
        clf.fit(Xtr, ytr)
        p_val = clf.predict_proba(Xva)[:, 1]
        p_test = clf.predict_proba(Xte)[:, 1]
        val_aucs.append(roc_auc_score(yva, p_val))
        test_aucs.append(roc_auc_score(yte, p_test))
        test_probs.append(p_test)
    ens = roc_auc_score(yte, np.mean(test_probs, axis=0))
    return (float(np.mean(val_aucs)), float(np.std(val_aucs)),
            float(np.mean(test_aucs)), float(np.std(test_aucs)), float(ens))


def sweep(args, csv_path: Path) -> list[dict]:
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Featurizing molhiv splits once (graph + descriptors) ...", flush=True)
    feat = cfg.data.get("featurizer", "ours")
    train_ds = MolHIVDataset(split="train", root=cfg.data.root, featurizer=feat)
    valid_ds = MolHIVDataset(split="valid", root=cfg.data.root, featurizer=feat,
                             desc_stats=train_ds.desc_stats)
    test_ds = MolHIVDataset(split="test", root=cfg.data.root, featurizer=feat,
                            desc_stats=train_ds.desc_stats)

    ckpts = sorted(Path(args.ckpt_dir).glob("epoch_*.pt"), key=_epoch_of)
    if args.max_epoch is not None:
        ckpts = [c for c in ckpts if _epoch_of(c) <= args.max_epoch]
    if args.every > 1:
        ckpts = [c for c in ckpts if _epoch_of(c) % args.every == 0 or c is ckpts[-1]]
    print(f"Probing {len(ckpts)} checkpoints from {args.ckpt_dir}\n", flush=True)

    # Stream rows to the CSV as they finish so a crash (e.g. a corrupt checkpoint)
    # never discards completed work — the partial CSV is reusable on the next run.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fields = ["epoch", "valid_mean", "valid_std", "test_mean", "test_std", "ensemble"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in ckpts:
            ep = _epoch_of(c)
            try:
                backbone = load_backbone(str(c), cfg, device)
            except (RuntimeError, EOFError) as e:
                # A truncated/corrupt checkpoint (e.g. an interrupted save) should not
                # abort the whole sweep — skip it and keep the curve going.
                print(f"  epoch {ep:3d}:  SKIPPED (corrupt checkpoint: {e})", flush=True)
                continue
            backbone.eval()
            Xtr, ytr = embed_split(backbone, train_ds, device, args.batch_size, "pooled")
            Xva, yva = embed_split(backbone, valid_ds, device, args.batch_size, "pooled")
            Xte, yte = embed_split(backbone, test_ds, device, args.batch_size, "pooled")
            vm, vs, tm, ts, ens = run_rf_probe(
                Xtr, ytr, Xva, yva, Xte, yte, args.n_models, args.n_estimators)
            row = dict(epoch=ep, valid_mean=vm, valid_std=vs,
                       test_mean=tm, test_std=ts, ensemble=ens)
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"  epoch {ep:3d}:  test {tm:.4f} ± {ts:.4f}   "
                  f"valid {vm:.4f}   ensemble {ens:.4f}", flush=True)
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: (int(v) if k == "epoch" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


_CURVE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def plot_series(series: list[dict], out_png: Path) -> None:
    """Plot one or more probe curves on shared axes + reference lines.

    series: list of {"rows": [...], "label": str, "color": str}. With a single
    entry this is the standalone curve; with two it is the desc-vs-nodesc overlay
    (the two curves sit on top of each other on the plateau = the ablation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # SSL reference band + key baselines (drawn once, behind the curves).
    ax.axhspan(*SSL_BAND, color="0.85", zorder=0,
               label=f"SSL band ({SSL_BAND[0]:.2f}-{SSL_BAND[1]:.2f})")
    ax.axhline(MORGAN_RF, color="tab:red", ls="--", lw=1.2,
               label=f"Morgan ECFP4 + RF ({MORGAN_RF:.3f})")
    ax.axhline(RANDOM_FLOOR, color="0.4", ls=":", lw=1.2,
               label=f"random-init floor ({RANDOM_FLOOR:.3f})")

    # Per-epoch error bars = ±1 s.d. over the RF seed ensemble — shows the late
    # wobble is seed noise on a flat plateau, not real epoch-to-epoch movement.
    for s in series:
        rows = s["rows"]
        ep = [r["epoch"] for r in rows]
        tm = np.array([r["test_mean"] for r in rows])
        ts = np.array([r["test_std"] for r in rows])
        ax.errorbar(ep, tm, yerr=ts, fmt="-o", color=s["color"], ms=3.5, lw=1.5,
                    elinewidth=1.0, capsize=2.5, capthick=1.0, ecolor=s["color"],
                    label=s["label"], zorder=3)

    ax.set_xlabel("pretraining epoch")
    ax.set_ylabel("molhiv test ROC-AUC (frozen probe, RF)")
    ax.set_title("Frozen-probe representation quality vs pretraining")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"\nWrote {out_png}", flush=True)


def plot(rows: list[dict], out_png: Path, label: str) -> None:
    plot_series([{"rows": rows, "label": label, "color": _CURVE_COLORS[0]}], out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/pretrain_chembl")
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--label", default="GPS + descriptors")
    ap.add_argument("--out", default="figures/molhiv_probe_curve",
                    help="output stem; writes <out>.csv and <out>.png")
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--every", type=int, default=1, help="probe every k-th epoch")
    ap.add_argument("--max-epoch", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-probe even if the CSV exists")
    ap.add_argument("--overlay", nargs="+", metavar="CSV:LABEL", default=None,
                    help="plot-only: overlay existing curve CSVs on shared axes, "
                         "e.g. --overlay desc.csv:'GPS + desc' nodesc.csv:'GPS no-desc'. "
                         "Writes <out>.png; no probing.")
    args = ap.parse_args()

    out = Path(args.out)
    if args.overlay:
        # Pure replot from cached CSVs — read each "path:label" spec and overlay.
        series = []
        for i, spec in enumerate(args.overlay):
            path_str, _, lbl = spec.partition(":")
            series.append({"rows": read_csv(Path(path_str)),
                           "label": lbl or Path(path_str).stem,
                           "color": _CURVE_COLORS[i % len(_CURVE_COLORS)]})
        plot_series(series, out.with_suffix(".png"))
        return

    csv_path = out.with_suffix(".csv")
    if csv_path.exists() and not args.force:
        print(f"Reusing {csv_path} (--force to re-probe)")
        rows = read_csv(csv_path)
    else:
        rows = sweep(args, csv_path)
        print(f"Wrote {csv_path}")
    plot(rows, out.with_suffix(".png"), args.label)


if __name__ == "__main__":
    main()
