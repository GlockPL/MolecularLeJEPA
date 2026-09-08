"""OGB-leaderboard-compliant run of the Morgan + LeJEPA-embedding random forest.

``concat_balanced_molhiv.py`` sweeps configurations and reports the cell selected on
validation. This script does one thing instead: it produces the exact numbers an OGB
leaderboard submission requires for the ALREADY-SELECTED configuration, fixing three
ways the sweep's output is not submission-ready.

1. THE FULL OFFICIAL SPLIT. ``MolHIVDataset`` drops molecules our graph featurizer
   rejects (2 of 4113 in valid and in test), so the sweep scores 4111 rows. A
   leaderboard number must cover every row of the official split, or it is not
   comparable with entries that do. Predictions here are scattered back into the full
   split via ``dataset.kept_idx`` and the dropped rows are filled with a constant (the
   training-set positive rate by default), which is the neutral choice for a molecule
   the model cannot featurize. Both numbers are printed, so the effect of the fill is
   visible rather than assumed: for a rank-based metric with 2 inactive molecules
   added, it is expected to be ~0.

2. THE UNBIASED STANDARD DEVIATION. OGB asks for ``torch.std``, which is ddof=1. The
   sweep uses ``np.std`` (ddof=0). At 10 seeds that is a factor of sqrt(10/9).

3. THE OFFICIAL EVALUATOR. ``ogb`` is a project dependency, so the reported number
   comes from ``ogb.graphproppred.Evaluator``. The script also computes
   ``sklearn.roc_auc_score`` on the same vectors and raises if the two disagree. For
   ogbg-molhiv they cannot: the Evaluator computes, per task, ``roc_auc_score`` over
   the rows whose label is not NaN and averages over tasks, and this dataset has one
   task and no missing labels. That equivalence is what lets the script still run, with
   a printed warning, where ``ogb`` is not installed - but it is checked on every
   evaluation rather than assumed.

Reported as the submission: per-model mean +/- unbiased std over seeds 0-9. Do NOT
submit the ensemble number - averaging the 10 seeds' predictions collapses the 10
required runs into a single model.

Usage:
    uv run python scripts/ogb_submission_molhiv.py
    uv run python scripts/ogb_submission_molhiv.py --random-init   # the control
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from scripts.spectrum import Spectrum  # noqa: F401  (also re-exported by spectrum_sweep_lbvs)


def ogb_rocauc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, str]:
    """ROC-AUC under the official OGB Evaluator when available, else the identical
    single-task computation. Returns (value, which path was used)."""
    try:
        from ogb.graphproppred import Evaluator
    except ImportError:
        return float(roc_auc_score(y_true, y_pred)), "sklearn (ogb not installed)"
    ev = Evaluator(name="ogbg-molhiv")
    official = float(ev.eval({"y_true": y_true.reshape(-1, 1),
                              "y_pred": y_pred.reshape(-1, 1)})["rocauc"])
    plain = float(roc_auc_score(y_true, y_pred))
    if abs(official - plain) > 1e-12:
        raise AssertionError(
            f"OGB Evaluator ({official!r}) and roc_auc_score ({plain!r}) disagree; "
            "the fallback path in this script would be wrong for this dataset")
    return official, "ogb.graphproppred.Evaluator"


def scatter_to_full(pred: np.ndarray, kept_idx: np.ndarray, n_full: int,
                    fill: float) -> np.ndarray:
    """Place per-kept-molecule predictions back into the full official split."""
    full = np.full(n_full, fill, dtype=np.float64)
    full[kept_idx] = pred
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_molhiv.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/pretrain_chembl/final.pt")
    # The configuration selected on VALIDATION by concat_balanced_molhiv.py. Changing
    # these means re-running that selection; they are not free knobs at this stage.
    ap.add_argument("--morgan-bits", type=int, default=1024)
    ap.add_argument("--embed-dims", type=int, default=32, help="PCA top-k, train-fitted")
    ap.add_argument("--max-features", default="0.1")
    ap.add_argument("--n-trees", type=int, default=1000)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=10, help="OGB requires 10 (0..9)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fill", default="prior",
                    help="score for molecules the featurizer rejects: 'prior' (the "
                         "training positive rate) or a float")
    ap.add_argument("--random-init", action="store_true",
                    help="untrained backbone control, same pipeline")
    ap.add_argument("--init-seed", type=int, default=100)
    ap.add_argument("--out", default="logs/ogb_submission_molhiv.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = cfg.data.get("featurizer", "ours")
    tr = MolHIVDataset(split="train", root=cfg.data.root, featurizer=feat)
    va = MolHIVDataset(split="valid", root=cfg.data.root, featurizer=feat,
                       desc_stats=tr.desc_stats)
    te = MolHIVDataset(split="test", root=cfg.data.root, featurizer=feat,
                       desc_stats=tr.desc_stats)

    # Labels for the KEPT rows (to fit / to score the subset) and for the FULL official
    # split (to score the submission). The full labels come straight from the OGB CSV.
    ytr = np.array([int(d.y[0, 0]) for d in tr._data_list])
    y_kept = {s: np.array([int(d.y[0, 0]) for d in ds._data_list])
              for s, ds in (("valid", va), ("test", te))}
    # Full-split labels come from the OGB CSV, which is the authoritative source and
    # includes the rows our featurizer rejected.
    from src.data.molhiv_dataset import ensure_molhiv, load_molhiv_split
    hiv_dir = ensure_molhiv(cfg.data.root)
    y_full = {}
    for split, ds in (("valid", va), ("test", te)):
        _, labs = load_molhiv_split(hiv_dir, split)
        full = np.asarray(labs, dtype=np.int64).reshape(-1)
        if len(full) != ds.n_original:
            raise ValueError(f"{split}: CSV has {len(full)} rows but the dataset saw "
                             f"{ds.n_original}")
        # THE alignment check for this whole script. If kept_idx were off by even one
        # position, predictions would be scattered onto the wrong molecules and the
        # submitted number would be silently wrong. Labels are the only thing we can
        # check it against, so check it.
        if not np.array_equal(full[ds.kept_idx], y_kept[split]):
            raise AssertionError(
                f"{split}: labels at kept_idx do not match the dataset's own labels - "
                "kept_idx is misaligned with _data_list, predictions would land on the "
                "wrong rows")
        y_full[split] = full
        missing = ds.n_original - len(ds.kept_idx)
        drop_pos = int(full.sum() - y_kept[split].sum())
        print(f"  [{split}] official {ds.n_original} rows, featurized "
              f"{len(ds.kept_idx)}, refilled {missing} ({drop_pos} of them active)"
              "  [kept_idx alignment verified]")

    fill = float(ytr.mean()) if args.fill == "prior" else float(args.fill)
    print(f"  fill score for unfeaturizable molecules: {fill:.6f}"
          f" ({'training positive rate' if args.fill == 'prior' else 'user-supplied'})")

    if args.random_init:
        torch.manual_seed(args.init_seed)
        bb = GPSTransformer(
            hidden_dim=cfg.model.hidden_dim, embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
            walk_length=cfg.model.get("walk_length", 20),
            dropout=cfg.model.get("dropout", 0.1),
            n_descriptors=cfg.model.get("n_descriptors", 0),
            readout=cfg.model.get("readout", "mean")).to(device)
        print(f"CONTROL: UNTRAINED backbone (seed {args.init_seed})")
    else:
        bb = load_backbone(args.checkpoint, cfg, device)
    bb.eval()
    # OGB asks for the parameter count of the model. The forest is not a torch module,
    # so we report the frozen encoder's parameters and describe the forest separately.
    n_params = sum(p.numel() for p in bb.parameters())

    E = {}
    for name, ds in (("train", tr), ("valid", va), ("test", te)):
        E[name], _ = embed_split(bb, ds, device, args.batch_size, "pooled")
    sp = Spectrum(E["train"])                       # PCA basis fitted on TRAIN only
    k = args.embed_dims
    P = {n: sp.apply(E[n], "center")[:, :k] for n in E}
    M = {n: featurize_morgan(ds.smiles, 2, args.morgan_bits)
         for n, ds in (("train", tr), ("valid", va), ("test", te))}
    X = {n: np.concatenate([M[n], P[n]], axis=1) for n in M}
    print(f"  design matrix: morgan({args.morgan_bits}) + embed PCA-{k}"
          f" = {X['train'].shape[1]} columns")

    rows, evaluator_path = [], ""
    for seed in range(args.seeds):
        clf = RandomForestClassifier(
            n_estimators=args.n_trees, min_samples_leaf=args.min_samples_leaf,
            max_features=_parse_max_features(args.max_features),
            n_jobs=-1, random_state=seed).fit(X["train"], ytr)
        r = {"seed": seed}
        for split, ds in (("valid", va), ("test", te)):
            p_kept = clf.predict_proba(X[split])[:, 1]
            p_full = scatter_to_full(p_kept, ds.kept_idx, ds.n_original, fill)
            auc_full, evaluator_path = ogb_rocauc(y_full[split], p_full)
            r[split] = auc_full
            r[f"{split}_subset"] = float(roc_auc_score(y_kept[split], p_kept))
        rows.append(r)
        print(f"  seed {seed}: valid {r['valid']:.4f}  test {r['test']:.4f}"
              f"   (subset-only test {r['test_subset']:.4f})", flush=True)

    def stat(key):
        v = np.array([r[key] for r in rows])
        # ddof=1 == torch.std, which is what the OGB submission form asks for.
        return float(v.mean()), float(v.std(ddof=1)), float(v.std(ddof=0))

    print(f"\n{'='*74}\n  OGB SUBMISSION VALUES  (evaluator: {evaluator_path})\n{'='*74}")
    out = {"method": ("Morgan FP + LeJEPA embedding (RF)"
                      + (" [random-init control]" if args.random_init else "")),
           "external_data": not args.random_init,
           "external_data_detail": ("self-supervised LeJEPA pretraining on ~2.9M ChEMBL "
                                    "molecules; no external labels"),
           "n_seeds": args.seeds, "evaluator": evaluator_path,
           "backbone_params": n_params,
           "hyperparameters": {
               "morgan_bits": args.morgan_bits, "morgan_radius": 2,
               "pca_dims": k, "pca_fitted_on": "train split only",
               "rf_n_estimators": args.n_trees,
               "rf_min_samples_leaf": args.min_samples_leaf,
               "rf_max_features": args.max_features},
           "unfeaturizable_fill": fill,
           "per_seed": rows}
    for split in ("test", "valid"):
        m, sd1, sd0 = stat(split)
        ms, _, _ = stat(f"{split}_subset")
        out[split] = {"mean": m, "std_ddof1": sd1, "std_ddof0": sd0,
                      "mean_subset_only": ms}
        print(f"  {split:5s} {m:.4f} +/- {sd1:.4f}   (ddof=0 would be {sd0:.4f};"
              f" subset-only mean {ms:.4f}, delta {m - ms:+.5f})")
    print(f"\n  Report on the form:  test {out['test']['mean']:.4f} +/- "
          f"{out['test']['std_ddof1']:.4f}   validation {out['valid']['mean']:.4f}")
    print(f"  Frozen encoder parameters: {n_params:,}"
          f"  (+ a {args.n_trees}-tree random forest, not a torch module)")
    print("  Tick the EXTERNAL DATA box: pretrained on ChEMBL." if not args.random_init
          else "  Control run - not for submission.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
