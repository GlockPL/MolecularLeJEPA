"""
Descriptor-only baseline (Phase 8.1 / the decisive diagnostic).

Trains models on the 217 normalized RDKit descriptors ALONE — NO graph network,
NO pretraining — on the IDENTICAL scaffold split used everywhere else (reuses
AntibioticDataset, so the descriptors, normalization, and train/val/test indices
are byte-identical to the GPS/D-MPNN finetune pipeline). Reports test AUPRC
(average precision) so it is directly comparable to chemprop's prc-auc.

WHY: the Chemprop-scale D-MPNN's best checkpoint was the linear probe over a
FROZEN RANDOM backbone + descriptors (test 0.3205, beating Chemprop's 0.312
ensemble). That implicates the descriptors + readout capacity — not the learned
graph representation — as the signal source. This script tests it head-on:

  GBM on descriptors ≈ 0.30+  → the graph network AND LeJEPA pretraining are
                                ~irrelevant for antibiotic; the signal is the
                                descriptors. Big reframe (but true).
  GBM on descriptors ≪ 0.30   → the (random/learned) graph features WERE
                                contributing; the representation has real value.

Anchors (antibiotic, scaffold test AUPRC): GPS-pretrained 0.260 |
D-MPNN small 0.234 | D-MPNN Chemprop-scale 0.3205 | Chemprop ens 0.312 / per-model 0.297.

Run:
  uv run python scripts/descriptor_baseline.py --task antibiotic
  uv run python scripts/descriptor_baseline.py --task all --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from rdkit import RDLogger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.data.antibiotic_dataset import AntibioticDataset, LABEL_COLS
from src.logutil import RunLogger


_RDKIT2D_GEN = None


def _rdkit2dnorm_matrix(smiles_list: list[str]) -> np.ndarray:
    """Chemprop's `rdkit_2d_normalized` features (200 descriptors, CDF/quantile-
    normalized to [0,1] via descriptastorus). Bad SMILES → zero row."""
    global _RDKIT2D_GEN
    if _RDKIT2D_GEN is None:
        from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
        _RDKIT2D_GEN = RDKit2DNormalized()
    gen = _RDKIT2D_GEN
    n = len(gen.columns)
    out = np.zeros((len(smiles_list), n), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        r = gen.process(smi)
        if r is not None and r[0]:
            out[i] = np.asarray(r[1:], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _xy(ds: AntibioticDataset, task: str, features: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack the descriptor matrix X and the task's binary labels y.

    features='rdkit217'  → our 217 z-scored RDKit descriptors (from global_feat).
    features='rdkit2dnorm' → Chemprop's 200 rdkit_2d_normalized (CDF-bounded) feats.
    """
    col = LABEL_COLS.index(task)
    y = torch.cat([d.y for d in ds._data_list], dim=0)[:, col].numpy()    # (N,)
    if features == "rdkit217":
        X = torch.cat([d.global_feat for d in ds._data_list], dim=0).numpy()  # (N, 217)
    elif features == "rdkit2dnorm":
        X = _rdkit2dnorm_matrix([d.smiles for d in ds._data_list])            # (N, 200)
    else:
        raise ValueError(f"unknown features {features!r}")
    return X, y.astype(int)


def _report(name: str, va_probs: list, te_probs: list,
            yva: np.ndarray, yte: np.ndarray) -> tuple[float, np.ndarray]:
    """Print per-model (mean ± std) and, for >1 seed, the ensemble AUPRC.

    Returns (ensemble_test_AUPRC, ensemble_test_probs) so callers can aggregate
    across splits and/or save per-compound probabilities.
    """
    singles = np.array([average_precision_score(yte, p) for p in te_probs])
    ens_te_probs = np.mean(te_probs, axis=0)
    if len(te_probs) > 1:
        ens_va = average_precision_score(yva, np.mean(va_probs, axis=0))
        ens_te = average_precision_score(yte, ens_te_probs)
        print(f"  {name:22s} per-model test {singles.mean():.4f} ± {singles.std():.4f} (n={len(te_probs)})")
        print(f"  {name:22s} ENSEMBLE       val {ens_va:.4f}   test {ens_te:.4f}")
        return ens_te, ens_te_probs
    va = average_precision_score(yva, va_probs[0])
    print(f"  {name:22s} val {va:.4f}   test {singles[0]:.4f}")
    return float(singles[0]), ens_te_probs


def _run_task(task: str, xlsx: str, split_method: str, test_size: float,
              val_size: float, split_seed: int, model_seeds: list[int],
              models: str, features: str) -> dict:
    """Run all models for one (task, split_seed). Returns a dict with per-model
    ensemble test AUPRC, the ensemble test probabilities, the test SMILES, and
    the test labels (for cross-split aggregation and --save-probs)."""
    print(f"\n{'='*64}\nTASK: {task}  (split={split_method}, seed={split_seed}, "
          f"features={features})\n{'='*64}")
    common = dict(test_size=test_size, val_size=val_size, seed=split_seed,
                  split_method=split_method, tasks=[task])
    train_ds = AntibioticDataset(xlsx, split="train", **common)
    # val/test normalized with TRAIN descriptor stats — identical to the NN pipeline.
    val_ds = AntibioticDataset(xlsx, split="val", desc_stats=train_ds.desc_stats, **common)
    test_ds = AntibioticDataset(xlsx, split="test", desc_stats=train_ds.desc_stats, **common)

    Xtr, ytr = _xy(train_ds, task, features)
    Xva, yva = _xy(val_ds, task, features)
    Xte, yte = _xy(test_ds, task, features)
    print(f"  X: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  "
          f"({N(ytr)} / {N(yva)} / {N(yte)} positives)")

    auprc: dict[str, float] = {}     # model -> ensemble test AUPRC
    probs: dict[str, np.ndarray] = {}  # model -> ensemble test probs

    # --- Logistic regression (linear reference over descriptors) ---
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    logreg.fit(Xtr, ytr)
    lr_val = average_precision_score(yva, logreg.predict_proba(Xva)[:, 1])
    lr_te_p = logreg.predict_proba(Xte)[:, 1]
    lr_te = average_precision_score(yte, lr_te_p)
    print(f"  LogisticRegression       val AUPRC {lr_val:.4f}   test AUPRC {lr_te:.4f}")
    auprc["LogReg"] = lr_te; probs["LogReg"] = lr_te_p

    # --- HistGradientBoosting (strong GBM over descriptors), per-seed + ensemble ---
    if models in ("hgb", "both"):
        va_p, te_p = [], []
        for s in model_seeds:
            gbm = HistGradientBoostingClassifier(
                class_weight="balanced", random_state=s,
                early_stopping=True, validation_fraction=0.15, max_iter=500,
            )
            gbm.fit(Xtr, ytr)
            va_p.append(gbm.predict_proba(Xva)[:, 1])
            te_p.append(gbm.predict_proba(Xte)[:, 1])
        auprc["HistGBM"], probs["HistGBM"] = _report("HistGBM", va_p, te_p, yva, yte)

    # --- XGBoost tuned for the actual metric: early-stop on OUR val with aucpr +
    #     scale_pos_weight for the 0.9%-prevalence imbalance. NOTE val is now the
    #     early-stopping selection target, so its AUPRC is optimistic — TEST is the
    #     honest number (test is never seen during fit/selection). ---
    if models in ("xgb", "both"):
        from xgboost import XGBClassifier
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        va_p, te_p = [], []
        for s in model_seeds:
            clf = XGBClassifier(
                n_estimators=2000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
                reg_lambda=1.0, scale_pos_weight=spw, tree_method="hist",
                eval_metric="aucpr", early_stopping_rounds=50,
                random_state=s, n_jobs=4,
            )
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            va_p.append(clf.predict_proba(Xva)[:, 1])
            te_p.append(clf.predict_proba(Xte)[:, 1])
        auprc["XGB"], probs["XGB"] = _report("XGB(aucpr,val-ES)", va_p, te_p, yva, yte)

    # --- MLP on descriptors (no graph): a NEURAL NET is sensitive to feature
    #     conditioning (unlike trees), so comparing this across --features
    #     rdkit217 vs rdkit2dnorm isolates the z-score-vs-CDF normalization effect. ---
    if models == "mlp":
        from sklearn.neural_network import MLPClassifier
        va_p, te_p = [], []
        for s in model_seeds:
            mlp = MLPClassifier(
                hidden_layer_sizes=(512, 512), activation="relu",
                alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                early_stopping=False, max_iter=300, random_state=s,
            )
            mlp.fit(Xtr, ytr)
            va_p.append(mlp.predict_proba(Xva)[:, 1])
            te_p.append(mlp.predict_proba(Xte)[:, 1])
        auprc["MLP"], probs["MLP"] = _report("MLP(512,512)", va_p, te_p, yva, yte)

    if task == "antibiotic":
        print("  ---")
        print("  anchors (scaffold test): GPS-pretrained 0.260 | D-MPNN-scale 0.3205 | "
              "Chemprop ens 0.312 / per-model 0.297")

    return {
        "auprc": auprc, "probs": probs,
        "smiles": [str(d.smiles) for d in test_ds._data_list],
        "yte": yte,
    }


def N(y: np.ndarray) -> int:
    return int(y.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="data/wang et al/41586_2023_6887_MOESM3_ESM.xlsx")
    ap.add_argument("--task", default="antibiotic",
                    choices=["all"] + LABEL_COLS,
                    help="single task or 'all' to loop every task")
    ap.add_argument("--split-method", default="scaffold", choices=["scaffold", "random"])
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--split-seeds", default=None,
                    help="Comma list of SPLIT seeds (e.g. 42,1,2,3,4). Runs them all in "
                         "ONE invocation and writes ONE log with a mean±std summary across "
                         "splits — instead of one log per split. Overrides --split-seed.")
    ap.add_argument("--save-probs", default=None,
                    help="Directory to save per-(task,split) ensemble test probabilities "
                         "(smiles,label,<model>_prob) for paired-bootstrap comparisons.")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated GBM seeds; >1 also reports an ensemble "
                         "(e.g. 0,1,2,3,4 for a fair vs-Chemprop-ensemble number)")
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgb", "mlp", "both"],
                    help="hgb = sklearn HistGradientBoosting (default, no extra dep); "
                         "xgb = XGBoost tuned for aucpr (early-stop on val, scale_pos_weight); "
                         "mlp = sklearn MLP (2×512) — a NEURAL NET, so comparing it across "
                         "--features rdkit217 vs rdkit2dnorm isolates the conditioning effect; "
                         "both = hgb+xgb. LogisticRegression (linear floor) always runs.")
    ap.add_argument("--features", default="rdkit217", choices=["rdkit217", "rdkit2dnorm"],
                    help="rdkit217 = our 217 z-scored RDKit descriptors (default); "
                         "rdkit2dnorm = Chemprop's 200 rdkit_2d_normalized (CDF-bounded) "
                         "features via descriptastorus. NB trees are normalization-invariant, "
                         "so a GBM difference between the two isolates the descriptor SET; "
                         "the conditioning effect only shows up in a neural net.")
    ap.add_argument("--log-dir", default="logs", help="directory for the auto-created log file")
    args = ap.parse_args()

    import csv
    from collections import defaultdict
    from pathlib import Path

    model_seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    tasks = LABEL_COLS if args.task == "all" else [args.task]
    split_seeds = ([int(s) for s in args.split_seeds.split(",") if s.strip() != ""]
                   if args.split_seeds else [args.split_seed])

    # Quiet RDKit's noisy C++ warnings so the log stays readable.
    RDLogger.DisableLog("rdApp.*")

    if args.save_probs:
        Path(args.save_probs).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # agg[task][model] = list of ensemble test AUPRC, one per split seed.
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with RunLogger(
        f"descriptor_baseline_{args.task}_{args.features}_{args.model}",
        log_dir=args.log_dir,
        title="AntibioticJEPA — Descriptor-only baseline",
        tasks=tasks,
        features=args.features,
        model=args.model,
        seeds=model_seeds,
        split_seeds=split_seeds,
        split=f"{args.split_method} (test={args.test_size}, val={args.val_size})",
        save_probs=args.save_probs or "(off)",
    ):
        for ss in split_seeds:
            for t in tasks:
                res = _run_task(t, args.xlsx, args.split_method, args.test_size,
                                args.val_size, ss, model_seeds, args.model, args.features)
                for model, ap_ in res["auprc"].items():
                    agg[t][model].append(ap_)
                if args.save_probs:
                    out = Path(args.save_probs) / f"descriptor_probs_{t}_seed{ss}.csv"
                    cols = ["smiles", "label"] + [f"{m}_prob" for m in res["probs"]]
                    with open(out, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f); w.writerow(cols)
                        for i, smi in enumerate(res["smiles"]):
                            w.writerow([smi, int(res["yte"][i])]
                                       + [f"{res['probs'][m][i]:.6f}" for m in res["probs"]])
                    print(f"  saved probs -> {out}")

        # ---- Consolidated cross-split summary (the one place to read results) ----
        if len(split_seeds) > 1:
            print(f"\n{'='*64}")
            print(f"SUMMARY across {len(split_seeds)} split seeds {split_seeds} "
                  f"({args.split_method}, model={args.model}, {len(model_seeds)} model seeds)")
            print(f"{'='*64}")
            for t in tasks:
                print(f"\nTASK: {t}   ensemble test AUPRC (mean ± std over splits)")
                for model, vals in agg[t].items():
                    a = np.array(vals)
                    per = "  ".join(f"{v:.3f}" for v in vals)
                    print(f"  {model:18s} {a.mean():.4f} ± {a.std():.4f}   [{per}]")

        elapsed = time.time() - t0
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\nElapsed      : {h}h {m}m {s}s")


if __name__ == "__main__":
    main()