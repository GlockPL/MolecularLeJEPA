"""XGBoost ensemble on RDKit descriptors for ogbg-molhiv — baseline reference.

Companion to ``scripts/finetune_molhiv.py``. Same canonical scaffold split and
the same 217 RDKit descriptors (``_mol_descriptors``) the GPS proj_head consumes,
but here they feed a plain XGBoost ensemble instead of a graph network. This is
the "are descriptors alone enough?" reference for the molhiv pivot.

Protocol (matches the OGB benchmark):
  * fixed CANONICAL scaffold split shipped with the dataset;
  * single binary task ``HIV_active`` (~3.5% positive);
  * metric = ROC-AUC; tune early-stopping on VALID, report TEST.

Features (``--features``):
  * ``descriptors`` — 217 RDKit 2D descriptors (global physicochemistry);
  * ``morgan``      — Morgan/ECFP circular fingerprint bits (local substructure
                      inventory; the classic molhiv baseline, usually stronger);
  * ``both``        — the two concatenated.

Ensemble = ``--n-models`` XGBoost classifiers differing only in random seed
(bagging-style); test scores are averaged probabilities. We report both the
ensemble TEST AUC and the per-model mean±std, mirroring how the chemprop and
GPS numbers are reported.

Usage:
    uv run python scripts/xgb_molhiv.py --features descriptors --n-models 5
    uv run python scripts/xgb_molhiv.py --features morgan --n-models 5
    uv run python scripts/xgb_molhiv.py --features both --n-models 5
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data.antibiotic_dataset import N_DESCRIPTORS, _mol_descriptors
from src.data.molhiv_dataset import ensure_molhiv, load_molhiv_split

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def featurize_descriptors(smiles: list[str]) -> np.ndarray:
    """SMILES list -> (N, 217) raw RDKit descriptor matrix (NaN/inf -> 0)."""
    feats = np.zeros((len(smiles), N_DESCRIPTORS), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            feats[i] = _mol_descriptors(mol)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def featurize_morgan(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """SMILES list -> (N, n_bits) Morgan/ECFP bit matrix (radius 2 = ECFP4)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    feats = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            feats[i] = gen.GetFingerprintAsNumPy(mol)
    return feats


def featurize(smiles: list[str], kind: str, radius: int, n_bits: int) -> np.ndarray:
    if kind == "descriptors":
        return featurize_descriptors(smiles)
    if kind == "morgan":
        return featurize_morgan(smiles, radius, n_bits)
    if kind == "both":
        return np.concatenate(
            [featurize_descriptors(smiles), featurize_morgan(smiles, radius, n_bits)],
            axis=1,
        )
    raise ValueError(f"unknown --features {kind!r}")


def load_split(split: str, kind: str, radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    hiv_dir = ensure_molhiv()
    smiles, labels = load_molhiv_split(hiv_dir, split)
    X = featurize(smiles, kind, radius, n_bits)
    y = labels.astype(np.int32)
    print(f"  [molhiv-{split}] {len(y)} compounds, {int(y.sum())} active "
          f"({100*y.mean():.1f}%)", flush=True)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["descriptors", "morgan", "both"],
                    default="descriptors", help="feature set fed to XGBoost")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (2 = ECFP4)")
    ap.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint length")
    add_model_args(ap)
    args = ap.parse_args()

    desc = {"descriptors": "217 RDKit descriptors",
            "morgan": f"Morgan ECFP r={args.radius}, {args.n_bits} bits",
            "both": f"217 descriptors + Morgan r={args.radius}/{args.n_bits} bits"}[args.features]
    print(f"Loading + featurizing molhiv splits ({desc}) ...", flush=True)
    Xtr, ytr = load_split("train", args.features, args.radius, args.n_bits)
    Xva, yva = load_split("valid", args.features, args.radius, args.n_bits)
    Xte, yte = load_split("test", args.features, args.radius, args.n_bits)

    run_ensemble(Xtr, ytr, Xva, yva, Xte, yte, args)


def add_model_args(ap: argparse.ArgumentParser) -> None:
    """Shared tabular-head knobs so both entry-point scripts expose the same flags."""
    ap.add_argument("--model", choices=["xgb", "rf", "hgb"], default="xgb",
                    help="xgb=XGBoost (valid early-stop) | rf=RandomForest (no early-stop, "
                         "matches the OGB MorganFP+RF leaderboard head) | hgb=HistGradientBoosting")
    ap.add_argument("--n-models", type=int, default=10, help="ensemble size (seeds)")
    ap.add_argument("--n-estimators", type=int, default=2000,
                    help="xgb boosting rounds / rf+hgb tree count")
    ap.add_argument("--max-depth", type=int, default=6, help="xgb/hgb depth (rf is unbounded)")
    ap.add_argument("--lr", type=float, default=0.05, help="xgb/hgb learning rate")
    ap.add_argument("--early-stopping", type=int, default=50, help="xgb only; 0 disables")
    ap.add_argument("--max-features", default="sqrt",
                    help="rf only: features considered per split ('sqrt' = sklearn default "
                         "and the leaderboard recipe | 'log2' | float fraction | int). Matters "
                         "for concatenated blocks: with 2048 Morgan bits + a 128-d embedding, "
                         "'sqrt' draws ~49 columns of which only ~3 are embedding dims.")
    ap.add_argument("--scale-pos-weight", default="auto",
                    help="xgb positive-class upweight: 'auto'=N_neg/N_pos (rebalanced) | "
                         "'1'=OFF (matches the RF recipe; best for AUC ranking) | float")
    ap.add_argument("--save-probs", default="",
                    help="write the ensemble TEST probabilities + labels to this .npz, so two "
                         "feature sets can be compared with scripts/paired_bootstrap_auc.py "
                         "(per-seed std ~0.004 makes small AUC gaps unreadable otherwise)")
    ap.add_argument("--cv-rounds", type=int, default=0,
                    help="xgb: if >0, pick n_estimators by k-fold CV on TRAIN (k=this value, "
                         "e.g. 5) instead of early-stopping on the tiny external valid — then "
                         "train fixed rounds with NO external ES. Avoids valid-selection overfit.")


def _parse_max_features(v):
    """'sqrt'/'log2'/None pass through; a numeric string becomes int (count) or float (fraction)."""
    if v is None or v in ("sqrt", "log2", "none", "None"):
        return None if v in ("none", "None") else v
    f = float(v)
    return int(f) if f == int(f) and f > 1 else f


def _build_model(seed: int, spw: float, args):
    """Construct one classifier per ``args.model``. Returns (clf, needs_eval_set)."""
    if args.model == "xgb":
        es = args.early_stopping if args.early_stopping > 0 else None
        spw_use = spw if args.scale_pos_weight == "auto" else float(args.scale_pos_weight)
        return XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.lr, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, scale_pos_weight=spw_use, eval_metric="auc",
            early_stopping_rounds=es, tree_method="hist",
            random_state=seed, n_jobs=-1,
        ), es is not None
    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # EXACT recipe from the OGB MorganFP+RF leaderboard entry (Cyrus Maher,
        # cyrusmaher/ogb-molecule-comp): plain RF, NO class_weight (AUC is
        # threshold-free so RF's probability ranking handles the 3.7% imbalance),
        # min_samples_leaf=2, default unbounded depth. Their process_y downsampling
        # is a no-op here (molhiv train ~33k < its 50k trigger).
        return RandomForestClassifier(
            n_estimators=args.n_estimators, min_samples_leaf=2,
            max_features=_parse_max_features(getattr(args, "max_features", "sqrt")),
            n_jobs=-1, random_state=seed,
        ), False
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators, learning_rate=args.lr,
            max_depth=None if args.max_depth <= 0 else args.max_depth,
            class_weight="balanced", l2_regularization=1.0,
            validation_fraction=0.1, early_stopping=True, random_state=seed,
        ), False
    raise ValueError(f"unknown --model {args.model!r}")


def _cv_best_rounds(Xtr, ytr, spw: float, args, nfold: int = 5, seed: int = 0) -> int:
    """Stratified k-fold CV on TRAIN to find the AUC-optimal xgb boosting round.

    Returns argmax of the mean test-AUC curve (+1 for count). Uses the same params
    as the seed models so the chosen round transfers."""
    import xgboost as xgb
    spw_use = spw if args.scale_pos_weight == "auto" else float(args.scale_pos_weight)
    params = dict(objective="binary:logistic", eval_metric="auc",
                  max_depth=args.max_depth, eta=args.lr, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=1,
                  scale_pos_weight=spw_use, tree_method="hist")
    cvres = xgb.cv(params, xgb.DMatrix(Xtr, label=ytr),
                   num_boost_round=args.n_estimators, nfold=nfold, stratified=True,
                   early_stopping_rounds=50, seed=seed, verbose_eval=False)
    col = next(c for c in cvres.columns if c.startswith("test-") and c.endswith("-mean"))
    return int(cvres[col].values.argmax()) + 1


def run_ensemble(Xtr, ytr, Xva, yva, Xte, yte, args) -> float:
    """Train ``args.n_models`` seed-varied models; print + return ensemble test AUC.

    Shared by the descriptor/Morgan featurizer (this file) and the frozen-backbone
    embedding variant (``xgb_molhiv_embed.py``) so both use an identical head/protocol.
    Head selectable via ``args.model`` (xgb / rf / hgb).
    """
    print(f"  feature dim = {Xtr.shape[1]}  |  head = {args.model}", flush=True)
    # ~3.5% positives -> rebalance the loss.
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"  scale_pos_weight = {spw:.1f}\n", flush=True)

    # Pick xgb round count by TRAIN cross-validation (stable: pools many more
    # positives per fold than the tiny external valid) instead of valid early
    # stopping, then train fixed rounds with no external ES — kills the
    # valid-selection overfit that opens the scaffold valid->test gap.
    if args.model == "xgb" and getattr(args, "cv_rounds", 0) > 0:
        best = _cv_best_rounds(Xtr, ytr, spw, args, nfold=args.cv_rounds)
        print(f"  CV({args.cv_rounds}-fold)-selected rounds = {best}  "
              f"(was n_estimators={args.n_estimators}, external ES disabled)\n", flush=True)
        args.n_estimators = best
        args.early_stopping = 0

    test_probs, val_aucs, test_aucs = [], [], []
    for seed in range(args.n_models):
        clf, needs_eval = _build_model(seed, spw, args)
        if needs_eval:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            best = f"best_iter={clf.best_iteration:4d}  "
        else:
            clf.fit(Xtr, ytr)
            best = ""
        p_val = clf.predict_proba(Xva)[:, 1]
        p_test = clf.predict_proba(Xte)[:, 1]
        va, ta = roc_auc_score(yva, p_val), roc_auc_score(yte, p_test)
        val_aucs.append(va); test_aucs.append(ta); test_probs.append(p_test)
        print(f"  seed {seed}: {best}valid AUC={va:.4f}  test AUC={ta:.4f}", flush=True)

    ens_probs = np.mean(test_probs, axis=0)
    ens_test = roc_auc_score(yte, ens_probs)
    if getattr(args, "save_probs", ""):
        from pathlib import Path
        Path(args.save_probs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_probs, probs=ens_probs, y=yte,
                 per_seed=np.asarray(test_probs), test_aucs=np.asarray(test_aucs))
        print(f"  saved ensemble test probs -> {args.save_probs}", flush=True)
    print("\n" + "=" * 60)
    print(f"Per-model   valid AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Per-model   test  AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"ENSEMBLE    test  AUC: {ens_test:.4f}  (avg of {args.n_models} models)")
    print("=" * 60)
    print("Reference bands: GIN scratch ~0.757 | MolCLR ~0.77 | "
          "Mole-BERT ~0.787 | OGB FP+leaderboard ~0.80+")
    return ens_test


if __name__ == "__main__":
    main()
