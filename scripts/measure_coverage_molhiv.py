"""Phase 11 gate — does the ChEMBL pretraining corpus cover ogbg-molhiv?

Before pivoting to ogbg-molhiv (PLAN.md Phase 11) we MUST repeat the Phase 5
coverage check for the NEW target. The antibiotic lesson was decisive: ZINC
covered only 21.7% of antibiotic TEST scaffolds → LeJEPA pretraining could not
transfer there. ChEMBL covered 58.6% and was greenlit. molhiv is a different
target — do NOT assume ChEMBL covers it; measure first.

This script:
  1. Downloads ogbg-molhiv directly from OGB's CSV mirror (no `ogb` dependency) —
     the zip carries the SMILES (mapping/mol.csv.gz) AND the canonical scaffold
     split (split/scaffold/{train,test}.csv.gz), so we evaluate coverage on the
     EXACT held-out test scaffolds the benchmark uses.
  2. Samples ChEMBL SMILES from data/chembl/*.smi (the pretraining corpus source).
  3. Reports, for molhiv-train and molhiv-TEST vs the ChEMBL sample:
       - scaffold overlap (Bemis-Murcko) — the key transfer number;
       - NN-Tanimoto (max ECFP4 to ChEMBL) — structural distance;
       - internal diversity, for context.

Coverage vs a ChEMBL SAMPLE is a LOWER bound (the full ~2.9M can only add
overlap). Bump --chembl-sample for a tighter estimate if a number is borderline.

CPU/RDKit-heavy. An ensemble training on the GPU is fine but shares CPU — lower
--workers if the machine is busy.

Usage:
    uv run python scripts/measure_coverage_molhiv.py
    uv run python scripts/measure_coverage_molhiv.py --chembl-sample 300000 --workers 16
"""

from __future__ import annotations

import argparse
import atexit
import io
import random
import urllib.request
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.logutil import RunLogger

RDLogger.DisableLog("rdApp.*")

OGB_MOLHIV_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"


# --- molhiv download ---------------------------------------------------------
def download_molhiv(dest: Path) -> Path:
    """Download + extract the OGB molhiv CSV bundle. Returns the 'hiv/' dir."""
    hiv_dir = dest / "hiv"
    if (hiv_dir / "mapping" / "mol.csv.gz").exists():
        print(f"  molhiv already present: {hiv_dir}")
        return hiv_dir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {OGB_MOLHIV_URL} ...", flush=True)
    with urllib.request.urlopen(OGB_MOLHIV_URL) as r:
        blob = r.read()
    print(f"    got {len(blob)/1e6:.1f} MB; extracting → {dest}", flush=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    if not (hiv_dir / "mapping" / "mol.csv.gz").exists():
        raise FileNotFoundError(f"Unexpected zip layout under {dest} (no hiv/mapping/mol.csv.gz)")
    return hiv_dir


def load_molhiv_splits(hiv_dir: Path) -> dict[str, list[str]]:
    """Return {'train': [...smiles], 'test': [...]} using the canonical scaffold split."""
    mol = pd.read_csv(hiv_dir / "mapping" / "mol.csv.gz")
    smi_col = next((c for c in mol.columns if c.lower() == "smiles"), None)
    if smi_col is None:
        raise KeyError(f"No 'smiles' column in mol.csv.gz (have {list(mol.columns)})")
    label_col = next((c for c in mol.columns if "hiv" in c.lower() or c.lower() == "activity"), None)
    smiles_all = mol[smi_col].astype(str).tolist()

    out: dict[str, list[str]] = {}
    for split in ("train", "test"):
        idx = pd.read_csv(hiv_dir / "split" / "scaffold" / f"{split}.csv.gz",
                          header=None)[0].to_numpy()
        out[split] = [smiles_all[i] for i in idx]
        n_pos = ""
        if label_col is not None:
            labs = mol[label_col].to_numpy()[idx]
            n_pos = f"  ({int(np.nansum(labs))} active, {100*np.nansum(labs)/max(len(idx),1):.1f}%)"
        print(f"  molhiv-{split}: {len(out[split]):,} compounds{n_pos}")
    return out


# --- ChEMBL sampling ---------------------------------------------------------
def sample_chembl(chembl_dir: Path, target: int, seed: int) -> list[str]:
    """Random sample of ~target ChEMBL SMILES spread across all .smi shards.

    target<=0 → use ALL SMILES (tighter scaffold estimate; heavier NN-Tanimoto)."""
    files = sorted(chembl_dir.glob("*.smi"))
    if not files:
        raise FileNotFoundError(f"No .smi files under {chembl_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)

    def _smis(f: Path) -> list[str]:
        with open(f) as fh:
            return [ln.split()[0] for ln in fh if ln.strip() and not ln.startswith("smiles")]

    if target <= 0:
        pool = [s for f in files for s in _smis(f)]
        print(f"  ChEMBL: ALL {len(pool):,} SMILES from {len(files)} shards")
        return pool

    per_file_cap = max(1, target // len(files) + 1)
    pool: list[str] = []
    for f in files:
        lines = _smis(f)
        if not lines:
            continue
        pool.extend(rng.sample(lines, min(per_file_cap, len(lines))))
        if len(pool) >= target:
            break
    if len(pool) > target:
        pool = rng.sample(pool, target)
    print(f"  ChEMBL sample: {len(pool):,} SMILES from {len(files)} shards")
    return pool


# --- fingerprints / scaffolds / similarity (self-contained) ------------------
def _fp(smiles: str, radius: int, nbits: int):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _fp_batch(smiles_list, radius, nbits):
    return [fp for s in smiles_list if (fp := _fp(s, radius, nbits)) is not None]


def fingerprints(smiles, radius, nbits, workers, label):
    print(f"  fingerprinting {label}: {len(smiles):,} ...", flush=True)
    if workers <= 1 or len(smiles) < 4000:
        fps = _fp_batch(smiles, radius, nbits)
    else:
        chunks = [smiles[i::workers] for i in range(workers)]
        with Pool(workers) as pool:
            fps = [fp for r in pool.map(partial(_fp_batch, radius=radius, nbits=nbits), chunks) for fp in r]
    print(f"    -> {len(fps):,} valid ({len(smiles)-len(fps)} dropped)", flush=True)
    return fps


def _murcko(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_set(smiles, workers, label):
    print(f"  scaffolds for {label}: {len(smiles):,} ...", flush=True)
    if workers <= 1:
        scaffs = [_murcko(s) for s in smiles]
    else:
        with Pool(workers) as pool:
            scaffs = pool.map(_murcko, smiles, chunksize=512)
    return {s for s in scaffs if s}, scaffs


def overlap(query_scaffs, ref_set):
    valid = [s for s in query_scaffs if s]
    if not valid:
        return 0.0, 0, 0
    hit = sum(1 for s in valid if s in ref_set)
    return hit / len(valid), hit, len(valid)


_REF_FPS: list = []


def _nn_init(ref_fps):
    global _REF_FPS
    _REF_FPS = ref_fps


def _nn_max(query_fp):
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, _REF_FPS)
    return max(sims) if sims else 0.0


def nn_tanimoto(query_fps, ref_fps, workers):
    if workers <= 1:
        _nn_init(ref_fps)
        return np.array([_nn_max(q) for q in query_fps])
    with Pool(workers, initializer=_nn_init, initargs=(ref_fps,)) as pool:
        return np.array(pool.map(_nn_max, query_fps, chunksize=64))


def internal_diversity(fps, sample_n, seed):
    if len(fps) < 2:
        return float("nan")
    rng = random.Random(seed)
    sub = fps if len(fps) <= sample_n else rng.sample(fps, sample_n)
    sims = []
    for i in range(len(sub) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(sub[i], sub[i + 1:]))
    return 1.0 - (sum(sims) / len(sims) if sims else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 11 gate: ChEMBL vs ogbg-molhiv coverage")
    ap.add_argument("--chembl-dir", default="data/chembl")
    ap.add_argument("--molhiv-dir", default="data/ogbg_molhiv",
                    help="Download/extract target for the OGB molhiv CSV bundle")
    ap.add_argument("--chembl-sample", type=int, default=200_000,
                    help="ChEMBL SMILES to sample (lower bound on coverage; <=0 = ALL)")
    ap.add_argument("--internal-sample", type=int, default=2000)
    ap.add_argument("--nn-threshold", type=float, default=0.4)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reviewer paper-trail: tee everything (incl. tracebacks) to a timestamped log.
    log = RunLogger(
        "measure_coverage_molhiv",
        title="Phase 11 coverage: ChEMBL vs ogbg-molhiv (scaffold split)",
        chembl_dir=args.chembl_dir,
        chembl_sample=args.chembl_sample,
        nn_threshold=args.nn_threshold,
        workers=args.workers,
        seed=args.seed,
    ).start()
    atexit.register(log.stop)

    print("=== Phase 11 coverage: ChEMBL sample vs ogbg-molhiv (scaffold split) ===\n")

    print("[molhiv]")
    hiv_dir = download_molhiv(Path(args.molhiv_dir))
    splits = load_molhiv_splits(hiv_dir)
    smi_train, smi_test = splits["train"], splits["test"]

    print("\n[ChEMBL]")
    chembl_smiles = sample_chembl(Path(args.chembl_dir), args.chembl_sample, args.seed)

    print("\n[fingerprints]")
    fp = partial(fingerprints, radius=args.radius, nbits=args.nbits, workers=args.workers)
    chembl_fps = fp(chembl_smiles, label="ChEMBL")
    train_fps = fp(smi_train, label="molhiv-train")
    test_fps = fp(smi_test, label="molhiv-test")

    print("\n[scaffolds]")
    chembl_scaff_set, _ = scaffold_set(chembl_smiles, args.workers, "ChEMBL")
    _, train_scaffs = scaffold_set(smi_train, args.workers, "molhiv-train")
    _, test_scaffs = scaffold_set(smi_test, args.workers, "molhiv-test")
    ov_tr = overlap(train_scaffs, chembl_scaff_set)
    ov_te = overlap(test_scaffs, chembl_scaff_set)
    print(f"  unique ChEMBL scaffolds in sample: {len(chembl_scaff_set):,}")

    print("\n[NN-Tanimoto vs ChEMBL]")
    nn_tr = nn_tanimoto(train_fps, chembl_fps, args.workers)
    print(f"  train done ({len(nn_tr):,})", flush=True)
    nn_te = nn_tanimoto(test_fps, chembl_fps, args.workers)
    print(f"  test  done ({len(nn_te):,})", flush=True)

    print("\n[internal diversity]")
    div_chembl = internal_diversity(chembl_fps, args.internal_sample, args.seed)
    div_tr = internal_diversity(train_fps, args.internal_sample, args.seed)
    div_te = internal_diversity(test_fps, args.internal_sample, args.seed)

    thr = args.nn_threshold
    print("\n" + "=" * 68)
    print(f"COVERAGE REPORT — ogbg-molhiv vs ChEMBL sample={len(chembl_fps):,} (lower bound)")
    print("=" * 68)
    print(f"{'metric':<34}{'molhiv-train':>16}{'molhiv-test':>16}")
    print("-" * 68)
    print(f"{'n compounds':<34}{len(train_fps):>16,}{len(test_fps):>16,}")
    print(f"{'scaffold overlap with ChEMBL':<34}{ov_tr[0]:>15.1%}{ov_te[0]:>16.1%}")
    print(f"{'  (scaffolds hit / total)':<34}"
          f"{f'{ov_tr[1]}/{ov_tr[2]}':>16}{f'{ov_te[1]}/{ov_te[2]}':>16}")
    print(f"{'NN-Tanimoto median':<34}{np.median(nn_tr):>16.3f}{np.median(nn_te):>16.3f}")
    print(f"{'NN-Tanimoto mean':<34}{nn_tr.mean():>16.3f}{nn_te.mean():>16.3f}")
    print(f"{f'frac NN >= {thr}':<34}{(nn_tr>=thr).mean():>15.1%}{(nn_te>=thr).mean():>16.1%}")
    print(f"{'internal diversity (1-meanT)':<34}{div_tr:>16.3f}{div_te:>16.3f}")
    print("-" * 68)
    print(f"ChEMBL-sample internal diversity (1-meanT): {div_chembl:.3f}")
    print("=" * 68)
    print("\nRead vs the antibiotic precedent (ZINC 21.7% = fail, ChEMBL 58.6% = greenlit):")
    print("HIGH molhiv-TEST scaffold overlap + NN-Tanimoto => ChEMBL covers molhiv =>")
    print("LeJEPA has a chance to transfer => pivot justified. LOW => same dead-end.")


if __name__ == "__main__":
    main()