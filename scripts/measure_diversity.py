"""Phase 5 — does the ZINC pretraining set cover the antibiotic scaffold split?

Cheap diagnostic that decides whether building a more diverse pretraining set is
worth it. Computes, against a random sample of the ZINC pretraining SMILES
(sourced from data/zinc20_remote — the source the cache was built from):

  1. Scaffold overlap   — fraction of antibiotic-train / antibiotic-test
     Bemis-Murcko scaffolds that ALSO appear in the ZINC sample. The TEST number
     is the key one: if pretraining never visited those scaffolds, the embedding
     can't transfer there regardless of model/head size.
  2. NN-Tanimoto        — for each antibiotic compound (train / test), the max
     ECFP4 (Morgan r=2, 2048-bit) Tanimoto to the ZINC sample. Reports median and
     the fraction with NN >= --nn-threshold. Low test NN-Tanimoto = pretraining
     is structurally far from what we evaluate on.
  3. Internal diversity — 1 - mean pairwise Tanimoto on a random sub-sample of
     each set, for context on how diverse each set is on its own.

ZINC coverage measured this way is a LOWER bound: the full 30M set can only add
overlap, never remove it. The antibiotic split here is identical to
`finetune ... --task <task>` (same load_xlsx + _scaffold_split).

NOTE: CPU/RDKit-heavy. Per PLAN.md, avoid running while pretraining is hammering
the disk/CPU; a Chemprop ensemble on the GPU is fine but will share CPU, so lower
--workers if the machine is busy.

Usage:
    uv run python scripts/measure_diversity.py                 # antibiotic, 100k ZINC sample
    uv run python scripts/measure_diversity.py --zinc-sample 200000 --workers 16
    uv run python scripts/measure_diversity.py --task hepg2
"""

from __future__ import annotations

import argparse
import random
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from src.data.antibiotic_dataset import (
    LABEL_COLS,
    _murcko_scaffold,
    _scaffold_split,
    load_xlsx,
)

RDLogger.DisableLog("rdApp.*")

# featurize_corpus.py lives alongside this script; importing its select_files
# lets us replay the EXACT deterministic file subset the cache was built from
# (sorted rglob -> seeded shuffle -> accumulate to --limit), so we sample the
# literal pretrained molecules rather than the larger downloaded superset.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize_corpus import select_files  # noqa: E402


# --- fingerprints ------------------------------------------------------------
def _fp(smiles: str, radius: int, nbits: int):
    """ECFP4 ExplicitBitVect, or None if SMILES is unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _fp_batch(smiles_list, radius, nbits):
    out = []
    for s in smiles_list:
        fp = _fp(s, radius, nbits)
        if fp is not None:
            out.append(fp)
    return out


def _fingerprints(smiles, radius, nbits, workers, label):
    """Parallel ECFP4 over a list of SMILES; drops unparseable."""
    print(f"  fingerprinting {label}: {len(smiles):,} SMILES ...", flush=True)
    if workers <= 1 or len(smiles) < 4000:
        fps = _fp_batch(smiles, radius, nbits)
    else:
        chunks = [smiles[i::workers] for i in range(workers)]
        with Pool(workers) as pool:
            results = pool.map(partial(_fp_batch, radius=radius, nbits=nbits), chunks)
        fps = [fp for r in results for fp in r]
    print(f"    -> {len(fps):,} valid fingerprints "
          f"({len(smiles) - len(fps)} dropped)", flush=True)
    return fps


# --- ZINC sampling -----------------------------------------------------------
def _sample_zinc_smiles(
    zinc_dir: Path, target: int, seed: int,
    pretrain_limit: int | None = None, pretrain_seed: int = 0,
) -> list[str]:
    """Random sample of ~target SMILES spread across many .smi shards/tranches.

    Caps lines drawn per file so the sample spans many files (and thus many
    tranches), rather than coming from one contiguous shard.

    If ``pretrain_limit`` is given, the candidate files are first restricted to
    the EXACT subset the cache was featurized from, by replaying
    ``featurize_corpus.select_files(sorted(rglob), pretrain_limit, pretrain_seed)``.
    Then the sample is drawn only from those files — the literal pretrained 30M,
    not the wider download. Requires zinc_dir to be unchanged since featurization.
    """
    all_files = sorted(zinc_dir.rglob("*.smi"))
    if not all_files:
        raise FileNotFoundError(f"No .smi files under {zinc_dir}")

    if pretrain_limit is not None:
        selected = select_files(all_files, pretrain_limit, pretrain_seed)
        files = [f for f, _ in selected]
        print(f"  reproduced pretrained subset: {len(files)}/{len(all_files)} "
              f"files (limit={pretrain_limit:,}, seed={pretrain_seed})", flush=True)
    else:
        files = list(all_files)
        print(f"  representative sample over all {len(files)} files "
              f"(superset of the pretrained subset)", flush=True)

    rng = random.Random(seed)
    rng.shuffle(files)
    per_file_cap = max(1, target // 30)

    pool: list[str] = []
    used = 0
    for f in files:
        with open(f) as fh:
            lines = [ln.split()[0] for ln in fh
                     if ln.strip() and not ln.startswith("smiles")]
        if not lines:
            continue
        k = min(per_file_cap, len(lines))
        pool.extend(rng.sample(lines, k))
        used += 1
        if len(pool) >= target:
            break
    if len(pool) > target:
        pool = rng.sample(pool, target)
    print(f"  ZINC sample: {len(pool):,} SMILES from {used} shard(s) "
          f"(of {len(files)} available)", flush=True)
    return pool


# --- scaffold overlap --------------------------------------------------------
def _scaffold_set(smiles, workers, label):
    print(f"  scaffolds for {label}: {len(smiles):,} SMILES ...", flush=True)
    if workers <= 1:
        scaffs = [_murcko_scaffold(s) for s in smiles]
    else:
        with Pool(workers) as pool:
            scaffs = pool.map(_murcko_scaffold, smiles, chunksize=512)
    return {s for s in scaffs if s}, scaffs


def _overlap(query_scaffs, zinc_scaff_set):
    """Fraction of query compounds whose Murcko scaffold is in the ZINC set."""
    valid = [s for s in query_scaffs if s]
    if not valid:
        return 0.0, 0, 0
    hit = sum(1 for s in valid if s in zinc_scaff_set)
    return hit / len(valid), hit, len(valid)


# --- nearest-neighbour Tanimoto ----------------------------------------------
_REF_FPS: list = []  # populated per-worker via initializer (copy-on-write on fork)


def _nn_init(ref_fps):
    global _REF_FPS
    _REF_FPS = ref_fps


def _nn_max(query_fp):
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, _REF_FPS)
    return max(sims) if sims else 0.0


def _nn_tanimoto(query_fps, ref_fps, workers):
    """Max Tanimoto of each query fp to the reference (ZINC) set."""
    if workers <= 1:
        _nn_init(ref_fps)
        return np.array([_nn_max(q) for q in query_fps])
    with Pool(workers, initializer=_nn_init, initargs=(ref_fps,)) as pool:
        sims = pool.map(_nn_max, query_fps, chunksize=64)
    return np.array(sims)


# --- internal diversity ------------------------------------------------------
def _internal_diversity(fps, sample_n, seed):
    """1 - mean pairwise Tanimoto over a random sub-sample."""
    if len(fps) < 2:
        return float("nan")
    rng = random.Random(seed)
    sub = fps if len(fps) <= sample_n else rng.sample(fps, sample_n)
    sims = []
    for i in range(len(sub) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(sub[i], sub[i + 1:]))
    return 1.0 - (sum(sims) / len(sims) if sims else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 ZINC-vs-antibiotic coverage")
    ap.add_argument("--xlsx", default="data/wang et al/41586_2023_6887_MOESM3_ESM.xlsx")
    ap.add_argument("--zinc-dir", default="data/chembl",
                    help="ZINC .smi source. Default is the frozen 396-file copy "
                         "that matches the cache's featurization input; "
                         "data/zinc20 (now 545 files) is a representative superset.")
    ap.add_argument("--task", default="antibiotic", choices=list(LABEL_COLS))
    ap.add_argument("--zinc-sample", type=int, default=100_000,
                    help="Number of ZINC SMILES to sample (lower bound on coverage)")
    ap.add_argument("--reproduce-subset", action="store_true",
                    help="Replay featurize_corpus.select_files to restrict the "
                         "sample to the EXACT pretrained file subset. Only valid "
                         "if --zinc-dir is unchanged since featurization "
                         "(use data/zinc20_remote, not the grown data/zinc20).")
    ap.add_argument("--pretrain-limit", type=int, default=30_000_000,
                    help="--limit used at featurization (for --reproduce-subset)")
    ap.add_argument("--pretrain-seed", type=int, default=0,
                    help="--seed used at featurization (for --reproduce-subset)")
    ap.add_argument("--internal-sample", type=int, default=2000,
                    help="Sub-sample size for internal-diversity (O(n^2))")
    ap.add_argument("--nn-threshold", type=float, default=0.4,
                    help="Report fraction of compounds with NN-Tanimoto >= this")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Phase 5 coverage: ZINC sample vs antibiotic '{args.task}' split ===\n")

    # 1. Antibiotic set + identical scaffold split.
    df = load_xlsx(Path(args.xlsx), tasks=[args.task])
    idx_tr, idx_va, idx_te = _scaffold_split(df, args.test_size, args.val_size)
    smi_train = df.iloc[idx_tr]["smiles"].astype(str).tolist()
    smi_test = df.iloc[idx_te]["smiles"].astype(str).tolist()
    print(f"  antibiotic-train: {len(smi_train):,}   antibiotic-test: {len(smi_test):,}\n")

    # 2. ZINC sample.
    zinc_smiles = _sample_zinc_smiles(
        Path(args.zinc_dir), args.zinc_sample, args.seed,
        pretrain_limit=args.pretrain_limit if args.reproduce_subset else None,
        pretrain_seed=args.pretrain_seed,
    )

    # 3. Fingerprints.
    print("\n[fingerprints]")
    fp = partial(_fingerprints, radius=args.radius, nbits=args.nbits, workers=args.workers)
    zinc_fps = fp(zinc_smiles, label="ZINC-sample")
    train_fps = fp(smi_train, label="antibiotic-train")
    test_fps = fp(smi_test, label="antibiotic-test")

    # 4. Scaffold overlap.
    print("\n[scaffolds]")
    zinc_scaff_set, _ = _scaffold_set(zinc_smiles, args.workers, "ZINC-sample")
    _, train_scaffs = _scaffold_set(smi_train, args.workers, "antibiotic-train")
    _, test_scaffs = _scaffold_set(smi_test, args.workers, "antibiotic-test")
    ov_tr = _overlap(train_scaffs, zinc_scaff_set)
    ov_te = _overlap(test_scaffs, zinc_scaff_set)
    print(f"  unique ZINC scaffolds in sample: {len(zinc_scaff_set):,}")

    # 5. NN-Tanimoto vs ZINC.
    print("\n[NN-Tanimoto vs ZINC]")
    nn_tr = _nn_tanimoto(train_fps, zinc_fps, args.workers)
    print(f"  train done ({len(nn_tr):,} queries)", flush=True)
    nn_te = _nn_tanimoto(test_fps, zinc_fps, args.workers)
    print(f"  test  done ({len(nn_te):,} queries)", flush=True)

    # 6. Internal diversity.
    print("\n[internal diversity]")
    div_zinc = _internal_diversity(zinc_fps, args.internal_sample, args.seed)
    div_tr = _internal_diversity(train_fps, args.internal_sample, args.seed)
    div_te = _internal_diversity(test_fps, args.internal_sample, args.seed)

    thr = args.nn_threshold
    print("\n" + "=" * 68)
    print(f"COVERAGE REPORT — task={args.task}, ZINC sample={len(zinc_fps):,} "
          f"(lower bound)")
    print("=" * 68)
    print(f"{'metric':<34}{'antibiotic-train':>16}{'antibiotic-test':>16}")
    print("-" * 68)
    print(f"{'n compounds':<34}{len(train_fps):>16,}{len(test_fps):>16,}")
    print(f"{'scaffold overlap with ZINC':<34}"
          f"{ov_tr[0]:>15.1%}{ov_te[0]:>16.1%}")
    print(f"{'  (scaffolds hit / total)':<34}"
          f"{f'{ov_tr[1]}/{ov_tr[2]}':>16}{f'{ov_te[1]}/{ov_te[2]}':>16}")
    print(f"{'NN-Tanimoto median':<34}{np.median(nn_tr):>16.3f}{np.median(nn_te):>16.3f}")
    print(f"{'NN-Tanimoto mean':<34}{nn_tr.mean():>16.3f}{nn_te.mean():>16.3f}")
    print(f"{f'frac NN >= {thr}':<34}"
          f"{(nn_tr >= thr).mean():>15.1%}{(nn_te >= thr).mean():>16.1%}")
    print(f"{'internal diversity (1-meanT)':<34}{div_tr:>16.3f}{div_te:>16.3f}")
    print("-" * 68)
    print(f"ZINC-sample internal diversity (1-meanT): {div_zinc:.3f}")
    print("=" * 68)
    print("\nRead: LOW test scaffold-overlap + LOW test NN-Tanimoto => pretraining")
    print("is structurally far from the test set => a more diverse ZINC set is")
    print("justified. HIGH coverage => the gap is readout/representation, not data.")


if __name__ == "__main__":
    main()