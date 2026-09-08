"""
Pre-featurize a (subset of the) ZINC20 .smi corpus into compact shard files.

RDKit parsing dominates the pretraining DataLoader (~0.7 ms/molecule, repeated
every epoch). This script runs that pass once: each input .smi file becomes one
or more compact shard .pt files that ZINC20CachedDataset reloads without ever
touching RDKit.

Each shard stores all its molecules concatenated into a few arrays plus an
offset index (ragged/CSR layout) — a shard is ~6 large tensors, not millions
of tiny ones. The compact form is ~500 B/molecule.

Use --limit to cap the corpus size (the full 95M molecules do not fit a single
rented instance's disk/RAM). A random subset of whole .smi files is selected to
hit the limit, so the subset still spans all molecular-weight tranches.

Use --global-shuffle to fix a tranche-ordering problem in standard mode: ZINC20
files are organized by molecular weight, so without shuffling batches grow
progressively slower (light → heavy molecules). With --global-shuffle all SMILES
are loaded into RAM first, shuffled globally, then featurized into
chunk_NNN.pt shards with uniform MW distribution. Use this when you observe
per-batch time increasing over training (a sure sign of tranche ordering).
Memory required: ~3 GB for 22M SMILES strings.

Use --chunk-size to control how many molecules go into each shard (and each
parallel featurization job). Larger chunks mean fewer shards but more IPC data.

Run from the repo root — standard mode:
    uv run python scripts/featurize_corpus.py \\
        --input data/chembl --output data/chembl_cache_desc \\
        --limit 22000000 --max-atoms 60 --workers 46

Global-shuffle mode (recommended for consistent batch times):
    uv run python scripts/featurize_corpus.py \\
        --input data/chembl --output data/chembl_cache_desc \\
        --limit 22000000 --max-atoms 60 --workers 46 --global-shuffle

Re-running skips shards that already exist.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors as _RDKitDescriptors

from src.data.featurize import mol_to_compact

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

_DESC_LIST = list(_RDKitDescriptors.descList)
_N_DESC = len(_DESC_LIST)


def _mol_descriptors(mol) -> np.ndarray:
    """NaN-safe 217-element descriptor vector. Failures and NaN returns → 0."""
    out = np.zeros(_N_DESC, dtype=np.float32)
    for i, (_, fn) in enumerate(_DESC_LIST):
        try:
            v = fn(mol)
            if v is not None and math.isfinite(float(v)):
                out[i] = float(v)
        except Exception:
            pass
    return out


def _iter_smiles(path: Path):
    """Yield SMILES strings from a ZINC20 .smi file (first token per line)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            if token.lower() == "smiles":  # per-file header
                continue
            yield token


def _count_lines(path: Path) -> int:
    """Fast newline count — used to estimate a file's molecule count."""
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 20)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def process_chunk(
    job: tuple[Path, int, int, Path, int | None, bool],
) -> tuple[str, int]:
    """Featurize one chunk of one .smi file. Returns (shard_name, n_written)."""
    in_path, start, count, out_path, max_atoms, compute_desc = job
    if out_path.exists():
        return out_path.name, -1  # already done

    x_idx_l, x_mass_l, ei_l, eidx_l, desc_l = [], [], [], [], []
    atom_off, edge_off = [0], [0]

    # islice consumes start molecules then yields the next count. Skipping
    # is text-level so cheap (~ms) compared to the per-molecule RDKit parse.
    for smi in islice(_iter_smiles(in_path), start, start + count):
        mol = Chem.MolFromSmiles(smi)
        compact = mol_to_compact(mol)
        if compact is None:
            continue
        x_idx, x_mass, edge_index, e_idx = compact
        if max_atoms is not None and x_idx.size(0) > max_atoms:
            continue
        x_idx_l.append(x_idx)
        x_mass_l.append(x_mass)
        ei_l.append(edge_index)
        eidx_l.append(e_idx)
        atom_off.append(atom_off[-1] + x_idx.size(0))
        edge_off.append(edge_off[-1] + edge_index.size(1))
        if compute_desc:
            desc_l.append(_mol_descriptors(mol))

    n = len(atom_off) - 1
    if n == 0:
        return out_path.name, 0

    shard = {
        "atom_offsets": torch.tensor(atom_off, dtype=torch.int64),
        "edge_offsets": torch.tensor(edge_off, dtype=torch.int64),
        "x_idx": torch.cat(x_idx_l, dim=0),
        "x_mass": torch.cat(x_mass_l, dim=0),
        "edge_index": torch.cat(ei_l, dim=1),
        "e_idx": torch.cat(eidx_l, dim=0),
    }
    if compute_desc and desc_l:
        # float16: 434 B/mol for 217 descriptors (vs 4 B for float32).
        # Normalization happens at load time from desc_stats.pt so precision
        # loss here is negligible — descriptors are just input features.
        shard["desc"] = torch.tensor(np.stack(desc_l, axis=0), dtype=torch.float16)

    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(shard, tmp)
    tmp.rename(out_path)
    return out_path.name, n


def process_smiles_list(
    job: tuple[list[str], Path, int | None, bool],
) -> tuple[str, int]:
    """Featurize a pre-shuffled list of SMILES strings. Returns (shard_name, n_written)."""
    smiles_list, out_path, max_atoms, compute_desc = job
    if out_path.exists():
        return out_path.name, -1  # already done

    x_idx_l, x_mass_l, ei_l, eidx_l, desc_l = [], [], [], [], []
    atom_off, edge_off = [0], [0]

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        compact = mol_to_compact(mol)
        if compact is None:
            continue
        x_idx, x_mass, edge_index, e_idx = compact
        if max_atoms is not None and x_idx.size(0) > max_atoms:
            continue
        x_idx_l.append(x_idx)
        x_mass_l.append(x_mass)
        ei_l.append(edge_index)
        eidx_l.append(e_idx)
        atom_off.append(atom_off[-1] + x_idx.size(0))
        edge_off.append(edge_off[-1] + edge_index.size(1))
        if compute_desc:
            desc_l.append(_mol_descriptors(mol))

    n = len(atom_off) - 1
    if n == 0:
        return out_path.name, 0

    shard = {
        "atom_offsets": torch.tensor(atom_off, dtype=torch.int64),
        "edge_offsets": torch.tensor(edge_off, dtype=torch.int64),
        "x_idx": torch.cat(x_idx_l, dim=0),
        "x_mass": torch.cat(x_mass_l, dim=0),
        "edge_index": torch.cat(ei_l, dim=1),
        "e_idx": torch.cat(eidx_l, dim=0),
    }
    if compute_desc and desc_l:
        shard["desc"] = torch.tensor(np.stack(desc_l, axis=0), dtype=torch.float16)

    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(shard, tmp)
    tmp.rename(out_path)
    return out_path.name, n


def _load_all_smiles(
    files_with_counts: list[tuple[Path, int]], limit: int | None
) -> list[str]:
    """Read all SMILES from selected files into a single list.

    Stops early once `limit` molecules are collected. Reading is text-only —
    no RDKit involved — so this is fast (~30s for 22M SMILES).
    """
    all_smiles: list[str] = []
    for f, _ in files_with_counts:
        for smi in _iter_smiles(f):
            all_smiles.append(smi)
            if limit is not None and len(all_smiles) >= limit:
                return all_smiles
    return all_smiles


def select_files(
    smi_files: list[Path], limit: int, seed: int
) -> list[tuple[Path, int]]:
    """Pick a random subset of whole files whose molecule count covers `limit`.

    Returns (file, n_molecules) pairs so the chunker doesn't have to re-count.
    """
    files = list(smi_files)
    random.Random(seed).shuffle(files)
    # Aim slightly above the limit — a few % of molecules are dropped by the
    # max_atoms filter, and counting newlines includes the header line.
    target = int(limit / 0.97) + 1
    selected: list[tuple[Path, int]] = []
    estimated = 0
    for f in files:
        if estimated >= target:
            break
        n = max(_count_lines(f) - 1, 0)
        selected.append((f, n))
        estimated += n
    print(f"Selected {len(selected)}/{len(files)} files (~{estimated:,} molecules)")
    return selected


def build_jobs(
    files_with_counts: list[tuple[Path, int]],
    out_root: Path,
    max_atoms: int | None,
    chunk_size: int,
    compute_desc: bool = False,
) -> list[tuple[Path, int, int, Path, int | None, bool]]:
    """Split big files into sub-jobs so they parallelize across Pool workers."""
    jobs = []
    for f, n_mols in files_with_counts:
        # Honor the original single-shard naming if a prior run wrote it.
        legacy = out_root / f"{f.parent.name}_{f.stem}.pt"
        if legacy.exists():
            continue
        if n_mols == 0:
            continue
        n_chunks = max(1, (n_mols + chunk_size - 1) // chunk_size)
        for c in range(n_chunks):
            start = c * chunk_size
            count = min(chunk_size, n_mols - start)
            # When a file fits in one chunk, keep the legacy name so this run
            # and the previous run produce identical shard names.
            if n_chunks == 1:
                out = legacy
            else:
                out = out_root / f"{f.parent.name}_{f.stem}_c{c}.pt"
            jobs.append((f, start, count, out, max_atoms, compute_desc))
    return jobs


def _compute_desc_stats(out_root: Path) -> None:
    """Compute global mean/std over all descriptor shards and save desc_stats.pt.

    Iterates shard-by-shard in float32 to avoid accumulation errors. Shards
    without "desc" are silently skipped (backward-compatible with old caches).
    """
    shards = sorted(out_root.glob("*.pt"))
    # exclude desc_stats.pt itself if it already exists
    shards = [p for p in shards if p.name != "desc_stats.pt"]

    n_total = 0
    desc_sum: torch.Tensor | None = None
    desc_sq_sum: torch.Tensor | None = None

    print(f"Computing descriptor stats from {len(shards)} shards ...")
    t0 = time.time()
    for i, p in enumerate(shards):
        shard = torch.load(p, map_location="cpu", weights_only=True)
        if "desc" not in shard:
            continue
        desc = shard["desc"].float()  # [N, 217]
        n = desc.shape[0]
        n_total += n
        if desc_sum is None:
            desc_sum = desc.sum(0)
            desc_sq_sum = (desc ** 2).sum(0)
        else:
            desc_sum += desc.sum(0)
            desc_sq_sum += (desc ** 2).sum(0)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(shards)}] {n_total:,} molecules  {time.time()-t0:.0f}s")

    if desc_sum is None or n_total == 0:
        print("No descriptor data found in shards — skipping desc_stats.pt")
        return

    mean = desc_sum / n_total
    var = (desc_sq_sum / n_total) - mean ** 2
    std = var.clamp(min=0).sqrt().clamp(min=1e-6)

    stats_path = out_root / "desc_stats.pt"
    torch.save({"mean": mean, "std": std}, stats_path)
    print(
        f"Saved descriptor stats: {n_total:,} molecules, "
        f"{mean.shape[0]} features → {stats_path}  ({time.time()-t0:.0f}s)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="ZINC20 .smi root directory")
    parser.add_argument("--output", required=True, help="output cache directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap total molecules (selects a random file subset)")
    parser.add_argument("--max-atoms", type=int, default=60,
                        help="skip molecules with more heavy atoms than this")
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="molecules per sub-job; big files split into chunks")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random file subset")
    parser.add_argument("--descriptors", action="store_true",
                        help="compute and store 217 normalized RDKit descriptors per molecule")
    parser.add_argument("--global-shuffle", action="store_true",
                        help=(
                            "load all SMILES into RAM, shuffle globally, then featurize. "
                            "Fixes tranche-ordering: without this, batch time grows as training "
                            "moves from light to heavy MW files. Needs ~3 GB RAM for 22M SMILES."
                        ))
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    smi_files = sorted(in_root.rglob("*.smi"))
    if not smi_files:
        sys.exit(f"No .smi files found under {in_root}")

    desc_note = " + 217 RDKit descriptors" if args.descriptors else ""

    if args.global_shuffle:
        # --- Global-shuffle path ---
        # Select files first (for --limit), then load ALL their SMILES into RAM,
        # shuffle once, chunk into shard-sized jobs. Each shard gets molecules from
        # all MW tranches uniformly — batch time stays constant throughout training.
        if args.limit is not None:
            files_with_counts = select_files(smi_files, args.limit, args.seed)
        else:
            files_with_counts = [(f, 0) for f in smi_files]  # counts unused in this path

        print(f"Loading SMILES from {len(files_with_counts)} files into RAM ...")
        t_load = time.time()
        all_smiles = _load_all_smiles(files_with_counts, args.limit)
        print(f"  {len(all_smiles):,} SMILES loaded in {time.time()-t_load:.0f}s")

        rng = random.Random(args.seed)
        rng.shuffle(all_smiles)
        print("  Globally shuffled.")

        chunk_size = args.chunk_size
        chunks = [all_smiles[i:i + chunk_size] for i in range(0, len(all_smiles), chunk_size)]
        all_jobs = [
            (chunk, out_root / f"chunk_{c:06d}.pt", args.max_atoms, args.descriptors)
            for c, chunk in enumerate(chunks)
        ]
        # Free the master list before forking workers — each chunk is a separate object
        # so this only frees the top-level list structure, not the strings themselves.
        del all_smiles, chunks

        print(
            f"Featurizing {len(all_jobs)} globally-shuffled shards "
            f"-> {out_root}  ({args.workers} workers, chunk_size={chunk_size:,}){desc_note}"
        )
        t0 = time.time()
        total_mols = done = skipped = 0
        with Pool(args.workers) as pool:
            for name, n in pool.imap_unordered(process_smiles_list, all_jobs):
                done += 1
                if n < 0:
                    skipped += 1
                else:
                    total_mols += n
                if done % 20 == 0 or done == len(all_jobs):
                    el = time.time() - t0
                    print(f"  [{done}/{len(all_jobs)}] {total_mols:,} molecules  {el:.0f}s")

    else:
        # --- Standard per-file path (original behavior) ---
        if args.limit is not None:
            files_with_counts = select_files(smi_files, args.limit, args.seed)
        else:
            print(f"Counting lines in {len(smi_files)} files...")
            files_with_counts = [(f, max(_count_lines(f) - 1, 0)) for f in smi_files]

        all_jobs = build_jobs(
            files_with_counts, out_root, args.max_atoms, args.chunk_size,
            compute_desc=args.descriptors,
        )
        print(
            f"Featurizing {len(all_jobs)} chunks from {len(files_with_counts)} files "
            f"-> {out_root}  ({args.workers} workers, chunk_size={args.chunk_size:,}){desc_note}"
        )
        t0 = time.time()
        total_mols = done = skipped = 0
        with Pool(args.workers) as pool:
            for name, n in pool.imap_unordered(process_chunk, all_jobs):
                done += 1
                if n < 0:
                    skipped += 1
                else:
                    total_mols += n
                if done % 20 == 0 or done == len(all_jobs):
                    el = time.time() - t0
                    print(f"  [{done}/{len(all_jobs)}] {total_mols:,} molecules  {el:.0f}s")

    print(
        f"Done: {total_mols:,} molecules, {len(all_jobs) - skipped} "
        f"chunks written ({skipped} already present), {time.time() - t0:.0f}s"
    )
    print(f"Set training steps_per_epoch to ~{total_mols // 1024} (= molecules / batch_size).")

    if args.descriptors:
        _compute_desc_stats(out_root)


if __name__ == "__main__":
    main()