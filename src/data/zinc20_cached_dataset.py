"""
Cached ZINC20 dataset for LeJEPA pretraining.

Drop-in replacement for ZINC20Dataset that reads the precomputed compact shards
written by scripts/featurize_corpus.py instead of parsing SMILES with RDKit
every epoch. RDKit parsing was ~0.7 ms/molecule of the DataLoader budget; the
cached path replaces it with a cheap vectorized one-hot reconstruction.

Each shard (.pt) holds all its molecules concatenated into a few arrays plus an
offset index — see scripts/featurize_corpus.py for the layout.

With in_memory=True the whole cache is loaded into RAM once, in the main
process, before the DataLoader forks its workers. Because a shard is a handful
of large tensors (not millions of small objects), the forked workers share that
memory copy-on-write — one physical copy total, not one per worker. Slicing the
tensors only reads them, so the shared pages are never duplicated.

Augmentation stays random per epoch, so it is never cached.

Usage:
    dataset = ZINC20CachedDataset(cache_dir="data/zinc20_cache", Vg=2, Vl=8)
    loader  = DataLoader(dataset, batch_size=1024, num_workers=24,
                         collate_fn=zinc20_collate)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
import torch.utils.data
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from src.data.augment import augment
from src.data.featurize import compact_to_data
from src.data.zinc20_dataset import ddp_partition, reservoir_shuffle


class ZINC20CachedDataset(IterableDataset):
    """
    Streaming dataset over precomputed compact ZINC20 shards.

    Args:
        cache_dir: directory of .pt shards from scripts/featurize_corpus.py.
        Vg:        number of global views per molecule.
        Vl:        number of local views per molecule.
        max_atoms: skip molecules with more than this many heavy atoms.
        max_mols:  stop after this many molecules (None = unlimited).
        shuffle_buffer: molecules held per worker for a reservoir-style stream
                   shuffle. 0 disables it.
        seed:      base seed for the per-epoch shard shuffle.
        in_memory: load every shard into RAM up front (shared across workers
                   copy-on-write). Use only when the cache fits in RAM.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        Vg: int = 2,
        Vl: int = 8,
        max_atoms: int = 100,
        max_mols: int | None = None,
        shuffle_buffer: int = 100_000,
        seed: int = 0,
        in_memory: bool = True,
        cover_frac: float = 0.6,
        desc_global_views: int | None = None,
        edge_corruption: str = "dropout",
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.Vg = Vg
        self.Vl = Vl
        self.cover_frac = cover_frac
        self.edge_corruption = edge_corruption
        # How many of the Vg global views carry descriptors (rest get zeros, like
        # the local views). None = all Vg (legacy). Phase-7 knob: setting this to 1
        # leaves the other global view descriptor-free so the global-view prediction
        # is non-trivial (both global views share an identical descriptor vector, so
        # with all-global descriptors pred_global ≈ 0 — the views are "free").
        self.desc_global_views = Vg if desc_global_views is None else desc_global_views
        self.max_atoms = max_atoms
        self.max_mols = max_mols
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.in_memory = in_memory
        # See ZINC20Dataset._epoch — incremented to 0 on the first __iter__.
        self._epoch = -1

        self._shards: list[Path] = sorted(
            p for p in self.cache_dir.glob("*.pt") if p.name != "desc_stats.pt"
        )
        if not self._shards:
            raise FileNotFoundError(
                f"No .pt shards found under {self.cache_dir}. "
                "Build the cache first:\n"
                "  uv run python scripts/featurize_corpus.py "
                f"--input data/zinc20_remote --output {self.cache_dir} "
                "--limit 22000000 --max-atoms 60 --workers 46"
            )

        # Load descriptor normalization stats if the cache was built with --descriptors.
        self._desc_mean: torch.Tensor | None = None
        self._desc_std: torch.Tensor | None = None
        desc_stats_path = self.cache_dir / "desc_stats.pt"
        if desc_stats_path.exists():
            ds = torch.load(desc_stats_path, map_location="cpu", weights_only=True)
            self._desc_mean = ds["mean"]
            self._desc_std = ds["std"]
            print(f"ZINC20 cache: loaded descriptor stats ({self._desc_mean.shape[0]} features)")

        # Preload the whole cache in the main process so forked DataLoader
        # workers can share it copy-on-write (see module docstring).
        self._mem: dict[Path, dict] | None = None
        if in_memory:
            self._mem = {}
            n_mols = 0
            for p in self._shards:
                shard = torch.load(p, map_location="cpu", weights_only=True)
                self._mem[p] = shard
                n_mols += shard["atom_offsets"].numel() - 1
            print(
                f"ZINC20 cache: loaded {len(self._shards)} shards / "
                f"{n_mols:,} molecules into RAM"
            )

    def _load_shard(self, path: Path) -> dict:
        if self._mem is not None:
            return self._mem[path]
        return torch.load(path, map_location="cpu", weights_only=True)

    def _iter_molecules(self, shards: list[Path]) -> Iterator[tuple]:
        """Yield per-molecule compact tuples (x_idx, x_mass, edge_index, e_idx, desc|None)."""
        for path in shards:
            shard = self._load_shard(path)
            atom_off = shard["atom_offsets"]
            edge_off = shard["edge_offsets"]
            x_idx, x_mass = shard["x_idx"], shard["x_mass"]
            edge_index, e_idx = shard["edge_index"], shard["e_idx"]
            desc_data = shard.get("desc", None)  # (N_mol, n_desc) float16 or None

            for i in range(atom_off.numel() - 1):
                a0, a1 = int(atom_off[i]), int(atom_off[i + 1])
                e0, e1 = int(edge_off[i]), int(edge_off[i + 1])
                # clone() so a buffered molecule does not keep the whole shard
                # alive (and, on the streaming path, can be freed).
                desc = desc_data[i].clone() if desc_data is not None else None
                yield (
                    x_idx[a0:a1].clone(),
                    x_mass[a0:a1].clone(),
                    edge_index[:, e0:e1].clone(),
                    e_idx[e0:e1].clone(),
                    desc,
                )

    def __iter__(self) -> Iterator[list[Data]]:
        self._epoch += 1
        # Per-epoch deterministic shard shuffle — see ZINC20Dataset.__iter__.
        rng = random.Random(self.seed + self._epoch)

        shards = list(self._shards)
        rng.shuffle(shards)

        # Partition across (DDP rank x DataLoader worker). Every replica reads the
        # same shuffled shard list and walks all molecules, but keeps only every
        # `total_parts`-th one — disjoint, evenly balanced (to within 1 molecule)
        # slices with full coverage and no cross-GPU duplication. We stride at the
        # molecule level (not the shard level) because with only ~15 shards over
        # 4 GPUs a shard-level split is badly unbalanced, which desyncs DDP.
        global_id, total_parts = ddp_partition()
        stream = self._iter_molecules(shards)
        if total_parts > 1:
            stream = (
                mol for i, mol in enumerate(stream) if i % total_parts == global_id
            )
        if self.shuffle_buffer > 0:
            stream = reservoir_shuffle(stream, self.shuffle_buffer, rng)

        count = 0
        for x_idx, x_mass, edge_index, e_idx, desc in stream:
            if self.max_mols is not None and count >= self.max_mols:
                return
            if x_idx.size(0) > self.max_atoms:
                continue

            data = compact_to_data(x_idx, x_mass, edge_index, e_idx)
            views = augment(data, Vg=self.Vg, Vl=self.Vl, cover_frac=self.cover_frac,
                            edge_corruption=self.edge_corruption)

            # Attach normalized descriptor vector to the first `desc_global_views`
            # GLOBAL views only; all other views (remaining global + every local)
            # get zeros so their prediction target is descriptor-free — forcing the
            # GNN to learn structural information rather than trivially matching on
            # the shared descriptor signal. desc_global_views < Vg additionally makes
            # the global-view prediction non-trivial (Phase-7 knob).
            # unsqueeze(0) → shape [1, n_desc] so PyG Batch.from_data_list stacks
            # to [B, n_desc] instead of concatenating to [B*n_desc].
            if desc is not None and self._desc_mean is not None:
                desc_norm = torch.nan_to_num(
                    (desc.float() - self._desc_mean) / self._desc_std,
                    nan=0.0, posinf=0.0, neginf=0.0,
                ).unsqueeze(0)
                zero_desc = torch.zeros_like(desc_norm)
                for j, view in enumerate(views):
                    view.global_feat = desc_norm if j < self.desc_global_views else zero_desc

            count += 1
            yield views

    def file_count(self) -> int:
        return len(self._shards)