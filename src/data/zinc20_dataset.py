"""
ZINC20 streaming dataset for LeJEPA pretraining.

ZINC20 file layout (from ZINC-downloader-2D-smi.curl):
  data/zinc20/AA/AAAA.smi
  data/zinc20/AA/AAAB.smi
  ...  (1916 files total across ~26 two-letter subdirectories)

Each .smi file has one molecule per line:
  SMILES<whitespace>ZINC_ID

Usage:
    dataset = ZINC20Dataset(zinc20_dir="data/zinc20", Vg=2, Vl=8)
    loader  = DataLoader(dataset, batch_size=512, num_workers=8,
                         collate_fn=zinc20_collate)

For DDP each worker processes a disjoint subset of files.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch.distributed as dist
import torch.utils.data
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from src.data.featurize import smiles_to_data
from src.data.augment import augment


def ddp_partition() -> tuple[int, int]:
    """
    Return (global_partition_id, total_partitions) for the current process.

    An IterableDataset is replicated to every (DDP rank x DataLoader worker)
    process, and each replica iterates the whole dataset unless we tell it which
    disjoint slice to take. The original code only split by DataLoader worker, so
    under DDP every rank read the *same* data — N_GPUS-fold duplication. We fold
    the DDP rank in here so the data is partitioned across ranks AND workers:

        global_id    = rank * num_workers + worker_id
        total_parts  = world_size * num_workers

    Falls back to (0, 1) for single-process / non-DDP runs.
    """
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_id, num_workers = worker_info.id, worker_info.num_workers
    else:
        worker_id, num_workers = 0, 1

    return rank * num_workers + worker_id, world_size * num_workers


def reservoir_shuffle(stream: Iterator, cap: int, rng: random.Random) -> Iterator:
    """Reservoir-style shuffle buffer — yields a random buffered item per pull."""
    buf: list = []
    for item in stream:
        if len(buf) < cap:
            buf.append(item)
            continue
        idx = rng.randrange(cap)
        yield buf[idx]
        buf[idx] = item
    rng.shuffle(buf)
    yield from buf


class ZINC20Dataset(IterableDataset):
    """
    Streaming dataset over the ZINC20 2D-SMILES file tree.

    Args:
        zinc20_dir: root directory containing the downloaded .smi files
                    (subdirectories AA/, BA/, CA/, etc.).
        Vg:        number of global views per molecule.
        Vl:        number of local views per molecule.
        max_atoms: skip molecules with more than this many heavy atoms.
        max_mols:  stop after this many valid molecules (None = unlimited).
                   Useful for debug runs.
        shuffle_buffer: number of SMILES held per worker for a reservoir-style
                   stream shuffle. 0 disables it.
        seed:      base seed for the per-epoch file shuffle.
    """

    def __init__(
        self,
        zinc20_dir: str | Path,
        Vg: int = 2,
        Vl: int = 8,
        max_atoms: int = 100,
        max_mols: int | None = None,
        shuffle_buffer: int = 100_000,
        seed: int = 0,
        cover_frac: float = 0.6,
        edge_corruption: str = "dropout",
    ):
        super().__init__()
        self.zinc20_dir = Path(zinc20_dir)
        self.Vg = Vg
        self.Vl = Vl
        self.cover_frac = cover_frac
        self.edge_corruption = edge_corruption
        self.max_atoms = max_atoms
        self.max_mols = max_mols
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        # Incremented to 0 on the first __iter__ call. Persistent workers keep
        # their own copy and re-call __iter__ once per epoch, so this stays in
        # lockstep across workers without needing set_epoch() from the parent.
        self._epoch = -1

        # File order here is irrelevant — __iter__ reshuffles every epoch.
        self._files: list[Path] = sorted(self.zinc20_dir.rglob("*.smi"))
        if not self._files:
            raise FileNotFoundError(
                f"No .smi files found under {self.zinc20_dir}. "
                "Run the ZINC20 curl downloader first from inside data/zinc20/."
            )

    def _iter_smiles(self, path: Path) -> Iterator[str]:
        """Yield SMILES strings from a ZINC20 .smi file (first token per line)."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split()[0]
                # Skip the header line present in each ZINC20 file
                if token.lower() == "smiles":
                    continue
                yield token

    def _iter_all_smiles(self, files: list[Path]) -> Iterator[str]:
        for path in files:
            yield from self._iter_smiles(path)

    def __iter__(self) -> Iterator[list[Data]]:
        self._epoch += 1
        # Per-epoch deterministic file shuffle. ZINC20 tranches are ordered by
        # molecular weight, so a sorted pass is a light→heavy curriculum: it
        # makes step time climb monotonically (heavy graphs cost O(N^2) in
        # RWPE/attention and CPU augmentation) and breaks per-batch i.i.d.,
        # which SIGReg assumes. Shuffling spreads heavy molecules evenly.
        rng = random.Random(self.seed + self._epoch)

        files = list(self._files)
        rng.shuffle(files)

        # Every replica shuffles identically (rank/worker-independent seed) then
        # takes a disjoint stride across (DDP rank x worker) — disjoint shards,
        # full coverage, no cross-GPU duplication.
        global_id, total_parts = ddp_partition()
        if total_parts > 1:
            files = files[global_id::total_parts]

        smiles = self._iter_all_smiles(files)
        if self.shuffle_buffer > 0:
            smiles = reservoir_shuffle(smiles, self.shuffle_buffer, rng)

        count = 0
        for smi in smiles:
            if self.max_mols is not None and count >= self.max_mols:
                return

            data = smiles_to_data(smi)
            if data is None:
                continue
            if data.x.size(0) > self.max_atoms:
                continue

            views = augment(data, Vg=self.Vg, Vl=self.Vl, cover_frac=self.cover_frac,
                            edge_corruption=self.edge_corruption)
            count += 1
            yield views

    def file_count(self) -> int:
        return len(self._files)


def zinc20_collate(batch: list[list[Data]]):
    """
    Collate a list of view-lists into a list of PyG Batch objects.

    Each element of `batch` is a list of (Vg + Vl) augmented Data objects
    for one molecule.  Returns a list of (Vg + Vl) Batch objects — one per
    view position — ready for lejepa_loss().

    The first Vg batches are global views; the remaining Vl are local views.
    """
    from torch_geometric.data import Batch

    V = len(batch[0])
    views_per_position: list[list[Data]] = [[] for _ in range(V)]
    for mol_views in batch:
        for v_idx, view in enumerate(mol_views):
            views_per_position[v_idx].append(view)

    return [Batch.from_data_list(views) for views in views_per_position]