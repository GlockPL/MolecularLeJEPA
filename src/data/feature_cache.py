"""Persistent per-SMILES featurization cache for the antibiotic dataset.

Featurizing a compound (PyG graph + ~200 RDKit 2D descriptors) is a *pure*
function of ``(smiles, featurizer)`` — it does not depend on the split, seed,
or task. Yet every ``finetune()`` call rebuilt all three splits from scratch,
recomputing descriptors for ~39k compounds (~390 s, ~8x the graph cost). A
10-member ensemble repeated that ~10x → ~70 min wasted per task.

This cache stores ``{smiles: (graph_without_y, raw_descriptor_vector)}`` on disk
(one file per featurizer, since the graph differs between the 'ours' and
'chemprop' atom/bond schemes; descriptors are scheme-independent but kept
alongside for locality). Misses are computed lazily and flushed atomically, so
the first build pays the cost once and every later build — across members,
seeds, splits, and tasks — is a dict lookup.

The cached graph carries no labels: ``get()`` returns a fresh ``clone()`` so the
caller can attach a per-row ``y`` (and ``global_feat``) without mutating the
shared cache entry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Cache format version — bump if the featurizers or descriptor list change in a
# way that would make on-disk entries stale.
_CACHE_VERSION = 1

DEFAULT_CACHE_DIR = Path("data/feature_cache")


class FeatureCache:
    """Lazy, disk-backed cache of (graph, raw-descriptors) keyed by SMILES.

    Args:
        featurizer:  'ours' | 'chemprop' — selects the graph featurizer and the
                     cache file, so the two schemes never collide.
        graph_fn:    SMILES -> Data | None (no labels). Called only on a miss.
        desc_fn:     RDKit Mol -> np.ndarray. Called only on a miss.
        cache_dir:   directory holding ``<featurizer>.pt`` (created if absent).
    """

    def __init__(
        self,
        featurizer: str,
        graph_fn: Callable[[str], Data | None],
        desc_fn: Callable,
        n_descriptors: int,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        self.featurizer = featurizer
        self.graph_fn = graph_fn
        self.desc_fn = desc_fn
        self.n_descriptors = n_descriptors
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / f"{featurizer}_v{_CACHE_VERSION}.pt"
        # smiles -> (Data | None, np.ndarray). A None graph marks an
        # unparseable SMILES so we don't retry RDKit on every build.
        self._cache: dict[str, tuple[Data | None, np.ndarray]] = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._cache = torch.load(self.path, weights_only=False)
            except Exception:
                # Corrupt/old cache — rebuild from scratch rather than crash.
                self._cache = {}

    def get(self, smiles: str) -> tuple[Data | None, np.ndarray]:
        """Return ``(graph_clone | None, raw_descriptors)`` for one SMILES.

        On a miss, featurizes and stores the result. The returned graph is a
        clone — safe to attach ``y`` / ``global_feat`` to.
        """
        entry = self._cache.get(smiles)
        if entry is None:
            data = self.graph_fn(smiles)
            if data is None:
                desc = np.zeros(self.n_descriptors, dtype=np.float32)
            else:
                # Strip any labels so the cached graph is split/task-agnostic.
                data.y = None
                mol = Chem.MolFromSmiles(smiles)
                desc = (
                    self.desc_fn(mol) if mol is not None
                    else np.zeros(self.n_descriptors, dtype=np.float32)
                )
            entry = (data, desc)
            self._cache[smiles] = entry
            self._dirty = True
        data, desc = entry
        return (data.clone() if data is not None else None), desc

    def save(self) -> None:
        """Atomically flush the cache to disk if anything new was computed."""
        if not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(self._cache, tmp)
        os.replace(tmp, self.path)
        self._dirty = False