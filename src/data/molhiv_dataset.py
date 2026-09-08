"""Dataset loader for OGB ogbg-molhiv (MoleculeNet HIV) — Phase 11 benchmark.

Benchmark rules (https://ogb.stanford.edu/docs/graphprop/#ogbg-mol):
  * fixed CANONICAL scaffold split (train/valid/test) shipped with the dataset;
  * single binary task ``HIV_active`` (inhibits HIV replication), ~3.5% positive;
  * metric = ROC-AUC (OGB Evaluator); select on VALID, report TEST.

To finetune our EXISTING LeJEPA backbones unchanged, this produces PyG ``Data``
in the same shape ``AntibioticDataset`` does — ``x / edge_index / edge_attr``
from our featurizer, ``y`` as ``(1, 1)``, and the 217 z-scored RDKit descriptors
as ``global_feat`` ``(1, 217)`` (the GPS proj_head + finetune heads consume them).
Descriptors are normalized with the TRAIN split's mean/std (pass
``train_ds.desc_stats`` to valid/test), identical to the antibiotic pipeline, so
the scratch-vs-pretrained comparison is apples-to-apples.

The (graph, descriptor) featurization is cached to ``data/molhiv_feature_cache/``
— a SEPARATE dir from the antibiotic cache so a concurrent antibiotic run can't
race it.

Data is auto-downloaded from OGB's CSV mirror (no ``ogb`` dependency); the zip
carries both the SMILES and the canonical split indices.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.antibiotic_dataset import N_DESCRIPTORS, _mol_descriptors
from src.data.featurize import smiles_to_data
from src.data.featurize_chemprop import smiles_to_data_cp
from src.data.feature_cache import FeatureCache

OGB_MOLHIV_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"
TASK = "hiv"  # single binary task; y is (N, 1)


def ensure_molhiv(root: str | Path = "data/ogbg_molhiv") -> Path:
    """Download + extract the OGB molhiv CSV bundle if absent. Returns the hiv/ dir."""
    root = Path(root)
    hiv_dir = root / "hiv"
    if (hiv_dir / "mapping" / "mol.csv.gz").exists():
        return hiv_dir
    root.mkdir(parents=True, exist_ok=True)
    print(f"  downloading ogbg-molhiv from {OGB_MOLHIV_URL} ...", flush=True)
    with urllib.request.urlopen(OGB_MOLHIV_URL) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(root)
    if not (hiv_dir / "mapping" / "mol.csv.gz").exists():
        raise FileNotFoundError(f"Unexpected zip layout under {root}")
    return hiv_dir


def load_molhiv_split(hiv_dir: Path, split: str) -> tuple[list[str], np.ndarray]:
    """Return (smiles, labels) for the canonical scaffold split.

    split: 'train' | 'valid' | 'test' ('val' accepted as alias for 'valid')."""
    split = "valid" if split == "val" else split
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be train/valid/test, got {split!r}")
    mol = pd.read_csv(hiv_dir / "mapping" / "mol.csv.gz")
    smi_col = next((c for c in mol.columns if c.lower() == "smiles"), None)
    if smi_col is None:
        raise KeyError(f"No 'smiles' column in mol.csv.gz (have {list(mol.columns)})")
    lab_col = next((c for c in mol.columns if "hiv" in c.lower() or c.lower() == "activity"), None)
    if lab_col is None:
        raise KeyError(f"No HIV label column in mol.csv.gz (have {list(mol.columns)})")
    smiles_all = mol[smi_col].astype(str).tolist()
    labels_all = mol[lab_col].to_numpy()
    idx = pd.read_csv(hiv_dir / "split" / "scaffold" / f"{split}.csv.gz",
                      header=None)[0].to_numpy()
    smiles = [smiles_all[i] for i in idx]
    labels = labels_all[idx].astype(np.float32)
    return smiles, labels


class MolHIVDataset(Dataset):
    """In-memory ogbg-molhiv split, PyG-Data compatible with our finetune harness.

    Args:
        split:      'train' | 'valid' | 'test'.
        root:       dataset root (auto-downloaded into ``<root>/hiv``).
        featurizer: 'ours' (164-dim atoms) or 'chemprop' (133-dim) atom/bond feats.
        desc_stats: (mean, std) from the TRAIN split for descriptor normalization;
                    pass ``train_ds.desc_stats`` to valid/test. None → computed here.
        use_cache:  persist (graph, descriptors) per SMILES (isolated molhiv cache).
    """

    def __init__(
        self,
        split: str = "train",
        root: str | Path = "data/ogbg_molhiv",
        featurizer: str = "ours",
        desc_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = True,
        cache_dir: str | Path = "data/molhiv_feature_cache",
    ):
        super().__init__()
        self.split = split
        self._featurize_fn = smiles_to_data_cp if featurizer == "chemprop" else smiles_to_data
        self._cache = (
            FeatureCache(featurizer=featurizer, graph_fn=self._featurize_fn,
                         desc_fn=_mol_descriptors, n_descriptors=N_DESCRIPTORS,
                         cache_dir=cache_dir)
            if use_cache else None
        )

        hiv_dir = ensure_molhiv(root)
        smiles, labels = load_molhiv_split(hiv_dir, split)

        data_list, raw_descs, kept_smiles, skipped = [], [], [], 0
        for smi, lab in zip(smiles, labels):
            if self._cache is not None:
                data, desc = self._cache.get(smi)
            else:
                data = self._featurize_fn(smi)
                import rdkit.Chem as _Chem
                mol = _Chem.MolFromSmiles(smi)
                desc = _mol_descriptors(mol) if mol is not None else np.zeros(N_DESCRIPTORS)
            if data is None:
                skipped += 1
                continue
            # y is (1, 1) so PyG batches graph labels to (B, 1).
            data.y = torch.tensor([[float(lab)]], dtype=torch.float)
            raw_descs.append(desc)
            kept_smiles.append(smi)
            data_list.append(data)
        if self._cache is not None:
            self._cache.save()

        n_pos = int(sum(d.y[0, 0].item() for d in data_list))
        print(f"  [molhiv-{split}] {len(data_list)} compounds, {n_pos} HIV-active "
              f"({100*n_pos/max(len(data_list),1):.1f}%)"
              + (f"  [skipped {skipped} unparseable]" if skipped else ""))

        self._data_list = data_list
        # SMILES of the KEPT molecules, index-aligned with ``self._data_list`` (so a
        # fingerprint block computed from these rows lines up with the embeddings —
        # ``load_molhiv_split`` order is NOT safe when anything was skipped).
        self.smiles = kept_smiles

        # Normalize descriptors (train stats shared to valid/test).
        raw = torch.tensor(np.nan_to_num(np.asarray(raw_descs, dtype=np.float32), nan=0.0),
                           dtype=torch.float32)
        if desc_stats is None:
            self.desc_mean = raw.mean(0)
            self.desc_std = raw.std(0).clamp(min=1e-6)
        else:
            self.desc_mean, self.desc_std = desc_stats
        norm = torch.nan_to_num((raw - self.desc_mean) / self.desc_std,
                                nan=0.0, posinf=0.0, neginf=0.0)
        for i, data in enumerate(self._data_list):
            data.global_feat = norm[i].unsqueeze(0)

    @property
    def desc_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.desc_mean, self.desc_std)

    @property
    def n_descriptors(self) -> int:
        return N_DESCRIPTORS

    def pos_weight(self) -> float:
        """N_neg / N_pos for the single HIV task (for optional weighted BCE)."""
        ys = torch.cat([d.y for d in self._data_list], dim=0)
        n_pos = ys.sum().item()
        n_neg = (ys == 0).sum().item()
        return n_neg / max(n_pos, 1.0)

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int):
        return self._data_list[idx]