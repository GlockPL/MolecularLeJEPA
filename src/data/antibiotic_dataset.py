"""
Dataset loader for the Wong et al. 2023 antibiotic screening dataset.

Source: Supplementary Data 1 (41586_2023_6887_MOESM3_ESM.xlsx)
        doi:10.1038/s41586-023-06887-8

Reads four sheets from the XLSX file and merges them on Compound_ID:
  'S. aureus growth inhibition' — binary antibiotic activity (Mean_50uM < 0.2)
  'HepG2 viability'            — binary HepG2 cytotoxicity (Mean_10uM < 0.9)
  'HSkMC viability'            — binary HSkMC cytotoxicity (Mean_10uM < 0.9)
  'IMR-90 viability'           — binary IMR-90 cytotoxicity (Mean_10uM < 0.9)

Cutoffs match the paper's Methods section:
  Active antibiotic  = mean relative growth    < 0.2  (80% growth inhibition)
  Cytotoxic          = mean relative viability < 0.9  (90% viability threshold)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.featurize import smiles_to_data
from src.data.featurize_chemprop import smiles_to_data_cp
from src.data.feature_cache import FeatureCache


# All RDKit 2D descriptors (~200). NaN-safe: _mol_descriptors fills failures with 0.0.
_DESCRIPTORS: list[tuple[str, callable]] = list(Descriptors.descList)
N_DESCRIPTORS = len(_DESCRIPTORS)


def _mol_descriptors(mol) -> np.ndarray:
    """Compute raw descriptor vector for one RDKit mol. Returns zeros for failures."""
    out = np.zeros(N_DESCRIPTORS, dtype=np.float32)
    for i, (_, fn) in enumerate(_DESCRIPTORS):
        try:
            v = fn(mol)
            # Some descriptors (e.g. MaxAbsPartialCharge) return float NaN
            # without raising — catch both None and non-finite returns.
            if v is not None and math.isfinite(float(v)):
                out[i] = float(v)
        except Exception:
            pass
    return out


LABEL_COLS = ["antibiotic", "hepg2", "hskmc", "imr90"]

# Binarization thresholds (from Wong et al. 2023 Methods)
ANTIBIOTIC_CUTOFF = 0.2   # Mean_50uM < 0.2 → active
CYTOTOX_CUTOFF = 0.9      # Mean_10uM < 0.9 → cytotoxic

# sheet/column metadata per task. The antibiotic sheet is the only one that
# carries SMILES, so it is always read; other sheets are merged only when their
# task is requested.
_SHEET_SPEC = {
    "antibiotic": ("S. aureus growth inhibition", "Mean_50uM", ANTIBIOTIC_CUTOFF),
    "hepg2":      ("HepG2 viability",             "Mean_10uM", CYTOTOX_CUTOFF),
    "hskmc":      ("HSkMC viability",             "Mean_10uM", CYTOTOX_CUTOFF),
    "imr90":      ("IMR-90 viability",            "Mean_10uM", CYTOTOX_CUTOFF),
}


@lru_cache(maxsize=None)
def _load_xlsx_cached(path_str: str, tasks_key: tuple[str, ...]) -> pd.DataFrame:
    """Memoized core of load_xlsx (keyed on hashable args). Returns the shared
    DataFrame — callers must treat it as read-only / copy before mutating."""
    return _load_xlsx_impl(Path(path_str), list(tasks_key))


def load_xlsx(xlsx_path: str | Path, tasks: list[str] | None = None) -> pd.DataFrame:
    """Load Wong et al. 2023 assay sheets, inner-joining the requested tasks.

    Memoized on (path, tasks): the same sheets are read once per process even
    though train/val/test each call this (3x per finetune, 30x per ensemble).
    """
    if tasks is None:
        tasks = list(LABEL_COLS)
    return _load_xlsx_cached(str(Path(xlsx_path)), tuple(tasks))


def _load_xlsx_impl(xlsx_path: str | Path, tasks: list[str] | None = None) -> pd.DataFrame:
    """
    Load Wong et al. 2023 assay sheets and inner-join only the requested tasks.

    Args:
        tasks: which task labels to require. Compounds missing any requested
               label are dropped (inner join). Defaults to all four — the
               original 4-way intersection. For a single-task antibiotic run,
               pass tasks=["antibiotic"] to keep all ~39k antibiotic-screened
               compounds instead of the ~12k subset that was also screened on
               every cytotox cell line.

    Returns:
        DataFrame with columns compound_id, smiles, antibiotic, hepg2, hskmc,
        imr90. Non-requested label columns are filled with 0.0 (placeholder;
        task-restricted heads ignore them via task_cols).
    """
    if tasks is None:
        tasks = list(LABEL_COLS)
    unknown = [t for t in tasks if t not in LABEL_COLS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; must be subset of {LABEL_COLS}")

    path = Path(xlsx_path)

    # Antibiotic sheet always read — it provides SMILES.
    ab_sheet, ab_raw, _ = _SHEET_SPEC["antibiotic"]
    df = pd.read_excel(
        path, sheet_name=ab_sheet,
        usecols=["Compound_ID", "SMILES", ab_raw],
    ).rename(columns={ab_raw: "antibiotic_raw"})

    # Inner-merge each additional requested cytotox sheet.
    for t in tasks:
        if t == "antibiotic":
            continue
        sheet, raw, _ = _SHEET_SPEC[t]
        d = pd.read_excel(
            path, sheet_name=sheet, usecols=["Compound_ID", raw],
        ).rename(columns={raw: f"{t}_raw"})
        df = df.merge(d, on="Compound_ID", how="inner")

    # Binarize requested labels; placeholder 0.0 for the rest so y stays (1, 4).
    for t in LABEL_COLS:
        if t in tasks:
            _, _, cutoff = _SHEET_SPEC[t]
            df[t] = (df[f"{t}_raw"] < cutoff).astype(float)
        else:
            df[t] = 0.0

    df = df.rename(columns={"SMILES": "smiles", "Compound_ID": "compound_id"})
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    n = len(df)
    counts = ", ".join(f"{int(df[t].sum())} {t}-active" for t in tasks)
    print(f"Loaded {n} compounds ({'+'.join(tasks)} inner-join): {counts}")
    return df[["compound_id", "smiles"] + LABEL_COLS]


def _murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES (ring systems + linkers); '' if unavailable."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def _scaffold_split(df: pd.DataFrame, test_size: float, val_size: float,
                    partition_seed: int | None = None):
    """Bemis-Murcko scaffold split — no scaffold shared across splits.

    Scaffold groups are assigned to whichever split is currently furthest below
    its target size. With ``partition_seed is None`` (the default, canonical
    partition) groups are visited largest-first: big common scaffolds land in
    train, rarer chemotypes fill val/test, so test measures extrapolation to
    unseen structural classes — the relevant signal for novel-class discovery.

    With an integer ``partition_seed`` the group visitation order is shuffled by
    that seed instead (a *random* scaffold split). The no-shared-scaffold
    invariant is preserved — whole groups are still assigned wholesale — but the
    composition of the held-out test set changes, yielding a *different* valid
    scaffold partition per seed. This is used to check that a scaffold-split
    result is robust across partitions rather than an artefact of the single
    canonical one. (Because train has the most room early on, the largest groups
    still tend to land in train regardless of order, so each partition keeps the
    same extrapolation character while holding out a different set of rarer
    chemotypes.)
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df["smiles"]):
        scaffolds[_murcko_scaffold(str(smi))].append(i)

    n = len(df)
    targets = {
        "train": (1.0 - val_size - test_size) * n,
        "val": val_size * n,
        "test": test_size * n,
    }
    groups = list(scaffolds.values())
    if partition_seed is None:
        ordered = sorted(groups, key=len, reverse=True)
    else:
        # Sort first so the shuffle is deterministic w.r.t. the seed regardless
        # of dict iteration order, then shuffle the visitation order.
        ordered = sorted(groups, key=len, reverse=True)
        random.Random(partition_seed).shuffle(ordered)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for group in ordered:
        # assign the whole scaffold group to the split with the most room left
        dest = max(splits, key=lambda s: targets[s] - len(splits[s]))
        splits[dest] += group
    return splits["train"], splits["val"], splits["test"]


def _random_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int,
                  stratify_col: str = "antibiotic"):
    """Random split, stratified by the (rare) label column for stratify_col."""
    idx = list(range(len(df)))
    strat = df[stratify_col].values.astype(int)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=strat, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size / (1.0 - test_size),
        stratify=strat[idx_trainval],
        random_state=seed,
    )
    return idx_train, idx_val, idx_test


class AntibioticDataset(Dataset):
    """
    In-memory dataset for the Wong et al. 2023 antibiotic compound library.

    Args:
        xlsx_path:    path to 41586_2023_6887_MOESM3_ESM.xlsx.
        split:        'train' | 'val' | 'test'
        test_size:    fraction held out for test (default 0.20, matching paper).
        val_size:     fraction of train+val used for validation.
        seed:         random seed (used only by the 'random' split method).
        split_method: 'scaffold' (no scaffold shared across splits — measures
                      extrapolation to novel chemotypes) or 'random' (stratified
                      by antibiotic label — optimistic, measures interpolation).
        desc_stats:   (mean, std) tensors from the training split for normalizing
                      RDKit descriptors. Pass train_ds.desc_stats to val/test
                      datasets so they are normalized on the same scale. If None,
                      statistics are computed from this split (correct for train).
        tasks:        which task labels to inner-join on. Defaults to all four
                      (the original 4-way intersection). Pass e.g.
                      ["antibiotic"] to keep every compound with an antibiotic
                      label instead of intersecting with cytotox screens.
    """

    def __init__(
        self,
        xlsx_path: str | Path,
        split: str = "train",
        test_size: float = 0.20,
        val_size: float = 0.10,
        seed: int = 42,
        split_method: str = "scaffold",
        desc_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        tasks: list[str] | None = None,
        featurizer: str = "ours",
        use_cache: bool = True,
        scaffold_partition_seed: int | None = None,
    ):
        super().__init__()
        self.xlsx_path = Path(xlsx_path)
        self.split = split
        self.split_method = split_method
        # None = canonical deterministic scaffold partition; an int selects a
        # different (still leakage-free) scaffold partition for robustness checks.
        self.scaffold_partition_seed = scaffold_partition_seed
        self.tasks = list(tasks) if tasks is not None else list(LABEL_COLS)
        # Atom/bond featurizer for x/edge_attr (the 217 global descriptors are
        # unchanged). 'chemprop' = byte-faithful Chemprop scheme (fidelity audit).
        self._featurize_fn = smiles_to_data_cp if featurizer == "chemprop" else smiles_to_data
        # Persistent (graph, descriptors) cache — featurization is a pure function
        # of (smiles, featurizer), so this is shared across splits/seeds/tasks and
        # the ~390 s/build descriptor cost is paid only on the first cold build.
        self._cache = (
            FeatureCache(
                featurizer=featurizer,
                graph_fn=self._featurize_fn,
                desc_fn=_mol_descriptors,
                n_descriptors=N_DESCRIPTORS,
            )
            if use_cache else None
        )
        # First requested task is the "primary" — used for stratification and
        # for the per-split actives count log line.
        self._primary_task = self.tasks[0]
        self._data_list, raw_descs = self._build(test_size, val_size, seed)

        # Normalize descriptors.
        raw = torch.tensor(np.nan_to_num(raw_descs, nan=0.0), dtype=torch.float32)
        if desc_stats is None:
            self.desc_mean = raw.mean(0)
            self.desc_std  = raw.std(0).clamp(min=1e-6)
        else:
            self.desc_mean, self.desc_std = desc_stats

        norm = torch.nan_to_num(
            (raw - self.desc_mean) / self.desc_std,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        for i, data in enumerate(self._data_list):
            # unsqueeze(0) so PyG's Batch.from_data_list stacks to (B, n_desc)
            # rather than concatenating to (B*n_desc,).
            data.global_feat = norm[i].unsqueeze(0)

    @property
    def desc_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.desc_mean, self.desc_std)

    @property
    def n_descriptors(self) -> int:
        return N_DESCRIPTORS

    def _build(self, test_size: float, val_size: float, seed: int) -> tuple[list[Data], np.ndarray]:
        df = load_xlsx(self.xlsx_path, tasks=self.tasks)

        if self.split_method == "scaffold":
            idx_train, idx_val, idx_test = _scaffold_split(
                df, test_size, val_size, partition_seed=self.scaffold_partition_seed,
            )
        elif self.split_method == "random":
            idx_train, idx_val, idx_test = _random_split(
                df, test_size, val_size, seed, stratify_col=self._primary_task,
            )
        else:
            raise ValueError(
                f"Unknown split_method {self.split_method!r} (use 'scaffold' or 'random')"
            )

        split_idx = {"train": idx_train, "val": idx_val, "test": idx_test}[self.split]
        subset = df.iloc[split_idx].reset_index(drop=True)

        n_active = int(subset[self._primary_task].sum())
        print(
            f"  [{self.split}] {self.split_method} split: {len(subset)} compounds, "
            f"{n_active} {self._primary_task}-active "
            f"({100 * n_active / max(len(subset), 1):.1f}%)"
        )

        data_list = []
        raw_descs = []
        skipped = 0
        for _, row in subset.iterrows():
            smi = str(row["smiles"])
            # y shape (1, num_tasks) so PyG's Batch.from_data_list concatenates
            # graph labels to (num_graphs, num_tasks), not flat (num_graphs*num_tasks,)
            y = torch.tensor([[float(row[c]) for c in LABEL_COLS]], dtype=torch.float)
            if self._cache is not None:
                data, desc = self._cache.get(smi)
            else:
                data = self._featurize_fn(smi)
                mol = Chem.MolFromSmiles(smi)
                desc = _mol_descriptors(mol) if mol is not None else np.zeros(N_DESCRIPTORS)
            if data is None:
                skipped += 1
                continue
            data.y = y
            raw_descs.append(desc)
            data_list.append(data)

        if self._cache is not None:
            self._cache.save()

        if skipped:
            print(f"  [{self.split}] Skipped {skipped} unparseable SMILES")
        print(f"  [{self.split}] {len(data_list)} molecules ready")
        return data_list, np.array(raw_descs, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]

    def pos_weights(self) -> torch.Tensor:
        """Per-task positive class weights (N_neg / N_pos) for weighted BCE."""
        ys = torch.cat([d.y for d in self._data_list], dim=0)  # (N, num_tasks)
        weights = []
        for t in range(ys.size(1)):
            n_pos = ys[:, t].sum().item()
            n_neg = (ys[:, t] == 0).sum().item()
            weights.append(n_neg / max(n_pos, 1.0))
        return torch.tensor(weights, dtype=torch.float)