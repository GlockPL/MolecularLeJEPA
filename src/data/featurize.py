"""
Molecular featurization following Wong et al. 2023 (Nature) Methods section.
Converts RDKit molecules into PyG Data objects with atom, bond, and global features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except ImportError as e:
    raise ImportError("rdkit is required: pip install rdkit") from e


# --- Atom feature vocabulary ---

_ATOMIC_NUMS = list(range(1, 119))  # H to Og; will one-hot with an "other" bucket
_ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_FORMAL_CHARGES = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
_NUM_HS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
_CHIRALITIES = [
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

# --- Bond feature vocabulary ---

_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
_BOND_STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

# O(1) lookup dicts — list.index() on 118-element lists was a hot path
_ATOMIC_NUM_IDX   = {v: i for i, v in enumerate(_ATOMIC_NUMS)}
_ATOM_DEGREE_IDX  = {v: i for i, v in enumerate(_ATOM_DEGREES)}
_FORMAL_CHARGE_IDX= {v: i for i, v in enumerate(_FORMAL_CHARGES)}
_NUM_HS_IDX       = {v: i for i, v in enumerate(_NUM_HS)}
_HYBRID_IDX       = {v: i for i, v in enumerate(_HYBRIDIZATIONS)}
_CHIRAL_IDX       = {v: i for i, v in enumerate(_CHIRALITIES)}
_BOND_TYPE_IDX    = {v: i for i, v in enumerate(_BOND_TYPES)}
_BOND_STEREO_IDX  = {v: i for i, v in enumerate(_BOND_STEREOS)}

# Keep old names as public aliases for external code
ATOMIC_NUMS   = _ATOMIC_NUMS
ATOM_DEGREES  = _ATOM_DEGREES
FORMAL_CHARGES= _FORMAL_CHARGES
NUM_HS        = _NUM_HS
HYBRIDIZATIONS= _HYBRIDIZATIONS
CHIRALITIES   = _CHIRALITIES
BOND_TYPES    = _BOND_TYPES
BOND_STEREOS  = _BOND_STEREOS


def _one_hot(value, choices: list, idx: dict) -> list[int]:
    """One-hot encode value against choices; last element is the 'other' bucket."""
    encoding = [0] * (len(choices) + 1)
    i = idx.get(value)
    if i is not None:
        encoding[i] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom: rdchem.Atom) -> list[float]:
    """Return a flat float vector of atom features (~72 dims)."""
    return (
        _one_hot(atom.GetAtomicNum(),    _ATOMIC_NUMS,    _ATOMIC_NUM_IDX)    # 119
        + _one_hot(atom.GetDegree(),     _ATOM_DEGREES,   _ATOM_DEGREE_IDX)   # 12
        + _one_hot(atom.GetFormalCharge(),_FORMAL_CHARGES,_FORMAL_CHARGE_IDX) # 12
        + _one_hot(atom.GetChiralTag(),  _CHIRALITIES,    _CHIRAL_IDX)        # 3
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS,         _NUM_HS_IDX)        # 10
        + _one_hot(atom.GetHybridization(),_HYBRIDIZATIONS,_HYBRID_IDX)       # 6
        + [int(atom.GetIsAromatic())]                                          # 1
        + [atom.GetMass() / 100.0]                                             # 1 (scaled)
    )


ATOM_DIM = (
    len(ATOMIC_NUMS) + 1
    + len(ATOM_DEGREES) + 1
    + len(FORMAL_CHARGES) + 1
    + len(CHIRALITIES) + 1
    + len(NUM_HS) + 1
    + len(HYBRIDIZATIONS) + 1
    + 1  # aromaticity
    + 1  # mass
)


def bond_features(bond: rdchem.Bond) -> list[float]:
    """Return a flat float vector of bond features (~10 dims)."""
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES,  _BOND_TYPE_IDX)   # 5
        + [int(bond.GetIsConjugated())]                               # 1
        + [int(bond.IsInRing())]                                      # 1
        + _one_hot(bond.GetStereo(), _BOND_STEREOS, _BOND_STEREO_IDX) # 7
    )


BOND_DIM = len(BOND_TYPES) + 1 + 1 + 1 + len(BOND_STEREOS) + 1


def mol_to_data(mol: rdchem.Mol, y: torch.Tensor | None = None, smiles: str = "") -> Data | None:
    """
    Convert an RDKit molecule into a PyG Data object.

    Args:
        mol: RDKit molecule (must have been sanitized).
        y: optional label tensor of shape (num_tasks,).
        smiles: original SMILES string (stored for reference).

    Returns:
        Data object with fields: x, edge_index, edge_attr, y (optional), smiles.
        Returns None if molecule is invalid.
    """
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    if not atom_feats:
        return None
    x = torch.tensor(atom_feats, dtype=torch.float)  # (num_atoms, ATOM_DIM)

    # Edge features (undirected: each bond → two directed edges)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bf, bf]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # molecule with no bonds (e.g. single atom)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
    )
    if y is not None:
        data.y = y
    return data


def smiles_to_data(smiles: str, y: torch.Tensor | None = None) -> Data | None:
    """Parse SMILES and return a PyG Data object, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)  # sanitizes implicitly
    if mol is None:
        return None
    return mol_to_data(mol, y=y, smiles=smiles)


# --- Compact featurization (for the precomputed shard cache) ---------------
#
# RDKit parsing is ~0.7 ms/molecule and is repeated every epoch. The compact
# form stores raw one-hot *bucket indices* (uint8) instead of expanded float
# vectors — ~25x smaller on disk than dense x — so the dataset can be
# pre-featurized once and reloaded without RDKit. compact_to_data()
# reconstructs the exact tensors mol_to_data() would have produced.

# One-hot block widths in the order atom_features() / bond_features() concat.
_ATOM_ONEHOT_W = (
    len(_ATOMIC_NUMS) + 1,    # 119
    len(_ATOM_DEGREES) + 1,   # 12
    len(_FORMAL_CHARGES) + 1, # 12
    len(_CHIRALITIES) + 1,    # 3
    len(_NUM_HS) + 1,         # 10
    len(_HYBRIDIZATIONS) + 1, # 6
)
_BOND_TYPE_W = len(_BOND_TYPES) + 1     # 5
_BOND_STEREO_W = len(_BOND_STEREOS) + 1 # 7


def _bucket(value, idx: dict, n_choices: int) -> int:
    """One-hot bucket index for `value`; the 'other' bucket sits at n_choices."""
    return idx.get(value, n_choices)


def mol_to_compact(mol: rdchem.Mol) -> tuple | None:
    """
    Convert an RDKit molecule into compact integer arrays for the shard cache.

    Returns (x_idx, x_mass, edge_index, e_idx), or None if invalid:
      x_idx      uint8   (N, 7)  6 one-hot bucket indices + aromatic flag
      x_mass     float32 (N,)    atom mass / 100
      edge_index int16   (2, B)  one entry per *bond* (undirected), local idx
      e_idx      uint8   (B, 4)  [bondtype idx, conjugated, in-ring, stereo idx]

    Edges are stored undirected (one column per bond) to halve the cache size;
    compact_to_data() expands each bond to its two directed edges.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x_rows: list[list[int]] = []
    mass: list[float] = []
    for atom in mol.GetAtoms():
        x_rows.append([
            _bucket(atom.GetAtomicNum(),     _ATOMIC_NUM_IDX,    len(_ATOMIC_NUMS)),
            _bucket(atom.GetDegree(),        _ATOM_DEGREE_IDX,   len(_ATOM_DEGREES)),
            _bucket(atom.GetFormalCharge(),  _FORMAL_CHARGE_IDX, len(_FORMAL_CHARGES)),
            _bucket(atom.GetChiralTag(),     _CHIRAL_IDX,        len(_CHIRALITIES)),
            _bucket(atom.GetTotalNumHs(),    _NUM_HS_IDX,        len(_NUM_HS)),
            _bucket(atom.GetHybridization(), _HYBRID_IDX,        len(_HYBRIDIZATIONS)),
            int(atom.GetIsAromatic()),
        ])
        mass.append(atom.GetMass() / 100.0)

    x_idx = torch.tensor(x_rows, dtype=torch.uint8)
    x_mass = torch.tensor(mass, dtype=torch.float32)

    edge_pairs: list[list[int]] = []  # one [begin, end] per bond
    edge_rows: list[list[int]] = []   # one feature row per bond
    for bond in mol.GetBonds():
        edge_pairs.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_rows.append([
            _bucket(bond.GetBondType(), _BOND_TYPE_IDX,   len(_BOND_TYPES)),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            _bucket(bond.GetStereo(),   _BOND_STEREO_IDX, len(_BOND_STEREOS)),
        ])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.int16).t().contiguous()
        e_idx = torch.tensor(edge_rows, dtype=torch.uint8)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int16)
        e_idx = torch.zeros((0, 4), dtype=torch.uint8)

    return x_idx, x_mass, edge_index, e_idx


def compact_to_data(
    x_idx: torch.Tensor,
    x_mass: torch.Tensor,
    edge_index: torch.Tensor,
    e_idx: torch.Tensor,
) -> Data:
    """
    Reconstruct the exact PyG Data that mol_to_data() would produce.

    edge_index/e_idx hold undirected bonds; each bond is expanded to its two
    directed edges (i->j, j->i) in the same column order mol_to_data() emits.
    """
    xi = x_idx.long()
    parts = [F.one_hot(xi[:, k], w).float() for k, w in enumerate(_ATOM_ONEHOT_W)]
    parts.append(xi[:, 6:7].float())               # aromatic flag
    parts.append(x_mass.view(-1, 1).float())       # scaled mass
    x = torch.cat(parts, dim=1)

    # Undirected (2, B) -> directed (2, 2B): columns [b0_ij, b0_ji, b1_ij, ...].
    b = edge_index.size(1)
    directed = torch.empty((2, 2 * b), dtype=torch.long)
    src, dst = edge_index[0].long(), edge_index[1].long()
    directed[0, 0::2], directed[0, 1::2] = src, dst
    directed[1, 0::2], directed[1, 1::2] = dst, src

    ei = e_idx.repeat_interleave(2, dim=0).long()  # each bond's row, twice
    edge_attr = torch.cat(
        [
            F.one_hot(ei[:, 0], _BOND_TYPE_W).float(),
            ei[:, 1:3].float(),                    # conjugated, in-ring
            F.one_hot(ei[:, 3], _BOND_STEREO_W).float(),
        ],
        dim=1,
    )
    return Data(x=x, edge_index=directed, edge_attr=edge_attr)
