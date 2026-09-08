"""
Chemprop-FAITHFUL atom/bond featurization (ported from chemprop v1
features/featurization.py), as a drop-in alternative to featurize.py for the
D-MPNN fidelity audit.

Why: our featurize.py ("following Wong et al. Methods") differs from Chemprop's
actual scheme in ways that feed the message passing — e.g. GetTotalDegree (incl.
H) vs our GetDegree, 4 chiral tags vs our 2, atomic_num over range(100) with the
`-1` offset, num_Hs [0..4]. After eliminating set/conditioning/scale/regime/metric/
proj_head as the hepg2 gap cause, the per-atom/bond features are the last untested
difference. This module reproduces Chemprop's exact vectors so a D-MPNN over them
isolates that variable.

Produces PyG Data with:
  x         = Chemprop atom_features  (ATOM_FDIM_CP = 133)
  edge_attr = Chemprop bond_features  (BOND_FDIM_CP = 14, the PURE bond part)
Edges are stored as consecutive (i->j, j->i) pairs, identical to featurize.py, so
DMPNNEncoder's reverse-edge map (arange(E) ^ 1) and its cat(x[src], edge_attr)
init (== Chemprop's cat(f_atom[a1], f_bond)) both hold unchanged.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import rdchem


_MAX_ATOMIC_NUM = 100
_ATOM_FEATURES = {
    "atomic_num": list(range(_MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ],
}

# sum(len(choices)+1) + 2 (aromatic, mass) = 131 + 2 = 133
ATOM_FDIM_CP = sum(len(c) + 1 for c in _ATOM_FEATURES.values()) + 2
BOND_FDIM_CP = 14


def _onek(value, choices: list) -> list[int]:
    """One-hot with an extra 'uncommon' bucket at the end (chemprop's onek_encoding_unk)."""
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else -1
    enc[idx] = 1
    return enc


def atom_features_cp(atom: rdchem.Atom) -> list[float]:
    return (
        _onek(atom.GetAtomicNum() - 1, _ATOM_FEATURES["atomic_num"])
        + _onek(atom.GetTotalDegree(), _ATOM_FEATURES["degree"])
        + _onek(atom.GetFormalCharge(), _ATOM_FEATURES["formal_charge"])
        + _onek(int(atom.GetChiralTag()), _ATOM_FEATURES["chiral_tag"])
        + _onek(int(atom.GetTotalNumHs()), _ATOM_FEATURES["num_Hs"])
        + _onek(int(atom.GetHybridization()), _ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )


def bond_features_cp(bond: rdchem.Bond) -> list[float]:
    bt = bond.GetBondType()
    fbond = [
        0,  # bond is not None
        int(bt == rdchem.BondType.SINGLE),
        int(bt == rdchem.BondType.DOUBLE),
        int(bt == rdchem.BondType.TRIPLE),
        int(bt == rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
    fbond += _onek(int(bond.GetStereo()), list(range(6)))
    return fbond


def mol_to_data_cp(mol: rdchem.Mol, y: torch.Tensor | None = None, smiles: str = "") -> Data | None:
    if mol is None:
        return None
    atom_feats = [atom_features_cp(a) for a in mol.GetAtoms()]
    if not atom_feats:
        return None
    x = torch.tensor(atom_feats, dtype=torch.float)  # (num_atoms, ATOM_FDIM_CP)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features_cp(bond)
        edge_indices += [[i, j], [j, i]]   # consecutive reverse pairs
        edge_attrs += [bf, bf]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FDIM_CP), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if y is not None:
        data.y = y
    return data


def smiles_to_data_cp(smiles: str, y: torch.Tensor | None = None) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_data_cp(mol, y=y, smiles=smiles)
