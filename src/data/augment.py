"""
Molecular graph augmentations for LeJEPA views.

Produces Vg global views (full molecule, lightly corrupted) and
Vl local views (contiguous subgraph samples) for each molecule.
"""

from __future__ import annotations

import random
from collections import deque

import torch
from torch_geometric.data import Data


def _mask_atom_features(data: Data, mask_prob: float = 0.15) -> Data:
    """Global view 1: randomly zero-out atom feature rows."""
    x = data.x.clone()
    mask = torch.rand(x.size(0)) < mask_prob
    x[mask] = 0.0
    return Data(
        x=x,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
    )


def _drop_edges(data: Data, drop_prob: float = 0.10) -> Data:
    """Global view 2 (DEFAULT, reproduces the paper): randomly remove bond pairs.

    Removes a random ``drop_prob`` fraction of bonds from the graph (both directed
    half-edges of each undirected bond), which can locally fragment the molecule.
    This is the edge-dropout augmentation used by the checkpoints reported in the
    paper; it is the default so the released code reproduces those results. A
    connectivity-preserving alternative is ``_mask_edge_features`` below
    (``edge_corruption="feature_mask"``).
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_edges = edge_index.size(1)

    if num_edges == 0:
        return Data(
            x=data.x.clone(),
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone(),
        )

    # Each undirected bond occupies two consecutive positions (i→j and j→i);
    # keep/drop by pairs so the graph stays symmetric.
    num_bonds = num_edges // 2
    keep_mask = torch.rand(num_bonds) >= drop_prob
    keep_directed = keep_mask.repeat_interleave(2)

    return Data(
        x=data.x.clone(),
        edge_index=edge_index[:, keep_directed],
        edge_attr=edge_attr[keep_directed],
    )


def _mask_edge_features(data: Data, drop_prob: float = 0.10) -> Data:
    """Global view 2 (Phase-7 variant): connectivity-preserving bond-feature mask.

    Zeroes the ``edge_attr`` of randomly chosen bonds while KEEPING ``edge_index``
    intact, so the molecule is never fragmented and RWPE reachability is preserved.
    This is NOT used by the reported checkpoints; select it with
    ``edge_corruption="feature_mask"`` (the Phase-7 SSL-tuning runs, e.g. cover025,
    used this). Masking is per undirected bond-pair so the graph stays symmetric.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_edges = edge_index.size(1)

    if num_edges == 0 or edge_attr is None:
        return Data(
            x=data.x.clone(),
            edge_index=edge_index.clone(),
            edge_attr=None if edge_attr is None else edge_attr.clone(),
        )

    num_bonds = num_edges // 2
    drop_mask = torch.rand(num_bonds) < drop_prob
    drop_directed = drop_mask.repeat_interleave(2)  # both directions of each bond
    new_attr = edge_attr.clone()
    new_attr[drop_directed] = 0.0

    return Data(
        x=data.x.clone(),
        edge_index=edge_index.clone(),
        edge_attr=new_attr,
    )


def _build_adjacency(edge_index: torch.Tensor, num_atoms: int) -> list[list[int]]:
    """Adjacency list built once per molecule and shared by every BFS view."""
    adj: list[list[int]] = [[] for _ in range(num_atoms)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
    return adj


def _bfs_subgraph(data: Data, adj: list[list[int]], cover_frac: float = 0.6) -> Data:
    """
    Local view: BFS subgraph from a random seed atom.

    Expands until at least cover_frac of atoms are included (or all atoms
    if the graph is small). Falls back to the full graph for tiny molecules.

    `adj` is precomputed once per molecule by augment() — do not rebuild it
    here, it was 8x redundant work per molecule.
    """
    num_atoms = data.x.size(0)

    if num_atoms <= 5:
        # For very small molecules, return full graph
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone(),
        )

    target_size = max(1, int(num_atoms * cover_frac))

    seed = random.randint(0, num_atoms - 1)
    visited: set[int] = {seed}
    queue: deque[int] = deque([seed])

    while queue and len(visited) < target_size:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target_size:
                    break

    nodes = torch.tensor(sorted(visited), dtype=torch.long)

    # Boolean membership mask — cheaper than torch.isin for edge filtering.
    member = torch.zeros(num_atoms, dtype=torch.bool)
    member[nodes] = True
    edge_index = data.edge_index
    in_subgraph = member[edge_index[0]] & member[edge_index[1]]

    # Vectorized old→new node index remap.
    remap = torch.full((num_atoms,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(nodes.numel(), dtype=torch.long)

    # Advanced indexing already returns fresh tensors (no alias to `data`).
    return Data(
        x=data.x[nodes],
        edge_index=remap[edge_index[:, in_subgraph]],
        edge_attr=data.edge_attr[in_subgraph],
    )


def augment(
    data: Data,
    Vg: int = 2,
    Vl: int = 8,
    cover_frac: float = 0.6,
    edge_corruption: str = "dropout",
) -> list[Data]:
    """
    Generate Vg global views + Vl local views for a molecule.

    Global views (full molecule, lightly corrupted):
      - Even indices: atom feature masking
      - Odd indices: bond corruption, selected by `edge_corruption`:
          * "dropout"      -> remove ~10% of bonds (DEFAULT; reproduces the paper)
          * "feature_mask" -> zero ~10% of bond features, keep connectivity (Phase 7)

    Local views:
      - BFS subgraph covering ~cover_frac of atoms each.

    `cover_frac` is the Phase-7 SSL-difficulty knob: lower = local views hold out
    MORE of the molecule, so the part→whole prediction task is harder (local
    pred_loss should rise). Sweep it (0.6 → 0.4 → 0.25) to test whether the
    invariance task is too easy.

    Returns a list of length Vg + Vl.
    """
    if edge_corruption == "dropout":
        _corrupt_edges = _drop_edges
    elif edge_corruption == "feature_mask":
        _corrupt_edges = _mask_edge_features
    else:
        raise ValueError(
            f"edge_corruption must be 'dropout' or 'feature_mask', got {edge_corruption!r}"
        )

    views: list[Data] = []

    # Global views
    for i in range(Vg):
        if i % 2 == 0:
            views.append(_mask_atom_features(data))
        else:
            views.append(_corrupt_edges(data))

    # Local views — adjacency built once, reused across all Vl BFS walks.
    adj = _build_adjacency(data.edge_index, data.x.size(0))
    for _ in range(Vl):
        views.append(_bfs_subgraph(data, adj, cover_frac=cover_frac))

    return views