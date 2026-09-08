"""
Random Walk Positional Encoding (RWPE) for graph nodes.

Computes k-step random walk landing probabilities from each node back to itself,
providing a relative structural position signal for the GPS Transformer.

Reference: Dwivedi et al. "Graph Neural Networks with Learnable Structural and
Positional Representations" (ICLR 2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops


def compute_rwpe(data: Data, walk_length: int = 20) -> Tensor:
    """
    Compute random walk positional encodings for all nodes in a graph.

    Args:
        data: PyG Data object (must have edge_index and x).
        walk_length: number of RW steps (= dimension of positional encoding).

    Returns:
        pe: Tensor of shape (num_nodes, walk_length).
    """
    num_nodes = data.x.size(0)
    edge_index = data.edge_index

    if edge_index.size(1) == 0:
        return torch.zeros(num_nodes, walk_length, device=data.x.device)

    # Add self-loops for numerical stability
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Compute degree
    deg = torch.zeros(num_nodes, device=data.x.device)
    deg.scatter_add_(0, edge_index_sl[0], torch.ones(edge_index_sl.size(1), device=data.x.device))

    # Row-normalised adjacency: D^{-1} A
    adj = to_dense_adj(edge_index_sl, max_num_nodes=num_nodes).squeeze(0)  # (N, N)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float("inf")] = 0.0
    rw = deg_inv.unsqueeze(1) * adj  # D^{-1} A, shape (N, N)

    # Accumulate k-step landing probabilities (diagonal of RW^k)
    pe_list = []
    rw_k = rw.clone()
    for _ in range(walk_length):
        pe_list.append(rw_k.diagonal())  # (N,)
        rw_k = rw_k @ rw

    pe = torch.stack(pe_list, dim=1)  # (N, walk_length)
    return pe


class RWPEEncoder(nn.Module):
    """
    Learnable linear projection of RWPE features into the node embedding space.

    The raw RWPE values are projected to `out_dim` and added to the node features
    before being passed to GPS layers.
    """

    def __init__(self, walk_length: int = 20, out_dim: int = 256):
        super().__init__()
        self.walk_length = walk_length
        self.proj = nn.Linear(walk_length, out_dim)

    def forward(self, pe: Tensor) -> Tensor:
        """pe: (num_nodes, walk_length) → (num_nodes, out_dim)."""
        return self.proj(pe)
