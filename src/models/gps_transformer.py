"""
GPS Graph Transformer backbone for molecular property learning.

Architecture (per layer):
  Local:  GINEConv (message-passing with edge features)
  Global: MultiheadAttention over all nodes in the graph
  Residuals + LayerNorm on both branches

After N GPS layers:
  Global mean pooling → MLP projection head
  → embed_dim-dimensional isotropic Gaussian embedding (target of LeJEPA)

Reference: Rampášek et al. "Recipe for a General, Powerful, Scalable Graph
Transformer" (NeurIPS 2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (GINEConv, global_add_pool, global_max_pool,
                                global_mean_pool)

# Graph readouts. 'mean' is the historical default and the only one any existing
# checkpoint was trained with. The readout is NOT a free choice at probe time: on
# molhiv the ranking inverts between untrained (max > sum > mean) and LeJEPA-pretrained
# (mean > max > sum), because pretraining optimizes the statistic it pools through.
# Comparing a trained to an untrained encoder therefore requires sweeping the readout
# on BOTH arms — see scripts/readout_probe_molhiv.py.
POOL_FNS = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


def _parse_readout(readout: str) -> list[str]:
    """'mean' | 'sum' | 'max', '+'-joined; returns the block list."""
    parts = [p for p in readout.split("+") if p]
    unknown = [p for p in parts if p not in POOL_FNS]
    if not parts or unknown:
        raise ValueError(f"bad readout {readout!r}: blocks must be "
                         f"{list(POOL_FNS)} joined by '+'")
    return parts
from torch_geometric.utils import to_dense_adj, to_dense_batch

from src.data.featurize import ATOM_DIM, BOND_DIM
from src.models.positional_encoding import RWPEEncoder
from src.profiling import PROFILER


class GPSLayer(nn.Module):
    """
    Single GPS layer: MPNN (local) + MultiheadAttention (global), residuals + LN.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Local: GINEConv requires edge_attr to have same dim as node features
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.mpnn = GINEConv(nn=mlp, edge_dim=hidden_dim)
        self.norm_mpnn = nn.LayerNorm(hidden_dim)

        # Global: standard MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        # --- Local branch (MPNN) ---
        h_local = self.mpnn(x, edge_index, edge_attr)
        x = self.norm_mpnn(x + self.dropout(h_local))

        # --- Global branch (attention) ---
        # Pack nodes into a dense (batch, max_nodes, hidden) tensor with mask
        x_dense, key_padding_mask = to_dense_batch(x, batch)  # (B, N_max, H)
        # key_padding_mask: True where *padding* — invert for nn.MultiheadAttention
        attn_mask = ~key_padding_mask  # True = valid, but MHA expects True = ignore
        h_global, _ = self.attn(x_dense, x_dense, x_dense, key_padding_mask=attn_mask)
        # Scatter back to sparse representation
        h_global_sparse = h_global[key_padding_mask]  # valid nodes only
        x = self.norm_attn(x + self.dropout(h_global_sparse))

        # --- Feed-forward ---
        x = self.norm_ff(x + self.dropout(self.ff(x)))
        return x


class GPSTransformer(nn.Module):
    """
    GPS Graph Transformer backbone.

    Input:  PyG batch (x, edge_index, edge_attr, batch)
    Output: (N_graphs, embed_dim) embedding tensor

    Args:
        atom_dim:    raw atom feature dimension (from featurize.ATOM_DIM)
        bond_dim:    raw bond feature dimension (from featurize.BOND_DIM)
        hidden_dim:  internal node/edge embedding dimension
        embed_dim:   final output embedding dimension (target for LeJEPA)
        num_layers:  number of GPS layers
        num_heads:   attention heads per GPS layer
        walk_length: RWPE steps
        dropout:     dropout rate in attention and FF layers
        readout:     graph pooling, '+'-joined blocks from {mean, sum, max}
                     ('mean' = the historical default; see POOL_FNS / _pool)
    """

    def __init__(
        self,
        atom_dim: int = ATOM_DIM,
        bond_dim: int = BOND_DIM,
        hidden_dim: int = 256,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        walk_length: int = 20,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        n_descriptors: int = 0,
        readout: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.walk_length = walk_length
        self.use_checkpoint = use_checkpoint
        self.n_descriptors = n_descriptors
        self.readout = readout
        # Widen proj_head for concat readouts so they are pretrainable too. Equal-width
        # readouts (mean/sum/max) leave the parameter count identical to the historical
        # model, so a max/sum run stays parameter-matched to the mean baseline.
        n_pool_blocks = len(_parse_readout(readout))

        # Input projections
        self.node_proj = nn.Linear(atom_dim, hidden_dim)
        self.edge_proj = nn.Linear(bond_dim, hidden_dim)

        # Positional encoding
        self.rwpe = RWPEEncoder(walk_length=walk_length, out_dim=hidden_dim)

        # GPS layers
        self.layers = nn.ModuleList(
            [GPSLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Graph-level projection head.
        # When n_descriptors > 0, normalized RDKit descriptors are concatenated to
        # the pooled GNN embedding before the first Linear. SIGReg only sees the
        # final embed_dim output so this is safe — the descriptors must be
        # z-score normalized (stored in desc_stats.pt) to keep the SIGReg
        # working range intact.
        proj_in = self._proj_in = n_pool_blocks * hidden_dim + n_descriptors
        self.proj_head = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _pool(self, x: Tensor, batch: Tensor) -> Tensor:
        """Graph readout. `self.readout` may be REASSIGNED after construction to probe
        alternatives on a frozen backbone (`backbone.readout = "max"`), which is how
        `scripts/readout_probe_molhiv.py` works. That is only safe when the width still
        matches what proj_head was built for, or when proj_head is bypassed (Identity,
        as embed_split's 'pooled' mode does) — otherwise we raise instead of emitting a
        silently wrong shape."""
        parts = _parse_readout(self.readout)
        if len(parts) * self.hidden_dim + self.n_descriptors != self._proj_in \
                and not isinstance(self.proj_head, nn.Identity):
            raise ValueError(
                f"readout={self.readout!r} yields {len(parts)}*hidden_dim features but "
                f"proj_head was built for proj_in={self._proj_in}. Reassigning .readout "
                "to a different WIDTH is frozen-probe only (bypass proj_head); to train "
                "with it, construct the model with readout= instead.")
        return torch.cat([POOL_FNS[p](x, batch) for p in parts], dim=1)

    def forward(self, data: Batch | Data) -> Tensor:
        """
        Args:
            data: PyG Batch (or single Data) with fields:
                  x, edge_index, edge_attr, batch

        Returns:
            z: (num_graphs, embed_dim)
        """
        x = data.x                          # (total_nodes, atom_dim)
        edge_index = data.edge_index        # (2, total_edges)
        edge_attr = data.edge_attr          # (total_edges, bond_dim)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project inputs to hidden_dim
        x = self.node_proj(x)             # (N, H)
        edge_attr = self.edge_proj(edge_attr)  # (E, H)

        # Add RWPE: compute per-graph, concatenate
        with PROFILER.section("rwpe"):
            pe = self._compute_rwpe_batch(data)  # (N, walk_length)
        x = x + self.rwpe(pe)

        # GPS layers (checkpointing trades ~30% time for ~50% activation memory)
        with PROFILER.section("gps_layers"):
            for layer in self.layers:
                if self.use_checkpoint and self.training:
                    x = checkpoint(layer, x, edge_index, edge_attr, batch, use_reentrant=False)
                else:
                    x = layer(x, edge_index, edge_attr, batch)

        # Readout → (num_graphs, H * n_blocks); 'mean' is the default and the only
        # one compatible with proj_head (which expects hidden_dim in). The others
        # are for FROZEN-PROBE use, where proj_head is bypassed — see _pool.
        graph_emb = self._pool(x, batch)

        # Concat normalized RDKit descriptors if available (set by dataset loader).
        # Shape: global_feat (B, n_desc) was batched from per-molecule [1, n_desc].
        if self.n_descriptors > 0 and hasattr(data, "global_feat") and data.global_feat is not None:
            graph_emb = torch.cat([graph_emb, data.global_feat], dim=1)

        z = self.proj_head(graph_emb)
        return z

    def _compute_rwpe_batch(self, data: Batch | Data) -> Tensor:
        """Vectorized RWPE for every graph in the batch — no Python loop.

        Builds a dense (G, Nmax, Nmax) random-walk transition matrix, raises it
        to powers 1..walk_length with batched matmul, and reads the diagonal
        (k-step return probability) at each step. Purely topological, so it
        runs under no_grad.
        """
        x = data.x
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            adj = to_dense_adj(data.edge_index, batch)          # (G, Nmax, Nmax)
            nmax = adj.size(1)
            adj = adj + torch.eye(nmax, device=adj.device)      # self-loops
            rw = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

            landings = []
            rw_k = rw
            for _ in range(self.walk_length):
                landings.append(torch.diagonal(rw_k, dim1=-2, dim2=-1))  # (G, Nmax)
                rw_k = torch.bmm(rw_k, rw)
            pe_dense = torch.stack(landings, dim=-1)            # (G, Nmax, walk_length)

            # Mask of valid (graph, position) slots, in sparse node order
            mask = to_dense_batch(x[:, :1], batch)[1]           # (G, Nmax) bool
        return pe_dense[mask]                                   # (num_nodes, walk_length)
