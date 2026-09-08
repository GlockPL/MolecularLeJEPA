"""
Directed Message Passing Neural Network (D-MPNN) encoder, native PyG.

A from-scratch reimplementation of Chemprop's D-MPNN (Yang et al. 2019,
"Analyzing Learned Molecular Representations for Property Prediction"), built as a
*drop-in* replacement for ``GPSTransformer``: identical constructor surface,
identical ``(Batch | Data) -> (B, embed_dim)`` forward, and identical descriptor
handling (normalized RDKit descriptors concatenated to the pooled embedding before
the projection head). The ONLY thing that differs from the GPS backbone is the
message-passing core — so swapping it in isolates the effect of *architecture*
while holding the featurization, descriptor pipeline, heads, and finetune harness
constant (Phase 9b in PLAN.md).

D-MPNN specifics:
  - Messages live on DIRECTED bonds, not atoms.
  - h0_{v->w} = ReLU(W_i · [x_v ‖ e_{vw}])               (init from SOURCE atom)
  - m_{v->w}  = Σ_{k∈N(v)\\w} h_{k->v}                     (exclude reverse edge)
  - h_{v->w}  = ReLU(h0_{v->w} + W_m · m_{v->w})          (residual to h0)
  - m_v       = Σ_{k∈N(v)} h_{k->v},  h_v = ReLU(W_a·[x_v ‖ m_v])
  - molecule  = mean-pool(h_v)  → (+descriptors) → proj_head

The "exclude reverse edge" message is just ``agg_at_v - h[reverse_edge]``. Our
featurizer (``featurize.mol_to_data`` / ``compact_to_data``) stores each bond as a
consecutive ``(i->j, j->i)`` pair, and ``Batch.from_data_list`` concatenates graphs
while preserving that pairing, so the reverse-edge map over the *batched*
edge_index is simply ``arange(E) ^ 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

from src.data.featurize import ATOM_DIM, BOND_DIM


class DMPNNEncoder(nn.Module):
    """Chemprop-style D-MPNN encoder with the GPSTransformer interface.

    Args mirror the GPS backbone so ``load_backbone`` can construct either from
    the same config block. GPS-only kwargs (``num_heads``, ``walk_length``,
    ``use_checkpoint``) are accepted and ignored via ``**_unused``.

    Args:
        atom_dim:      raw atom feature dim (featurize.ATOM_DIM).
        bond_dim:      raw bond feature dim (featurize.BOND_DIM).
        hidden_dim:    message / atom hidden width (Chemprop ``hidden_size``, 300).
        embed_dim:     final embedding dim (proj_head output). Keep <= hidden_dim.
        num_layers:    D-MPNN message-passing depth (Chemprop ``depth``, 3).
        dropout:       dropout on messages / atom readout (Chemprop default 0.0).
        n_descriptors: if > 0, concat data.global_feat to the pooled embedding
                       before the proj_head — exactly as GPSTransformer does.
    """

    def __init__(
        self,
        atom_dim: int = ATOM_DIM,
        bond_dim: int = BOND_DIM,
        hidden_dim: int = 300,
        embed_dim: int = 300,
        num_layers: int = 3,
        dropout: float = 0.0,
        n_descriptors: int = 0,
        use_proj_head: bool = True,
        **_unused,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = max(1, num_layers)
        self.n_descriptors = n_descriptors
        self.use_proj_head = use_proj_head

        # Directed-edge message network.
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Atom readout from aggregated incoming messages + raw atom features.
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if use_proj_head:
            # GPSTransformer-style head: concat descriptors to the pooled embedding,
            # then Linear→GELU→Linear to embed_dim (the LeJEPA-compatible path).
            proj_in = hidden_dim + n_descriptors
            self.proj_head = nn.Sequential(
                nn.Linear(proj_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.embed_dim = embed_dim
        else:
            # FAITHFUL-Chemprop mode: return the raw pooled graph embedding
            # (hidden_dim). No proj_head bottleneck, no internal descriptor concat —
            # the finetune harness concatenates descriptors ONCE into the FFN heads,
            # giving Chemprop's exact `D-MPNN(hidden) ⊕ features → FFN` structure.
            self.proj_head = None
            self.embed_dim = hidden_dim

    def forward(self, data: Batch | Data) -> Tensor:
        x = data.x                       # (N, atom_dim)
        edge_index = data.edge_index     # (2, E), consecutive (i->j, j->i) pairs
        edge_attr = data.edge_attr       # (E, bond_dim)
        batch = data.batch if getattr(data, "batch", None) is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        N = x.size(0)
        E = edge_index.size(1)

        if E == 0:
            # Batch of bond-less molecules (e.g. single atoms): no messages.
            agg = x.new_zeros((N, self.hidden_dim))
        else:
            src, dst = edge_index[0], edge_index[1]
            rev = torch.arange(E, device=x.device) ^ 1          # reverse-edge index

            # Init directed-edge hidden from the SOURCE atom + bond features.
            h0 = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=1)))   # (E, H)
            h = h0
            for _ in range(self.depth - 1):
                # Sum of edge hiddens incoming to each atom v, then the per-edge
                # message into (v->w): incoming to v minus the reverse edge (w->v).
                agg_v = scatter(h, dst, dim=0, dim_size=N, reduce="sum")      # (N, H)
                m = agg_v[src] - h[rev]                                       # (E, H)
                h = self.dropout(self.act(h0 + self.W_m(m)))
            # Atom-level aggregation of final incoming edge messages.
            agg = scatter(h, dst, dim=0, dim_size=N, reduce="sum")           # (N, H)

        h_atom = self.dropout(self.act(self.W_a(torch.cat([x, agg], dim=1))))  # (N, H)
        graph_emb = global_mean_pool(h_atom, batch)                           # (B, H)

        if not self.use_proj_head:
            # Raw pooled graph embedding; the harness concatenates descriptors
            # once into the FFN heads (Chemprop-faithful).
            return graph_emb

        # Concat normalized RDKit descriptors if present (same as GPSTransformer).
        if self.n_descriptors > 0 and getattr(data, "global_feat", None) is not None:
            graph_emb = torch.cat([graph_emb, data.global_feat], dim=1)

        return self.proj_head(graph_emb)