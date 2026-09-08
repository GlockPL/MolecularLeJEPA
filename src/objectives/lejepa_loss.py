"""
LeJEPA combined loss for self-supervised molecular representation learning.

Implements Algorithm 2 from Balestriero & LeCun (2025):

  L_LeJEPA = λ · (1/V) Σ_v SIGReg(z_v) + (1-λ) · (1/B) Σ_n ||μ_n - z_n||²

where:
  - μ_n = mean of global view embeddings for molecule n
  - z_n,v = embedding of view v for molecule n
  - λ balances the SIGReg regularisation vs the prediction objective

No teacher-student network, no stop-gradients, no predictor — this is the
key simplification that LeJEPA's theory enables.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from src.objectives.sigreg import sigreg
from src.profiling import PROFILER


def _stack_batches(batches: list[Batch]) -> Batch:
    """Concatenate V PyG Batch objects into one mega-Batch without to_data_list overhead."""
    node_offset = 0
    graph_offset = 0
    xs, edge_indices, edge_attrs, batch_vecs = [], [], [], []

    for b in batches:
        xs.append(b.x)
        edge_indices.append(b.edge_index + node_offset)
        if b.edge_attr is not None:
            edge_attrs.append(b.edge_attr)
        batch_vecs.append(b.batch + graph_offset)
        node_offset += b.num_nodes
        graph_offset += b.num_graphs

    out = Batch()
    out.x = torch.cat(xs)
    out.edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs:
        out.edge_attr = torch.cat(edge_attrs)
    out.batch = torch.cat(batch_vecs)
    out._num_graphs = graph_offset
    # ptr kept for PyG Batch consistency (pooling/RWPE now use `batch` directly)
    counts = torch.bincount(out.batch, minlength=graph_offset)
    out.ptr = torch.cat([out.batch.new_zeros(1), counts.cumsum(dim=0)])
    # Pass through graph-level descriptor features if present (B, n_desc)
    if hasattr(batches[0], "global_feat") and batches[0].global_feat is not None:
        out.global_feat = torch.cat([b.global_feat for b in batches], dim=0)
    return out


@torch.no_grad()
def embedding_stats(z: Tensor) -> dict[str, float]:
    """Cheap collapse-detection metrics on a (B, K) embedding batch.

    Healthy isotropic-Gaussian embeddings have eff_rank near K, mean_abs near 0,
    std near 1. A falling eff_rank is the earliest sign of dimensional collapse.
    """
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / z.size(0)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = eigvals / eigvals.sum()
    eff_rank = torch.exp(-(p * p.log()).sum())
    return {
        "eff_rank": eff_rank.item(),
        "mean_abs": z.mean(dim=0).abs().mean().item(),
        "std": z.std(dim=0).mean().item(),
    }


def lejepa_loss(
    all_batches: list[Batch],
    encoder: nn.Module,
    lambd: float,
    global_step: int,
    Vg: int = 2,
    num_slices: int = 1024,
    chunk_size: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the LeJEPA loss for a batch of molecules.

    Args:
        all_batches: list of (Vg + Vl) PyG Batch objects.
                     First Vg are global views; remaining Vl are local views.
        encoder:     GPS Transformer (or any encoder) mapping Batch → (B, K).
        lambd:       λ ∈ [0, 1] — weight on SIGReg term.
        global_step: used by SIGReg to seed random projections consistently.
        Vg:          number of global views (first Vg items in all_batches).
        num_slices:  SIGReg projection dimension.

    Returns:
        total_loss:  scalar LeJEPA loss.
        pred_loss:   scalar prediction loss (for logging).
        sig_loss:    scalar SIGReg loss (for logging).
        stats:       dict of embedding health metrics (eff_rank, mean_abs, std).
    """
    n_views = len(all_batches)
    B = all_batches[0].num_graphs

    # Encode views in chunks to bound peak GPU memory from to_dense_batch.
    # chunk_size=1 → one view at a time (safest); higher → fewer kernel launches.
    embeddings: list[Tensor] = []
    for i in range(0, n_views, chunk_size):
        chunk = all_batches[i : i + chunk_size]
        with PROFILER.section("stack_batches"):
            mega = _stack_batches(chunk)
        chunk_z = encoder(mega)  # (chunk*B, K) — times rwpe/gps_layers internally
        embeddings.extend(chunk_z.split(B, dim=0))

    # Global view embeddings: (Vg, B, K)
    global_embs = torch.stack(embeddings[:Vg], dim=0)

    # Centroid of global views: (B, K)
    mu = global_embs.mean(dim=0)

    # Prediction loss: mean squared distance from centroid to each view embedding
    per_view = [((mu - z) ** 2).mean() for z in embeddings]
    pred_loss = torch.stack(per_view).mean()

    # SIGReg loss: average over all views
    with PROFILER.section("sigreg"):
        sig_loss = torch.stack(
            [sigreg(z, global_step, num_slices=num_slices) for z in embeddings]
        ).mean()

    total_loss = (1.0 - lambd) * pred_loss + lambd * sig_loss
    stats = embedding_stats(embeddings[0])
    # Phase-7 instrumentation: split pred_loss into the (near-trivial) global-view
    # component vs the local-view (part→whole) component. Expect global ≈ 0 and
    # local to carry the task — a rising local pred_loss when cover_frac drops
    # confirms the invariance task is getting harder.
    stats["pred_global"] = torch.stack(per_view[:Vg]).mean().item()
    stats["pred_local"] = (
        torch.stack(per_view[Vg:]).mean().item() if len(per_view) > Vg else 0.0
    )
    return total_loss, pred_loss, sig_loss, stats
