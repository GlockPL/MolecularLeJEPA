"""
Sketched Isotropic Gaussian Regularization (SIGReg).

Implements Algorithm 1 from LeJEPA (Balestriero & LeCun, 2025):
  - Uses the Epps-Pulley characteristic function test
  - O(N) time and memory complexity
  - DDP-compatible via all_reduce on empirical characteristic functions
  - Seeds random projections from global_step for reproducibility across ranks

The test statistic measures how far the embedding distribution deviates
from an isotropic standard Gaussian N(0, I_K).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def sigreg(
    x: Tensor,
    global_step: int,
    num_slices: int = 1024,
    t_range: tuple[float, float] = (-5.0, 5.0),
    t_points: int = 17,
) -> Tensor:
    """
    Compute the SIGReg loss for a batch of embeddings.

    Args:
        x:           (N, K) embedding tensor (should not be detached — gradients flow).
        global_step: training step used to seed the random projection directions,
                     ensuring the same directions are used across all DDP ranks.
        num_slices:  number of 1-D random projection directions (|A| in the paper).
        t_range:     integration interval for the Epps-Pulley test.
        t_points:    number of quadrature points for numerical integration.

    Returns:
        Scalar SIGReg loss value.
    """
    N, K = x.shape

    # Sample random unit-norm projection directions, seeded by global_step
    # so all DDP ranks use identical directions.
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn((K, num_slices), generator=g, device=x.device, dtype=x.dtype)
    A = A / A.norm(p=2, dim=0, keepdim=True)  # (K, num_slices) — unit columns

    # Project embeddings: (N, num_slices)
    proj = x @ A  # (N, M)

    # Quadrature points
    t = torch.linspace(t_range[0], t_range[1], t_points, device=x.device, dtype=x.dtype)

    # Empirical characteristic function (ECF): E[exp(i t z)] over the batch
    # proj: (N, M), t: (T,) → x_t: (N, M, T)
    x_t = proj.unsqueeze(2) * t  # (N, M, T)

    # exp(i x_t) = cos(x_t) + i sin(x_t); store as complex
    ecf_real = torch.cos(x_t).mean(0)  # (M, T)
    ecf_imag = torch.sin(x_t).mean(0)  # (M, T)

    # Sync ECF across DDP ranks (average over all GPUs = average over larger batch)
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ecf_real, op=dist.ReduceOp.AVG)
        dist.all_reduce(ecf_imag, op=dist.ReduceOp.AVG)
        world_size = dist.get_world_size()

    # Theoretical CF of N(0,1): phi(t) = exp(-0.5 * t^2)
    phi = torch.exp(-0.5 * t ** 2)  # (T,)

    # Epps-Pulley weighted L2 distance between ECF and target CF
    # Weight function: w(t) = exp(-0.5 t^2) (Gaussian window)
    err_real = (ecf_real - phi) ** 2   # (M, T)
    err_imag = ecf_imag ** 2           # (M, T)  [phi is real-valued, imag part = 0]
    err = (err_real + err_imag) * phi  # (M, T) — weighted

    # Numerical integration via trapezoidal rule over t
    T_stat = torch.trapezoid(err, t, dim=1)  # (M,)

    # Scale by effective batch size (N * world_size) to match paper's N factor
    effective_N = N * world_size
    loss = T_stat.mean() * effective_N
    return loss
