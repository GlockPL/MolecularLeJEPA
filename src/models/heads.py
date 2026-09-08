"""
Binary classification heads for antibiotic activity and cytotoxicity.

Independent binary MLP heads, one per task. Full task list (TASK_NAMES) also
defines the column order of the dataset's y tensor:
  - antibiotic: activity against S. aureus RN4220
  - hepg2:      cytotoxicity in HepG2 liver carcinoma cells
  - hskmc:      cytotoxicity in human skeletal muscle cells
  - imr90:      cytotoxicity in IMR-90 lung fibroblasts

Pass `tasks=` to train only a subset. Wong et al. 2023 trained one Chemprop
model per task rather than a single multi-task network, so e.g.
tasks=["antibiotic"] is the more faithful reproduction.

Each head predicts a logit; sigmoid + threshold at 0.5 gives the binary label.
Loss: weighted binary cross-entropy (pos_weight accounts for class imbalance).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """MLP readout: `num_layers` × (Linear → GELU → Dropout), then a final
    Linear to out_dim.

    num_layers is the number of *hidden* blocks. num_layers=1 reproduces the
    original Linear → GELU → Dropout → Linear layout exactly (same submodule
    indices), so checkpoints saved before this became configurable still load.
    Increase it (e.g. 3 × 1600) to approach the capacity of Chemprop's FFN
    readout over the concatenated RDKit descriptors.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AntibioticHeads(nn.Module):
    """
    Binary classification heads operating on backbone embeddings.

    Args:
        embed_dim:  dimension of the backbone embedding (input to heads).
        hidden_dim: hidden layer width in each MLP head.
        dropout:    dropout rate in each head.
        tasks:      optional subset of TASK_NAMES. Defaults to all four
                    (multi-task). Single-task example: tasks=["antibiotic"].
        num_layers: number of hidden blocks per head (default 1). Raise it to
                    grow readout capacity (e.g. num_layers=3, hidden_dim=1600
                    to approach Chemprop's FFN readout).

    Attributes set after init:
        self.tasks     — active task names, in their TASK_NAMES order.
        self.task_cols — index of each active task in the full y tensor.
    """

    TASK_NAMES = ["antibiotic", "hepg2", "hskmc", "imr90"]

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        tasks: Iterable[str] | None = None,
        num_layers: int = 1,
    ):
        super().__init__()
        if tasks is None:
            self.tasks = list(self.TASK_NAMES)
        else:
            self.tasks = list(tasks)
            unknown = [t for t in self.tasks if t not in self.TASK_NAMES]
            if unknown:
                raise ValueError(
                    f"Unknown task(s) {unknown}; must be a subset of {self.TASK_NAMES}"
                )
        # task_cols[i] = column of self.tasks[i] in the dataset's y tensor.
        self.task_cols = [self.TASK_NAMES.index(t) for t in self.tasks]
        self.heads = nn.ModuleDict(
            {name: MLP(embed_dim, hidden_dim, 1, dropout, num_layers)
             for name in self.tasks}
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        tasks: Iterable[str] | None = None,
        dropout: float = 0.1,
    ) -> "AntibioticHeads":
        """Rebuild heads with the exact architecture encoded in a saved
        state_dict, inferring in_dim / hidden_dim / num_layers from the Linear
        weights. Use this at load time so checkpoints trained with any head
        width or depth reconstruct correctly without reading the config.
        """
        import re

        task_list = list(tasks) if tasks is not None else list(cls.TASK_NAMES)
        t0 = task_list[0]
        pat = re.compile(rf"heads\.{re.escape(t0)}\.net\.(\d+)\.weight")
        lin_idx = sorted(
            int(m.group(1)) for k in state_dict if (m := pat.match(k))
        )
        if not lin_idx:
            raise KeyError(
                f"No head Linear weights for task {t0!r} found in state_dict"
            )
        first = state_dict[f"heads.{t0}.net.{lin_idx[0]}.weight"]
        in_dim, hidden_dim = first.shape[1], first.shape[0]
        num_layers = len(lin_idx) - 1  # drop the final out-projection Linear
        return cls(
            embed_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            tasks=task_list,
            num_layers=num_layers,
        )

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """Returns {active task name → (B, 1) logit}."""
        return {name: head(z) for name, head in self.heads.items()}

    def loss(
        self,
        logits: dict[str, Tensor],
        targets: Tensor,
        pos_weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            logits:      dict of {active task name → (B, 1) logit}.
            targets:     (B, len(TASK_NAMES)) full label tensor from the dataset.
            pos_weights: (len(TASK_NAMES),) indexed by full-y column. May be None.
        """
        per_task: dict[str, Tensor] = {}
        for name, col in zip(self.tasks, self.task_cols):
            logit = logits[name].squeeze(1)
            target = targets[:, col]
            pw = None
            if pos_weights is not None:
                pw = pos_weights[col].unsqueeze(0).to(logit.device)
            per_task[name] = nn.functional.binary_cross_entropy_with_logits(
                logit, target, pos_weight=pw
            )
        total = sum(per_task.values())
        return total, per_task

    def predict_proba(self, z: Tensor) -> Tensor:
        """Returns (B, len(self.tasks)) probabilities in self.tasks order."""
        logits = self.forward(z)
        return torch.cat(
            [torch.sigmoid(logits[name]) for name in self.tasks], dim=1
        )
