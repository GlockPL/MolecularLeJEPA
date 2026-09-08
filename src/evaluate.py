"""
Evaluation utilities: AUPRC with bootstrap confidence intervals,
precision-recall curves, and virtual screening ranking.

Matches the evaluation methodology of Wong et al. 2023:
- AUPRC (area under precision-recall curve) as primary metric
- 95% CI via 100 bootstrap subsamples (equal-sized to test set with replacement)
- One curve per task
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve


def compute_auprc(
    probs: torch.Tensor,
    targets: torch.Tensor,
    task_names: list[str],
    n_bootstrap: int = 100,
) -> dict[str, float]:
    """
    Compute AUPRC for each task, with 95% bootstrap CI.

    Args:
        probs:      (N, num_tasks) predicted probabilities in [0, 1].
        targets:    (N, num_tasks) binary ground-truth labels.
        task_names: list of task names (length = num_tasks).
        n_bootstrap: number of bootstrap replicates for CI estimation.

    Returns:
        dict mapping task name → AUPRC value (point estimate).
        Also prints 95% CIs to stdout.
    """
    probs_np = probs.numpy()
    targets_np = targets.numpy()
    results: dict[str, float] = {}

    for t_idx, name in enumerate(task_names):
        y_true = targets_np[:, t_idx]
        y_score = probs_np[:, t_idx]

        # Point estimate
        auprc = average_precision_score(y_true, y_score)
        results[name] = auprc

        # Bootstrap CI
        n = len(y_true)
        rng = np.random.default_rng(seed=42)
        boot_auprcs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if y_true[idx].sum() == 0:
                continue  # skip bootstrap samples with no positives
            boot_auprcs.append(average_precision_score(y_true[idx], y_score[idx]))

        if boot_auprcs:
            ci_lo = np.percentile(boot_auprcs, 2.5)
            ci_hi = np.percentile(boot_auprcs, 97.5)
            print(f"  {name}: AUPRC={auprc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f"  {name}: AUPRC={auprc:.4f}  (CI unavailable)")

    return results


def precision_recall_curves_data(
    probs: torch.Tensor,
    targets: torch.Tensor,
    task_names: list[str],
) -> dict[str, dict]:
    """
    Compute precision-recall curve data for all tasks.

    Returns:
        dict mapping task name → {"precision": array, "recall": array, "thresholds": array}
    """
    probs_np = probs.numpy()
    targets_np = targets.numpy()
    curves = {}
    for t_idx, name in enumerate(task_names):
        p, r, thresh = precision_recall_curve(targets_np[:, t_idx], probs_np[:, t_idx])
        curves[name] = {"precision": p, "recall": r, "thresholds": thresh}
    return curves


@torch.no_grad()
def screen_library(
    smiles_list: list[str],
    backbone,
    heads,
    device: torch.device,
    batch_size: int = 512,
    antibiotic_threshold: float = 0.4,
    cytotox_threshold: float = 0.2,
) -> list[dict]:
    """
    Virtual screening: rank a compound library by predicted antibiotic activity,
    filtering out high-cytotoxicity compounds.

    Args:
        smiles_list:          list of SMILES strings to screen.
        backbone:             pretrained GPSTransformer.
        heads:                AntibioticHeads.
        device:               torch device.
        batch_size:           inference batch size.
        antibiotic_threshold: minimum antibiotic score to retain a compound.
        cytotox_threshold:    maximum cytotoxicity score (any cell line) to retain.

    Returns:
        List of dicts sorted by descending antibiotic score, each with:
        {smiles, antibiotic, hepg2, hskmc, imr90}
    """
    from src.data.featurize import smiles_to_data
    from src.models.heads import AntibioticHeads
    from torch_geometric.data import Batch

    backbone.eval()
    heads.eval()

    results = []
    buf: list = []

    def _flush():
        if not buf:
            return
        batch = Batch.from_data_list([d for d, _ in buf]).to(device)
        smi_batch = [s for _, s in buf]
        z = backbone(batch)
        probs = heads.predict_proba(z).cpu()  # (B, len(heads.tasks))
        for i, smi in enumerate(smi_batch):
            p = probs[i]
            row = {"smiles": smi}
            for j, name in enumerate(heads.tasks):
                row[name] = p[j].item()
            results.append(row)
        buf.clear()

    for smi in smiles_list:
        data = smiles_to_data(smi)
        if data is None:
            continue
        buf.append((data, smi))
        if len(buf) >= batch_size:
            _flush()
    _flush()

    # Filter on whichever scores are present — task-restricted checkpoints
    # (e.g. antibiotic-only) skip the cytotox filters they cannot score.
    def _passes(r):
        if "antibiotic" in r and r["antibiotic"] < antibiotic_threshold:
            return False
        for cyto in ("hepg2", "hskmc", "imr90"):
            if cyto in r and r[cyto] >= cytotox_threshold:
                return False
        return True

    hits = [r for r in results if _passes(r)]
    hits.sort(key=lambda r: r.get("antibiotic", 0.0), reverse=True)
    return hits
