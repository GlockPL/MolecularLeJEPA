"""
Evaluate an ensemble of finetuned models on the val or test set.

Discovers all member_*/best.pt checkpoints under --ensemble-dir, runs
inference with each model, averages predicted probabilities, then computes
AUPRC with 95% bootstrap CI.

Optionally prints individual-member AUPRC first so you can see how much the
ensemble actually helps (--individual flag).

Usage:
    uv run python scripts/ensemble_eval.py \\
        --config configs/finetune.yaml \\
        --ensemble-dir checkpoints/ensemble

    # also show per-member results:
    uv run python scripts/ensemble_eval.py \\
        --config configs/finetune.yaml \\
        --ensemble-dir checkpoints/ensemble \\
        --individual

    # evaluate on val set instead:
    uv run python scripts/ensemble_eval.py \\
        --config configs/finetune.yaml \\
        --ensemble-dir checkpoints/ensemble \\
        --split val
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from omegaconf import OmegaConf

from src.data.antibiotic_dataset import AntibioticDataset
from src.models.gps_transformer import GPSTransformer
from src.models.heads import AntibioticHeads
from src.evaluate import compute_auprc
from src.finetune import load_backbone
from src.logutil import RunLogger


@torch.no_grad()
def _collect_probs(
    backbone: GPSTransformer,
    heads: AntibioticHeads,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    heads.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        z = backbone(batch)
        if hasattr(batch, "global_feat"):
            z = torch.cat([z, batch.global_feat], dim=1)
        # predict_proba returns (B, len(heads.tasks)); slice the (B, 4) label
        # tensor to the same active task columns so probs and targets align.
        all_probs.append(heads.predict_proba(z).cpu())
        all_targets.append(batch.y[:, heads.task_cols].cpu())
    return torch.cat(all_probs, 0), torch.cat(all_targets, 0)


def _load_member(
    ckpt_path: Path,
    cfg,
    device: torch.device,
) -> tuple[GPSTransformer, AntibioticHeads]:
    # Build architecture only (no pretrain weights — we load best.pt directly).
    backbone = load_backbone(None, cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Rebuild the exact head this member was trained with — task set, input dim,
    # width and depth are all inferred from the saved weights, so any head
    # capacity (incl. a wide/deep capacity-sweep head) loads correctly. Falls
    # back to all four tasks for legacy checkpoints with no "tasks" key.
    heads = AntibioticHeads.from_state_dict(
        ckpt["heads"], tasks=ckpt.get("tasks"),
    ).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    heads.load_state_dict(ckpt["heads"])
    return backbone, heads


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a finetuning ensemble")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--ensemble-dir", type=str, required=True,
                        help="Directory containing member_NN/ subdirectories")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--individual", action="store_true",
                        help="Print per-member AUPRC in addition to ensemble")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for the auto-named run log (default: logs/)")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Override cfg.data.seed (split seed). MUST match the "
                             "--data-seed the ensemble was trained with, so the test "
                             "set is identical.")
    parser.add_argument("--partition-seed", type=int, default=None,
                        help="Override cfg.data.scaffold_partition_seed. MUST match the "
                             "--partition-seed the ensemble was trained with, so the "
                             "scaffold test set is identical.")
    parser.add_argument("--save-probs", nargs="?", const="", default=None,
                        help="Save per-compound ensemble probabilities + labels to CSV "
                             "(for a paired bootstrap vs another model on the SAME test "
                             "set). Optional path; default "
                             "<ensemble-dir>/ensemble_probs_<split>.csv")
    args = parser.parse_args()

    # Tee all stdout/stderr (ours + compute_auprc's internal prints) to a run log.
    ens_tag = Path(args.ensemble_dir).name
    log = RunLogger(
        f"ensemble_eval_{ens_tag}_{args.split}",
        log_dir=args.log_dir,
        title="AntibioticJEPA — Ensemble evaluation",
        ensemble_dir=args.ensemble_dir,
        config=args.config,
        split=args.split,
        data_seed=args.data_seed,
    ).start()

    cfg = OmegaConf.load(args.config)
    if args.data_seed is not None:
        cfg.data.seed = args.data_seed
        print(f"Split (data) seed overridden to {args.data_seed}\n")
    if args.partition_seed is not None:
        cfg.data.scaffold_partition_seed = args.partition_seed
        print(f"Scaffold partition seed overridden to {args.partition_seed}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover member checkpoints. finetune.py saves the best model as
    # best_<task_tag>.pt (e.g. best_all.pt); older runs used best.pt. Prefer
    # the legacy name, otherwise take the single best_*.pt in the member dir.
    def _member_ckpt(d: Path) -> Path | None:
        legacy = d / "best.pt"
        if legacy.exists():
            return legacy
        tagged = sorted(d.glob("best_*.pt"))
        return tagged[0] if tagged else None

    ensemble_dir = Path(args.ensemble_dir)
    member_dirs = sorted(ensemble_dir.glob("member_*/"))
    ckpt_paths = [p for d in member_dirs if (p := _member_ckpt(d)) is not None]
    missing = [d for d in member_dirs if _member_ckpt(d) is None]
    for d in missing:
        print(f"  Warning: no best.pt / best_*.pt in {d}, skipping")

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No member_*/best.pt found in {ensemble_dir}\n"
            f"Run: uv run python scripts/train_ensemble.py --out {ensemble_dir}"
        )

    print(f"Found {len(ckpt_paths)} member checkpoints in {ensemble_dir}")

    # Load descriptor normalization stats from the first checkpoint (same for all members).
    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=True)
    desc_stats = None
    if "desc_mean" in first_ckpt:
        desc_stats = (first_ckpt["desc_mean"], first_ckpt["desc_std"])
        print(f"Descriptor stats loaded from checkpoint ({desc_stats[0].shape[0]} features)")

    # CRUCIAL: rebuild the SAME compound set + split as training. A single-task
    # ensemble (e.g. --task antibiotic) was split over the ~39k antibiotic
    # compounds; defaulting tasks=None here would inner-join all 4 tasks (~12k)
    # and split THAT — a different test set. The trained task list is stored in
    # the checkpoint, so use it (None → legacy 4-task, matching old behavior).
    ckpt_tasks = first_ckpt.get("tasks")
    if ckpt_tasks is not None:
        print(f"Tasks (from checkpoint): {list(ckpt_tasks)}")

    # Build dataset (same split seed/method/tasks/featurizer as training → same test set).
    data_cfg = cfg.data
    common = dict(
        test_size=data_cfg.get("test_size", 0.20),
        val_size=data_cfg.get("val_size", 0.10),
        seed=data_cfg.get("seed", 42),
        split_method=data_cfg.get("split_method", "scaffold"),
        tasks=list(ckpt_tasks) if ckpt_tasks is not None else None,
        featurizer=data_cfg.get("featurizer", "ours"),
        scaffold_partition_seed=data_cfg.get("scaffold_partition_seed", None),
    )
    ds = AntibioticDataset(data_cfg.xlsx_path, split=args.split,
                           desc_stats=desc_stats, **common)
    loader = DataLoader(
        ds, batch_size=cfg.training.get("batch_size", 64),
        shuffle=False, num_workers=4, collate_fn=Batch.from_data_list,
    )
    print(f"Evaluating on {args.split} set ({len(ds)} compounds)\n")

    # Collect per-member probabilities
    all_probs: list[torch.Tensor] = []
    targets: torch.Tensor | None = None
    task_names: list[str] | None = None  # active tasks (same across members)

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Running member {i:02d}: {ckpt_path}")
        backbone, heads = _load_member(ckpt_path, cfg, device)
        probs, tgts = _collect_probs(backbone, heads, loader, device)
        all_probs.append(probs)
        if targets is None:
            targets = tgts
            task_names = list(heads.tasks)

        if args.individual:
            print(f"  Member {i:02d} AUPRC:")
            compute_auprc(probs, targets, task_names=task_names)
            print()

        del backbone, heads
        torch.cuda.empty_cache()

    # Ensemble: average predicted probabilities across all members
    ensemble_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (N, 4)

    print(f"\n{'='*64}")
    print(f"Ensemble ({len(ckpt_paths)} members) — {args.split} set AUPRC:")
    print(f"{'='*64}")
    compute_auprc(ensemble_probs, targets, task_names=task_names)

    if args.save_probs is not None:
        # SMILES in dataset order (loader is shuffle=False, so probs/targets align).
        smiles = [str(ds[i].smiles) for i in range(len(ds))]
        assert len(smiles) == ensemble_probs.size(0), (
            f"smiles ({len(smiles)}) != probs ({ensemble_probs.size(0)})"
        )
        probs_out = (
            Path(args.save_probs) if args.save_probs
            else Path(args.ensemble_dir) / f"ensemble_probs_{args.split}.csv"
        )
        probs_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["smiles"]
        for t in task_names:
            cols += [f"{t}_prob", f"{t}_label"]
        with open(probs_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i, smi in enumerate(smiles):
                row = [smi]
                for ti in range(len(task_names)):
                    row += [f"{ensemble_probs[i, ti].item():.6f}",
                            int(targets[i, ti].item())]
                w.writerow(row)
        print(f"\nSaved per-compound ensemble probs → {probs_out}  ({len(smiles)} rows)")

    log.stop()


if __name__ == "__main__":
    main()