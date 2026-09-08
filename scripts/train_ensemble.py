"""
Train an ensemble of N finetuned models from the same pretrained backbone.

Each member trains with a different random seed, which controls head weight
initialisation and training stochasticity (dropout masks, batch ordering).
The data-split seed stays fixed at cfg.data.seed so every member sees the
same scaffold-split train / val / test partition.

Usage:
    uv run python scripts/train_ensemble.py \\
        --config configs/finetune.yaml \\
        --checkpoint checkpoints/pretrain_local/final.pt \\
        --n 10 \\
        --out checkpoints/ensemble

    # or supply explicit seeds:
    uv run python scripts/train_ensemble.py \\
        --config configs/finetune.yaml \\
        --checkpoint checkpoints/pretrain_local/final.pt \\
        --seeds 0 1 2 3 4 \\
        --out checkpoints/ensemble

Each member is saved to {out}/member_NN/best.pt.
Evaluate the ensemble afterwards with:
    uv run python scripts/ensemble_eval.py \\
        --config configs/finetune.yaml \\
        --ensemble-dir checkpoints/ensemble
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from omegaconf import OmegaConf

from src.finetune import finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a finetuning ensemble")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Pretrained backbone checkpoint (omit to finetune from scratch)",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of ensemble members (ignored if --seeds is given)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit list of seeds, one per member",
    )
    parser.add_argument(
        "--out", type=str, default="checkpoints/ensemble",
        help="Root output directory; members saved to {out}/member_NN/",
    )
    parser.add_argument(
        "--data-seed", type=int, default=None,
        help="Override cfg.data.seed (the SPLIT seed). Use to train one ensemble "
             "per random split for a multi-seed significance test. Leaves the "
             "per-member model seeds (--seeds/--n) independent of the split seed.",
    )
    parser.add_argument(
        "--partition-seed", type=int, default=None,
        help="Override cfg.data.scaffold_partition_seed: select a DIFFERENT (still "
             "leakage-free) scaffold partition for a multi-partition robustness "
             "check. None (default) = the canonical deterministic partition.",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "antibiotic", "hepg2", "hskmc", "imr90"],
        help="Single task (Wong et al. style — one model per task, e.g. "
             "'antibiotic' to match the Chemprop antibiotic-only ensemble) or "
             "'all' for the multi-task model (default).",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n))
    cfg_base = OmegaConf.load(args.config)
    if args.data_seed is not None:
        cfg_base.data.seed = args.data_seed
    if args.partition_seed is not None:
        cfg_base.data.scaffold_partition_seed = args.partition_seed

    print(f"Ensemble training: {len(seeds)} members  seeds={seeds}")
    print(f"Backbone checkpoint: {args.checkpoint or '(none — scratch)'}")
    print(f"Task(s): {args.task}")
    print(f"Split (data) seed: {cfg_base.data.seed}")
    print(f"Output root: {args.out}\n")

    t0_total = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n{'='*64}")
        print(f"  Member {i:02d}/{len(seeds)}  seed={seed}")
        print(f"{'='*64}")
        t0 = time.time()

        # Deep-copy config so mutations don't carry over between members.
        cfg = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=True))
        cfg.checkpoint_dir = str(Path(args.out) / f"member_{i:02d}")

        finetune(cfg, args.checkpoint, seed=seed, tasks=args.task)

        elapsed = time.time() - t0
        print(f"  Member {i:02d} done in {elapsed/60:.1f} min")

    total = time.time() - t0_total
    print(f"\nAll {len(seeds)} members trained in {total/3600:.2f} h")
    print(f"Evaluate with:")
    print(f"  uv run python scripts/ensemble_eval.py \\")
    print(f"      --config {args.config} \\")
    print(f"      --ensemble-dir {args.out}")


if __name__ == "__main__":
    main()