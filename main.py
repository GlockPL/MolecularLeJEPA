"""
AntibioticJEPA — CLI entry point.

Commands:
  pretrain   Pretrain GPS Graph Transformer on ZINC20 using LeJEPA
  finetune   Finetune on the Wong et al. 2023 antibiotic dataset
  screen     Virtual screen a SMILES file with a finetuned model

Examples:
  # Single-GPU pretraining
  python main.py pretrain --config configs/pretrain.yaml

  # Multi-GPU pretraining (HPC)
  torchrun --nproc_per_node=8 main.py pretrain --config configs/pretrain.yaml

  # Finetuning from a pretrained checkpoint
  python main.py finetune --config configs/finetune.yaml --checkpoint checkpoints/pretrain/final.pt

  # Virtual screening
  python main.py screen --smiles library.smi \\
      --checkpoint checkpoints/finetune/best.pt \\
      --config configs/finetune.yaml \\
      --out hits.csv
"""

from __future__ import annotations

import argparse
import sys


def cmd_pretrain(args):
    from src.pretrain import main as pretrain_main
    argv = ["pretrain", "--config", args.config]
    if args.resume:
        argv += ["--resume", args.resume]
    sys.argv = argv
    pretrain_main()


def cmd_finetune(args):
    from src.finetune import main as finetune_main
    sys.argv = ["finetune", "--config", args.config,
                *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
                "--task", args.task,
                "--seed", str(args.seed),
                "--train-frac", str(args.train_frac),
                "--fewshot-seed", str(args.fewshot_seed)]
    finetune_main()


def cmd_screen(args):
    import torch
    import pandas as pd
    from omegaconf import OmegaConf
    from src.models.gps_transformer import GPSTransformer
    from src.models.heads import AntibioticHeads
    from src.evaluate import screen_library

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = GPSTransformer(
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Reconstruct head architecture (task set, input dim, width, depth) from the
    # saved weights. tasks key is absent in old checkpoints → defaults to all 4.
    heads = AntibioticHeads.from_state_dict(
        ckpt["heads"], tasks=ckpt.get("tasks"),
    ).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    heads.load_state_dict(ckpt["heads"])

    with open(args.smiles) as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]

    print(f"Screening {len(smiles_list)} compounds...")
    hits = screen_library(
        smiles_list, backbone, heads, device,
        antibiotic_threshold=args.antibiotic_threshold,
        cytotox_threshold=args.cytotox_threshold,
    )
    print(f"Found {len(hits)} hits passing filters.")

    df = pd.DataFrame(hits)
    df.to_csv(args.out, index=False)
    print(f"Saved hits to {args.out}")


def main():
    parser = argparse.ArgumentParser(
        prog="antibioticjepa",
        description="Graph Transformer pretrained with LeJEPA for antibiotic discovery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # pretrain
    p_pt = sub.add_parser("pretrain", help="Pretrain on ZINC20 with LeJEPA")
    p_pt.add_argument("--config", default="configs/pretrain.yaml")
    p_pt.add_argument("--resume", default=None,
                      help="Checkpoint path to resume from, or 'auto' for latest in checkpoint_dir")

    # finetune
    p_ft = sub.add_parser("finetune", help="Finetune on antibiotic dataset")
    p_ft.add_argument("--config", default="configs/finetune.yaml")
    p_ft.add_argument("--checkpoint", default=None, help="Pretrained backbone checkpoint")
    p_ft.add_argument("--task", default="all",
                      choices=["all", "antibiotic", "hepg2", "hskmc", "imr90"],
                      help="Single task name (Wong et al. style — one model per task) "
                           "or 'all' for the multi-task model (default).")
    p_ft.add_argument("--train-frac", type=float, default=1.0,
                      help="Few-shot: fraction of TRAIN to keep (stratified; val/test fixed)")
    p_ft.add_argument("--fewshot-seed", type=int, default=42,
                      help="Seed for the few-shot train subsample")
    p_ft.add_argument("--seed", type=int, default=42,
                      help="Random seed for head init and training stochasticity")

    # screen
    p_sc = sub.add_parser("screen", help="Virtual screening of a SMILES library")
    p_sc.add_argument("--smiles", required=True, help="SMILES file (one per line)")
    p_sc.add_argument("--checkpoint", required=True, help="Finetuned model checkpoint")
    p_sc.add_argument("--config", default="configs/finetune.yaml")
    p_sc.add_argument("--out", default="hits.csv", help="Output CSV path")
    p_sc.add_argument("--antibiotic-threshold", type=float, default=0.4)
    p_sc.add_argument("--cytotox-threshold", type=float, default=0.2)

    args = parser.parse_args()

    if args.command == "pretrain":
        cmd_pretrain(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "screen":
        cmd_screen(args)


if __name__ == "__main__":
    main()
