"""
LeJEPA pretraining on ZINC20.

Launch with torchrun for multi-GPU DDP:
    torchrun --nproc_per_node=<N_GPUS> -m src.pretrain --config configs/pretrain.yaml

Or single-GPU:
    python -m src.pretrain --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing

# Avoid "received 0 items of ancdata" — FDs explode when many tensors per batch
# (large batch × Vg+Vl views × num_workers) outpace the per-process FD limit.
torch.multiprocessing.set_sharing_strategy("file_system")

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.data.zinc20_dataset import ZINC20Dataset, zinc20_collate
from src.data.zinc20_cached_dataset import ZINC20CachedDataset
from src.models.gps_transformer import GPSTransformer
from src.models.dmpnn import DMPNNEncoder
from src.data.featurize import ATOM_DIM as _DEF_ATOM_DIM, BOND_DIM as _DEF_BOND_DIM
from src.objectives.lejepa_loss import lejepa_loss
from src.profiling import PROFILER


def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def build_model(cfg: DictConfig) -> torch.nn.Module:
    # model.backbone selects the encoder; defaults to 'gps' so existing pretrain
    # configs are unchanged. 'dmpnn' = native PyG Chemprop-style D-MPNN (use_proj_head
    # MUST stay True for pretraining — SIGReg operates on the proj_head's embed_dim
    # output, and descriptors concat there on global views, same as the GPS path).
    backbone = cfg.model.get("backbone", "gps")
    if backbone == "gps":
        return GPSTransformer(
            hidden_dim=cfg.model.hidden_dim,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            walk_length=cfg.model.get("walk_length", 20),
            dropout=cfg.model.get("dropout", 0.1),
            use_checkpoint=cfg.model.get("use_checkpoint", False),
            n_descriptors=cfg.model.get("n_descriptors", 0),
        )
    elif backbone == "dmpnn":
        return DMPNNEncoder(
            atom_dim=cfg.model.get("atom_dim", _DEF_ATOM_DIM),
            bond_dim=cfg.model.get("bond_dim", _DEF_BOND_DIM),
            hidden_dim=cfg.model.hidden_dim,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.get("dropout", 0.0),
            n_descriptors=cfg.model.get("n_descriptors", 0),
            use_proj_head=cfg.model.get("use_proj_head", True),
        )
    raise ValueError(f"Unknown model.backbone {backbone!r} (use 'gps' or 'dmpnn')")


def build_scheduler(optimizer, cfg: DictConfig, steps_per_epoch: int):
    warmup_steps = int(cfg.training.warmup_epochs * steps_per_epoch)
    total_steps = cfg.training.epochs * steps_per_epoch

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _unwrap_model(model):
    """Strip DDP and torch.compile wrappers to get the raw nn.Module."""
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile wrapper
        model = model._orig_mod
    return model


def save_checkpoint(model, optimizer, scheduler, step: int, epoch: int, path: Path):
    raw = _unwrap_model(model)
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def _resolve_resume_path(resume_arg: str, checkpoint_dir: Path) -> Path:
    """Resolve --resume <path> or --resume auto to a concrete checkpoint file."""
    if resume_arg == "auto":
        candidates = sorted(checkpoint_dir.glob("step_*.pt")) + sorted(checkpoint_dir.glob("epoch_*.pt"))
        final = checkpoint_dir / "final.pt"
        if final.exists():
            candidates.append(final)
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return candidates[-1]
    return Path(resume_arg)


def train(cfg: DictConfig, resume_path: str | None = None):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # File logging on the MAIN RANK ONLY (DDP) — captures the full pretrain
    # stdout/stderr (incl. per-step pred_g/pred_l/rank) to a timestamped,
    # self-documenting log so a run is easy to share/inspect afterwards.
    if is_main(rank):
        import atexit
        from src.logutil import RunLogger
        _plog = RunLogger(
            f"pretrain_{Path(cfg.checkpoint_dir).name}",
            title=f"LeJEPA pretraining — {cfg.get('project', 'antibioticjepa')}",
            backbone=cfg.model.get("backbone", "gps"),
            hidden_dim=cfg.model.get("hidden_dim"),
            embed_dim=cfg.model.get("embed_dim"),
            lambd=cfg.jepa.lambd, Vg=cfg.jepa.Vg, Vl=cfg.jepa.Vl,
            cover_frac=cfg.jepa.get("cover_frac", 0.6),
            edge_corruption=cfg.jepa.get("edge_corruption", "dropout"),
            desc_global_views=cfg.jepa.get("desc_global_views", None),
            batch_size=cfg.training.batch_size,
            world_size=world_size,
            checkpoint_dir=cfg.checkpoint_dir,
        ).start()
        atexit.register(_plog.stop)

    # Logging (only main rank)
    use_wandb = cfg.get("wandb", False) and is_main(rank)
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("project", "antibioticjepa"), config=OmegaConf.to_container(cfg))

    # Dataset & DataLoader. cache_dir set -> read precomputed compact shards
    # (no RDKit in the loader); otherwise parse .smi files on the fly.
    cache_dir = cfg.data.get("cache_dir", None)
    common = dict(
        Vg=cfg.jepa.Vg,
        Vl=cfg.jepa.Vl,
        max_atoms=cfg.data.get("max_atoms", 100),
        max_mols=cfg.data.get("max_mols", None),
        shuffle_buffer=cfg.data.get("shuffle_buffer", 100_000),
        seed=cfg.get("seed", 0),
        cover_frac=cfg.jepa.get("cover_frac", 0.6),
        edge_corruption=cfg.jepa.get("edge_corruption", "dropout"),
    )
    if cache_dir:
        dataset = ZINC20CachedDataset(
            cache_dir=cache_dir,
            in_memory=cfg.data.get("cache_in_memory", True),
            desc_global_views=cfg.jepa.get("desc_global_views", None),
            **common,
        )
    else:
        dataset = ZINC20Dataset(zinc20_dir=cfg.data.zinc20_dir, **common)
    if is_main(rank):
        kind = "cached shards" if cache_dir else ".smi files"
        print(f"ZINC20: {dataset.file_count()} {kind} found")

    num_workers = cfg.training.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        collate_fn=zinc20_collate,
        pin_memory=False,
        prefetch_factor=cfg.training.get("prefetch_factor", 4) if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = build_model(cfg).to(device)
    if cfg.get("compile", True) and hasattr(torch, "compile"):
        if is_main(rank):
            print("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Estimate steps_per_epoch from data (rough; streaming dataset has unknown length)
    steps_per_epoch = cfg.training.get("steps_per_epoch", 10_000)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    PROFILER.enabled = cfg.get("profile", False)

    global_step = 0
    start_epoch = 0
    Vg = cfg.jepa.Vg
    lambd = cfg.jepa.lambd
    num_slices = cfg.jepa.get("num_slices", 1024)
    chunk_size = cfg.jepa.get("chunk_size", 1)

    if resume_path:
        ckpt_path = _resolve_resume_path(resume_path, checkpoint_dir)
        if is_main(rank):
            print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        _unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        if is_main(rank):
            print(f"Resumed at step {global_step}, starting epoch {start_epoch+1}")

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        t0 = time.time()

        if is_main(rank):
            print(f"\n=== Epoch {epoch+1}/{cfg.training.epochs} ===")

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch", disable=not is_main(rank))
        prev_iter_end = time.perf_counter()
        for batch_idx, all_batches in enumerate(pbar):
            # Bound every epoch to exactly steps_per_epoch optimizer steps. Under
            # DDP the per-rank molecule counts differ by ~1 at the stream tail, so
            # letting the loader run dry would desync the collective all-reduce and
            # hang the whole job. A fixed cap keeps all ranks in lockstep (and the
            # LR schedule, which is built from steps_per_epoch, stays accurate).
            # Config must size steps_per_epoch below the per-rank molecule budget
            # (total_molecules / world_size / batch_size) so no rank exhausts early.
            if batch_idx >= steps_per_epoch:
                break
            PROFILER.add("data_wait", time.perf_counter() - prev_iter_end)

            with PROFILER.section("to_device"):
                all_batches = [b.to(device) for b in all_batches]

            optimizer.zero_grad()
            loss, pred_loss, sig_loss, emb_stats = lejepa_loss(
                all_batches,
                encoder=model,
                lambd=lambd,
                global_step=global_step,
                Vg=Vg,
                num_slices=num_slices,
                chunk_size=chunk_size,
            )
            with PROFILER.section("backward"):
                loss.backward()
            with PROFILER.section("optim"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            PROFILER.end_step()
            global_step += 1

            if is_main(rank):
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    pred=f"{pred_loss.item():.4f}",
                    sig=f"{sig_loss.item():.4f}",
                    rank=f"{emb_stats['eff_rank']:.0f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if is_main(rank) and global_step % cfg.training.get("log_every", 50) == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"\n[epoch {epoch+1} step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"pred={pred_loss.item():.4f} "
                    f"sig={sig_loss.item():.4f} "
                    f"rank={emb_stats['eff_rank']:.1f} "
                    f"|mean|={emb_stats['mean_abs']:.3f} "
                    f"std={emb_stats['std']:.2f} "
                    f"pred_g={emb_stats.get('pred_global', 0.0):.4f} "
                    f"pred_l={emb_stats.get('pred_local', 0.0):.4f} "
                    f"lr={lr:.2e} "
                    f"t={elapsed:.1f}s"
                )
                if PROFILER.enabled:
                    print("Step profile (last window):\n" + PROFILER.report())
                    PROFILER.reset()
                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "loss/total": loss.item(),
                            "loss/pred": pred_loss.item(),
                            "loss/sigreg": sig_loss.item(),
                            "emb/eff_rank": emb_stats["eff_rank"],
                            "emb/mean_abs": emb_stats["mean_abs"],
                            "emb/std": emb_stats["std"],
                            "loss/pred_global": emb_stats.get("pred_global", 0.0),
                            "loss/pred_local": emb_stats.get("pred_local", 0.0),
                            "lr": lr,
                            "step": global_step,
                        }
                    )
                t0 = time.time()

            if is_main(rank) and global_step % cfg.training.get("save_every", 5000) == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    checkpoint_dir / f"step_{global_step:08d}.pt",
                )

            prev_iter_end = time.perf_counter()

        # End of epoch checkpoint — store epoch+1 so resume continues at the next epoch
        if is_main(rank):
            save_checkpoint(
                model, optimizer, scheduler, global_step, epoch + 1,
                checkpoint_dir / f"epoch_{epoch+1:04d}.pt",
            )

    if is_main(rank):
        save_checkpoint(
            model, optimizer, scheduler, global_step, cfg.training.epochs,
            checkpoint_dir / "final.pt",
        )
        if use_wandb:
            import wandb
            wandb.finish()

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' for latest")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
