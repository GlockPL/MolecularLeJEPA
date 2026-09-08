"""
Two-stage finetuning on the Wong et al. 2023 antibiotic dataset.

Stage 1 — Linear probe:
    Freeze backbone, train task heads only (fast, ~10 epochs).

Stage 2 — Full finetuning:
    Unfreeze backbone, lower learning rate, train end-to-end (~50 epochs).

Usage:
    python -m src.finetune --config configs/finetune.yaml --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import copy
import datetime
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from omegaconf import OmegaConf, DictConfig

from src.data.antibiotic_dataset import AntibioticDataset
from src.data.featurize import ATOM_DIM as _DEF_ATOM_DIM, BOND_DIM as _DEF_BOND_DIM
from src.models.gps_transformer import GPSTransformer
from src.models.dmpnn import DMPNNEncoder
from src.models.heads import AntibioticHeads
from src.evaluate import compute_auprc


# ANSI colors for AUPRC reporting: red by default, green when a task beats
# its running best. Codes are stripped before writing to the log file.
_RED = "\033[31m"
_GREEN = "\033[32m"
_RESET = "\033[0m"
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _colored_metrics(metrics: dict[str, float], best_per_task: dict[str, float],
                     tasks: list[str], width: int = 10) -> str:
    """Format each task's AUPRC in red, or green if it beats that task's best.

    Mutates ``best_per_task`` in place so a value only shows green on the epoch
    it sets a new high-water mark for that task.
    """
    cells = []
    for t in tasks:
        v = metrics[t]
        better = v > best_per_task.get(t, -float("inf"))
        if better:
            best_per_task[t] = v
        color = _GREEN if better else _RED
        cells.append(f"{color}{v:{width}.4f}{_RESET}")
    return "  ".join(cells)


class _Logger:
    """Tees log lines to stdout and a UTF-8 text file simultaneously.

    ANSI color codes are kept for stdout but stripped from the file so logs
    stay readable in editors and grep.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._f.write(_ANSI_RE.sub("", msg) + "\n")

    def close(self) -> None:
        self._f.close()


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


class _PlateauRollback:
    """Keras-style ReduceLROnPlateau + restore_best_weights for stage 2.

    Each epoch, the caller reports whether val metric improved. When `patience`
    consecutive non-improving epochs accumulate, we reload the best checkpoint
    (weights + optimizer state, so Adam moments don't carry overfitting
    momentum) and multiply LR by `factor`. Returns 'stop' once the next LR
    drop would fall below `min_lr`, so the caller breaks the loop.
    """

    def __init__(
        self,
        optimizer,
        ckpt_path: Path,
        backbone,
        heads,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.ckpt_path = Path(ckpt_path)
        self.backbone = backbone
        self.heads = heads
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.bad_epochs = 0

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def step(self, improved: bool) -> str:
        """Returns 'improved' | 'plateau' | 'reduced' | 'stop'."""
        if improved:
            self.bad_epochs = 0
            return "improved"
        self.bad_epochs += 1
        if self.bad_epochs <= self.patience:
            return "plateau"
        new_lr = self.lr * self.factor
        if new_lr < self.min_lr:
            return "stop"
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.heads.load_state_dict(ckpt["heads"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restored the original LR; override to the decayed value.
        self._set_lr(new_lr)
        self.bad_epochs = 0
        return "reduced"


class _NoamLR:
    """Chemprop-style per-step LR: linear warmup init_lr→max_lr over warmup_steps,
    then exponential decay max_lr→final_lr over the remaining steps. Stepped every
    batch (not epoch). Used for end-to-end-from-scratch finetuning (scheduler=noam),
    which Chemprop uses and our two-stage probe→plateau does not.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 init_lr: float, max_lr: float, final_lr: float):
        self.opt = optimizer
        self.warmup = max(1, int(warmup_steps))
        self.total = max(self.warmup + 1, int(total_steps))
        self.init_lr, self.max_lr, self.final_lr = init_lr, max_lr, final_lr
        decay_steps = max(1, self.total - self.warmup)
        self.gamma = (final_lr / max_lr) ** (1.0 / decay_steps)
        self.step_num = 0
        self._set(init_lr)

    def _set(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.init_lr + (self.max_lr - self.init_lr) * self.step_num / self.warmup
        else:
            lr = max(self.final_lr, self.max_lr * (self.gamma ** (self.step_num - self.warmup)))
        self._set(lr)


def _count_actives(ds: AntibioticDataset, col: int = 0) -> int:
    return int(sum(d.y[0, col].item() for d in ds._data_list))


def load_backbone(checkpoint_path: str | Path, cfg: DictConfig, device: torch.device) -> nn.Module:
    # model.backbone selects the encoder; defaults to 'gps' so every existing
    # config loads unchanged. 'dmpnn' = native PyG Chemprop-style D-MPNN, a pure
    # architecture swap (same featurization / descriptors / heads / harness).
    backbone_kind = cfg.model.get("backbone", "gps")
    if backbone_kind == "gps":
        model = GPSTransformer(
            hidden_dim=cfg.model.hidden_dim,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            walk_length=cfg.model.get("walk_length", 20),
            dropout=cfg.model.get("dropout", 0.1),
            n_descriptors=cfg.model.get("n_descriptors", 0),
            readout=cfg.model.get("readout", "mean"),
        )
    elif backbone_kind == "dmpnn":
        model = DMPNNEncoder(
            atom_dim=cfg.model.get("atom_dim", _DEF_ATOM_DIM),   # 133 for chemprop feats
            bond_dim=cfg.model.get("bond_dim", _DEF_BOND_DIM),
            hidden_dim=cfg.model.hidden_dim,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,        # D-MPNN message-passing depth
            dropout=cfg.model.get("dropout", 0.0),
            n_descriptors=cfg.model.get("n_descriptors", 0),
            use_proj_head=cfg.model.get("use_proj_head", True),
        )
    else:
        raise ValueError(f"Unknown model.backbone {backbone_kind!r} (use 'gps' or 'dmpnn')")
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        # Strip DDP 'module.' prefix if present
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded backbone from {checkpoint_path}")
    return model.to(device)


def train_one_epoch(
    backbone: GPSTransformer,
    heads: AntibioticHeads,
    loader: DataLoader,
    optimizer,
    pos_weights: torch.Tensor,
    device: torch.device,
    step_scheduler=None,
) -> float:
    backbone.train()
    heads.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = backbone(batch)
        if hasattr(batch, "global_feat"):
            z = torch.cat([z, batch.global_feat], dim=1)
        logits = heads(z)
        loss, _ = heads.loss(logits, batch.y, pos_weights.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(heads.parameters()), 1.0
        )
        optimizer.step()
        # Per-step LR schedule (e.g. Noam warmup) steps every batch, not epoch.
        if step_scheduler is not None:
            step_scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    backbone: GPSTransformer,
    heads: AntibioticHeads,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    backbone.eval()
    heads.eval()
    all_probs = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        z = backbone(batch)
        if hasattr(batch, "global_feat"):
            z = torch.cat([z, batch.global_feat], dim=1)
        probs = heads.predict_proba(z)  # (B, len(heads.tasks))
        all_probs.append(probs.cpu())
        # Slice targets to active task columns so they align with probs/heads.tasks.
        all_targets.append(batch.y[:, heads.task_cols].cpu())
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_auprc(probs, targets, task_names=heads.tasks)


def _resolve_tasks(task_arg: str | list[str] | None) -> list[str]:
    """'all' or None → all 4; otherwise a single name or list of names."""
    if task_arg is None or (isinstance(task_arg, str) and task_arg == "all"):
        return list(AntibioticHeads.TASK_NAMES)
    if isinstance(task_arg, str):
        return [task_arg]
    return list(task_arg)


def finetune(
    cfg: DictConfig,
    checkpoint_path: str | None,
    seed: int = 42,
    tasks: str | list[str] | None = None,
    train_frac: float = 1.0,
    fewshot_seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed controls head init + training stochasticity.
    # Data split seed is always from cfg.data.seed so all ensemble members
    # share the same scaffold split and test set.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    active_tasks = _resolve_tasks(tasks)
    task_tag = "all" if len(active_tasks) == len(AntibioticHeads.TASK_NAMES) else "_".join(active_tasks)
    # Best-checkpoint tracking uses antibiotic when present (the discovery
    # signal); otherwise fall back to the first active task.
    primary_task = "antibiotic" if "antibiotic" in active_tasks else active_tasks[0]
    primary_col = AntibioticHeads.TASK_NAMES.index(primary_task)

    # --- Logger ---
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.get("log_dir", "logs"))
    ckpt_tag = Path(checkpoint_path).stem if checkpoint_path else "scratch"
    log_path = log_dir / f"finetune_{run_ts}_seed{seed}_{ckpt_tag}_{task_tag}.txt"
    log = _Logger(log_path)
    t0 = time.time()

    log("AntibioticJEPA — Finetuning")
    log("=" * 60)
    log(f"Started    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file   : {log_path}")
    log(f"Checkpoint : {checkpoint_path or '(none — scratch)'}")
    log(f"Seed       : {seed}")
    log(f"Device     : {device}")
    log(f"Tasks      : {active_tasks}  (best ckpt tracks '{primary_task}')")
    log()
    log("--- Config ---")
    log(OmegaConf.to_yaml(cfg).rstrip())

    use_wandb = cfg.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("project", "antibioticjepa-ft"), config=OmegaConf.to_container(cfg))

    # Datasets
    data_cfg = cfg.data
    xlsx_path = data_cfg.xlsx_path
    common = dict(
        test_size=data_cfg.get("test_size", 0.20),
        val_size=data_cfg.get("val_size", 0.10),
        seed=data_cfg.get("seed", 42),
        split_method=data_cfg.get("split_method", "scaffold"),
        tasks=active_tasks,
        featurizer=data_cfg.get("featurizer", "ours"),
        scaffold_partition_seed=data_cfg.get("scaffold_partition_seed", None),
    )
    train_ds = AntibioticDataset(xlsx_path, split="train", **common)
    # Few-shot: stratified subsample of TRAIN (val/test fixed; descriptor stats kept
    # from the FULL train so normalization is consistent across budgets). Stratified
    # on the primary task so the rare positives are preserved at small fractions.
    if train_frac < 1.0:
        from sklearn.model_selection import train_test_split
        prim_col = AntibioticHeads.TASK_NAMES.index(primary_task)
        ys = np.array([int(d.y[0, prim_col].item()) for d in train_ds._data_list])
        keep, _ = train_test_split(np.arange(len(ys)), train_size=train_frac,
                                   stratify=ys, random_state=fewshot_seed)
        train_ds._data_list = [train_ds._data_list[i] for i in keep]
        log(f"  few-shot: train subsampled to {len(keep)} ({train_frac:.1%}), "
            f"{int(ys[keep].sum())} {primary_task}-active (seed {fewshot_seed})")
    # Val/test use train-set descriptor statistics so normalization is consistent.
    val_ds   = AntibioticDataset(xlsx_path, split="val",  desc_stats=train_ds.desc_stats, **common)
    test_ds  = AntibioticDataset(xlsx_path, split="test", desc_stats=train_ds.desc_stats, **common)

    bs = cfg.training.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, collate_fn=Batch.from_data_list)

    pos_weights = train_ds.pos_weights()  # (num_tasks,)
    heads_in_dim = cfg.model.embed_dim + train_ds.n_descriptors

    log()
    log("--- Dataset ---")
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_active = _count_actives(ds, primary_col)
        log(f"  {split_name:5s}: {len(ds):6d} compounds  "
            f"{n_active:4d} {primary_task}-active ({100*n_active/max(len(ds),1):.1f}%)")
    log(f"  RDKit descriptors : {train_ds.n_descriptors}")
    log(f"  Heads input dim   : {heads_in_dim}  (backbone {cfg.model.embed_dim} + desc {train_ds.n_descriptors})")
    log(f"  Pos weights       : {[round(w,2) for w in pos_weights.tolist()]}")

    # Model — heads input dim = backbone embed + RDKit descriptors
    backbone = load_backbone(checkpoint_path, cfg, device)
    heads = AntibioticHeads(
        embed_dim=heads_in_dim,
        hidden_dim=cfg.model.get("head_hidden_dim", 256),
        num_layers=cfg.model.get("head_num_layers", 1),
        tasks=active_tasks,
    ).to(device)

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints/finetune"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Linear probe (backbone frozen) ---
    # skip_probe=True trains end-to-end from scratch with no frozen-probe warmup —
    # the Chemprop-style regime (pair with scheduler=noam). Use when the backbone
    # must actually learn the representation (e.g. cytotox tasks where descriptors
    # are insufficient), not just ride the descriptor skip-connection.
    skip_probe = bool(cfg.training.get("skip_probe", False))
    best_probe_auprc = -float("inf")
    best_probe_epoch = 0
    best_probe_heads = None
    if skip_probe:
        log()
        log("--- Stage 1: SKIPPED (skip_probe=true) — training end-to-end from scratch ---")
    else:
        probe_epochs = cfg.training.get("probe_epochs", 10)
        lr_probe = cfg.training.get("lr_probe", 1e-3)
        log()
        log(f"--- Stage 1: Linear probe ({probe_epochs} epochs, lr={lr_probe}) ---")
        log(f"  {'Ep':>3}  {'Loss':>7}  " + "  ".join(f"{t:>10}" for t in heads.tasks))

        for p in backbone.parameters():
            p.requires_grad_(False)

        opt1 = AdamW(heads.parameters(), lr=lr_probe,
                     weight_decay=cfg.training.get("weight_decay", 1e-2))
        sch1 = CosineAnnealingLR(opt1, T_max=probe_epochs)

        # Snapshot the best probe heads so we can carry them into stage 2 as the
        # starting "best" — if full finetuning never beats the probe (common when
        # the positive count is tiny), test eval falls back to these weights.
        probe_best_per_task: dict[str, float] = {}
        for epoch in range(probe_epochs):
            loss = train_one_epoch(backbone, heads, train_loader, opt1, pos_weights, device)
            sch1.step()
            metrics = evaluate(backbone, heads, val_loader, device)
            val_auprc = metrics.get(primary_task, 0.0)
            is_best = val_auprc > best_probe_auprc
            if is_best:
                best_probe_auprc = val_auprc
                best_probe_epoch = epoch + 1
                best_probe_heads = copy.deepcopy(heads.state_dict())
            row = (f"  {epoch+1:3d}  {loss:7.4f}  " +
                   _colored_metrics(metrics, probe_best_per_task, heads.tasks) +
                   ("  *" if is_best else ""))
            log(row)
            if use_wandb:
                import wandb
                wandb.log({"stage": 1, "epoch": epoch + 1, "train_loss": loss,
                           **{f"val/{k}": v for k, v in metrics.items()}})

        # Restore the best probe heads (backbone was frozen, so it is unchanged).
        if best_probe_heads is not None:
            heads.load_state_dict(best_probe_heads)
        log(f"  Best probe {primary_task} AUPRC: {best_probe_auprc:.4f}  (epoch {best_probe_epoch})")

    # --- Stage 2: Full finetuning ---
    ft_epochs = cfg.training.get("finetune_epochs", 50)
    lr_ft = cfg.training.get("lr_finetune", 1e-4)
    scheduler_kind = cfg.training.get("scheduler", "cosine")
    patience = cfg.training.get("patience", 5)
    lr_factor = cfg.training.get("lr_factor", 0.5)
    min_lr = float(cfg.training.get("min_lr", 1e-6))

    log()
    log(f"--- Stage 2: Full finetuning ({ft_epochs} epochs, lr={lr_ft}, scheduler={scheduler_kind}) ---")
    if scheduler_kind == "plateau":
        log(f"  Plateau scheduler: patience={patience}, lr_factor={lr_factor}, min_lr={min_lr:.0e}")
    log(f"  {'Ep':>3}  {'Loss':>7}  {'LR':>9}  " + "  ".join(f"{t:>10}" for t in heads.tasks))

    for p in backbone.parameters():
        p.requires_grad_(True)

    all_params = list(backbone.parameters()) + list(heads.parameters())
    opt2 = AdamW(all_params, lr=lr_ft,
                 weight_decay=cfg.training.get("weight_decay", 1e-2))

    best_ckpt_path = checkpoint_dir / f"best_{task_tag}.pt"

    def _save_best() -> None:
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "heads": heads.state_dict(),
                "tasks": heads.tasks,
                "optimizer": opt2.state_dict(),
                "desc_mean": train_ds.desc_mean,
                "desc_std": train_ds.desc_std,
            },
            best_ckpt_path,
        )

    sch2 = None
    plateau = None
    step_sched = None
    if scheduler_kind == "cosine":
        sch2 = CosineAnnealingLR(opt2, T_max=ft_epochs)
    elif scheduler_kind == "plateau":
        plateau = _PlateauRollback(
            opt2, best_ckpt_path, backbone, heads,
            patience=patience, factor=lr_factor, min_lr=min_lr,
        )
    elif scheduler_kind == "noam":
        # Chemprop-style: linear warmup init_lr→lr_finetune(=max) over warmup_epochs,
        # then exp decay to final_lr. Stepped per batch. Pair with skip_probe=true.
        steps_per_epoch = max(1, len(train_loader))
        warmup_epochs = float(cfg.training.get("warmup_epochs", 2))
        init_lr = float(cfg.training.get("init_lr", 1e-4))
        final_lr = float(cfg.training.get("final_lr", 1e-4))
        step_sched = _NoamLR(
            opt2,
            warmup_steps=int(warmup_epochs * steps_per_epoch),
            total_steps=ft_epochs * steps_per_epoch,
            init_lr=init_lr, max_lr=lr_ft, final_lr=final_lr,
        )
        log(f"  Noam scheduler: warmup_epochs={warmup_epochs}, init_lr={init_lr:.0e}, "
            f"max_lr={lr_ft:.0e}, final_lr={final_lr:.0e}")
    else:
        raise ValueError(f"Unknown scheduler {scheduler_kind!r} (use 'cosine', 'plateau', or 'noam')")

    # Seed the best from the linear probe so full finetuning only overwrites
    # the checkpoint when it genuinely improves. best_epoch == 0 denotes the
    # probe. Save it now so test eval (and plateau rollback) has a valid
    # checkpoint even if FT never beats the probe.
    best_auprc = best_probe_auprc
    best_epoch = 0
    _save_best()
    ft_best_per_task: dict[str, float] = {}
    for epoch in range(ft_epochs):
        # Capture the LR that was used to produce this epoch's loss/metrics
        # (cosine.step would advance it before logging).
        cur_lr = opt2.param_groups[0]["lr"]
        loss = train_one_epoch(backbone, heads, train_loader, opt2, pos_weights, device,
                               step_scheduler=step_sched)
        if sch2 is not None:
            sch2.step()
        metrics = evaluate(backbone, heads, val_loader, device)
        val_auprc = metrics.get(primary_task, 0.0)

        is_best = val_auprc > best_auprc
        if is_best:
            best_auprc = val_auprc
            best_epoch = epoch + 1
            _save_best()

        row = (f"  {epoch+1:3d}  {loss:7.4f}  {cur_lr:9.2e}  " +
               _colored_metrics(metrics, ft_best_per_task, heads.tasks) +
               ("  *" if is_best else ""))
        log(row)

        if use_wandb:
            import wandb
            wandb.log({"stage": 2, "epoch": epoch + 1, "train_loss": loss, "lr": cur_lr,
                       **{f"val/{k}": v for k, v in metrics.items()}})

        if plateau is not None:
            action = plateau.step(is_best)
            if action == "reduced":
                lbl = "probe" if best_epoch == 0 else f"epoch {best_epoch}"
                log(f"       ↺ Plateau: rolled back to {lbl} "
                    f"(AUPRC={best_auprc:.4f}), LR → {plateau.lr:.2e}")
            elif action == "stop":
                log(f"       ✕ Plateau: next LR drop would be < min_lr={min_lr:.0e}, stopping early.")
                break

    log()
    _best_lbl = "probe" if best_epoch == 0 else f"epoch {best_epoch}"
    log(f"  Best val {primary_task} AUPRC: {best_auprc:.4f}  ({_best_lbl})")

    # --- Final test evaluation ---
    log()
    log("--- Test set evaluation (best checkpoint) ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    backbone.load_state_dict(best_ckpt["backbone"])
    heads.load_state_dict(best_ckpt["heads"])
    test_metrics = evaluate(backbone, heads, test_loader, device)
    for task, val in test_metrics.items():
        log(f"  {task:10s}: {val:.4f}")

    elapsed = time.time() - t0
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    log()
    log(f"Finished   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Elapsed    : {h}h {m}m {s}s")
    log(f"Log saved  : {log_path}")
    log.close()

    if use_wandb:
        import wandb
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained backbone checkpoint (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for head init and training stochasticity")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all"] + AntibioticHeads.TASK_NAMES,
                        help="Train heads for one task (Wong et al. style) or "
                             "'all' for the multi-task model (default).")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="Few-shot: keep this fraction of TRAIN (stratified on the "
                             "primary task; val/test fixed). <1.0 probes the low-label "
                             "regime where pretraining should help most.")
    parser.add_argument("--fewshot-seed", type=int, default=42,
                        help="Seed for the few-shot train subsample (reproducible).")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    finetune(cfg, args.checkpoint, seed=args.seed, tasks=args.task,
             train_frac=args.train_frac, fewshot_seed=args.fewshot_seed)


if __name__ == "__main__":
    main()
