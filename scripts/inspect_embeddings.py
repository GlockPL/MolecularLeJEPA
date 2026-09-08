"""
Inspect a pretrained LeJEPA encoder's embedding distribution.

Produces:
  - per_dim_std.png    : bar chart of std per output dim (should be tight)
  - covariance.png     : correlation heatmap (should ≈ identity)
  - pca.png            : 2D PCA scatter + eigenvalue spectrum
  - tsne_props.png     : 2D UMAP/t-SNE colored by MolWt, LogP, NumRings

Usage:
  uv run python scripts/inspect_embeddings.py \\
      --checkpoint checkpoints/pretrain_chembl/final.pt \\
      --config configs/pretrain_chembl.yaml \\
      --smiles data/chembl \\
      --no-descriptors \\
      --n 5000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from torch_geometric.data import Batch
from tqdm import tqdm

from src.data.featurize import smiles_to_data
from src.models.gps_transformer import GPSTransformer

_DESC_FUNS = [fn for _, fn in Descriptors.descList]


def _mol_descriptors(mol) -> list[float]:
    out = []
    for fn in _DESC_FUNS:
        try:
            v = fn(mol)
            out.append(float(v) if v is not None and np.isfinite(v) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    # Reuse the pretrain backbone dispatch so this works for both gps and dmpnn
    # configs (dmpnn has no num_heads). Strip any DDP 'module.' prefix on load.
    from src.pretrain import build_model
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}, epoch {ckpt.get('epoch', '?')+1}")
    return model, cfg


def _read_smis(path: Path, max_atoms: int, cap: int) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()[0]
            if tok.lower() == "smiles":
                continue
            mol = Chem.MolFromSmiles(tok)
            if mol is None or mol.GetNumHeavyAtoms() > max_atoms:
                continue
            out.append(tok)
            if len(out) >= cap:
                break
    return out


def sample_smiles(smiles_path: Path, max_atoms: int, n: int, seed: int = 0) -> list[str]:
    """Sample n SMILES for embedding diagnostics.

    CRITICAL: ZINC .smi files are binned by molecular weight / logP (the 2-letter
    tranche codes), so a *single* file is structurally near-homogeneous. Embedding
    one file collapses the covariance to a low effective rank — a SAMPLING
    artifact that looks exactly like dimensional collapse but is not. When
    smiles_path is a directory, sample a few molecules from many tranche files so
    the probe set matches the diversity the model was trained on (global-shuffled
    cache). Pass a single file only if you specifically want that tranche.
    """
    if smiles_path.is_dir():
        files = sorted(smiles_path.rglob("*.smi"))
        if not files:
            raise FileNotFoundError(f"No .smi files under {smiles_path}")
        random.Random(seed).shuffle(files)
        per_file = max(1, n // 100)  # spread across >=~100 tranches
        out: list[str] = []
        for fp in files:
            out += _read_smis(fp, max_atoms, per_file)
            if len(out) >= n:
                break
        return out[:n]
    return _read_smis(smiles_path, max_atoms, n)


@torch.no_grad()
def compute_embeddings(smiles_list, model, device, batch_size=128,
                       desc_mean=None, desc_std=None, zero_desc=False, n_desc=0):
    # zero_desc: attach a ZERO descriptor vector (keeps the proj_head input width
    # of hidden+n_desc valid) so the visualized embedding is the PURE structural
    # representation. The 217 RDKit descriptors trivially encode MolWt/LogP/rings
    # and leak through even a random proj_head, making both pretrained and random
    # init look organized; zeroing them isolates what the GNN actually learned.
    use_real_desc = desc_mean is not None and desc_std is not None
    attach_zeros = zero_desc and n_desc > 0
    Zs, kept = [], []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Embedding"):
        chunk = smiles_list[i : i + batch_size]
        datas, valid = [], []
        for smi in chunk:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            d = smiles_to_data(smi)
            if d is None:
                continue
            if attach_zeros:
                d.global_feat = torch.zeros(1, n_desc, dtype=torch.float32)
            elif use_real_desc:
                raw = torch.tensor(_mol_descriptors(mol), dtype=torch.float32)
                d.global_feat = torch.nan_to_num(
                    (raw - desc_mean) / desc_std, nan=0.0, posinf=0.0, neginf=0.0
                ).unsqueeze(0)
            datas.append(d)
            valid.append(smi)
        if not datas:
            continue
        batch = Batch.from_data_list(datas).to(device)
        z = model(batch)
        Zs.append(z.cpu().numpy())
        kept.extend(valid)
    return np.concatenate(Zs, axis=0), kept


def compute_rdkit_props(smiles_list):
    mw, logp, n_rings = [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        mw.append(Descriptors.MolWt(m))
        logp.append(Descriptors.MolLogP(m))
        n_rings.append(m.GetRingInfo().NumRings())
    return np.array(mw), np.array(logp), np.array(n_rings)


def plot_per_dim_std(Z, out_path):
    std = Z.std(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(std)), np.sort(std))
    ax.set_xlabel("dimension (sorted by std)")
    ax.set_ylabel("std")
    ax.set_title(
        f"Per-dim std: mean={std.mean():.3f}, "
        f"range [{std.min():.3f}, {std.max():.3f}], "
        f"collapsed (<0.01): {(std < 0.01).sum()}/{len(std)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_covariance(Z, out_path):
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / Z.shape[0]
    s = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / np.outer(s, s)
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(
        f"Correlation matrix (off-diag |mean|={np.abs(off_diag).mean():.3f}, "
        f"max={np.abs(off_diag).max():.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pca(Z, out_path):
    pca_full = PCA().fit(Z)
    Z2 = pca_full.transform(Z)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Z2[:, 0], Z2[:, 1], s=2, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title(
        f"PCA 2D (PC1 {pca_full.explained_variance_ratio_[0]:.1%}, "
        f"PC2 {pca_full.explained_variance_ratio_[1]:.1%})"
    )
    axes[1].plot(pca_full.explained_variance_ratio_)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("variance ratio (log)")
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k50 = int(np.searchsorted(cumvar, 0.5)) + 1
    k90 = int(np.searchsorted(cumvar, 0.9)) + 1
    axes[1].set_title(f"Spectrum (50% in {k50} dims, 90% in {k90} dims)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def project_2d(Z):
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(Z), "t-SNE"


def plot_props(Z, props, names, out_path):
    emb_2d, method = project_2d(Z)
    n = len(props)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, vals, name in zip(axes, props, names):
        # Clip extremes for better color contrast
        lo, hi = np.percentile(vals, [2, 98])
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=vals, cmap="viridis",
                        vmin=lo, vmax=hi, s=3, alpha=0.6)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_title(f"{method} — colored by {name}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_props_compare(Z_pre, Z_rand, props, names, out_path):
    """2-row property-colored projection: PRETRAINED (top) vs RANDOM-INIT (bottom).

    Each row is projected with its OWN UMAP/t-SNE (we compare whether each
    embedding organizes by property, not aligning the two layouts). If the
    pretrained row shows a smooth property gradient and the random row is a
    structureless blob, pretraining demonstrably organized the manifold.
    """
    pre_2d, method = project_2d(Z_pre)
    rand_2d, _ = project_2d(Z_rand)
    n = len(props)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, (vals, name) in enumerate(zip(props, names)):
        lo, hi = np.percentile(vals, [2, 98])
        for row, (emb, label) in enumerate(((pre_2d, "Pretrained"), (rand_2d, "Random init"))):
            ax = axes[row, col]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis",
                            vmin=lo, vmax=hi, s=3, alpha=0.6)
            plt.colorbar(sc, ax=ax, label=name)
            ax.set_title(f"{label} — {method} by {name}")
            # Abstract embedding axes: label them (the projected components) but
            # keep ticks off since the coordinates carry no absolute meaning.
            ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_summary(Z):
    std = Z.std(0)
    mean_abs = np.abs(Z.mean(0)).mean()
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / Z.shape[0]
    s = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / np.outer(s, s)
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    # Effective rank from PCA eigenvalues
    eigvals = np.linalg.eigvalsh(cov).clip(min=1e-12)
    p = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    print("\n=== Embedding statistics ===")
    print(f"  N samples            : {Z.shape[0]}")
    print(f"  embed_dim            : {Z.shape[1]}")
    print(f"  |mean|               : {mean_abs:.4f}  (should be ~0)")
    print(f"  std mean             : {std.mean():.4f}")
    print(f"  std range            : [{std.min():.4f}, {std.max():.4f}]  (tight = isotropic)")
    print(f"  collapsed dims       : {(std < 0.01).sum()}/{Z.shape[1]}  (<0.01)")
    print(f"  off-diag |corr| mean : {np.abs(off_diag).mean():.4f}  (should be small)")
    print(f"  off-diag |corr| max  : {np.abs(off_diag).max():.4f}")
    print(f"  effective rank       : {eff_rank:.1f} / {Z.shape[1]}  (closer to embed_dim = better)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smiles", default="data/chembl", type=Path,
                        help="A directory (sampled across MANY .smi shards — the "
                             "correct diverse default) or a single .smi file. Use the "
                             "same corpus the encoder was pretrained on (ChEMBL).")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--out", default="figures/embeddings", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compare-random", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Also embed a RANDOM-INIT backbone (same arch) on the "
                             "same molecules and emit tsne_props_compare.png "
                             "(pretrained vs random). --no-compare-random to skip.")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Seed for the random-init comparison backbone.")
    parser.add_argument("--no-descriptors", action="store_true",
                        help="Zero the RDKit descriptors before the projection head so "
                             "the figure shows the PURE learned structural embedding. "
                             "Recommended for the pretrained-vs-random figure: the "
                             "descriptors trivially encode MolWt/LogP/rings and leak "
                             "through even a random proj_head, making both rows look "
                             "organized.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    desc_mean, desc_std = None, None
    if cfg.model.get("n_descriptors", 0) > 0:
        cache_dir = cfg.data.get("cache_dir", None)
        stats_path = Path(cache_dir) / "desc_stats.pt" if cache_dir else None
        if stats_path and stats_path.exists():
            ds = torch.load(stats_path, map_location="cpu", weights_only=True)
            desc_mean = ds["mean"]
            desc_std = ds["std"]
            print(f"Loaded descriptor stats from {stats_path}")
        else:
            print("WARNING: n_descriptors>0 but desc_stats.pt not found — descriptors will be zero")

    n_desc = cfg.model.get("n_descriptors", 0)
    if args.no_descriptors:
        print("Descriptors ZEROED — visualizing the pure learned structural embedding.")
        desc_mean = desc_std = None   # do not attach real descriptors

    print(f"Sampling SMILES from {args.smiles}...")
    smiles = sample_smiles(args.smiles, cfg.data.get("max_atoms", 100), args.n)
    print(f"Got {len(smiles)} valid SMILES (max_atoms={cfg.data.get('max_atoms', 100)})")

    Z, kept = compute_embeddings(smiles, model, device, args.batch_size,
                                 desc_mean=desc_mean, desc_std=desc_std,
                                 zero_desc=args.no_descriptors, n_desc=n_desc)
    print(f"Embeddings: {Z.shape}")

    print("Computing RDKit properties for coloring...")
    mw, logp, n_rings = compute_rdkit_props(kept)

    print_summary(Z)

    print("\nGenerating plots...")
    plot_per_dim_std(Z, args.out / "per_dim_std.png")
    plot_covariance(Z, args.out / "covariance.png")
    plot_pca(Z, args.out / "pca.png")
    plot_props(Z, [mw, logp, n_rings], ["MolWt", "LogP", "NumRings"],
               args.out / "tsne_props.png")

    if args.compare_random:
        from src.pretrain import build_model
        print("\n[random-init backbone] embedding the SAME molecules for comparison...")
        torch.manual_seed(args.random_seed)
        rand_model = build_model(cfg).to(device).eval()
        Zr, _ = compute_embeddings(kept, rand_model, device, args.batch_size,
                                   desc_mean=desc_mean, desc_std=desc_std,
                                   zero_desc=args.no_descriptors, n_desc=n_desc)
        print("[random-init backbone] statistics:")
        print_summary(Zr)
        plot_props_compare(Z, Zr, [mw, logp, n_rings], ["MolWt", "LogP", "NumRings"],
                           args.out / "tsne_props_compare.png")
        print(f"Saved pretrained-vs-random comparison to {args.out}/tsne_props_compare.png")
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()