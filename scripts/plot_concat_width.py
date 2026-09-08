"""Figure: why the learned embedding adds to a fingerprint (paper Fig. "concat-width").

Panel (a) WIDTH. Every frozen probe in the paper pits a 128-dim dense embedding
    against a 2048-bit sparse fingerprint under a random forest, a head that
    thrives on many weak decorrelated features. Controlling for width shows the
    two representations have very different dimension-efficiency: Morgan keeps
    improving to ~1024 bits, while the embedding saturates by ~16-32 dims. At the
    MATCHED width of 128 the ranking INVERTS between the splits - the fingerprint
    leads on validation and the embedding leads on test, because the fingerprint
    loses 0.040 ROC-AUC across the scaffold shift and the embedding loses none.

Panel (b) WHERE THE GAIN LIVES. Concatenating the truncated embedding with the
    fingerprint leaves VALIDATION essentially unchanged (+0.002) but changes how
    much is lost crossing to the test scaffolds: the fingerprint alone drops
    0.042 valid->test, the concatenation drops 0.017. The gain is therefore not
    extra fit that validation rewards - it is robustness to the scaffold shift.
    Restoring the diluting dimensions (PCA-64) removes the protection, which is
    the control identifying dilution as the mechanism.

The two panels use different forest budgets (500 trees x 5 seeds for the width
sweep, 1000 x 10 for the concatenation), so they are internally comparable but
not against each other.

Usage:
    # from the shipped reference numbers (no GPU, no re-run needed)
    uv run python scripts/plot_concat_width.py

    # from your own re-run
    uv run python scripts/plot_concat_width.py \\
        --csv logs/width_probe_molhiv.csv \\
        --concat-log logs/concat_balanced.out
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reproduce_molhiv" / "reference"

# Panel (b) reference: reproduce_molhiv/reference/concat_pretrained.out, Morgan 1024
# at max_features=0.1, as (valid, test). --concat-log re-reads these from a fresh run.
SLOPE_REF = {
    "Morgan 1024 alone": (0.8453, 0.8032),
    "+ embedding PCA-32": (0.8471, 0.8298),
    "+ embedding PCA-64 (control)": (0.8313, 0.8110),
}
SLOPE_STYLE = [
    ("Morgan 1024 alone", "tab:red", "o", "-"),
    ("+ embedding PCA-32", "tab:purple", "o", "-"),
    ("+ embedding PCA-64 (control)", "0.55", "s", "--"),
]
# The concat-log row each panel-(b) line is read from, when --concat-log is given.
SLOPE_ROWS = {
    "Morgan 1024 alone": "morgan 1024 [mf=0.1]",
    "+ embedding PCA-32": "morgan 1024 + embed PCA-32 [mf=0.1]",
    "+ embedding PCA-64 (control)": "morgan 1024 + embed PCA-64 [mf=0.1]",
}
# "  <name padded to 38>  <dim> <test mean> +/- <std>   <ens>   <valid>"
ROW_RE = re.compile(r"^\s+(.+?)\s{2,}(\d+)\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_concat_log(path: Path) -> dict[str, tuple[float, float]]:
    """Pull (valid, test) for the panel-(b) rows out of a concat_balanced_molhiv.py log."""
    found: dict[str, tuple[float, float]] = {}
    wanted = {v: k for k, v in SLOPE_ROWS.items()}
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if m and m.group(1) in wanted:
            found[wanted[m.group(1)]] = (float(m.group(6)), float(m.group(3)))
    missing = set(SLOPE_ROWS) - set(found)
    if missing:
        raise SystemExit(f"{path}: could not find row(s) for {sorted(missing)} - was the "
                         "sweep run with the default --morgan-bits/--embed-dims/--max-features?")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REF / "width_probe_molhiv.csv"),
                    help="width sweep CSV from scripts/width_probe_molhiv.py (panel a)")
    ap.add_argument("--concat-log", default="",
                    help="stdout of scripts/concat_balanced_molhiv.py; if given, panel (b) "
                         "is re-read from it instead of the shipped reference numbers")
    ap.add_argument("--out", default="figures/molhiv_concat_width.png")
    args = ap.parse_args()

    # ------------------------------------------------------------ panel (a)
    with open(args.csv) as fh:
        raw = list(csv.DictReader(fh))
    has_valid = "valid_mean" in raw[0]
    if not has_valid:
        print(f"WARNING: {args.csv} has no valid_mean column - panel (a) falls back to "
              "TEST. Re-run scripts/width_probe_molhiv.py to fix.")

    rows: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in raw:
        for split in ("valid", "test"):
            if split == "valid" and not has_valid:
                continue
            rows.setdefault((r["kind"], split), []).append(
                (int(r["width"]), float(r[f"{split}_mean"])))

    # Colour encodes the representation, line style encodes the split. The point of
    # the panel is that the two representations cross over BETWEEN splits, so both
    # have to be on the same axes; the PCA truncation is left to the table.
    curves = [
        ("morgan", "valid", "Morgan fingerprint, validation", "tab:red", "o", "-"),
        ("morgan", "test", "Morgan fingerprint, test", "tab:red", "o", ":"),
        ("embed_raw", "valid", "LeJEPA embedding, validation", "tab:blue", "s", "-"),
        ("embed_raw", "test", "LeJEPA embedding, test", "tab:blue", "s", ":"),
    ]

    # ------------------------------------------------------------ panel (b)
    slope = parse_concat_log(Path(args.concat_log)) if args.concat_log else SLOPE_REF

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    for kind, split, label, colour, marker, ls in curves:
        if (kind, split) not in rows:
            continue
        xs, ms = zip(*sorted(rows[(kind, split)]))
        axa.plot(xs, ms, label=label, color=colour, marker=marker, ls=ls, ms=4.5,
                 lw=1.7, alpha=1.0 if split == "valid" else 0.75,
                 mfc="white" if split == "test" else colour)

    V = {k: dict(rows[(k, "valid")]) for k in ("morgan", "embed_raw")} if has_valid else {}
    T = {k: dict(rows[(k, "test")]) for k in ("morgan", "embed_raw")}
    axa.axvline(128, color="0.55", lw=1.0, ls=":", zorder=0)
    if has_valid:
        # The crossover, stated without spin: neither representation is simply better.
        axa.annotate(
            f"at matched width (128) the ranking depends on the split:\n"
            f"fingerprint ahead on validation "
            f"({V['morgan'][128]:.3f} vs {V['embed_raw'][128]:.3f}),\n"
            f"embedding ahead on test "
            f"({T['embed_raw'][128]:.3f} vs {T['morgan'][128]:.3f})",
            xy=(136, (V["embed_raw"][128] + T["morgan"][128]) / 2),
            xytext=(205, 0.7375), fontsize=8, color="0.25", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9,
                            connectionstyle="arc3,rad=0.25"))
    axa.annotate("embedding saturates early\n($\\sim$16-32 dims, both splits)",
                 xy=(24, T["embed_raw"][32]), xytext=(8.6, 0.842), fontsize=8.5,
                 color="tab:blue",
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.9))
    axa.set_xscale("log", base=2)
    axa.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    axa.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1k", "2k", "4k"])
    axa.set_xlabel("Representation dimensions retained")
    axa.set_ylabel("ROC-AUC")
    axa.set_ylim(0.700, 0.855)
    axa.set_title("(a) Dimension-efficiency, and a split-dependent ranking",
                  fontsize=10.5, loc="left")
    axa.legend(fontsize=7.8, loc="lower right", framealpha=0.95)
    axa.grid(alpha=0.25, lw=0.6)

    for label, colour, marker, ls in SLOPE_STYLE:
        v, t = slope[label]
        axb.plot([0, 1], [v, t], color=colour, marker=marker, ls=ls, ms=7, lw=2.0)
        axb.annotate(f"  {label}", xy=(1, t), fontsize=9, color=colour, va="center")
        axb.annotate(f"$-${v - t:.3f}  ", xy=(0.5, (v + t) / 2), fontsize=8.5,
                     color=colour, ha="center", va="bottom",
                     bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

    axb.set_xlim(-0.13, 1.78)
    axb.set_ylim(0.800, 0.8575)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Validation scaffolds", "Test scaffolds"])
    axb.set_ylabel("ROC-AUC")
    axb.set_title("(b) The gain is robustness to scaffold shift, not extra fit",
                  fontsize=10.5, loc="left")
    axb.annotate("fingerprint alone and fingerprint\n+ PCA-32 are indistinguishable here",
                 xy=(0.10, 0.8568), fontsize=8.5, color="0.25", va="top")
    axb.grid(alpha=0.25, lw=0.6, axis="y")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()