"""Export the GPS model's exact antibiotic scaffold split to Chemprop CSVs.

Reproduces the *identical* train/val/test partition the GPS model uses for the
antibiotic-only task: same MOESM3 source + label cutoffs (via ``load_xlsx``),
same deterministic Bemis-Murcko ``_scaffold_split``. This lets a Chemprop v1
model be trained/evaluated head-to-head with the GPS model on the same held-out
test compounds.

The matching GPS run is:
    uv run python main.py finetune --config configs/finetune_desc_scaffold.yaml \
        --checkpoint <ckpt> --task antibiotic

Output (default ``splits_out/scaffold_split_<task>/``):
    train.csv, val.csv, test.csv  with columns: smiles,<task>

Usage:
    uv run python scripts/export_chemprop_splits.py --task antibiotic
    uv run python scripts/export_chemprop_splits.py --task hepg2
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Import the real loader + split so the partition is identical to finetune.py.
from src.data.antibiotic_dataset import (
    LABEL_COLS, load_xlsx, _scaffold_split, _random_split,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/wang et al/41586_2023_6887_MOESM3_ESM.xlsx")
    ap.add_argument("--task", default="antibiotic", choices=list(LABEL_COLS),
                    help="Single task to export (inner-joined to its own "
                         "compound set, matching `finetune ... --task <task>`).")
    ap.add_argument("--split-method", default="scaffold", choices=["scaffold", "random"],
                    help="Must match the finetune config's data.split_method so the "
                         "Chemprop control trains on the SAME split as our model. "
                         "'random' reproduces Wong et al.'s 80/20 protocol.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Split seed (random split only) — must equal data.seed in the "
                         "finetune config (default 42) for member-identical splits.")
    ap.add_argument("--out", default=None,
                    help="Output dir (default splits_out/<split-method>_split_<task>)")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.10)
    args = ap.parse_args()

    task = args.task
    out = Path(args.out) if args.out else Path(f"splits_out/{args.split_method}_split_{task}")
    out.mkdir(parents=True, exist_ok=True)

    # Single-task inner-join -> keeps every compound screened for this task,
    # matching `finetune ... --task <task>` (AntibioticDataset uses tasks=[task]).
    df = load_xlsx(Path(args.xlsx), tasks=[task])
    if args.split_method == "scaffold":
        idx_train, idx_val, idx_test = _scaffold_split(df, args.test_size, args.val_size)
    else:
        # Same call AntibioticDataset makes -> identical random split + test set.
        idx_train, idx_val, idx_test = _random_split(
            df, args.test_size, args.val_size, args.seed, stratify_col=task,
        )

    for name, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        sub = df.iloc[idx][["smiles", task]].reset_index(drop=True)
        sub[task] = sub[task].astype(int)
        sub.to_csv(out / f"{name}.csv", index=False)
        n_pos = int(sub[task].sum())
        print(f"{name:5s}: {len(sub):6d} rows  {task}_pos={n_pos:4d} "
              f"({100*n_pos/max(len(sub),1):.2f}%)  -> {out / f'{name}.csv'}")

    total = len(idx_train) + len(idx_val) + len(idx_test)
    print(f"total : {total} rows")


if __name__ == "__main__":
    main()