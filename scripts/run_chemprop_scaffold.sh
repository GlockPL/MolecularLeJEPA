#!/usr/bin/env bash
# Replicate the Wong et al. 2023 Chemprop v1 model on the GPS model's exact
# scaffold split for a given task, for a head-to-head AUPRC comparison.
#
# Task is selected via the TASK env var (default antibiotic):
#     TASK=antibiotic bash scripts/run_chemprop_scaffold.sh
#     TASK=hepg2      bash scripts/run_chemprop_scaffold.sh
#     TASK=hskmc      bash scripts/run_chemprop_scaffold.sh
#     TASK=imr90      bash scripts/run_chemprop_scaffold.sh
#
# Idempotent / resumable: if test_preds.csv already exists for the task, the
# train+predict steps are skipped and only scoring is re-run. Run from a stable
# terminal.
#
# Produces the comparison number at the end (AUPRC + 95% CI on the test split
# that is IDENTICAL to `main.py finetune --task <TASK>` with split_method:
# scaffold).
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-antibiotic}"
VENV=.venv-cp1
SPLIT="splits_out/scaffold_split_${TASK}"
HP=configs/chemprop_hyperparameters.json   # depth5 dropout.35 ffn3 hidden1600
OUT="checkpoints/chemprop_scaffold_${TASK}"

# Back-compat: the original antibiotic run used unsuffixed dirs. Reuse them so
# the completed antibiotic ensemble isn't retrained.
if [ "$TASK" = "antibiotic" ] && [ -d splits_out/scaffold_split ] \
   && [ ! -d "$SPLIT" ]; then
  SPLIT=splits_out/scaffold_split
  OUT=checkpoints/chemprop_scaffold
fi

echo "=== Chemprop scaffold run — task=${TASK}  split=${SPLIT}  out=${OUT} ==="

# 1. Toolchain: chemprop v1 in an isolated Python 3.8 venv.
if [ ! -f "$VENV/bin/chemprop_train" ]; then
  [ -d "$VENV" ] || uv venv "$VENV" --python 3.8
  # If this step is killed (exit 144) in a constrained sandbox, run it in a
  # plain terminal — it just needs network + time to fetch torch.
  # setuptools provides pkg_resources, which chemprop v1's deps (hyperopt) need
  # and which uv venvs do not include by default.
  uv pip install --python "$VENV" "chemprop>=1.6,<2" descriptastorus setuptools
fi

# 2. Split CSVs (deterministic; identical to the GPS --task $TASK split).
[ -f "$SPLIT/test.csv" ] || \
  uv run python scripts/export_chemprop_splits.py --task "$TASK" --out "$SPLIT"

# 3+4. Train ensemble of 10 + predict — skipped if predictions already exist.
if [ -f "$OUT/test_preds.csv" ]; then
  echo "Found $OUT/test_preds.csv — skipping train/predict, re-scoring only."
else
  rm -rf "$OUT"
  "$VENV/bin/chemprop_train" \
    --data_path "$SPLIT/train.csv" \
    --separate_val_path "$SPLIT/val.csv" \
    --separate_test_path "$SPLIT/test.csv" \
    --dataset_type classification \
    --smiles_columns smiles --target_columns "$TASK" \
    --config_path "$HP" \
    --ensemble_size 10 \
    --features_generator rdkit_2d_normalized --no_features_scaling \
    --metric prc-auc --extra_metrics auc \
    --gpu 0 \
    --save_dir "$OUT"

  "$VENV/bin/chemprop_predict" \
    --test_path "$SPLIT/test.csv" \
    --checkpoint_dir "$OUT" \
    --smiles_columns smiles \
    --features_generator rdkit_2d_normalized --no_features_scaling \
    --preds_path "$OUT/test_preds.csv"
fi

# 5. Score with the repo's own AUPRC + bootstrap CI (comparable to GPS numbers).
uv run python scripts/score_chemprop.py \
  --labels "$SPLIT/test.csv" --preds "$OUT/test_preds.csv" --target "$TASK"