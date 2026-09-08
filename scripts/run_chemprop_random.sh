#!/usr/bin/env bash
# Replicate the Wong et al. 2023 Chemprop v1 model on a RANDOM 80/20 split —
# the paper's OWN protocol (the paper reports AUPRC 0.364 on a random split;
# scaffold splitting appears only later for the G1-G5 rationale analysis). This
# is the control that confirms the published ~0.36 number and is the head-to-
# head partner for our random-split GPS/D-MPNN ensemble.
#
# Twin of run_chemprop_scaffold.sh — ONLY the split differs (random, stratified,
# seed 42 to match data.seed in the finetune configs). Task via TASK env:
#     TASK=antibiotic bash scripts/run_chemprop_random.sh
#
# Idempotent / resumable: if test_preds.csv exists, only scoring re-runs.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-antibiotic}"
SEED="${SEED:-42}"                                      # must equal data.seed in finetune config
VENV=.venv-cp1
# SPLIT and OUT are overridable (the multi-seed orchestrator passes seed-suffixed
# dirs so nothing is overwritten); defaults keep the original single-run behavior.
SPLIT="${SPLIT:-splits_out/random_split_${TASK}}"
HP=configs/chemprop_hyperparameters.json   # depth5 dropout.35 ffn3 hidden1600
OUT="${OUT:-checkpoints/chemprop_random_${TASK}}"

echo "=== Chemprop RANDOM run — task=${TASK}  seed=${SEED}  split=${SPLIT}  out=${OUT} ==="

# 1. Toolchain: chemprop v1 in an isolated Python 3.8 venv (shared with scaffold run).
if [ ! -f "$VENV/bin/chemprop_train" ]; then
  [ -d "$VENV" ] || uv venv "$VENV" --python 3.8
  uv pip install --python "$VENV" "chemprop>=1.6,<2" descriptastorus setuptools
fi

# 2. Split CSVs — RANDOM, stratified, seed-matched to the GPS/D-MPNN finetune split.
[ -f "$SPLIT/test.csv" ] || \
  uv run python scripts/export_chemprop_splits.py \
    --task "$TASK" --split-method random --seed "$SEED" --out "$SPLIT"

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