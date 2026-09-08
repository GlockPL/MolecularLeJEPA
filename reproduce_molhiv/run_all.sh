#!/usr/bin/env bash
# Reproduce the ogbg-molhiv results of the paper end to end.
#
# What it produces, in order:
#   probe    frozen-probe table   - pretrained 0.788 / random-init 0.665 / Morgan 0.807
#   width    width sweep          - Figure (a) data -> logs/width_probe_molhiv.csv
#   concat   the headline result  - Morgan 1024 + LeJEPA PCA-32 under one random forest
#   control  the same sweep with an UNTRAINED backbone (the random-init control)
#   stats    the five paired bootstraps quoted in the paper
#   figure   the two-panel figure -> figures/molhiv_concat_width.png
#
# Run everything (default), or a subset:
#   bash reproduce_molhiv/run_all.sh
#   STEPS="concat stats" bash reproduce_molhiv/run_all.sh
#   STEPS="stats figure" PROBS=reproduce_molhiv/reference/probs \
#       bash reproduce_molhiv/run_all.sh        # verify our numbers, no forests refit
#
# Knobs (env): STEPS CHECKPOINT CONFIG NMODELS NTREES PROBE_TREES PROBS
# Lowering NMODELS/NTREES makes a smoke test cheap, at the cost of comparability
# with the published numbers - the paper's budgets are the defaults here.
#
# Cost: the two concat sweeps dominate at roughly 3 h each on a 16-core/32-thread
# desktop CPU (one 1000-tree forest over the 1056-column matrix takes ~103 s at
# max_features=0.1, ~29 s at sqrt, and the sweep fits 190 of them). The GPU is used
# only for three forward passes of the frozen encoder, well under a minute.
set -uo pipefail
cd "$(dirname "$0")/.."          # repo root

CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_chembl/final.pt}"
CONFIG="${CONFIG:-configs/finetune_molhiv.yaml}"
NMODELS="${NMODELS:-10}"
NTREES="${NTREES:-1000}"      # concat + width sweeps
PROBE_TREES="${PROBE_TREES:-2000}"   # the frozen-probe table uses a larger forest
STEPS="${STEPS:-probe width concat control stats figure}"
# Where the .npz ensemble predictions for the bootstraps come from. Point it at
# reproduce_molhiv/reference/probs to re-check OUR numbers without refitting.
PROBS="${PROBS:-logs}"

mkdir -p logs figures
SUMMARY="logs/molhiv_reproduce_summary.txt"
: > "$SUMMARY"

want() { [[ " $STEPS " == *" $1 "* ]]; }
say()  { echo -e "\n\033[1m=== $* ===\033[0m"; }

if want probe || want width || want concat; then
  if [ ! -f "$CHECKPOINT" ]; then
    echo "FATAL: no checkpoint at $CHECKPOINT."
    echo "It is tracked with Git LFS - run 'git lfs install && git lfs pull'."
    exit 1
  fi
fi

# ---------------------------------------------------------------- 1. frozen probe
if want probe; then
  say "1/6 frozen probe (Table: molhiv frozen probe; ~20 min)"
  uv run python scripts/xgb_molhiv_embed.py --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" --embed-mode pooled \
      --model rf --n-estimators "$PROBE_TREES" --n-models "$NMODELS" \
      2>&1 | tee logs/molhiv_probe_pretrained.out
  uv run python scripts/xgb_molhiv_embed.py --config "$CONFIG" --random-init \
      --embed-mode pooled \
      --model rf --n-estimators "$PROBE_TREES" --n-models "$NMODELS" \
      2>&1 | tee logs/molhiv_probe_randinit.out
  uv run python scripts/xgb_molhiv.py --features morgan --n-bits 2048 \
      --model rf --n-estimators "$PROBE_TREES" --n-models "$NMODELS" \
      2>&1 | tee logs/molhiv_probe_morgan.out
  grep -H "Per-model\|ENSEMBLE" logs/molhiv_probe_*.out | tee -a "$SUMMARY"
fi

# ---------------------------------------------------------------- 2. width sweep
if want width; then
  say "2/6 width sweep (Figure panel a; 500 trees x 5 seeds, ~20 min)"
  uv run python scripts/width_probe_molhiv.py --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" --n-models 5 --n-trees 500 \
      --out logs/width_probe_molhiv.csv 2>&1 | tee logs/width_probe_molhiv.out
fi

# ---------------------------------------------------------------- 3. the result
# Writes logs/concat_<tag>.npz per cell; the pretrained and random-init runs use
# different filename prefixes, so neither overwrites the other.
if want concat; then
  say "3/6 concat sweep, PRETRAINED backbone (the headline result, ~3 h)"
  uv run python scripts/concat_balanced_molhiv.py --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" --n-models "$NMODELS" --n-trees "$NTREES" \
      2>&1 | tee logs/concat_pretrained.out
fi

# ---------------------------------------------------------------- 4. the control
if want control; then
  say "4/6 concat sweep, UNTRAINED backbone (random-init control, ~3 h)"
  uv run python scripts/concat_balanced_molhiv.py --config "$CONFIG" --random-init \
      --n-models "$NMODELS" --n-trees "$NTREES" \
      2>&1 | tee logs/concat_randinit.out
fi

# ---------------------------------------------------------------- 5. significance
# All five contrasts are scored against the SAME valid-selected fingerprint baseline
# (Morgan 1024, max_features=0.1), except the second, which uses the fingerprint cell
# that is strongest on TEST - a deliberately unfavourable reference for us.
if want stats; then
  say "5/6 paired bootstraps (10k resamples over the shared 4111 test rows)"
  boot() {   # <label> <ours.npz> <other.npz>
    echo -e "\n--- $1 ---" | tee -a "$SUMMARY"
    if [ ! -f "$PROBS/$2.npz" ] || [ ! -f "$PROBS/$3.npz" ]; then
      echo "  skipped: missing $PROBS/$2.npz or $PROBS/$3.npz" | tee -a "$SUMMARY"; return
    fi
    uv run python scripts/paired_bootstrap_auc.py \
        --ours "$PROBS/$2.npz" --other "$PROBS/$3.npz" 2>&1 | tee -a "$SUMMARY"
  }
  boot "headline: +PCA-32 vs fingerprint (both valid-selected)   [paper +0.027, p=0.014]" \
       concat_c1024_32_0.1 concat_morgan1024_0.1
  boot "same, against the fingerprint cell strongest on TEST     [paper +0.025, p=0.019]" \
       concat_c1024_32_0.1 concat_morgan1024_sqrt
  boot "dilution control: +PCA-64 vs fingerprint                 [paper +0.008, p=0.29]" \
       concat_c1024_64_0.1 concat_morgan1024_0.1
  boot "random-init control, winning config                      [paper -0.003, p=0.59]" \
       concat_randinit_c1024_32_0.1 concat_morgan1024_0.1
  boot "random-init control, its own valid-selected cell         [paper -0.013, p=0.77]" \
       concat_randinit_c1024_64_0.1 concat_morgan1024_0.1
fi

# ---------------------------------------------------------------- 6. figure
if want figure; then
  say "6/6 figure"
  CSV="logs/width_probe_molhiv.csv"
  [ -f "$CSV" ] || CSV="reproduce_molhiv/reference/width_probe_molhiv.csv"
  LOG=""
  [ -f logs/concat_pretrained.out ] && LOG="--concat-log logs/concat_pretrained.out"
  # shellcheck disable=SC2086
  uv run python scripts/plot_concat_width.py --csv "$CSV" $LOG \
      --out figures/molhiv_concat_width.png
fi

say "done"
echo "Bootstrap summary : $SUMMARY"
echo "Compare against   : reproduce_molhiv/reference/  (our exact run)"
