#!/usr/bin/env bash
# Multi-seed random-split significance run: train our ensemble AND a faithful
# Chemprop ensemble on K independent random splits, then a single POOLED paired
# bootstrap of the AUPRC difference. Pooling ~K-multiplies the positive count
# (single split has only 102 antibiotic positives → ΔAUPRC CI spans 0); this is
# the protocol that can turn the +0.04 point-win into a significant one.
#
# Design (per your requirements):
#   * NOTHING OVERWRITTEN — every seed writes to its own *_seed<N> dirs; the
#     existing unsuffixed dirs (ens_random_pretrained, chemprop_random_antibiotic,
#     random_split_antibiotic) are left untouched.
#   * RESUMABLE — each step is skipped if its output already exists, so a crash or
#     a half-finished night just continues on rerun.
#   * OUR ENSEMBLES FIRST — Phase A does all our (GPU) ensembles before any
#     Chemprop, so "at least my ensemble trains" is guaranteed.
#   * FULLY LOGGED — a master log + per-seed step logs + a MANIFEST documenting
#     params/git-commit/results. (finetune.py and ensemble_eval.py also write
#     their own logs to logs/, so coverage is triple.)
#   * FAULT-TOLERANT — a single seed's failure is logged and skipped, not fatal;
#     the final bootstrap pools whatever pairs completed.
#
# Usage (defaults shown; override via env):
#   bash scripts/run_random_multiseed.sh
#   SEEDS="42 1 2 3 4 5 6" N=10 bash scripts/run_random_multiseed.sh
#   CHECKPOINT="" bash scripts/run_random_multiseed.sh        # scratch instead of pretrained
#   RUN_CHEMPROP=0 bash scripts/run_random_multiseed.sh       # our side only
set -uo pipefail                         # NOT -e: we want to survive a single-seed failure
cd "$(dirname "$0")/.."

# ---- Parameters (all overridable) -------------------------------------------
SEEDS="${SEEDS:-42 1 2 3 4}"             # the K split seeds to pool
CONFIG="${CONFIG:-configs/finetune_desc.yaml}"          # GPS random-split config
CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_chembl/final.pt}"  # "" => scratch
TASK="${TASK:-antibiotic}"
N="${N:-10}"                             # ensemble members per seed
RUN_CHEMPROP="${RUN_CHEMPROP:-1}"        # 1 = also train Chemprop control
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -n "$CHECKPOINT" ]; then TAG=pretrained; CKPT_ARG=(--checkpoint "$CHECKPOINT")
else                          TAG=scratch;    CKPT_ARG=(); fi

mkdir -p logs
RUNDIR="checkpoints/multiseed_${TASK}_${TAG}_${STAMP}"
mkdir -p "$RUNDIR"
MASTER_LOG="logs/multiseed_${TASK}_${TAG}_${STAMP}.log"
MANIFEST="$RUNDIR/MANIFEST.txt"

# ---- Logging helpers --------------------------------------------------------
log()  { echo "$@" | tee -a "$MASTER_LOG"; }
# run <steplog> <description> <cmd...> : full output → steplog; status → master.
run() {
  local logf="$1"; local desc="$2"; shift 2
  log "  + $desc"
  log "    \$ $*"
  if "$@" >>"$logf" 2>&1; then log "    ok"; return 0
  else log "    !! FAILED (exit $?) — see $logf"; return 1; fi
}

# ---- Manifest / header ------------------------------------------------------
{
  echo "=== Multi-seed random-split significance run ==="
  echo "timestamp     : $STAMP"
  echo "git commit    : $(git rev-parse HEAD 2>/dev/null || echo '(not a git repo)')"
  echo "git branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "git dirty     : $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "task          : $TASK"
  echo "seeds         : $SEEDS"
  echo "config        : $CONFIG"
  echo "checkpoint    : ${CHECKPOINT:-'(none — scratch)'}   (tag=$TAG)"
  echo "members (N)   : $N"
  echo "run chemprop  : $RUN_CHEMPROP"
  echo "run dir       : $RUNDIR"
  echo "master log    : $MASTER_LOG"
  echo "================================================"
} | tee "$MANIFEST" | tee -a "$MASTER_LOG"

# Per-seed path helpers (kept consistent across all phases).
split_dir() { echo "splits_out/random_split_${TASK}_seed$1"; }
ens_dir()   { echo "checkpoints/ens_random_${TASK}_${TAG}_seed$1"; }
cp_dir()    { echo "checkpoints/chemprop_random_${TASK}_seed$1"; }
probs_csv() { echo "$(ens_dir "$1")/ensemble_probs_test.csv"; }
preds_csv() { echo "$(cp_dir "$1")/test_preds.csv"; }

# =============================================================================
# PHASE A — our ensembles (GPU). Done first so they finish even if the night is short.
# =============================================================================
log ""
log "########## PHASE A — our $TAG ensembles ($TASK) ##########"
for SEED in $SEEDS; do
  SPLIT="$(split_dir "$SEED")"; ENS="$(ens_dir "$SEED")"; PROBS="$(probs_csv "$SEED")"
  STEPLOG="logs/multiseed_${TASK}_${TAG}_seed${SEED}_ours_${STAMP}.log"
  log ""
  log "--- seed $SEED : our ensemble → $ENS ---"

  # 1. Export the split CSVs (documents the exact split; Chemprop trains on these,
  #    and OUR model regenerates the IDENTICAL split internally from --data-seed).
  if [ -f "$SPLIT/test.csv" ]; then log "  = split exists, skip ($SPLIT)"
  else run "$STEPLOG" "export split seed=$SEED" \
        uv run python scripts/export_chemprop_splits.py \
          --task "$TASK" --split-method random --seed "$SEED" --out "$SPLIT"; fi

  # 2. Train the ensemble (skip if the last member's best ckpt already exists).
  LAST_MEMBER="$ENS/member_$(printf '%02d' $((N-1)))/best_${TASK}.pt"
  if [ -f "$LAST_MEMBER" ]; then log "  = ensemble complete, skip ($ENS)"
  else run "$STEPLOG" "train ensemble seed=$SEED n=$N" \
        uv run python scripts/train_ensemble.py \
          --config "$CONFIG" "${CKPT_ARG[@]}" --data-seed "$SEED" \
          --task "$TASK" --n "$N" --out "$ENS"; fi

  # 3. Evaluate + save per-compound probs (skip if probs already saved).
  if [ -f "$PROBS" ]; then log "  = probs exist, skip ($PROBS)"
  else run "$STEPLOG" "eval + save-probs seed=$SEED" \
        uv run python scripts/ensemble_eval.py \
          --config "$CONFIG" --data-seed "$SEED" \
          --ensemble-dir "$ENS" --save-probs "$PROBS" --individual; fi
done

# =============================================================================
# PHASE B — Chemprop control (one ensemble per split). Optional / slower.
# =============================================================================
if [ "$RUN_CHEMPROP" = "1" ]; then
  log ""
  log "########## PHASE B — Chemprop control ($TASK) ##########"
  for SEED in $SEEDS; do
    SPLIT="$(split_dir "$SEED")"; CPOUT="$(cp_dir "$SEED")"; PREDS="$(preds_csv "$SEED")"
    STEPLOG="logs/multiseed_${TASK}_seed${SEED}_chemprop_${STAMP}.log"
    log ""
    log "--- seed $SEED : chemprop → $CPOUT ---"

    if [ -f "$PREDS" ]; then log "  = chemprop preds exist, skip ($PREDS)"; continue; fi

    # Reuse the already-trained seed-42 antibiotic Chemprop (3h artifact) instead
    # of retraining — copy it into the suffixed dir (originals untouched, documented).
    OLD_PREDS="checkpoints/chemprop_random_antibiotic/test_preds.csv"
    if [ "$SEED" = "42" ] && [ "$TASK" = "antibiotic" ] && [ -f "$OLD_PREDS" ]; then
      mkdir -p "$CPOUT"
      cp "$OLD_PREDS" "$PREDS"
      log "  = reused existing seed-42 chemprop preds: $OLD_PREDS → $PREDS"
      continue
    fi

    [ -f "$SPLIT/test.csv" ] || run "$STEPLOG" "export split seed=$SEED" \
        uv run python scripts/export_chemprop_splits.py \
          --task "$TASK" --split-method random --seed "$SEED" --out "$SPLIT"
    run "$STEPLOG" "chemprop train+predict seed=$SEED" \
        env SEED="$SEED" SPLIT="$SPLIT" OUT="$CPOUT" TASK="$TASK" \
          bash scripts/run_chemprop_random.sh
  done
fi

# =============================================================================
# PHASE C — pooled paired bootstrap over every seed where BOTH sides finished.
# =============================================================================
log ""
log "########## PHASE C — pooled paired bootstrap ##########"
OURS_LIST=(); OTHER_LIST=(); USED_SEEDS=()
for SEED in $SEEDS; do
  P="$(probs_csv "$SEED")"; C="$(preds_csv "$SEED")"
  if [ -f "$P" ] && [ -f "$C" ]; then
    OURS_LIST+=("$P"); OTHER_LIST+=("$C"); USED_SEEDS+=("$SEED")
  else
    log "  (skip seed $SEED — missing $( [ -f "$P" ] || echo ours ) $( [ -f "$C" ] || echo chemprop ))"
  fi
done

if [ "${#OURS_LIST[@]}" -eq 0 ]; then
  log "  No complete seed pairs yet — rerun after training finishes."
  exit 0
fi

log "  Pooling seeds: ${USED_SEEDS[*]}"
{
  echo ""; echo "=== POOLED PAIRED BOOTSTRAP (seeds ${USED_SEEDS[*]}) ==="
  uv run python scripts/paired_bootstrap.py --target "$TASK" \
      --ours "${OURS_LIST[@]}" --other "${OTHER_LIST[@]}"
} 2>&1 | tee -a "$MASTER_LOG" | tee -a "$MANIFEST"

log ""
log "Done. Master log: $MASTER_LOG   Manifest: $MANIFEST"