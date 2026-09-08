#!/usr/bin/env bash
# Full ogbg-molhiv benchmark run: scratch vs LeJEPA-pretrained, the OGB way.
#
#   Phase A  sweep lr (scratch) — 1 seed each
#   Phase B  sweep lr x probe-epochs (pretrained) — 1 seed each
#   Phase C  pick the best-VALIDATION config per model, run it at all final seeds
#            (mean ± std), then print the scratch-vs-pretrained delta.
#
# OGB rules: fixed scaffold split, ROC-AUC, **select on VALID / report TEST**. lr is
# NOT prescribed → swept here, selected by val (never test). scratch & pretrained are
# tuned INDEPENDENTLY (a shared high lr can wash out pretrained features = unfair).
#
# Design (same as run_random_multiseed.sh):
#   * fully logged — master log + per-config step logs + MANIFEST (git commit, picks);
#   * resumable — a config whose step log already has a result is skipped;
#   * fault-tolerant — set -uo pipefail (no -e); a failed config is logged & skipped.
#
# Usage (defaults shown; override via env):
#   bash scripts/run_molhiv_full.sh
#   LRS="1e-3 5e-4 1e-4" PROBES="0 15" SEEDS_FINAL="0,1,2,3,4" bash scripts/run_molhiv_full.sh
#   RUN_PRETRAINED=0 bash scripts/run_molhiv_full.sh        # scratch only
#   CHECKPOINT=checkpoints/pretrain_chembl/epoch_0030.pt bash scripts/run_molhiv_full.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- Parameters -------------------------------------------------------------
CONFIG="${CONFIG:-configs/finetune_molhiv.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_chembl/final.pt}"
LRS="${LRS:-1e-3 5e-4 1e-4 5e-5}"      # lr grid (both models)
PROBES="${PROBES:-0 15}"               # frozen-probe epochs (pretrained only)
SEEDS_FINAL="${SEEDS_FINAL:-0,1,2,3,4}"  # seeds for the final reported run
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"

SWEEP="logs/molhiv_sweep_${STAMP}"
mkdir -p "$SWEEP" logs
MASTER_LOG="logs/molhiv_full_${STAMP}.log"
MANIFEST="$SWEEP/MANIFEST.txt"

log() { echo "$@" | tee -a "$MASTER_LOG"; }

# Run one finetune_molhiv invocation, tee full output to a step log. Resumable:
# skips if the step log already contains a parseable result.
# args: <steplog> <python args...>
run_one() {
  local steplog="$1"; shift
  if [ -f "$steplog" ] && grep -q "best val" "$steplog"; then
    log "    = cached, skip ($steplog)"; return 0
  fi
  uv run python scripts/finetune_molhiv.py --config "$CONFIG" "$@" >"$steplog" 2>&1 \
    || log "    !! run failed (see $steplog)"
}

# Best per-seed VALIDATION ROC-AUC printed by the script ("best val X.XXXX ...").
# (Single-seed sweep → one value; take the max if several.)
parse_val() { grep -oE 'best val [0-9.]+' "$1" 2>/dev/null | awk '{print $3}' | sort -gr | head -1; }
# Final multi-seed line: "  mean ± std = X ± Y".
parse_mean() { grep -E 'mean .* std' "$1" 2>/dev/null | tail -1 | sed 's/.*= //'; }

{
  echo "=== ogbg-molhiv full run ==="
  echo "timestamp   : $STAMP"
  echo "git commit  : $(git rev-parse HEAD 2>/dev/null || echo '?')  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "git dirty   : $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "config      : $CONFIG"
  echo "checkpoint  : $CHECKPOINT"
  echo "lr grid     : $LRS"
  echo "probe grid  : $PROBES (pretrained only)"
  echo "final seeds : $SEEDS_FINAL"
  echo "sweep dir   : $SWEEP"
  echo "master log  : $MASTER_LOG"
  echo "============================"
} | tee "$MANIFEST" | tee -a "$MASTER_LOG"

# =============================================================================
# PHASE A — scratch lr sweep (1 seed)
# =============================================================================
best_scratch_lr=""; best_scratch_val="-1"
if [ "$RUN_SCRATCH" = "1" ]; then
  log ""; log "########## PHASE A — scratch lr sweep ##########"
  for LR in $LRS; do
    steplog="$SWEEP/scratch_lr${LR}.log"
    log "  scratch lr=$LR"
    run_one "$steplog" --seeds 0 --lr "$LR"
    val="$(parse_val "$steplog")"; val="${val:-0}"
    log "    best val = $val"
    if awk "BEGIN{exit !($val > $best_scratch_val)}"; then
      best_scratch_val="$val"; best_scratch_lr="$LR"
    fi
  done
  log "  >> best scratch: lr=$best_scratch_lr (val $best_scratch_val)"
fi

# =============================================================================
# PHASE B — pretrained lr x probe sweep (1 seed)
# =============================================================================
best_pre_lr=""; best_pre_probe=""; best_pre_val="-1"
if [ "$RUN_PRETRAINED" = "1" ]; then
  log ""; log "########## PHASE B — pretrained lr x probe sweep ##########"
  if [ ! -f "$CHECKPOINT" ]; then
    log "  !! checkpoint $CHECKPOINT not found — skipping pretrained."
    RUN_PRETRAINED=0
  fi
fi
if [ "$RUN_PRETRAINED" = "1" ]; then
  for LR in $LRS; do
    for PE in $PROBES; do
      steplog="$SWEEP/pre_lr${LR}_probe${PE}.log"
      log "  pretrained lr=$LR probe=$PE"
      run_one "$steplog" --checkpoint "$CHECKPOINT" --seeds 0 --lr "$LR" --probe-epochs "$PE"
      val="$(parse_val "$steplog")"; val="${val:-0}"
      log "    best val = $val"
      if awk "BEGIN{exit !($val > $best_pre_val)}"; then
        best_pre_val="$val"; best_pre_lr="$LR"; best_pre_probe="$PE"
      fi
    done
  done
  log "  >> best pretrained: lr=$best_pre_lr probe=$best_pre_probe (val $best_pre_val)"
fi

# =============================================================================
# PHASE C — final multi-seed runs at the best-val config(s)
# =============================================================================
log ""; log "########## PHASE C — final ${SEEDS_FINAL} runs at best-val configs ##########"
scratch_mean=""; pre_mean=""

if [ "$RUN_SCRATCH" = "1" ] && [ -n "$best_scratch_lr" ]; then
  steplog="$SWEEP/FINAL_scratch_lr${best_scratch_lr}.log"
  log "  FINAL scratch: lr=$best_scratch_lr seeds=$SEEDS_FINAL"
  if [ -f "$steplog" ] && grep -q "mean" "$steplog"; then log "    = cached"; else
    uv run python scripts/finetune_molhiv.py --config "$CONFIG" \
        --seeds "$SEEDS_FINAL" --lr "$best_scratch_lr" >"$steplog" 2>&1 \
      || log "    !! failed (see $steplog)"
  fi
  scratch_mean="$(parse_mean "$steplog")"
  log "    scratch TEST ROC-AUC = $scratch_mean"
fi

if [ "$RUN_PRETRAINED" = "1" ] && [ -n "$best_pre_lr" ]; then
  steplog="$SWEEP/FINAL_pretrained_lr${best_pre_lr}_probe${best_pre_probe}.log"
  log "  FINAL pretrained: lr=$best_pre_lr probe=$best_pre_probe seeds=$SEEDS_FINAL"
  if [ -f "$steplog" ] && grep -q "mean" "$steplog"; then log "    = cached"; else
    uv run python scripts/finetune_molhiv.py --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --seeds "$SEEDS_FINAL" --lr "$best_pre_lr" --probe-epochs "$best_pre_probe" >"$steplog" 2>&1 \
      || log "    !! failed (see $steplog)"
  fi
  pre_mean="$(parse_mean "$steplog")"
  log "    pretrained TEST ROC-AUC = $pre_mean"
fi

# ---- Summary ----------------------------------------------------------------
{
  echo ""
  echo "================= ogbg-molhiv RESULT ================="
  echo "scratch   : best-val lr=$best_scratch_lr  ->  TEST ROC-AUC = ${scratch_mean:-NA}"
  echo "pretrained: best-val lr=$best_pre_lr probe=$best_pre_probe  ->  TEST ROC-AUC = ${pre_mean:-NA}"
  echo "(GNN-SSL band ~0.75-0.79; GIN-scratch ~0.757, Mole-BERT ~0.787)"
  echo "Delta (pretrained - scratch) = the LeJEPA-helps signal."
  echo "======================================================"
} | tee -a "$MASTER_LOG" | tee -a "$MANIFEST"

log ""; log "Done. Master log: $MASTER_LOG   Manifest: $MANIFEST"