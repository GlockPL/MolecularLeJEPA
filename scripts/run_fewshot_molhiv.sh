#!/usr/bin/env bash
# Few-shot ogbg-molhiv: does LeJEPA pretraining help in the LOW-LABEL regime?
#
# The frozen linear probe showed the pretrained GPS embedding encodes antibiotic-
# relevant structure that random init lacks, yet full finetuning on data-rich tasks
# ties scratch (gain redundant with end-to-end supervision). Prediction: the
# pretrained advantage should RE-APPEAR as labels get scarce. This sweeps the train
# fraction and runs scratch vs pretrained at each budget (val/test fixed), so you get
# a TEST-ROC-AUC-vs-#labels curve with error bars.
#
# Fixed at the best-val finetune configs from the full grid (only train_frac varies):
#   scratch    : lr=SCRATCH_LR
#   pretrained : lr=PRE_LR, probe=PRE_PROBE, --checkpoint CHECKPOINT
#
# Logged (master + per-run + each run's own RunLogger), resumable (skips done runs).
#
# Usage:
#   bash scripts/run_fewshot_molhiv.sh
#   FRACS="0.02 0.05 0.1 0.25 0.5 1.0" SEEDS=0,1,2,3,4 bash scripts/run_fewshot_molhiv.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/finetune_molhiv.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_chembl/final.pt}"
FRACS="${FRACS:-0.02 0.05 0.1 0.25 0.5 1.0}"
SEEDS="${SEEDS:-0,1,2,3,4}"
SCRATCH_LR="${SCRATCH_LR:-5e-4}"          # best-val scratch config
PRE_LR="${PRE_LR:-5e-4}"                  # best-val pretrained config
PRE_PROBE="${PRE_PROBE:-15}"
STAMP="$(date +%Y%m%d_%H%M%S)"

SWEEP="logs/fewshot_molhiv_${STAMP}"
mkdir -p "$SWEEP" logs
MASTER_LOG="logs/fewshot_molhiv_${STAMP}.log"
MANIFEST="$SWEEP/MANIFEST.txt"

log() { echo "$@" | tee -a "$MASTER_LOG"; }
parse_mean() { grep -E 'mean .* std' "$1" 2>/dev/null | tail -1 | sed 's/.*= //'; }

{
  echo "=== Few-shot ogbg-molhiv (scratch vs pretrained vs #labels) ==="
  echo "timestamp : $STAMP"
  echo "git       : $(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "config    : $CONFIG     checkpoint: $CHECKPOINT"
  echo "fracs     : $FRACS"
  echo "seeds     : $SEEDS"
  echo "scratch lr: $SCRATCH_LR    pretrained lr=$PRE_LR probe=$PRE_PROBE"
  echo "================================================================"
} | tee "$MANIFEST" | tee -a "$MASTER_LOG"

run_one() {  # <steplog> <python args...>
  local steplog="$1"; shift
  if [ -f "$steplog" ] && grep -q "mean" "$steplog"; then log "    = cached"; return 0; fi
  uv run python scripts/finetune_molhiv.py --config "$CONFIG" "$@" >"$steplog" 2>&1 \
    || log "    !! failed (see $steplog)"
}

declare -A S_MEAN P_MEAN
for FRAC in $FRACS; do
  log ""; log "########## train_frac = $FRAC ##########"

  s_log="$SWEEP/scratch_frac${FRAC}.log"
  log "  scratch  frac=$FRAC"
  run_one "$s_log" --seeds "$SEEDS" --lr "$SCRATCH_LR" --train-frac "$FRAC"
  S_MEAN[$FRAC]="$(parse_mean "$s_log")"
  log "    scratch    = ${S_MEAN[$FRAC]:-NA}"

  p_log="$SWEEP/pretrained_frac${FRAC}.log"
  log "  pretrained  frac=$FRAC"
  run_one "$p_log" --checkpoint "$CHECKPOINT" --seeds "$SEEDS" \
      --lr "$PRE_LR" --probe-epochs "$PRE_PROBE" --train-frac "$FRAC"
  P_MEAN[$FRAC]="$(parse_mean "$p_log")"
  log "    pretrained = ${P_MEAN[$FRAC]:-NA}"
done

{
  echo ""
  echo "============== FEW-SHOT CURVE (molhiv TEST ROC-AUC) =============="
  printf "%-10s %-22s %-22s\n" "frac" "scratch" "pretrained"
  for FRAC in $FRACS; do
    printf "%-10s %-22s %-22s\n" "$FRAC" "${S_MEAN[$FRAC]:-NA}" "${P_MEAN[$FRAC]:-NA}"
  done
  echo "Prediction: pretrained − scratch should be LARGEST at small frac and"
  echo "shrink toward 0 at frac=1.0 (where we already measured a tie)."
  echo "================================================================="
} | tee -a "$MASTER_LOG" | tee -a "$MANIFEST"

log ""; log "Done. Master log: $MASTER_LOG   Manifest: $MANIFEST"