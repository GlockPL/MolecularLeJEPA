#!/usr/bin/env bash
# =============================================================================
# Multi-partition scaffold robustness driver.
#
# The headline scaffold result rests on a single canonical Bemis-Murcko
# partition. This driver re-runs the per-partition pretrained-vs-scratch
# multiseed (scripts/run_scaffold_multiseed.sh, 5 model seeds per arm) on
# several DIFFERENT, leakage-free scaffold partitions, then pools the canonical
# partition together with the new ones into a single paired bootstrap. The
# question it answers: is the pretraining gain robust across partitions, or an
# artefact of the one canonical split?
#
# The canonical partition (partition 0) is assumed already run; this driver adds
# partitions $PARTITIONS and pools everything at the end.
#
# Usage:
#   bash scripts/run_scaffold_partitions.sh                       # GPS, parts 1-4
#   SCRATCH_TAG=dmpnn512 CONFIG=configs/finetune_dmpnn512_scaffold.yaml \
#     CHECKPOINT=checkpoints/pretrain_chembl_dmpnn/final.pt \
#     bash scripts/run_scaffold_partitions.sh                     # D-MPNN
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/finetune_desc_scaffold.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_chembl/final.pt}"
SCRATCH_TAG="${SCRATCH_TAG:-}"            # set e.g. dmpnn512 for the D-MPNN backbone
PARTITIONS="${PARTITIONS:-1 2 3 4}"      # additional scaffold partitions (0 = canonical)
SEEDS="${SEEDS:-0 1 2 3 4}"              # model seeds per arm (matches canonical)
TASK="${TASK:-antibiotic}"

CKPT_TAG="$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT" .pt)"
CKPT_TAG="${CKPT_TAG// /_}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
DRIVERLOG="logs/scaffold_partitions_${TASK}_${CKPT_TAG}_${STAMP}.log"
log() { echo "$@" | tee -a "$DRIVERLOG"; }

log "=== Multi-partition scaffold robustness driver ==="
log "config=$CONFIG  checkpoint=$CHECKPOINT  scratch_tag='${SCRATCH_TAG}'"
log "partitions=$PARTITIONS  seeds=$SEEDS  task=$TASK"
log "driver log: $DRIVERLOG"

# 1. Run each additional partition (the per-partition runner does both arms).
for P in $PARTITIONS; do
  log ""; log "################## PARTITION $P ##################"
  PARTITION="$P" CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" SCRATCH_TAG="$SCRATCH_TAG" \
    SEEDS="$SEEDS" TASK="$TASK" bash scripts/run_scaffold_multiseed.sh 2>&1 | tee -a "$DRIVERLOG"
done

# 2. Pool canonical (partition 0) + all new partitions into ONE paired bootstrap.
log ""; log "################## POOLED ACROSS PARTITIONS (0 + $PARTITIONS) ##################"
CAN=$(ls -dt checkpoints/scaffold_ms_${TASK}_${CKPT_TAG}_2* 2>/dev/null | grep -v '_part' | head -1)
if [ -z "$CAN" ] || [ ! -f "$CAN/pretrained_ens_probs.csv" ]; then
  log "!! canonical pooled probs not found (looked in '$CAN'); skipping pooled bootstrap."
  log "   Run the canonical partition first (PARTITION unset) so partition 0 exists."
  exit 0
fi
OURS=("$CAN/pretrained_ens_probs.csv"); OTHER=("$CAN/scratch_ens_probs.csv")
for P in $PARTITIONS; do
  RD=$(ls -dt checkpoints/scaffold_ms_${TASK}_${CKPT_TAG}_part${P}_2* 2>/dev/null | head -1)
  if [ -n "$RD" ] && [ -f "$RD/pretrained_ens_probs.csv" ]; then
    OURS+=("$RD/pretrained_ens_probs.csv"); OTHER+=("$RD/scratch_ens_probs.csv")
  else
    log "!! partition $P pooled probs missing — excluded from pool."
  fi
done
log "pooling ${#OURS[@]} partitions (pretrained vs scratch)"
uv run python scripts/paired_bootstrap.py --target "$TASK" \
   --ours "${OURS[@]}" --other "${OTHER[@]}" 2>&1 | tee -a "$DRIVERLOG"
log ""; log "=== done. driver log: $DRIVERLOG ==="
