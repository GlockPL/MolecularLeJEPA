#!/usr/bin/env bash
# Multi-seed SCAFFOLD-split recheck of the pretrained-vs-scratch GPS finetune.
#
# WHY: the paper reports a single-seed scaffold lift (scratch 0.226 -> pretrained
# 0.260, Δ=+0.034 "at a single seed"). That number is flagged as suspect. This
# wrapper re-runs it across K independent MODEL seeds on the FIXED scaffold split
# and reports mean ± s.d. so the +0.034 is either confirmed or replaced with a
# clean multi-seed null.
#
# KEY DIFFERENCE vs run_random_multiseed.sh: the scaffold split is the single
# canonical Bemis-Murcko split — there is nothing to pool across. So here the
# seeds vary the MODEL (head init + training stochasticity) while the split stays
# fixed at cfg.data.seed. We get (a) per-seed test AUPRC -> mean±s.d. per arm and
# the per-seed paired Δ, and (b) a paired bootstrap on the K-seed mean ("ensemble")
# probabilities over the one shared scaffold test set (~164 positives).
#
# Design (same conventions as run_random_multiseed.sh):
#   * NOTHING OVERWRITTEN — each (arm, seed) writes its own dir.
#   * RESUMABLE — a step is skipped if its output already exists.
#   * PRETRAINED FIRST — the arm under test trains before the scratch control.
#   * FULLY LOGGED — master log + per-step logs + MANIFEST (git commit, params).
#   * FAULT-TOLERANT — set -uo pipefail (no -e); a single seed failure is skipped.
#
# Usage (defaults shown; override via env):
#   bash scripts/run_scaffold_multiseed.sh
#   SEEDS="0 1 2 3 4 5 6" bash scripts/run_scaffold_multiseed.sh
#   CHECKPOINT=checkpoints/pretrain_chembl/final.pt bash scripts/run_scaffold_multiseed.sh
#   CONFIG=configs/finetune_desc_scaffold.yaml TASK=antibiotic bash scripts/run_scaffold_multiseed.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- Parameters (all overridable) -------------------------------------------
SEEDS="${SEEDS:-0 1 2 3 4}"                                  # MODEL seeds (split is fixed)
CONFIG="${CONFIG:-configs/finetune_desc_scaffold.yaml}"      # GPS scaffold config
CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain_gps_p7_cover025/final.pt}"  # arm under test
TASK="${TASK:-antibiotic}"
# The scratch arm is checkpoint-independent, but it IS architecture-dependent.
# Namespace it so a different backbone (e.g. D-MPNN) does not silently reuse the
# GPS scratch dirs. Leave empty for the default GPS run; set e.g. dmpnn512 for the
# D-MPNN control. CONFIG/CHECKPOINT must match this architecture.
SCRATCH_TAG="${SCRATCH_TAG:-}"
# Scaffold PARTITION seed. Empty = the canonical deterministic partition (the
# paper's headline split). An integer selects a different, leakage-free scaffold
# partition (multi-partition robustness check); all output dirs and the split are
# namespaced by it so partitions never collide. Pass the SAME value to both the
# pretrained and scratch arms (handled automatically here).
PARTITION="${PARTITION:-}"
if [ -n "$PARTITION" ]; then PART_ARG=(--partition-seed "$PARTITION"); PART_SFX="_part${PARTITION}"
else                        PART_ARG=();                                PART_SFX=""; fi
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$CHECKPOINT" ]; then
  echo "!! checkpoint not found: $CHECKPOINT" >&2; exit 1
fi

# Tag derived from the checkpoint so the PRETRAINED arm's output dirs are unique
# per checkpoint. Without this, a rerun with a different --checkpoint would silently
# reuse another checkpoint's pretrained dirs (resume keys on (arm,seed) only). The
# SCRATCH arm has no checkpoint, so it stays untagged and is correctly shared/reused.
CKPT_TAG="$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT" .pt)"
CKPT_TAG="${CKPT_TAG// /_}"

mkdir -p logs
RUNDIR="checkpoints/scaffold_ms_${TASK}_${CKPT_TAG}${PART_SFX}_${STAMP}"
mkdir -p "$RUNDIR"
MASTER_LOG="logs/scaffold_ms_${TASK}_${CKPT_TAG}${PART_SFX}_${STAMP}.log"
MANIFEST="$RUNDIR/MANIFEST.txt"

# ---- Logging helpers --------------------------------------------------------
log()  { echo "$@" | tee -a "$MASTER_LOG"; }
run() {  # run <steplog> <description> <cmd...>
  local logf="$1"; local desc="$2"; shift 2
  log "  + $desc"; log "    \$ $*"
  if "$@" >>"$logf" 2>&1; then log "    ok"; return 0
  else log "    !! FAILED (exit $?) — see $logf"; return 1; fi
}

# ---- Manifest / header ------------------------------------------------------
{
  echo "=== Multi-seed SCAFFOLD-split pretrained-vs-scratch recheck ==="
  echo "timestamp     : $STAMP"
  echo "git commit    : $(git rev-parse HEAD 2>/dev/null || echo '(not a git repo)')"
  echo "git branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "git dirty     : $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "task          : $TASK"
  echo "model seeds   : $SEEDS   (scaffold split FIXED at cfg.data.seed)"
  echo "config        : $CONFIG"
  echo "checkpoint    : $CHECKPOINT  (pretrained arm; tag=$CKPT_TAG)"
  echo "run dir       : $RUNDIR"
  echo "master log    : $MASTER_LOG"
  echo "================================================"
} | tee "$MANIFEST" | tee -a "$MASTER_LOG"

# Per-(arm,seed) path helpers. PRETRAINED is namespaced by CKPT_TAG; SCRATCH is
# checkpoint-independent (shared across checkpoint reruns — same config/seeds/split).
arm_dir() {  # $1=arm $2=seed
  if [ "$1" = "pretrained" ]; then echo "checkpoints/scaffold_ms_${TASK}_pretrained_${CKPT_TAG}${PART_SFX}_seed$2"
  else                             echo "checkpoints/scaffold_ms_${TASK}_scratch${SCRATCH_TAG:+_$SCRATCH_TAG}${PART_SFX}_seed$2"; fi
}
member_ck() { echo "$(arm_dir "$1" "$2")/member_00/best_${TASK}.pt"; }
probs_csv() { echo "$(arm_dir "$1" "$2")/ensemble_probs_test.csv"; }

# =============================================================================
# Train + eval both arms across the model seeds.
# pretrained first (the arm under test), then scratch (control).
# =============================================================================
for ARM in pretrained scratch; do
  if [ "$ARM" = "pretrained" ]; then CKPT_ARG=(--checkpoint "$CHECKPOINT"); else CKPT_ARG=(); fi
  log ""
  log "########## ARM: $ARM ##########"
  for SEED in $SEEDS; do
    DIR="$(arm_dir "$ARM" "$SEED")"; CK="$(member_ck "$ARM" "$SEED")"; PROBS="$(probs_csv "$ARM" "$SEED")"
    STEPLOG="logs/scaffold_ms_${TASK}_${ARM}${PART_SFX}_seed${SEED}_${STAMP}.log"
    log ""
    log "--- $ARM seed $SEED -> $DIR ---"

    # 1. Train a single model (1-member ensemble) at this model seed on the fixed split.
    MARKER="$DIR/SOURCE_CHECKPOINT.txt"
    if [ -f "$CK" ]; then
      # Belt-and-suspenders: refuse to reuse a pretrained model that was trained
      # from a DIFFERENT checkpoint than the one requested now (the bug this guards).
      if [ "$ARM" = "pretrained" ] && [ -f "$MARKER" ] \
         && [ "$(cat "$MARKER")" != "$CHECKPOINT" ]; then
        log "  !! $DIR was trained from '$(cat "$MARKER")', not '$CHECKPOINT' — aborting."
        log "     Remove $DIR (or rerun with a distinct checkpoint) and try again."
        exit 1
      fi
      log "  = model exists, skip ($CK)"
    else
      run "$STEPLOG" "train $ARM seed=$SEED" \
          uv run python scripts/train_ensemble.py \
            --config "$CONFIG" "${CKPT_ARG[@]}" "${PART_ARG[@]}" --seeds "$SEED" \
            --task "$TASK" --out "$DIR"
      [ "$ARM" = "pretrained" ] && echo "$CHECKPOINT" > "$MARKER"
    fi

    # 2. Eval on the test split + save this seed's per-compound probs.
    if [ -f "$PROBS" ]; then log "  = probs exist, skip ($PROBS)"
    else run "$STEPLOG" "eval + save-probs $ARM seed=$SEED" \
          uv run python scripts/ensemble_eval.py \
            --config "$CONFIG" --ensemble-dir "$DIR" "${PART_ARG[@]}" \
            --save-probs "$PROBS" --individual; fi
  done
done

# =============================================================================
# Aggregate: per-seed AUPRC -> mean±s.d. per arm + per-seed paired Δ; then write
# each arm's K-seed-mean probs and run the paired bootstrap on the shared split.
# =============================================================================
log ""
log "########## AGGREGATE ##########"
PRE_ENS="$RUNDIR/pretrained_ens_probs.csv"
SCR_ENS="$RUNDIR/scratch_ens_probs.csv"

AGG_LOG="$RUNDIR/aggregate.txt"
uv run python - "$TASK" "$RUNDIR" "$PRE_ENS" "$SCR_ENS" "$CKPT_TAG" "$SCRATCH_TAG" "$PART_SFX" $SEEDS <<'PY' 2>&1 | tee -a "$AGG_LOG" | tee -a "$MASTER_LOG"
import sys, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

task, rundir, pre_ens, scr_ens, ckpt_tag, scratch_tag, part_sfx, *seeds = sys.argv[1:]
prob_col, label_col = f"{task}_prob", f"{task}_label"

def arm_csv(arm, seed):
    # Must mirror arm_dir() in the bash driver EXACTLY: pretrained is namespaced
    # by the checkpoint tag, scratch by SCRATCH_TAG, and BOTH by the partition
    # suffix. (Hardcoding these without the tags silently reads the canonical
    # GPS dirs — the bug that contaminated tagged/partitioned runs.)
    if arm == "pretrained":
        return f"checkpoints/scaffold_ms_{task}_pretrained_{ckpt_tag}{part_sfx}_seed{seed}/ensemble_probs_test.csv"
    st = f"_{scratch_tag}" if scratch_tag else ""
    return f"checkpoints/scaffold_ms_{task}_scratch{st}{part_sfx}_seed{seed}/ensemble_probs_test.csv"

def load(arm, seed):
    f = arm_csv(arm, seed)
    if not os.path.exists(f): return None
    df = pd.read_csv(f)
    return df[["smiles", prob_col, label_col]].dropna()

per = {"pretrained": {}, "scratch": {}}
ref = None
for arm in ("pretrained", "scratch"):
    for s in seeds:
        df = load(arm, s)
        if df is None:
            print(f"  (missing {arm} seed {s})"); continue
        if ref is None: ref = df[["smiles", label_col]].copy()
        per[arm][s] = df.set_index("smiles")[prob_col]

if ref is None:
    print("  No probs found — nothing to aggregate."); sys.exit(0)

y = ref.set_index("smiles")[label_col].astype(int)

def auprc_of(probs):
    p = probs.reindex(y.index)
    m = p.notna()
    return float(average_precision_score(y[m], p[m]))

print(f"\nPer-seed test AUPRC ({task}, scaffold split, {len(y)} cpds, {int(y.sum())} pos):")
print(f"  {'seed':>6}  {'scratch':>9}  {'pretrained':>11}  {'Δ(pre−scr)':>11}")
deltas, scr_v, pre_v, common = [], [], [], []
for s in seeds:
    if s in per["scratch"] and s in per["pretrained"]:
        a_scr = auprc_of(per["scratch"][s]); a_pre = auprc_of(per["pretrained"][s])
        d = a_pre - a_scr
        print(f"  {s:>6}  {a_scr:9.4f}  {a_pre:11.4f}  {d:+11.4f}")
        scr_v.append(a_scr); pre_v.append(a_pre); deltas.append(d); common.append(s)

if deltas:
    scr_v, pre_v, deltas = map(np.asarray, (scr_v, pre_v, deltas))
    n = len(deltas)
    print(f"\n  scratch     : {scr_v.mean():.4f} ± {scr_v.std(ddof=1):.4f}  (n={n})")
    print(f"  pretrained  : {pre_v.mean():.4f} ± {pre_v.std(ddof=1):.4f}  (n={n})")
    print(f"  Δ per-seed  : {deltas.mean():+.4f} ± {deltas.std(ddof=1):.4f}")
    print(f"  seeds with pretrained > scratch: {int((deltas>0).sum())}/{n}")
    # paired t-ish: mean / (sd/sqrt n)
    se = deltas.std(ddof=1)/np.sqrt(n) if n > 1 else float('nan')
    if n > 1 and se > 0:
        print(f"  Δ mean / SE : {deltas.mean()/se:+.2f}  (|t|>~2 ⇒ ~95% the sign is real)")

# K-seed-mean ("ensemble") probabilities per arm, aligned to the shared test set.
def ens_frame(arm):
    cols = [per[arm][s].reindex(y.index) for s in common]
    mean = pd.concat(cols, axis=1).mean(axis=1)
    out = pd.DataFrame({"smiles": y.index, f"{task}_prob": mean.values,
                        f"{task}_label": y.values})
    return out.dropna()

if common:
    ens_frame("pretrained").to_csv(pre_ens, index=False)
    ens_frame("scratch").to_csv(scr_ens, index=False)
    print(f"\n  wrote ensemble probs: {pre_ens} , {scr_ens}")
    print(f"  ensemble AUPRC  scratch={average_precision_score(pd.read_csv(scr_ens)[f'{task}_label'], pd.read_csv(scr_ens)[f'{task}_prob']):.4f}"
          f"  pretrained={average_precision_score(pd.read_csv(pre_ens)[f'{task}_label'], pd.read_csv(pre_ens)[f'{task}_prob']):.4f}")
PY

# Paired bootstrap on the K-seed-mean probs over the shared scaffold test set.
if [ -f "$PRE_ENS" ] && [ -f "$SCR_ENS" ]; then
  log ""
  log "--- paired bootstrap (pretrained ens vs scratch ens, shared scaffold test) ---"
  {
    uv run python scripts/paired_bootstrap.py --target "$TASK" \
        --ours "$PRE_ENS" --other "$SCR_ENS" --other-prob-col "${TASK}_prob"
  } 2>&1 | tee -a "$MASTER_LOG" | tee -a "$MANIFEST"
else
  log "  (no ensemble probs written — skipping paired bootstrap)"
fi

log ""
log "Done. Master log: $MASTER_LOG   Manifest: $MANIFEST   Aggregate: $AGG_LOG"