# Reproducing the `ogbg-molhiv` results

This directory reproduces every `ogbg-molhiv` number in the paper, and in particular
the main positive result: **loading the pretrained GPS backbone, pooling each molecule
to a frozen embedding, truncating that embedding to its informative subspace, and
concatenating it with a Morgan fingerprint under one random forest raises test
ROC-AUC from 0.803 to 0.830** (paired bootstrap Δ = +0.027, 95% CI [+0.003, +0.054],
one-sided p = 0.014).

No pretraining is repeated: the backbone is loaded from
`checkpoints/pretrain_chembl/final.pt`, which ships with the repository. The dataset
downloads itself on first use. Everything else is a random forest on CPU.

The code lives in `../scripts` and `../src`, alongside the rest of the project, so
there is only ever one copy of it. This directory holds the driver, the steps, and
our own run's outputs to compare against.

---

## Contents

```
run_all.sh                    driver for all six steps (each can be run alone)
reference/
├── concat_pretrained.out     our exact concat sweep, pretrained backbone
├── concat_randinit.out       the same sweep with an untrained backbone (control)
├── width_probe_molhiv.csv    the width sweep behind Figure (a)
└── probs/*.npz               per-cell ensemble test predictions, for the bootstraps
```

`run_all.sh` writes its own outputs to `../logs/` (one `.out` per step, `.npz` per
cell, and `logs/molhiv_reproduce_summary.txt` collecting the bootstraps) and the
figure to `../figures/molhiv_concat_width.png`, where our version is already
committed as the expected output.

Scripts used, all under `../scripts/`:

| Script | Role |
|--------|------|
| `xgb_molhiv_embed.py` | frozen backbone -> feature blocks (`pooled`/`desc`/`morgan`) -> forest |
| `xgb_molhiv.py` | the fingerprint-only reference through the identical head |
| `concat_balanced_molhiv.py` | **the main experiment**: the fingerprint x truncation sweep |
| `width_probe_molhiv.py` | ROC-AUC against retained dimensions, for both representations |
| `width_randinit_molhiv.py` | untrained-backbone width control |
| `spectrum.py` | train-fitted PCA basis used for the truncation |
| `paired_bootstrap_auc.py` | paired bootstrap of the ROC-AUC difference |
| `plot_concat_width.py` | the two-panel figure |

---

## Prerequisites

```bash
uv sync                       # environment from pyproject.toml + uv.lock
git lfs install && git lfs pull   # pulls checkpoints/pretrain_chembl/final.pt (21 MB)
```

`ogbg-molhiv` is fetched automatically from OGB's CSV mirror into `data/ogbg_molhiv/`
on first run (no `ogb` package needed); the graph featurization is then cached in
`data/molhiv_feature_cache/` (~0.9 GB, built once, a few minutes).

**Hardware.** A GPU is optional and barely used - it only runs three forward passes of
the frozen encoder, under a minute. The cost is the forests, on CPU. Measured on a
16-core / 32-thread desktop: one 1000-tree forest over the 1056-column concatenated
matrix takes ~103 s at `max_features=0.1` and ~29 s at `sqrt`, and each concat sweep
fits 190 of them, so **budget ~3 h per sweep, ~6 h for the sweep plus its control**.
The other steps are ~20 min each.

---

## Quick start

```bash
bash reproduce_molhiv/run_all.sh                 # everything, ~7 h
STEPS="concat stats" bash reproduce_molhiv/run_all.sh    # just the headline result
```

To check our reported numbers without refitting anything (seconds, CPU only), run the
bootstraps against the shipped predictions:

```bash
STEPS="stats figure" PROBS=reproduce_molhiv/reference/probs \
    bash reproduce_molhiv/run_all.sh
```

---

## Step by step

### 1. Frozen probe: is the pretrained representation better than the untrained one?

```bash
# pretrained GPS + LeJEPA embedding
uv run python scripts/xgb_molhiv_embed.py \
    --checkpoint checkpoints/pretrain_chembl/final.pt \
    --embed-mode pooled --model rf --n-estimators 2000 --n-models 10
# random-init control (architecture floor)
uv run python scripts/xgb_molhiv_embed.py --random-init \
    --embed-mode pooled --model rf --n-estimators 2000 --n-models 10
# Morgan fingerprint reference through the identical head
uv run python scripts/xgb_molhiv.py --features morgan --n-bits 2048 \
    --model rf --n-estimators 2000 --n-models 10
```

| Frozen features + RF (2000 trees x 10 seeds) | Valid | Test |
|---|---|---|
| GPS + LeJEPA, pooled (128 d) | 0.7826 | **0.7881 ± 0.0025** |
| Same architecture, random init | 0.6982 | 0.6654 ± 0.0039 |
| Morgan ECFP4, 2048 bits | 0.8437 | 0.8072 ± 0.0043 |

Pretraining is worth +0.123 over the untrained encoder and still loses to a plain
fingerprint. That gap is what step 3 addresses.

### 2. Width sweep: the two representations are not comparable at face value

```bash
uv run python scripts/width_probe_molhiv.py \
    --checkpoint checkpoints/pretrain_chembl/final.pt \
    --n-models 5 --n-trees 500 --out logs/width_probe_molhiv.csv
# untrained-backbone width control (paper: 0.661 / 0.691 / 0.687)
uv run python scripts/width_randinit_molhiv.py --widths 128 256 512 --readouts mean
```

Writes `logs/width_probe_molhiv.csv` (compare to `reference/width_probe_molhiv.csv`).
The fingerprint improves with width to ~1024 bits; the embedding saturates by 16-32
dimensions. At the matched width of 128 the ranking **inverts between splits** -
fingerprint 0.799 valid / 0.759 test, embedding 0.782 valid / 0.788 test - because the
fingerprint loses 0.040 across the scaffold shift and the embedding loses none.

Note this sweep uses a smaller forest (500 trees x 5 seeds) than step 1, so it reads
about 0.005 low in absolute terms; it is internally consistent, not comparable to the
table above.

### 3. The main experiment: fingerprint + truncated embedding

```bash
# pretrained backbone (~3 h)
uv run python scripts/concat_balanced_molhiv.py \
    --checkpoint checkpoints/pretrain_chembl/final.pt \
    --n-models 10 --n-trees 1000 | tee logs/concat_pretrained.out

# random-init control: same architecture, same truncation, same selection (~3 h)
uv run python scripts/concat_balanced_molhiv.py --random-init \
    --n-models 10 --n-trees 1000 | tee logs/concat_randinit.out
```

The sweep crosses fingerprint width {512, 1024} x PCA truncation k ∈ {16, 32, 64} x
`max_features` ∈ {sqrt, 0.1}, and prints per-cell valid and test ROC-AUC plus the delta
against that cell's own fingerprint-only baseline. Per-cell ensemble test predictions
are written to `logs/concat_*.npz` for step 4.

Expected (paper Table "concat", `max_features = 0.1`; ours in
`reference/concat_pretrained.out` and `reference/concat_randinit.out`):

| Frozen features + RF (1000 trees x 10 seeds) | dim | Valid | Test |
|---|---|---|---|
| LeJEPA embedding, PCA-32 | 32 | 0.769 | 0.789 ± 0.004 |
| Random-init embedding, PCA-32 | 32 | 0.754 | 0.681 ± 0.003 |
| Morgan 1024 alone | 1024 | 0.845 | 0.803 ± 0.004 |
| + LeJEPA PCA-16 | 1040 | 0.841 | 0.827 ± 0.003 |
| **+ LeJEPA PCA-32** (valid-selected) | 1056 | **0.847** | **0.830 ± 0.004** |
| + LeJEPA PCA-64 (dilution control) | 1088 | 0.831 | 0.811 ± 0.003 |
| + random-init PCA-32 (control) | 1056 | 0.836 | 0.800 ± 0.004 |
| + random-init PCA-64 (control) | 1088 | 0.842 | 0.791 ± 0.004 |
| Morgan 512 alone | 512 | 0.821 | 0.790 ± 0.005 |
| + LeJEPA PCA-32 | 544 | 0.815 | 0.821 ± 0.002 |
| + random-init PCA-32 (control) | 544 | 0.821 | 0.773 ± 0.004 |

The script ends by printing the cell selected **on validation** and its honest delta:

```
  VALID-SELECTED cell: morgan 1024 + embed PCA-32 [mf=0.1]
    valid 0.8471  ->  TEST 0.8298 (ens 0.8323)
    its morgan-alone baseline: test 0.8032 (valid 0.8453)
    honest delta = +0.0266
```

Three numbers per cell, all in the reference log: the table above quotes the
**per-model mean** (0.8032 -> 0.8298, Δ +0.0266), the paper's abstract quotes the same
cells as **ten-model ensembles** (0.8048 -> 0.8323, the `ens` column, Δ +0.0275), and
the step-4 bootstrap resamples those ensemble predictions (Δ +0.0275). The `VALID`
column is what selects the cell.

### 4. Significance

```bash
uv run python scripts/paired_bootstrap_auc.py \
    --ours logs/concat_c1024_32_0.1.npz --other logs/concat_morgan1024_0.1.npz
```

Paired over the shared 4111 test rows, 10 000 resamples. All five contrasts (the
driver runs them all, into `logs/molhiv_reproduce_summary.txt`):

| Contrast | Δ ROC-AUC | 95% CI | p (one-sided) |
|---|---|---|---|
| + PCA-32 vs fingerprint, both selected on valid | **+0.027** | [+0.003, +0.054] | **0.014** |
| the same, against the fingerprint cell strongest on **test** | +0.025 | [+0.001, +0.051] | 0.019 |
| + PCA-64 (dilution control) vs fingerprint | +0.008 | [-0.021, +0.039] | 0.29 |
| + random-init PCA-32, winning config | -0.003 | [-0.029, +0.026] | 0.59 |
| + random-init PCA-64, its own valid-selected cell | -0.013 | [-0.045, +0.021] | 0.77 |

The `.npz` filename encodes the cell: `concat_c<bits>_<k>_<max_features>.npz` for a
concatenation, `concat_morgan<bits>_<max_features>.npz` for a fingerprint-only
baseline, and a `randinit_` infix for the control run.

### 5. Figure

```bash
uv run python scripts/plot_concat_width.py \
    --csv logs/width_probe_molhiv.csv \
    --concat-log logs/concat_pretrained.out \
    --out figures/molhiv_concat_width.png
```

With no arguments it draws the shipped reference numbers, so the figure can be
rebuilt without running anything.

---

## OGB leaderboard submission

The sweep above selects a configuration; `scripts/ogb_submission_molhiv.py` runs that
one configuration under the OGB leaderboard's rules and prints the exact values the
submission form asks for.

```bash
uv run python scripts/ogb_submission_molhiv.py          # writes logs/ogb_submission_molhiv.json
uv add ogb                                              # optional, see below
```

It differs from the sweep in three ways that matter for a valid submission.

**It scores the full official split.** `MolHIVDataset` drops molecules our graph
featurizer rejects - test rows 29 and 302, both inactive, both hypervalent Al/B
complexes - so the sweep reports on 4111 of 4113 rows. A leaderboard number has to
cover every row. Predictions are scattered back through `dataset.kept_idx` and the
rejected rows get a constant (the training positive rate). The script prints the
subset and full-split numbers side by side; for a rank-based metric with two inactive
molecules restored the difference is ~0.

**It reports the unbiased standard deviation.** OGB asks for `torch.std` (ddof=1);
the sweep uses `np.std` (ddof=0). Both are printed.

**It uses the official Evaluator when available.** `ogb` is not a dependency of this
project. For ogbg-molhiv the Evaluator reduces to `roc_auc_score` over the full split
(one task, no missing labels), so the fallback is exact - and when `ogb` *is*
installed the script computes both and asserts they agree. Install it if you want the
official code path to be the one that produced your number.

Two things to get right on the form: report the **per-model mean over seeds 0-9**, not
the ensemble (averaging the seeds collapses the ten required runs into one model), and
**declare external data** - the backbone is pretrained on ~2.9M ChEMBL molecules.

---

## How to read the result

Three things have to be said together, or the finding is overstated.

1. **The gain is robustness to the scaffold shift, not extra fit.** On the validation
   scaffolds the concatenation is not better than the fingerprint alone (0.847 vs
   0.845, well inside noise). The two arms separate only on the test scaffolds,
   because the fingerprint alone loses 0.042 ROC-AUC from validation to test while the
   concatenation loses 0.017. The embedding is not adding signal the validation
   chemotypes reward; it is adding signal that survives a change of chemotype. The
   corollary is a real limitation: because the gain is invisible on validation, it
   cannot be selected *for*.

2. **The truncation is doing the work, and it is a fitted hyperparameter.** Appending
   all 128 embedding dimensions to a 2048-bit fingerprint is null - the forest samples
   ~47 columns per split, so under one *informative* embedding column is typically
   available while the rest supply noise. Restoring the diluting dimensions (k = 64)
   removes most of the effect, which is what identifies dilution as the mechanism. The
   truncation dimension is chosen on validation, which is legitimate, but it is fitted,
   not given.

3. **It requires pretraining, not merely extra dense columns.** The same sweep with an
   untrained backbone gives -0.003 in the winning configuration, and the pretrained
   embedding beats its untrained twin in all twelve cells while the untrained columns
   are net harmful in most.

Scope: one benchmark, one split, 130 test actives. The confidence interval runs from
+0.003 to +0.054, so the honest phrasing is "about +0.03, possibly as little as
+0.003", not a firm +0.027.

---

## Reproducibility notes

- **Selection protocol.** OGB rules: fixed canonical scaffold split, ROC-AUC, select on
  **validation**, report test. Both arms are selected independently on validation. The
  scripts print valid and test side by side precisely so a reader can check that no
  cell was picked by its test score.
- **The PCA rotation is fitted on TRAIN only** and then applied to valid and test. It is
  part of the feature extractor; fitting it on anything else would leak.
- **Determinism.** The forests are seeded (`random_state = 0..9`), so the fingerprint-only
  rows reproduce bit for bit. The embedding rows depend on a GPU forward pass, which is
  not bit-reproducible across hardware or cuDNN versions; in our re-runs this moves
  ROC-AUC by ≤ 0.0003. `--random-init` in `concat_balanced_molhiv.py` is seeded
  (`--init-seed 100`) so the control is reproducible too; the `--random-init` flag of
  `xgb_molhiv_embed.py` is *not* separately seeded, so its floor moves by ~0.005 between
  runs.
- **Row counts.** `xgb_molhiv.py` works from SMILES and keeps 4113 test molecules;
  everything that goes through the graph featurizer (`MolHIVDataset`) drops two more
  unparseable structures and keeps 4111. Comparisons are only ever made within one of
  these, and the bootstrap asserts identical label vectors before pairing.
- **Why the Morgan baseline differs between step 1 (0.807) and step 3 (0.803).** They
  are different by design: 2048 bits / `max_features=sqrt` / 2000 trees / 4113 rows in
  step 1, versus 1024 bits / `max_features=0.1` / 1000 trees / 4111 rows in step 3,
  where the fingerprint width and `max_features` were themselves selected on validation.
  Every delta quoted above is computed inside one protocol.
