                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      # Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA

Code, configurations, splits, and checkpoints for the paper:

> **Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA**
> Michał Kulczykowski, Rafał Łabędzki (deepsense.ai)

We adapt **LeJEPA** (a predictor-free joint-embedding predictive architecture
regularised by Sketched Isotropic Gaussian Regularisation, SIGReg) to molecular
graphs and ask: *does self-supervised pretraining on a large unlabelled corpus
improve downstream molecular property prediction?* We pretrain a GPS graph
transformer (and a Chemprop-style D-MPNN control) on ~2.9M bioactivity-curated
molecules from **ChEMBL**, and evaluate on the **Wong et al. antibiotic** dataset
(random and scaffold splits) and **ogbg-molhiv**.

**Headline finding.** Pretraining yields a measurably better *representation*: a
frozen probe on the pretrained embedding far exceeds the same architecture at
random initialisation on both tasks (on ogbg-molhiv, ROC-AUC 0.788 vs 0.665, a
+0.123 lift reaching the published self-supervised band). This advantage does **not**
robustly convert into a downstream *finetuning* gain. On a single canonical
antibiotic scaffold partition it appears to (ΔAUPRC = +0.041, one-sided p = 0.010),
but it does not survive replication across five scaffold partitions, where it ranges
from -0.033 to +0.054 and the pooled estimate is only +0.013 (p = 0.095, n.s.).
Finetuning likewise gives no gain on the random split, on ogbg-molhiv, or with the
D-MPNN backbone. The benefit is real at the level of the representation but, under
finetuning, weak and partition-dependent.

The representational edge is nonetheless recoverable by other means. Truncating the
frozen embedding to its informative subspace (~16-32 dimensions) and concatenating it
with a 1024-bit Morgan fingerprint under one random forest raises ogbg-molhiv test
ROC-AUC from 0.803 to 0.830 (Δ = +0.027, 95% CI [+0.003, +0.054], p = 0.014), while
an untrained encoder put through the identical pipeline gains nothing (Δ = -0.003).
The gain is robustness to the scaffold shift rather than extra fit: the two arms are
indistinguishable on validation and separate only on the shifted test scaffolds. See
[`reproduce_molhiv/`](reproduce_molhiv/README.md) to reproduce it.

---

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # create the environment from pyproject.toml + uv.lock
```

Model weights and label spreadsheets are tracked with **Git LFS** - install it
and run `git lfs pull` after cloning:

```bash
git lfs install
git lfs pull
```

---

## Repository layout

```
src/
├── data/        featurize.py, augment.py, antibiotic_dataset.py,
│                zinc20_cached_dataset.py (generic cached corpus loader),
│                molhiv_dataset.py, featurize_chemprop.py
├── models/      gps_transformer.py, dmpnn.py, heads.py, positional_encoding.py
├── objectives/  sigreg.py, lejepa_loss.py
├── pretrain.py  finetune.py  evaluate.py
configs/         pretrain + finetune configs (see table below)
scripts/         data prep, baselines, multi-seed runners, stats, molhiv, diagnostics
checkpoints/     pretrained backbones + finetuned ensembles (Git LFS)
splits/          exported deterministic train/val/test CSVs (random + scaffold)
data/            Wong et al. label xlsx + data-prep instructions (data/README.md)
reproduce_molhiv/  step-by-step reproduction of every ogbg-molhiv number,
                 including the fingerprint+embedding concatenation, with our
                 own run's logs and predictions to compare against
main.py          CLI entry point (pretrain | finetune | screen)
```

---

## Data

See [`data/README.md`](data/README.md). In short:

```bash
bash scripts/download_chembl.sh                 # ChEMBL 37 -> data/chembl/*.smi
uv run python scripts/featurize_corpus.py \
    --input data/chembl --output data/chembl_cache_desc \
    --max-atoms 100 --workers 32 --descriptors --global-shuffle
```

The Wong et al. label spreadsheet (`data/wong et al/...MOESM3...xlsx`) is included.

---

## Pretraining

GPS transformer (primary, ~2M params) and the D-MPNN control are pretrained with
the same LeJEPA objective (λ = 0.2, Vg = 2, Vl = 8, Ns = 1024, no LR warmup).

```bash
# Multi-GPU (DDP); the paper used 4 GPUs
torchrun --nproc_per_node=4 main.py pretrain --config configs/pretrain_chembl.yaml
torchrun --nproc_per_node=4 main.py pretrain --config configs/pretrain_chembl_dmpnn.yaml

# Single GPU
uv run python main.py pretrain --config configs/pretrain_chembl.yaml
```

Final backbones are provided at `checkpoints/pretrain_chembl/final.pt` (GPS) and
`checkpoints/pretrain_chembl_dmpnn/final.pt` (D-MPNN), so finetuning can be run
without repeating pretraining.

---

## Reproducing the paper

### Antibiotic activity (Wong et al.)

Multi-seed finetuning + a faithful retrained-Chemprop baseline on identical
splits, with paired-bootstrap significance:

```bash
# Random split (5 split seeds): GPS+LeJEPA ensemble + Chemprop
bash scripts/run_random_multiseed.sh

# Scaffold split (5 seeds): pretrained vs scratch, canonical partition
bash scripts/run_scaffold_multiseed.sh                 # pretrained (default ckpt)
CHECKPOINT="" bash scripts/run_scaffold_multiseed.sh   # from scratch

# Multi-partition robustness check (paper Table "scaffold partitions"):
# repeats the 5-seed pretrained-vs-scratch comparison on four further
# leakage-free scaffold partitions, then pools all five with a paired bootstrap.
# Each partition is selected by a seed (--partition-seed / PARTITION env var);
# the no-shared-scaffold invariant is preserved per partition.
bash scripts/run_scaffold_partitions.sh                # canonical + partitions 1-4, pooled
```

Descriptor baselines (LogReg / HistGBM / XGBoost) and the frozen linear probe:

```bash
uv run python scripts/descriptor_baseline.py           # Table: random-split descriptors
uv run python scripts/linear_probe.py                  # Table: frozen probe (scaffold)
```

Paired-bootstrap comparisons:

```bash
uv run python scripts/paired_bootstrap.py --help
```

### ogbg-molhiv

```bash
bash scripts/run_molhiv_full.sh        # lr x probe sweep, then 5-seed final at best-val
bash scripts/run_fewshot_molhiv.sh     # few-shot (2-100% label budgets)
```

Selected-final result: scratch lr=5e-4 → ROC-AUC 0.717 ± 0.014; pretrained
lr=5e-4 probe=15 → 0.722 ± 0.010 (5 seeds; see
`logs/molhiv_full_20260604_172508.log`).

**Frozen probe (representation quality).** The headline molhiv result is a
*frozen* probe: freeze the backbone, mean-pool each molecule to its 128-d
embedding, and fit the leaderboard random forest with no gradient reaching the
encoder. This reproduces the "ogbg-molhiv frozen probe" table — pretrained 0.788
vs random-init 0.665 vs Morgan 0.807:

```bash
# pretrained GPS+LeJEPA embedding (0.788)
uv run python scripts/xgb_molhiv_embed.py \
    --checkpoint checkpoints/pretrain_chembl/final.pt \
    --model rf --embed-mode pooled --n-estimators 2000 --n-models 10
# random-init control / architecture floor (0.665)
uv run python scripts/xgb_molhiv_embed.py --random-init \
    --model rf --embed-mode pooled --n-estimators 2000 --n-models 10
# Morgan-fingerprint reference, same forest (0.807)
uv run python scripts/xgb_molhiv.py --features morgan --n-bits 2048 \
    --model rf --n-estimators 2000 --n-models 10
```

**Fingerprint + embedding concatenation (the positive result).** Truncating the
frozen embedding to its informative subspace and concatenating it with a
1024-bit Morgan fingerprint raises test ROC-AUC from 0.803 to 0.830
(Δ = +0.027, 95% CI [+0.003, +0.054], p = 0.014), where an untrained encoder put
through the same pipeline gains nothing. This is the "concat" table and the
width/robustness figure. Full instructions, expected numbers, the random-init
control and our own run's outputs are in
**[`reproduce_molhiv/`](reproduce_molhiv/README.md)**:

```bash
bash reproduce_molhiv/run_all.sh                       # everything, ~7 h on CPU
STEPS="concat stats" bash reproduce_molhiv/run_all.sh  # just the headline result
# re-check our published numbers without refitting (seconds):
STEPS="stats figure" PROBS=reproduce_molhiv/reference/probs \
    bash reproduce_molhiv/run_all.sh
```

**Descriptor-free ablation + learning curve.** A second backbone pretrained with
descriptors removed (`pretrain_chembl_nodesc/final.pt`, identical settings but
`n_descriptors: 0`) reproduces the descriptor-conditioned probe epoch-for-epoch —
the "frozen-probe representation quality vs pretraining" figure:

```bash
# no-descriptor backbone, same frozen probe (n_descriptors=0 config)
uv run python scripts/xgb_molhiv_embed.py \
    --config configs/finetune_molhiv_nodesc.yaml \
    --checkpoint checkpoints/pretrain_chembl_nodesc/final.pt \
    --model rf --embed-mode pooled --n-estimators 1000 --n-models 10

# regenerate the overlay figure from the provided per-epoch CSVs (no re-pretraining)
uv run python scripts/probe_curve_molhiv.py \
    --overlay figures/molhiv_probe_curve_desc.csv:"GPS + descriptors" \
              figures/molhiv_probe_curve_nodesc.csv:"GPS (no descriptors)" \
    --out figures/molhiv_probe_curve_ablation
```

The per-epoch curves are shipped as `figures/molhiv_probe_curve_{desc,nodesc}.csv`.
Rebuilding them from scratch (instead of the cached CSVs) needs the full set of
per-epoch checkpoints: re-run pretraining and point `probe_curve_molhiv.py
--ckpt-dir` at the resulting `epoch_*.pt` directory — only `final.pt` is shipped
here.

### Pretraining-corpus coverage analysis (Discussion)

```bash
uv run python scripts/measure_diversity.py --help          # scaffold/Tanimoto coverage
uv run python scripts/measure_coverage_molhiv.py --help
```

### Embedding figure (pretrained vs random init)

```bash
uv run python scripts/inspect_embeddings.py \
    --checkpoint checkpoints/pretrain_chembl/final.pt \
    --config configs/pretrain_chembl.yaml \
    --smiles data/chembl --no-descriptors --n 5000
```

`--no-descriptors` is required for the comparison figure: the RDKit descriptors
trivially encode MolWt/LogP/rings and would organise even a random-init encoder.

---

## Configs

| Config | Purpose |
|--------|---------|
| `pretrain_chembl.yaml` | GPS LeJEPA pretraining on ChEMBL (primary backbone) |
| `pretrain_chembl_nodesc.yaml` | Same, **descriptor-free** (`n_descriptors: 0`) — molhiv ablation |
| `pretrain_chembl_1080.yaml` | Same, tuned for Pascal GPUs (cu118, no `torch.compile`) |
| `pretrain_chembl_dmpnn.yaml` | D-MPNN LeJEPA pretraining (architecture control) |
| `finetune_desc.yaml` | Antibiotic finetune, **random** split (descriptor-augmented heads) |
| `finetune_desc_scaffold.yaml` | Antibiotic finetune, **scaffold** split |
| `finetune_dmpnn512_random.yaml` / `_scaffold.yaml` | D-MPNN finetune controls |
| `finetune_molhiv.yaml` | ogbg-molhiv finetune (LR / probe set on the CLI) |
| `finetune_molhiv_nodesc.yaml` | ogbg-molhiv probe config for the descriptor-free backbone |

---

## Checkpoints

The three **pretrained backbones** are tracked here with Git LFS (55 MB). The
**finetuned** weights below (~4.2 GB) are archived separately - but each of their
directories keeps its per-compound `ensemble_probs_test.csv` **in this repository**,
and those predictions, not the weights, are what every reported metric and bootstrap
is computed from. See [`checkpoints/README.md`](checkpoints/README.md) for the
archive link and for when you would actually need the weights.

| Path | Weights | What |
|------|---------|------|
| `pretrain_chembl/final.pt` | tracked | GPS LeJEPA backbone (the paper's pretrained model) |
| `pretrain_chembl_nodesc/final.pt` | tracked | GPS LeJEPA backbone, descriptor-free (molhiv ablation) |
| `pretrain_chembl_dmpnn/final.pt` | tracked | D-MPNN LeJEPA backbone |
| `scaffold_ms_antibiotic_pretrained_pretrain_chembl_final_seed{0..4}` | archived | GPS antibiotic scaffold, pretrained (0.271 ± 0.015) |
| `scaffold_ms_antibiotic_scratch_seed{0..4}` | archived | GPS antibiotic scaffold, from scratch (0.232 ± 0.009) |
| `scaffold_ms_antibiotic_pretrained_pretrain_chembl_dmpnn_final_seed{0..4}` | archived | D-MPNN antibiotic scaffold, pretrained (0.247 ± 0.015) |
| `scaffold_ms_antibiotic_scratch_dmpnn512_seed{0..4}` | archived | D-MPNN antibiotic scaffold, from scratch (0.255 ± 0.030) |
| `ens_random_antibiotic_pretrained_seed{42,1,2,3,4}` | archived | Antibiotic random, GPS+LeJEPA ensemble (pooled 0.406) |
| `chemprop_random_antibiotic_seed{42,1,2,3,4}` | archived | Retrained Chemprop, random split |
| `chemprop_scaffold_antibiotic` | archived | Retrained Chemprop, scaffold split |
| `descriptor_probs/` | n/a | Descriptor-baseline per-compound predictions (CSV only) |
| `finetune_molhiv_scratch_lr0.0005/seed_{00..04}` | archived | molhiv from scratch (0.717) |
| `finetune_molhiv_pretrained_lr0.0005_probe15/seed_{00..04}` | archived | molhiv pretrained (0.722) |

Every `scaffold_ms_*` / `ens_random_*` directory keeps its `ensemble_probs_test.csv`
(per-compound test predictions used by the bootstrap) in this repository even where
the weights are archived; the pretrained arms additionally carry
`SOURCE_CHECKPOINT.txt` recording the backbone they were finetuned from. The two
`finetune_molhiv_*` directories hold weights only, so they appear in the archive
alone. The D-MPNN scaffold control (pretrained vs
`scratch_dmpnn512`) is null: the 5-seed ensemble scores 0.277 (pretrained) vs 0.278
(scratch), ΔAUPRC = -0.000, 95% CI [-0.045, +0.046], one-sided p = 0.48 (2/5 seeds
improved) - the scaffold effect does not carry across architectures.

---

## Splits

`splits/` holds the exact, deterministic partitions used in the paper, exported
as `train.csv` / `val.csv` / `test.csv` (`smiles,<label>`):

- `splits/scaffold_split_antibiotic/` - Bemis-Murcko scaffold split (no scaffold
  shared between train and test).
- `splits/random_split_antibiotic_seed{42,1,2,3,4}/` - the five stratified random
  splits.

These are provided for inspection and head-to-head baselines. The training code
regenerates the identical partitions on the fly from the Wong xlsx (scaffold is
deterministic; random is seeded), and `scripts/export_chemprop_splits.py`
re-exports them.

---

## Citation

```bibtex
@misc{kulczykowski2026selfsupervisedpretrainingmoleculargraph,
      title={Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA}, 
      author={Michał Kulczykowski and Rafał Łabędzki},
      year={2026},
      eprint={2609.04261},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2609.04261}, 
}
```

Please also cite LeJEPA (Balestriero & LeCun, 2025) and the Wong et al. (2023)
antibiotic dataset.

## License

MIT - see [LICENSE](LICENSE). The Wong et al. spreadsheets in `data/wong et al/`
are redistributed from the original paper's Supplementary Data for
reproducibility; cite Wong et al. (2023), *Nature* 626, 177-185.
