# Checkpoints

This directory ships the **pretrained backbones** (55 MB, Git LFS) and the
**per-compound predictions** of every finetuned model in the paper. The finetuned
weights themselves (~4.2 GB) are archived separately.

That split is deliberate: the predictions, not the weights, are what the paper's
statistics are computed from. Every ROC-AUC, AUPRC, bootstrap confidence interval and
p-value in the paper reruns from this repository alone, with no large download.

## Tracked here

| Path | Size | What |
|------|------|------|
| `pretrain_chembl/final.pt` | 21 MB | GPS + LeJEPA backbone - **the paper's pretrained model** |
| `pretrain_chembl_nodesc/final.pt` | 21 MB | same, descriptor-free (molhiv ablation) |
| `pretrain_chembl_dmpnn/final.pt` | 16 MB | D-MPNN + LeJEPA backbone (architecture control) |
| `*/ensemble_probs_test.csv` | 25 files | per-compound test predictions of each finetuned ensemble |
| `*/SOURCE_CHECKPOINT.txt` | 10 files | which backbone each pretrained arm was finetuned from |
| `chemprop_*/{args.json,test_preds.csv,test_scores.csv,*.log}` | | retrained Chemprop baseline config + predictions |
| `descriptor_probs/*.csv` | 5 files | descriptor-baseline per-compound predictions |

The three backbones are all you need to reproduce the frozen probes, the
`reproduce_molhiv/` results, and any finetuning run from scratch.

## Archived separately

The finetuned weights are not in git:

| Group | Size | Files |
|-------|------|-------|
| `chemprop_*` | 2.1 GB | retrained Chemprop ensembles (10 models x 5 seeds, random + scaffold) |
| `scaffold_ms_*` | 1.1 GB | GPS and D-MPNN antibiotic scaffold ensembles, 5 seeds x 5 partitions |
| `ens_random_*` | 1.0 GB | GPS antibiotic random-split ensembles, 5 seeds |
| `finetune_molhiv_*` | 65 MB | molhiv finetuning, scratch and pretrained, 5 seeds each |

<!-- TODO: replace with the Zenodo DOI once the archive is minted. -->
Download: **[archive URL / DOI to be added]**

Unpack it over this directory to restore the full tree; the directory names match
what the scripts expect, so nothing else changes.

## When you actually need the archived weights

Only to **re-score** a finetuned model, that is, to regenerate an
`ensemble_probs_test.csv` from weights rather than trusting the one committed here.
Everything else is already covered:

- **significance tests** - `scripts/paired_bootstrap.py` and
  `scripts/paired_bootstrap_auc.py` read the prediction CSVs / `.npz` files directly;
- **reported metrics** - recomputable from the same predictions;
- **finetuning from the pretrained backbone** - needs only `pretrain_chembl/final.pt`;
- **the molhiv results** - see [`../reproduce_molhiv/`](../reproduce_molhiv/README.md),
  which uses the backbone plus a random forest and touches no finetuned weights.