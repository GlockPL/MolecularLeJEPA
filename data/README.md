# Data

This directory holds the inputs needed to reproduce the experiments. Large
corpora and feature caches are **not** committed; regenerate them with the
scripts below.

## What is committed

- `wong et al/41586_2023_6887_MOESM3_ESM.xlsx` - Wong et al. (2023) training
  labels (39,312 compounds, four binary endpoints). This is the file the
  antibiotic finetuning and the deterministic splits are built from.
- `wong et al/41586_2023_6887_MOESM4_ESM.xlsx` - Wong et al. (2023) large-scale
  virtual-screening predictions (provided for completeness).

These spreadsheets are Supplementary Data from Wong et al. (2023), *Nature*
626, 177-185 (https://doi.org/10.1038/s41586-023-06887-8) and are redistributed
here only to make the pipeline runnable end to end; please cite the original
paper.

## What you regenerate locally

### ChEMBL pretraining corpus (~2.9M molecules)

```bash
# 1. Download ChEMBL 37 and write SMILES shards into data/chembl/
bash scripts/download_chembl.sh

# 2. Featurize into a descriptor-augmented cache (chunk_*.pt + desc_stats.pt)
uv run python scripts/featurize_corpus.py \
    --input data/chembl --output data/chembl_cache_desc \
    --max-atoms 100 --workers 32 --descriptors --global-shuffle
```

This produces `data/chembl_cache_desc/`, the cache the pretraining configs point
at (`data.cache_dir`).

### ogbg-molhiv

Downloaded automatically by the OGB library on first use into
`data/ogbg_molhiv/` (see `scripts/finetune_molhiv.py`).
