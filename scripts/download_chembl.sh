#!/usr/bin/env bash
# Download ChEMBL canonical SMILES and convert to sharded .smi files, so the
# existing corpus-agnostic tools (scaffold_ceiling.py, measure_diversity.py,
# featurize_corpus.py) can read it exactly like ZINC.
#
# ChEMBL is the candidate bioactivity-relevant pretraining corpus: unlike ZINC20
# (make-on-demand), it is measured-bioactivity compounds (drugs, natural-product-
# like, screening hits) — the chemical space the Wong et al. antibiotic actives
# live in. We check its scaffold/NN coverage of the antibiotic TEST set BEFORE
# committing to pretraining (same discipline that killed the diverse-ZINC idea).
#
# Output: data/chembl/chembl_000.smi .. chembl_0NN.smi  (lines: "<SMILES> <id>")
#
# Usage:
#     bash scripts/download_chembl.sh
#     VERSION=37 SHARDS=32 bash scripts/download_chembl.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-37}"
SHARDS="${SHARDS:-32}"
OUT="${OUT:-data/chembl}"
URL="${URL:-https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_${VERSION}_chemreps.txt.gz}"
GZ="$OUT/chembl_${VERSION}_chemreps.txt.gz"

mkdir -p "$OUT"

# 1. Download (resumable).
if [ ! -f "$GZ" ]; then
  echo "Downloading ChEMBL $VERSION chemreps -> $GZ"
  curl -L -C - --fail -o "$GZ" "$URL"
else
  echo "Found $GZ — skipping download."
fi

# 2. Convert: chemreps is TSV (chembl_id, canonical_smiles, inchi, inchikey) with
#    a header row. Emit "<SMILES> <id>" round-robin into SHARDS files for parallel
#    scanning. Drop rows with an empty SMILES field.
if ls "$OUT"/chembl_[0-9]*.smi >/dev/null 2>&1; then
  echo "Shards already present in $OUT — skipping conversion."
else
  echo "Converting to $SHARDS .smi shards ..."
  zcat "$GZ" | awk -F'\t' -v out="$OUT" -v n="$SHARDS" '
    NR>1 && $2!="" {
      s = (NR % n);
      print $2" "$1 > (out "/chembl_" sprintf("%03d", s) ".smi")
    }'
  total=$(cat "$OUT"/chembl_[0-9]*.smi | wc -l)
  echo "Wrote $total SMILES across $SHARDS shards in $OUT"
fi

echo "Done. Next, run the coverage check (see commands printed by the assistant)."