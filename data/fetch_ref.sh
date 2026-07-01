#!/usr/bin/env bash
# Download the annotation reference files needed by the ANNOTATE pipeline step.
#
# What we pull (chr22 only, matching the validation VCF):
#   chr22.gff3.gz / .tbi   -- Ensembl GRCh37 r87 gene annotation for bcftools csq
#   chr22.fa / .fai        -- chr22 FASTA for bcftools csq reference base calls
#
# Why chr22: the cached 1000 Genomes VCF is chr22. These references are
# deliberately chromosome-scoped so the validation run stays lightweight.
# For a full-cohort run, replace with whole-genome GFF3 and FASTA.
#
# Chromosome naming: Ensembl GRCh37 uses plain "22" (no chr prefix),
# matching 1000 Genomes phase 3 VCF chromosome names. No renaming needed.
#
# Prerequisites: bgzip, tabix, samtools (htslib tools).
#   macOS:  brew install htslib samtools
#   Linux:  apt-get install bcftools samtools  (includes bgzip/tabix)
#   Docker: alternatively run this inside the bcftools container:
#     docker run --rm -v "$(pwd)":/repo \
#       quay.io/biocontainers/bcftools:1.18--h8b25389_0 \
#       bash /repo/data/fetch_ref.sh
#
# Idempotent: skips files that are already cached and non-empty.

set -euo pipefail

DEST="${DATA_CACHE_DIR:-$(dirname "$0")/cache/ref}"
mkdir -p "$DEST"

# ── GFF3 (Ensembl GRCh37 release 87, chr22 only) ───────────────────────────
# We stream the full-genome GFF3 (~800 MB compressed) and filter to chr22
# on the fly so we only store the ~15 MB chr22 slice.
GFF_URL="http://ftp.ensembl.org/pub/grch37/release-87/gff3/homo_sapiens/Homo_sapiens.GRCh37.87.chr.gff3.gz"
GFF_OUT="${DEST}/chr22.gff3.gz"

if [[ -s "$GFF_OUT" ]]; then
  echo "already cached: $GFF_OUT"
else
  echo "downloading chr22 GFF3 (streaming ~800 MB -> filtering to chr22 -> ~15 MB cached) ..."
  curl -sL "$GFF_URL" \
    | zcat \
    | awk '$1 == "22" || /^#/' \
    | bgzip \
    > "$GFF_OUT"
  tabix -p gff "$GFF_OUT"
  echo "GFF3 ready: $GFF_OUT"
fi

# ── FASTA (chr22 only, GRCh37 / hg19) ──────────────────────────────────────
# bcftools csq requires the FASTA to verify reference bases when computing
# protein-level consequences. chr22 uncompressed is ~51 MB.
FA_URL="http://ftp.ensembl.org/pub/grch37/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.22.fa.gz"
FA_OUT="${DEST}/chr22.fa"

if [[ -s "$FA_OUT" ]]; then
  echo "already cached: $FA_OUT"
else
  echo "downloading chr22 FASTA (~12 MB compressed -> ~51 MB) ..."
  curl -sL "$FA_URL" | zcat > "$FA_OUT"
  samtools faidx "$FA_OUT"
  echo "FASTA ready: $FA_OUT"
fi

echo ""
echo "Reference files ready in: $DEST"
echo "  ${GFF_OUT}"
echo "  ${FA_OUT}"
