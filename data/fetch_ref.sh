#!/usr/bin/env bash
# Download the annotation reference files needed by the ANNOTATE pipeline step,
# for a single chromosome.
#
# Usage: fetch_ref.sh [chrom]
#   chrom defaults to 22, matching fetch_1000genomes.sh's default. Pass any
#   autosome number 1-22 to fetch references for that chromosome instead,
#   e.g. fetch_ref.sh 1
#
# What we pull, chromosome-scoped so each run stays lightweight:
#   chr<N>.gff3.gz / .tbi   -- Ensembl GRCh37 r87 gene annotation for bcftools csq
#   chr<N>.fa / .fai        -- chr<N> FASTA for bcftools csq reference base calls
#
# The GFF3 step re-downloads and streams the ~800 MB whole-genome file on
# every invocation, since there is no local cache of the unfiltered file.
# That is fine for a handful of chromosomes; if this is extended to all 22
# autosomes it would be worth caching the raw download once and filtering
# it locally per chromosome instead.
#
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

CHROM="${1:-22}"

DEST="${DATA_CACHE_DIR:-$(dirname "$0")/cache/ref}"
mkdir -p "$DEST"

# ── GFF3 (Ensembl GRCh37 release 87, single chromosome) ────────────────────
# We stream the full-genome GFF3 (~800 MB compressed) and filter to the
# target chromosome on the fly so we only store its slice.
GFF_URL="http://ftp.ensembl.org/pub/grch37/release-87/gff3/homo_sapiens/Homo_sapiens.GRCh37.87.chr.gff3.gz"
GFF_OUT="${DEST}/chr${CHROM}.gff3.gz"

if [[ -s "$GFF_OUT" ]]; then
  echo "already cached: $GFF_OUT"
else
  echo "downloading chr${CHROM} GFF3 (streaming ~800 MB -> filtering to chr${CHROM} -> cached) ..."
  TMP_GFF="${DEST}/.tmp_chr${CHROM}.gff3"
  # -f makes curl exit non-zero on HTTP 4xx/5xx instead of silently piping an
  # HTML error page into zcat. Without -f this only failed by accident, when
  # zcat rejected the non-gzip error body; -f makes that fail-loud on purpose.
  curl -fsSL "$GFF_URL" \
    | zcat \
    | awk -v c="$CHROM" '$1 == c || /^#/' \
    > "$TMP_GFF"

  if [[ ! -s "$TMP_GFF" ]]; then
    echo "ERROR: filtered GFF3 for chr${CHROM} is empty. Either chr${CHROM} has" >&2
    echo "  no records in the source file, or the awk chromosome filter (\$1 == \"$CHROM\") is wrong." >&2
    rm -f "$TMP_GFF"
    exit 1
  fi

  # tabix requires input sorted by chromosome then start position. Ensembl's
  # GFF3 is ordered hierarchically (a gene's transcript/exon child records
  # immediately follow it), not strictly by position -- overlapping or
  # nested gene models break the non-decreasing start-position order tabix
  # needs. chr22 happens to have few enough overlaps that this went
  # unnoticed, but chr1's much higher gene density makes an unsorted file
  # far more likely to trip tabix (or worse, index silently wrong). Sort
  # explicitly rather than relying on the source file's existing order.
  (grep '^#' "$TMP_GFF"; grep -v '^#' "$TMP_GFF" | sort -k1,1 -k4,4n) \
    | bgzip \
    > "$GFF_OUT"
  rm -f "$TMP_GFF"

  tabix -p gff "$GFF_OUT"
  echo "GFF3 ready: $GFF_OUT"
fi

# ── FASTA (single chromosome, GRCh37 / hg19) ───────────────────────────────
# bcftools csq requires the FASTA to verify reference bases when computing
# protein-level consequences.
FA_URL="http://ftp.ensembl.org/pub/grch37/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.${CHROM}.fa.gz"
FA_OUT="${DEST}/chr${CHROM}.fa"

if [[ -s "$FA_OUT" ]]; then
  echo "already cached: $FA_OUT"
else
  echo "downloading chr${CHROM} FASTA ..."
  curl -fsSL "$FA_URL" | zcat > "$FA_OUT"

  if [[ ! -s "$FA_OUT" ]]; then
    echo "ERROR: FASTA download for chr${CHROM} is empty. Check $FA_URL is reachable." >&2
    rm -f "$FA_OUT"
    exit 1
  fi

  samtools faidx "$FA_OUT"
  echo "FASTA ready: $FA_OUT"
fi

echo ""
echo "Reference files ready in: $DEST"
echo "  ${GFF_OUT}"
echo "  ${FA_OUT}"
