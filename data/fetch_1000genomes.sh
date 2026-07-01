#!/usr/bin/env bash
# Download a 1000 Genomes subset for the single-sample DAG-validation run.
#
# Source: s3://1000genomes (AWS Open Data, us-east-1).
# Run this script from an EC2 instance or AWS CloudShell in us-east-1 so
# S3-to-S3 (or S3-to-local) traffic stays in region and costs nothing.
#
# What we pull:
#   - Phase 3 VCF for chromosome 22 (smallest autosome, ~60 MB compressed).
#   - Its tabix index.
#   - The phase 3 sample panel (lists population, superpopulation per sample).
#
# These files are enough to run one Nextflow annotation process end-to-end
# and validate the full DAG before any quota increase.
#
# Idempotent: skips each file if the local copy already exists and is
# non-empty. Re-run freely; it will not re-download what it already has.

set -euo pipefail

DEST="${DATA_CACHE_DIR:-$(dirname "$0")/cache/1000genomes}"
BUCKET="s3://1000genomes"

# chr22, phase 3 final callset. GRCh37/hg19 -- must match the Ensembl GRCh37 r87
# references used by fetch_ref.sh; mismatched builds cause silent bcftools csq miscalls.
#
# Confirmed key pattern for autosomes chr1-22: release/20130502/ prefix, v5a suffix.
# chrX (v1b) and chrY have different version strings and key patterns; update this
# script separately before extending to sex chromosomes.
VCF_KEY="release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
TBI_KEY="${VCF_KEY}.tbi"
# Population panel: sample ID -> population -> superpopulation mapping.
# Verified present at this path (the old phase3/ prefix key does not exist).
PANEL_KEY="release/20130502/integrated_call_samples_v3.20130502.ALL.panel"

mkdir -p "$DEST"

download_if_missing() {
  local s3_uri="$1"
  local local_path="$2"
  if [[ -s "$local_path" ]]; then
    echo "already cached: $local_path"
    return
  fi
  echo "downloading: $s3_uri -> $local_path"
  aws s3 cp --no-progress "$s3_uri" "$local_path"
}

download_if_missing "${BUCKET}/${VCF_KEY}"    "${DEST}/chr22.vcf.gz"
download_if_missing "${BUCKET}/${TBI_KEY}"    "${DEST}/chr22.vcf.gz.tbi"
download_if_missing "${BUCKET}/${PANEL_KEY}"  "${DEST}/integrated_call_samples_v3.panel"

# Ensure the tabix index exists. The S3 download above covers the normal path;
# this catches a fresh clone where the VCF is present but the .tbi is absent.
if [[ ! -s "${DEST}/chr22.vcf.gz.tbi" ]]; then
  if ! command -v tabix &>/dev/null; then
    echo "ERROR: tabix not found. Install htslib (brew install htslib) and re-run." >&2
    exit 1
  fi
  echo "indexing: ${DEST}/chr22.vcf.gz"
  tabix -p vcf "${DEST}/chr22.vcf.gz"
fi

echo "1000 Genomes subset ready in: $DEST"
