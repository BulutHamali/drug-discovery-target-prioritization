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

# chr22, phase 3 final callset
VCF_KEY="phase3/data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
TBI_KEY="${VCF_KEY}.tbi"
PANEL_KEY="phase3/20130502.phase3.analysis.sequence.index"

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
download_if_missing "${BUCKET}/${PANEL_KEY}"  "${DEST}/phase3_samples.index"

echo "1000 Genomes subset ready in: $DEST"
