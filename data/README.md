# data/

Scripts that download and cache the public inputs for the pipeline.
All sources are in-region (us-east-1) or served over HTTP with no AWS egress charge.
No large data files are committed; `cache/` is gitignored.

## Scripts

| Script | Source | What it pulls |
|--------|--------|---------------|
| `fetch_1000genomes.sh` | `s3://1000genomes` (AWS Open Data, us-east-1) | chr22 VCF + index + sample panel (~60 MB) |
| `fetch_open_targets.py` | Open Targets FTP (EBI) | `associationByOverallDirect` + `targets` Parquet |
| `fetch_chembl_known_drugs.py` | Open Targets FTP (EBI) | `knownDrugsAggregated` Parquet (label source) |

## Usage

```bash
# 1000 Genomes — run from EC2 in us-east-1 (requires AWS CLI + credentials)
bash data/fetch_1000genomes.sh

# Open Targets evidence (association scores + gene metadata)
python3 data/fetch_open_targets.py --release 24.12

# ChEMBL known-drug evidence (the label source)
python3 data/fetch_chembl_known_drugs.py --release 24.12
```

All scripts are idempotent. Re-running skips files already present in `cache/`.

## Cache layout

```
data/cache/
  1000genomes/
    chr22.vcf.gz
    chr22.vcf.gz.tbi
    phase3_samples.index
  open_targets/24.12/
    associationByOverallDirect/   (Parquet partitions)
    targets/                      (Parquet partitions)
    knownDrugsAggregated/         (Parquet partitions)
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_CACHE_DIR` | `data/cache` | Override the cache root |
| `OT_RELEASE` | `24.12` | Open Targets release to pull |

## In-region note

Run `fetch_1000genomes.sh` from an EC2 instance or AWS CloudShell in us-east-1.
The 1000 Genomes bucket is in us-east-1; pulling from the same region costs nothing.
The Open Targets FTP is served from EBI (UK) over standard HTTPS — no AWS egress.
