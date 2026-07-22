"""
Download gnomAD v2.1.1 gene constraint metrics and cache to ml/cache/.

URL used (confirmed HTTP 200, ~4.6 MB bgzipped):
  https://gnomad-public-us-east-1.s3.amazonaws.com/release/2.1.1/constraint/
  gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz

Why gnomAD v2.1.1 and not v4.x:
  - The pipeline and 1000 Genomes source data are GRCh37. v2.1.1 is also
    GRCh37, keeping the genomic build consistent throughout. Constraint
    features are joined on gene symbol/ID (not position), so the build
    difference would not cause incorrect joins, but using the same build
    avoids any gene-ID version drift between assemblies.
  - v2.1.1 column names (pLI, oe_lof_upper) match DESIGN.md exactly and
    are the values cited in virtually every constraint paper. An interviewer
    will recognize them immediately.
  - v4.1 is available at the same S3 bucket if needed later; switching is
    a one-line URL change plus column renames.

The source file has one row per transcript. We deduplicate to one row per
Ensembl gene ID by taking the transcript with the highest pLI. Rationale:
if any isoform of a gene is highly intolerant to LoF, the gene carries that
constraint signal. Using the most constrained transcript is the conservative,
biologically appropriate choice and matches the gnomAD team's own convention
for per-gene summaries.

Output columns:
    gene        gene symbol  (primary join key to match other sources)
    gene_id     Ensembl gene ID  (ENSG..., secondary join key)
    pLI         probability of LoF intolerance, range [0, 1]
    loeuf       oe_lof_upper -- 90th-percentile upper bound of obs/exp LoF
                Lower LOEUF = more constrained. Preferred over pLI for
                continuous modeling because it is not threshold-dependent.
    oe_lof      obs/exp LoF point estimate
    oe_lof_lower  lower bound of 90% CI (for completeness)
    oe_mis      obs/exp missense ratio

Run:  python3 ml/fetch_gnomad.py
Output: ml/cache/gnomad_constraint.parquet
"""

import os
import sys
import urllib.request

import pandas as pd

GNOMAD_URL = (
    "https://gnomad-public-us-east-1.s3.amazonaws.com"
    "/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
)

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")
RAW_FILE  = os.path.join(CACHE_DIR, "gnomad_v2.1.1_lof_metrics.txt.bgz")
OUT_FILE  = os.path.join(CACHE_DIR, "gnomad_constraint.parquet")

# Columns we keep from the source file.
KEEP_COLS = [
    "gene",
    "gene_id",
    "pLI",
    "oe_lof_upper",    # LOEUF
    "oe_lof",
    "oe_lof_lower",
    "oe_mis",
    "gene_type",       # used for protein-coding filter, then dropped
]

NUMERIC_COLS = ["pLI", "oe_lof_upper", "oe_lof", "oe_lof_lower", "oe_mis"]

# Minimum fraction of protein-coding genes (post-dedup) that must have a real
# pLI value. Observed coverage in this project is 97.4% of the raw file
# (19,183 / 19,689 protein-coding rows). This is a conservative floor well
# below that, so a genuinely dead/truncated source fails loudly instead of
# silently handing build_features.py a near-empty column to median-impute.
MIN_PLI_COVERAGE = 0.85


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"already cached: {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading gnomAD constraint (~4.6 MB):\n  {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        print(
            "If the URL has moved, check https://gnomad.broadinstitute.org/downloads",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"saved: {dest}  ({os.path.getsize(dest):,} bytes)")


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)
    download(GNOMAD_URL, RAW_FILE)

    print("loading constraint file ...")
    # bgzip uses the same DEFLATE format as gzip; pandas does not recognise
    # the .bgz extension automatically, so we declare the compression explicitly.
    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        compression="gzip",
        usecols=lambda c: c in KEEP_COLS,
        dtype={"gene": str, "gene_id": str, "gene_type": str},
        low_memory=False,
    )

    missing = set(KEEP_COLS) - set(df.columns)
    if missing:
        print(f"ERROR: expected columns not found in file: {missing}", file=sys.stderr)
        print(f"columns present: {df.columns.tolist()}", file=sys.stderr)
        print(
            "gnomAD may have changed its schema. Inspect the raw file and update KEEP_COLS.",
            file=sys.stderr,
        )
        sys.exit(1)

    assert len(df) > 0, (
        f"FAIL: {GNOMAD_URL} downloaded to {RAW_FILE} but parsed to zero rows. "
        f"A 200 status with an empty or unparseable body must still fail; "
        f"inspect the raw file by hand before re-running."
    )

    # Filter to protein-coding genes; other gene types (lncRNA, pseudogene, etc.)
    # are not in the label universe and would just add noise rows.
    before = len(df)
    df = df[df["gene_type"] == "protein_coding"].copy()
    print(f"protein-coding transcripts: {len(df):,}  (from {before:,} total rows)")
    assert len(df) > 0, (
        f"FAIL: 0 protein_coding rows after filtering {before:,} rows from "
        f"{RAW_FILE}. gene_type schema may have changed; inspect the raw file."
    )

    # Parse numeric columns; a few transcripts have '.' for missing data.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate: one row per Ensembl gene ID, keeping the highest-pLI transcript.
    # NaN pLI rows sort last so they are only kept when a gene has no pLI at all.
    df = (
        df.sort_values("pLI", ascending=False, na_position="last")
        .groupby("gene_id", sort=False)
        .first()
        .reset_index()
    )
    print(f"unique genes after deduplication:  {len(df):,}")

    # Rename for a clean output schema.
    df = df.rename(columns={"oe_lof_upper": "loeuf"})
    df = df.drop(columns=["gene_type"])

    # Sanity checks.
    n_pli   = df["pLI"].notna().sum()
    n_loeuf = df["loeuf"].notna().sum()
    pli_coverage = n_pli / len(df)
    print(f"\nSanity checks:")
    print(f"  genes with pLI:   {n_pli:,} / {len(df):,}  ({pli_coverage:.1%})")
    print(f"  genes with LOEUF: {n_loeuf:,} / {len(df):,}  ({n_loeuf/len(df):.1%})")
    print(f"  pLI  range: [{df['pLI'].min():.3f}, {df['pLI'].max():.3f}]")
    print(f"  LOEUF range: [{df['loeuf'].min():.3f}, {df['loeuf'].max():.3f}]")
    assert pli_coverage >= MIN_PLI_COVERAGE, (
        f"FAIL: pLI coverage {pli_coverage:.1%} is below the "
        f"{MIN_PLI_COVERAGE:.0%} floor (observed coverage historically is "
        f"97.4%). {GNOMAD_URL} may be truncated or schema-shifted. Refusing "
        f"to write a parquet file that would get silently median-imputed "
        f"downstream."
    )

    # pLI must be bounded [0, 1]; anything outside that means we grabbed the
    # wrong column or the file format has changed.
    bad_pli = df["pLI"].dropna()
    assert bad_pli.between(0, 1).all(), \
        f"pLI out of [0,1] -- wrong column? min={bad_pli.min()}, max={bad_pli.max()}"

    df.to_parquet(OUT_FILE, index=False)
    print(f"\nwrote {OUT_FILE}")
    print(f"  rows: {len(df):,}  columns: {df.columns.tolist()}")


if __name__ == "__main__":
    build()
