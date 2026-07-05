"""
Gene-family grouping for the leakage-safe split (DESIGN.md section 6.1).

Why this exists:
A random train/test split leaks. Paralogs (e.g. members of the same gene
family) share sequence, domain, and constraint features, so if one paralog is
in train and another in test, the model sees near-duplicate rows across the
split and the test score is inflated. The fix is to split by gene FAMILY, so
an entire family lands in one fold only. That requires a gene -> family map.

Source:
HGNC publishes a complete dataset TSV with a pre-joined `gene_group` /
`gene_group_id` column. We use that rather than the separate many-to-many
group table because the grouping comes attached per gene, and we also get the
symbol and locus type in one file.

The subtlety this file handles:
A gene can belong to MULTIPLE groups (the field is pipe-delimited), and many
genes belong to NONE. For GroupKFold each gene needs exactly ONE grouping key.
So we deliberately collapse to a single key:
  - multi-group gene  -> use the first group id (stable, deterministic)
  - no-group gene      -> its own singleton group keyed on the gene itself,
                          so it is never dropped and never merged with others
This choice is defensible and explicit. An interviewer will ask about it, so
the reasoning lives here in the code, not just in someone's head.

Run:  python3 ml/gene_families.py
Output: ml/cache/gene_families.parquet  (columns: symbol, ensembl_id, group_key)
"""

import argparse
import os
import io
import sys
import urllib.request
import pandas as pd

# HGNC complete dataset, tab-separated, hosted on their public GCS bucket.
# This URL is the "complete set" archive current file. If it 404s, check
# https://www.genenames.org/download/ for the current path (they moved from
# FTP to a GCS bucket; paths occasionally change).
HGNC_TSV_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/"
    "tsv/tsv/hgnc_complete_set.txt"
)

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")


def fetch_hgnc(url, dest):
    """Download the HGNC complete-set TSV once, cache it."""
    raw_path = os.path.join(dest, "hgnc_complete_set.txt")
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        print(f"already cached: {raw_path}")
        return raw_path
    os.makedirs(dest, exist_ok=True)
    print(f"downloading HGNC complete set from {url} ...")
    try:
        urllib.request.urlretrieve(url, raw_path)
    except Exception as e:
        print(f"[!] download failed: {e}")
        print("    Check the current URL at https://www.genenames.org/download/")
        sys.exit(1)
    print(f"cached: {raw_path}")
    return raw_path


def collapse_to_single_group(row_group_id, symbol):
    """
    Reduce a possibly-multi-valued group id to ONE grouping key.

    HGNC pipe-delimits multiple group ids in one field. We take the first,
    deterministically. Genes with no group get their own singleton key so they
    stay in the data as their own 'family of one'.
    """
    if isinstance(row_group_id, str) and row_group_id.strip():
        first = row_group_id.split("|")[0].strip()
        if first:
            return f"grp_{first}"
    # No group: singleton keyed on the gene symbol so it is unique.
    return f"solo_{symbol}"


def build(url=HGNC_TSV_URL, cache_dir=CACHE_DIR):
    raw_path = fetch_hgnc(url, cache_dir)

    # HGNC TSV is large and has many columns; read only what we need.
    # Column names per HGNC: 'symbol', 'ensembl_gene_id', 'gene_group_id',
    # 'locus_group'. gene_group_id is the pipe-delimited numeric id field.
    usecols = ["symbol", "ensembl_gene_id", "gene_group_id", "locus_group"]
    df = pd.read_csv(
        raw_path,
        sep="\t",
        usecols=lambda c: c in usecols,
        dtype=str,
        low_memory=False,
    )

    missing = set(usecols) - set(df.columns)
    if missing:
        print(f"[!] expected columns missing from HGNC file: {missing}")
        print(f"    columns present: {df.columns.tolist()}")
        print("    HGNC may have renamed fields; inspect the file and adjust usecols.")
        sys.exit(1)

    # Keep protein-coding genes: the label universe is protein-coding, so the
    # grouping only needs to cover those. Non-coding/pseudogenes drop out at
    # the join later anyway, but filtering here keeps the family file focused.
    before = len(df)
    df = df[df["locus_group"] == "protein-coding gene"].copy()
    print(f"protein-coding genes: {len(df):,} (from {before:,} total HGNC rows)")

    # Collapse to a single group key per gene.
    df["group_key"] = [
        collapse_to_single_group(gid, sym)
        for gid, sym in zip(df["gene_group_id"], df["symbol"])
    ]

    out = df[["symbol", "ensembl_gene_id", "group_key"]].rename(
        columns={"ensembl_gene_id": "ensembl_id"}
    )

    # Diagnostics: how much of the universe actually got a real (non-solo) group?
    n_total = len(out)
    n_real_group = (~out["group_key"].str.startswith("solo_")).sum()
    n_families = out.loc[~out["group_key"].str.startswith("solo_"), "group_key"].nunique()
    print(f"\ngenes with a real HGNC family: {n_real_group:,} / {n_total:,} "
          f"({n_real_group / n_total:.1%})")
    print(f"distinct real families: {n_families:,}")
    print(f"singleton (no-group) genes: {n_total - n_real_group:,}")

    # Sanity check: no gene should be null-keyed.
    assert out["group_key"].notna().all(), "found null group_key"
    assert out["symbol"].notna().all(), "found null symbol"

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "gene_families.parquet")
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}: {len(out):,} genes")

    print("\nVERDICT:")
    frac = n_real_group / n_total
    if frac < 0.4:
        print("  [!] Under 40% of genes got a real family. That is low; most genes")
        print("      would be singletons and the family-split would behave almost")
        print("      like a random split. Check the gene_group_id column parsed right.")
    else:
        print(f"  OK: {frac:.0%} of protein-coding genes have a real HGNC family, so")
        print("      the family-based split will meaningfully prevent paralog leakage.")
        print("      Singletons split as themselves, which is correct: a gene with no")
        print("      known family cannot leak into a paralog it does not have.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=HGNC_TSV_URL)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()
    build(url=args.url, cache_dir=args.cache_dir)
