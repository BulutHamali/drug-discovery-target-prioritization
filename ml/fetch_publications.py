"""
Download NCBI gene-publication associations and cache per-gene publication
metadata to ml/cache/. This is the deliberate confounder feature from
DESIGN.md section 5: "Publication count and year-first-described. Included
specifically so the model can be shown not to be riding it."

Sources:
  NCBI gene2pubmed (curated Entrez GeneID <-> PubMed ID links, all
  organisms, ~272 MB gzipped):
    https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz

  HGNC complete set (same file gene_families.py already caches) for the
  Entrez GeneID -> gene symbol mapping:
    https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/
    hgnc_complete_set.txt

  NCBI eutils esummary, batched, to resolve the small set of "first PMID
  per gene" values to an actual publication year:
    https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi

Features produced:
    pub_count             number of distinct PubMed records linked to the
                          gene's Entrez GeneID (human, tax_id 9606).
    year_first_described  publication year of the gene's earliest linked
                          PubMed record.

Why gene2pubmed and not a free-text search (Europe PMC / PubMed keyword
query on gene symbol):
  Many gene symbols collide with common English words or abbreviations
  (SET, FOR, CAN, CAT, CIC, ...). A naive title/abstract keyword search
  would massively over-count these genes with unrelated literature.
  gene2pubmed is NCBI-curated: each row is a real, reviewed association
  between an Entrez Gene record and a PubMed record, so it does not have
  this collision problem. This is the standard method used in gene
  "fame" / study-bias bibliometrics (e.g. Pandey et al. 2014).

Why the minimum PMID as the "first described" proxy, resolved via esummary
rather than a hand-built PMID-to-year lookup table:
  PubMed IDs are assigned roughly sequentially as records are indexed, so
  the lowest PMID linked to a gene is a reasonable proxy for its earliest
  literature appearance. Rather than approximate the PMID -> year mapping
  with a hardcoded table, we resolve the (small, deduplicated) set of
  minimum PMIDs through NCBI's esummary endpoint in batches of 200 IDs,
  which gives the real publication year directly at negligible extra cost
  (tens of requests, not one per gene).

Join key: gene symbol, via the Entrez GeneID -> symbol mapping in the HGNC
  complete set (restricted to protein-coding genes, consistent with the
  universe built in gene_families.py).

Label independence: publication count and first-description year are
  literature metadata, not drug or clinical evidence. They are included
  specifically to let the study-bias check (train_eval.py) test whether
  the model's ranking is actually just tracking gene fame. Safe to use as
  a feature exactly because it is a known confounder we want to detect.

Run:  python3 ml/fetch_publications.py
Output: ml/cache/publication_features.parquet
"""

import gzip
import os
import re
import sys
import time
import urllib.request
import urllib.parse

import pandas as pd

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")

GENE2PUBMED_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz"
GENE2PUBMED_RAW = os.path.join(CACHE_DIR, "gene2pubmed.gz")

# Same URL and cache path gene_families.py uses -- if it already ran,
# download() below just reuses the cached file.
HGNC_TSV_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/"
    "tsv/tsv/hgnc_complete_set.txt"
)
HGNC_RAW = os.path.join(CACHE_DIR, "hgnc_complete_set.txt")

ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
ESUMMARY_BATCH = 200
# NCBI's rate limit without an API key is 3 requests/second.
ESUMMARY_SLEEP = 0.34

HUMAN_TAXID = "9606"

OUT_FILE = os.path.join(CACHE_DIR, "publication_features.parquet")


def download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"already cached: {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {label} ...")
    print(f"  {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"saved: {dest}  ({os.path.getsize(dest):,} bytes)")


def load_entrez_to_symbol(hgnc_path):
    df = pd.read_csv(
        hgnc_path, sep="\t", dtype=str, low_memory=False,
        usecols=["symbol", "entrez_id", "locus_group"],
    )
    df = df[(df["locus_group"] == "protein-coding gene") & df["entrez_id"].notna()]
    return dict(zip(df["entrez_id"], df["symbol"]))


def load_human_gene2pubmed(path):
    """
    gene2pubmed spans every organism NCBI tracks (~hundreds of millions of
    rows). Stream it in chunks and keep only tax_id == 9606 rows so the
    full multi-species file is never held in memory at once.
    """
    kept = []
    n_seen = 0
    reader = pd.read_csv(
        path, sep="\t", compression="gzip", dtype=str,
        chunksize=5_000_000,
    )
    for chunk in reader:
        chunk.columns = [c.lstrip("#") for c in chunk.columns]
        n_seen += len(chunk)
        human = chunk[chunk["tax_id"] == HUMAN_TAXID]
        if len(human):
            kept.append(human[["GeneID", "PubMed_ID"]])
    df = pd.concat(kept, ignore_index=True)
    df["PubMed_ID"] = pd.to_numeric(df["PubMed_ID"], errors="coerce")
    df = df.dropna(subset=["PubMed_ID"])
    df["PubMed_ID"] = df["PubMed_ID"].astype(int)
    print(f"  scanned {n_seen:,} gene2pubmed rows (all organisms), "
          f"kept {len(df):,} human (tax_id {HUMAN_TAXID}) rows")
    return df


def resolve_years(pmids):
    """Batch-resolve PubMed IDs to publication year via NCBI esummary."""
    pmids = sorted(set(int(p) for p in pmids))
    year_by_pmid = {}
    n_batches = (len(pmids) + ESUMMARY_BATCH - 1) // ESUMMARY_BATCH

    for i in range(0, len(pmids), ESUMMARY_BATCH):
        batch = pmids[i:i + ESUMMARY_BATCH]
        if (i // ESUMMARY_BATCH) % 10 == 0:
            print(f"  esummary batch {i // ESUMMARY_BATCH + 1}/{n_batches} ...")

        params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(str(p) for p in batch),
            "retmode": "json",
        })
        try:
            with urllib.request.urlopen(f"{ESUMMARY_URL}?{params}", timeout=30) as r:
                import json
                data = json.load(r)
        except Exception as exc:
            print(f"  [!] esummary batch failed, skipping {len(batch)} PMIDs: {exc}",
                  file=sys.stderr)
            time.sleep(ESUMMARY_SLEEP)
            continue

        result = data.get("result", {})
        for uid in result.get("uids", []):
            pubdate = result.get(uid, {}).get("pubdate", "")
            m = re.match(r"(\d{4})", pubdate)
            if m:
                year_by_pmid[int(uid)] = int(m.group(1))

        time.sleep(ESUMMARY_SLEEP)

    return year_by_pmid


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)

    download(HGNC_TSV_URL, HGNC_RAW, "HGNC complete set (~17 MB)")
    entrez_to_symbol = load_entrez_to_symbol(HGNC_RAW)
    print(f"  {len(entrez_to_symbol):,} protein-coding genes with an Entrez GeneID")

    download(GENE2PUBMED_URL, GENE2PUBMED_RAW, "NCBI gene2pubmed (~272 MB)")
    print("\nloading gene2pubmed (filtering to human) ...")
    g2p = load_human_gene2pubmed(GENE2PUBMED_RAW)

    print("\naggregating per gene ...")
    per_gene = (
        g2p.groupby("GeneID")["PubMed_ID"]
        .agg(pub_count="nunique", min_pmid="min")
        .reset_index()
    )
    per_gene["symbol"] = per_gene["GeneID"].map(entrez_to_symbol)
    before = len(per_gene)
    per_gene = per_gene.dropna(subset=["symbol"]).copy()
    print(f"  {len(per_gene):,} / {before:,} genes mapped to a protein-coding symbol")

    print(f"\nresolving {per_gene['min_pmid'].nunique():,} unique first-PMIDs to "
          f"publication years via esummary ...")
    year_by_pmid = resolve_years(per_gene["min_pmid"].unique())
    per_gene["year_first_described"] = per_gene["min_pmid"].map(year_by_pmid)
    n_resolved = per_gene["year_first_described"].notna().sum()
    print(f"  resolved {n_resolved:,} / {len(per_gene):,} years")

    out = per_gene[["symbol", "pub_count", "year_first_described"]].copy()

    # A gene symbol could map from more than one Entrez GeneID (rare, e.g.
    # a withdrawn/merged record); sum pub_count and keep the earliest year.
    before = len(out)
    n_dupe_syms = out["symbol"].duplicated().sum()
    if n_dupe_syms:
        out = out.groupby("symbol", as_index=False).agg(
            pub_count=("pub_count", "sum"),
            year_first_described=("year_first_described", "min"),
        )
    print(f"  {n_dupe_syms:,} duplicate gene symbols merged: {before:,} -> {len(out):,}")

    print(f"\nSanity checks:")
    print(f"  genes with pub_count:            {out['pub_count'].notna().sum():,} / {len(out):,}")
    print(f"  genes with year_first_described: {out['year_first_described'].notna().sum():,} / {len(out):,}")
    print(f"  pub_count range: [{out['pub_count'].min():.0f}, {out['pub_count'].max():.0f}]  "
          f"median: {out['pub_count'].median():.0f}")
    yr = out["year_first_described"].dropna()
    print(f"  year_first_described range: [{yr.min():.0f}, {yr.max():.0f}]  median: {yr.median():.0f}")

    n_dupes = out["symbol"].duplicated().sum()
    assert n_dupes == 0, f"FAIL: {n_dupes} duplicate gene symbols in output"

    out.to_parquet(OUT_FILE, index=False)
    print(f"\nwrote {OUT_FILE}")
    print(f"  rows: {len(out):,}  columns: {out.columns.tolist()}")


if __name__ == "__main__":
    build()
