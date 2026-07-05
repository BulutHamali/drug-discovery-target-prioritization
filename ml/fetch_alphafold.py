"""
Download human protein features from UniProt Swiss-Prot and cache to ml/cache/.

Source: UniProt Swiss-Prot, human proteome (organism_id:9606, reviewed:true)
URL: https://rest.uniprot.org/uniprotkb/stream (REST API v2)

Why Swiss-Prot (reviewed) and not TrEMBL:
  Swiss-Prot entries are manually curated. Gene name mappings are reliable.
  TrEMBL (unreviewed) gene names are machine-predicted and often ambiguous,
  which would corrupt the symbol join to the HGNC universe.

Features produced:
    protein_length       number of amino acids in the canonical sequence
    disorder_fraction    fraction of residues with pLDDT < 50 (AlphaFold proxy
                         for structural disorder). OFF by default; pass --disorder
                         to enable. See [DISORDER HOOK] below.

Join key:
  Gene symbol (first token of UniProt "Gene Names" field). The UniProt Ensembl
  cross-reference field returns transcript IDs (ENST...), not gene IDs (ENSG...),
  so symbol is the reliable join key here. This is consistent with the burden
  join in build_features.py.

  If a gene symbol appears in multiple UniProt entries (e.g. redundant isoforms),
  we keep the entry with the longest sequence -- this is the canonical isoform
  by convention.

Disorder fraction hook:
  Download one per-protein confidence JSON from the AlphaFold EBI endpoint:
    https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-confidence_v4.json
  For ~20,000 proteins this is ~20,000 HTTP requests (~50 min). Implemented
  below and gated behind --disorder. Per-protein JSON files are cached so
  interrupted runs can resume. Pass --disorder to enable.

Label independence: protein length and pLDDT-based disorder are intrinsic
  sequence/structure properties. They are derived entirely from genome sequence
  and predicted structure. Neither depends on drug history, clinical trials, or
  the label source (knownDrugsAggregated). Safe to use as features.

Run:  python3 ml/fetch_alphafold.py
      python3 ml/fetch_alphafold.py --disorder   # adds ~50 min for pLDDT
Output: ml/cache/alphafold_features.parquet
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")
OUT_FILE  = os.path.join(CACHE_DIR, "alphafold_features.parquet")

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=organism_id%3A9606+AND+reviewed%3Atrue"
    "&format=tsv"
    "&fields=accession%2Cgene_names%2Clength"
)
UNIPROT_RAW = os.path.join(CACHE_DIR, "uniprot_human_swissprot.tsv")

# AlphaFold EBI per-protein confidence endpoint.
AF_CONF_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-confidence_v4.json"
)

# Residues below this pLDDT score are considered disordered (Jumper et al. 2021).
PLDDT_DISORDER_THRESHOLD = 50


def download_uniprot(dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"already cached: {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading UniProt human Swiss-Prot TSV (~3-5 MB) ...")
    print(f"  {UNIPROT_URL}")
    try:
        urllib.request.urlretrieve(UNIPROT_URL, dest)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"saved: {dest}  ({os.path.getsize(dest):,} bytes)")


def parse_uniprot(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)
    print(f"loaded {len(df):,} UniProt entries, columns: {df.columns.tolist()}")

    df = df.rename(columns={
        "Entry":       "uniprot_id",
        "Gene Names":  "gene_names_raw",
        "Length":      "protein_length_str",
    })

    df["protein_length"] = pd.to_numeric(df["protein_length_str"], errors="coerce").astype("Int64")

    # Primary gene symbol: first whitespace-delimited token of Gene Names.
    # UniProt lists synonyms after the primary symbol separated by spaces.
    df["symbol"] = (
        df["gene_names_raw"]
        .fillna("")
        .str.split()
        .str[0]
        .str.strip()
    )
    # Drop rows with no usable gene symbol (e.g. hypothetical proteins).
    df = df[df["symbol"].str.len() > 0].copy()

    return df[["uniprot_id", "symbol", "protein_length"]].copy()


def fetch_disorder(uniprot_ids, cache_dir):
    """
    [DISORDER HOOK]
    Fetches per-residue pLDDT from AlphaFold EBI and returns disorder_fraction
    (fraction of residues with pLDDT < PLDDT_DISORDER_THRESHOLD) per protein.

    Per-protein JSON files are cached under cache_dir/af_confidence/ so that
    interrupted runs can resume without re-downloading completed proteins.
    """
    conf_dir = os.path.join(cache_dir, "af_confidence")
    os.makedirs(conf_dir, exist_ok=True)

    result = {}
    n = len(uniprot_ids)
    for i, uid in enumerate(uniprot_ids):
        if i % 500 == 0:
            print(f"  pLDDT {i:,}/{n:,} ...")

        cache_path = os.path.join(conf_dir, f"{uid}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            url = AF_CONF_URL.format(uid=uid)
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    data = json.load(r)
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                time.sleep(0.15)  # respect EBI rate limits (~6-7 req/s)
            except Exception:
                result[uid] = float("nan")
                continue

        scores = data.get("confidenceScore", [])
        if scores:
            n_disordered = sum(1 for v in scores if isinstance(v, (int, float)) and v < PLDDT_DISORDER_THRESHOLD)
            result[uid] = n_disordered / len(scores)
        else:
            result[uid] = float("nan")

    return pd.Series(result, name="disorder_fraction")


def build(fetch_disorder_flag=False, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)

    raw_path = os.path.join(cache_dir, "uniprot_human_swissprot.tsv")
    out_path = os.path.join(cache_dir, "alphafold_features.parquet")

    download_uniprot(raw_path)
    df = parse_uniprot(raw_path)

    if fetch_disorder_flag:
        print(f"\nfetching disorder_fraction from AlphaFold EBI (~50 min) ...")
        dis = fetch_disorder(df["uniprot_id"].tolist(), cache_dir)
        df["disorder_fraction"] = df["uniprot_id"].map(dis)
        n_dis = df["disorder_fraction"].notna().sum()
        print(f"  disorder_fraction: {n_dis:,} / {len(df):,} proteins")
    else:
        df["disorder_fraction"] = float("nan")
        print(
            "\n[DISORDER HOOK] disorder_fraction not fetched (pass --disorder to enable).\n"
            "  Column is present but NaN. build_features.py will median-impute it and\n"
            "  set has_af_disorder=0 so the model can distinguish real from imputed."
        )

    # Deduplicate: if multiple UniProt entries share a gene symbol, keep the
    # entry with the longest sequence. The longest isoform is typically canonical
    # and carries the most domain content, making its length most informative.
    before = len(df)
    df = (
        df.sort_values("protein_length", ascending=False, na_position="last")
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )
    print(f"\ndeduplicated to one entry per gene symbol: {before:,} -> {len(df):,}")

    print(f"\nSanity checks:")
    n_len = df["protein_length"].notna().sum()
    assert n_len > 15_000, f"only {n_len} proteins with length -- check TSV parsing"
    assert (df["protein_length"].dropna() > 0).all(), "non-positive protein length found"
    print(f"  genes with protein_length:  {n_len:,} / {len(df):,}")
    print(f"  length range: [{int(df['protein_length'].min())} .. {int(df['protein_length'].max())}] aa")
    print(f"  median length: {df['protein_length'].median():.0f} aa")

    if fetch_disorder_flag:
        n_dis = df["disorder_fraction"].notna().sum()
        print(f"  genes with disorder_fraction: {n_dis:,} / {len(df):,}")
        print(f"  median disorder_fraction: {df['disorder_fraction'].median():.3f}")

    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  rows: {len(df):,}  columns: {df.columns.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--disorder", action="store_true",
        help="fetch pLDDT-based disorder_fraction from AlphaFold EBI (~50 min, 20k requests)"
    )
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()
    build(fetch_disorder_flag=args.disorder, cache_dir=args.cache_dir)
