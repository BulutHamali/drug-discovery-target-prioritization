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
  Query one per-protein prediction summary from the AlphaFold DB API:
    https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}
  This returns fractionPlddtVeryLow directly (fraction of residues with
  pLDDT < 50, AlphaFold's own "very low confidence" bucket boundary), so no
  per-residue array needs to be fetched or parsed; it is used as
  disorder_fraction unchanged. (The old versioned bulk file URL,
  alphafold.ebi.ac.uk/files/AF-{uid}-F1-confidence_v4.json, is dead across
  every version number as of this writing, 404/NoSuchKey on every request;
  AlphaFold DB moved to this API. Confirmed via curl against known human
  UniProt IDs before switching.) For ~20,000 proteins this is ~20,000 HTTP
  requests. Implemented below and gated behind --disorder. Per-protein JSON
  responses are cached so interrupted runs can resume. Pass --disorder to
  enable.

Label independence: protein length and pLDDT-based disorder are intrinsic
  sequence/structure properties. They are derived entirely from genome sequence
  and predicted structure. Neither depends on drug history, clinical trials, or
  the label source (knownDrugsAggregated). Safe to use as features.

Run:  python3 ml/fetch_alphafold.py
      python3 ml/fetch_alphafold.py --disorder   # adds time for ~20k API calls
Output: ml/cache/alphafold_features.parquet
"""

import argparse
import json
import os
import sys
import time
import urllib.error
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

# AlphaFold DB per-protein prediction API. Replaces the old versioned bulk
# file URL (alphafold.ebi.ac.uk/files/AF-{uid}-F1-confidence_v4.json), which
# is dead (404/NoSuchKey on every version, confirmed by hand before this
# fix). This endpoint returns fractionPlddtVeryLow directly, no per-residue
# parsing needed.
AF_CONF_URL = (
    "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
)

# Residues below this pLDDT score are considered disordered (Jumper et al. 2021).
PLDDT_DISORDER_THRESHOLD = 50

# Minimum fraction of real (non-NaN) disorder_fraction values required after a
# --disorder run, checked against the final deduplicated gene-symbol table.
# Observed coverage in this project is 98.2% of the gene universe (98.9% of
# the raw UniProt table). This threshold exists so a dead or broken endpoint
# fails the run instead of silently writing a near-empty column that later
# gets median-imputed into a useless constant (see git history: this is
# exactly what the original --disorder bug did).
MIN_DISORDER_COVERAGE = 0.90

# If this many consecutive live network requests fail, stop immediately
# instead of grinding through ~20,000 requests against a dead endpoint before
# finding out. A handful of individual 404s for real (e.g. withdrawn/obsolete
# UniProt IDs) is expected and tolerated; a long unbroken failure streak means
# the endpoint itself is down.
MAX_CONSECUTIVE_FAILURES = 50


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
    Queries the AlphaFold DB prediction API per protein and returns
    disorder_fraction = fractionPlddtVeryLow (AlphaFold's own "pLDDT < 50"
    confidence bucket, matching PLDDT_DISORDER_THRESHOLD exactly), read
    straight from the API response, no per-residue array needed.

    Per-protein JSON responses are cached under cache_dir/af_confidence/ so
    that interrupted runs can resume without re-querying completed proteins.
    AlphaFold DB does not have a model for every Swiss-Prot accession (very
    large proteins and some others are genuinely absent); a confirmed 404 is
    cached as a permanent negative result (a distinct JSON sentinel) so it is
    not re-requested over the network on every future run, and so it does not
    get confused with an endpoint failure below.

    Fail-loud guard, and why it is not "any exception == dead endpoint":
    a plain 404 for a specific accession is the server correctly answering
    "no model for this protein," a normal and expected outcome documented at
    roughly 98% coverage, not a sign anything is broken. Only failures where
    the server did NOT give that clean answer (timeouts, connection errors,
    5xx, malformed JSON on what should be a 200) count toward the "is the
    endpoint itself broken" circuit breaker: a long unbroken run of THOSE
    means the endpoint is down, and raises immediately instead of silently
    NaN-ing thousands of requests (this is exactly how the original
    --disorder bug produced a fully-empty column that got median-imputed
    without error). An earlier version of this guard treated every 404 as a
    failure too, which misfired: the handful of genes AlphaFold DB has never
    modeled are never cached as failures (nothing to cache, they errored),
    so they get retried every run, and when several happen to land next to
    each other in iteration order they can trip a naive "N consecutive
    failures" counter even though the endpoint is perfectly healthy.
    """
    conf_dir = os.path.join(cache_dir, "af_confidence")
    os.makedirs(conf_dir, exist_ok=True)
    NOT_FOUND = {"__not_found__": True}

    result = {}
    n = len(uniprot_ids)
    consecutive_endpoint_failures = 0
    n_attempted = 0
    n_endpoint_failed = 0
    for i, uid in enumerate(uniprot_ids):
        if i % 500 == 0:
            print(f"  pLDDT {i:,}/{n:,} ...")

        cache_path = os.path.join(conf_dir, f"{uid}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
        else:
            n_attempted += 1
            url = AF_CONF_URL.format(uid=uid)
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    payload = json.load(r)
                data = payload[0] if isinstance(payload, list) and payload else None
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                time.sleep(0.05)  # respect EBI rate limits
                consecutive_endpoint_failures = 0
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    # Confirmed answer: this accession has no AlphaFold model.
                    # Cache it as a permanent negative so it is never
                    # re-requested, and do not count it against the endpoint's
                    # health.
                    data = NOT_FOUND
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                    time.sleep(0.05)
                    consecutive_endpoint_failures = 0
                else:
                    n_endpoint_failed += 1
                    consecutive_endpoint_failures += 1
                    if consecutive_endpoint_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"AlphaFold DB prediction API: {consecutive_endpoint_failures} "
                            f"consecutive non-404 failures (last: {uid}, HTTP "
                            f"{exc.code}, URL pattern {AF_CONF_URL}, error: {exc}). "
                            f"Stopping instead of grinding through the remaining "
                            f"{n - i - 1:,} proteins against what looks like a broken "
                            f"endpoint. Run python3 ml/check_endpoints.py to confirm."
                        ) from exc
                    result[uid] = float("nan")
                    continue
            except Exception as exc:
                n_endpoint_failed += 1
                consecutive_endpoint_failures += 1
                if consecutive_endpoint_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"AlphaFold DB prediction API: {consecutive_endpoint_failures} "
                        f"consecutive live requests failed with no HTTP response "
                        f"(last: {uid}, URL pattern {AF_CONF_URL}, error: {exc}). "
                        f"Stopping instead of grinding through the remaining "
                        f"{n - i - 1:,} proteins against what looks like a dead "
                        f"endpoint. Run python3 ml/check_endpoints.py to confirm."
                    ) from exc
                result[uid] = float("nan")
                continue

        # Works identically whether data is a fresh NOT_FOUND dict or one
        # reloaded from cache (a different object, same content): neither
        # has "fractionPlddtVeryLow", so both fall through to NaN correctly.
        if data and "fractionPlddtVeryLow" in data:
            result[uid] = data["fractionPlddtVeryLow"]
        else:
            result[uid] = float("nan")

    if n_attempted > 0 and n_endpoint_failed / n_attempted > 0.5:
        raise RuntimeError(
            f"AlphaFold DB prediction API: {n_endpoint_failed:,} / {n_attempted:,} "
            f"live requests failed with no clean HTTP response (>50%). Endpoint "
            f"{AF_CONF_URL} is likely broken or rate-limiting. Refusing to write "
            f"a mostly-empty disorder_fraction column. Run "
            f"python3 ml/check_endpoints.py to confirm."
        )

    return pd.Series(result, name="disorder_fraction")


def build(fetch_disorder_flag=False, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)

    raw_path = os.path.join(cache_dir, "uniprot_human_swissprot.tsv")
    out_path = os.path.join(cache_dir, "alphafold_features.parquet")

    download_uniprot(raw_path)
    df = parse_uniprot(raw_path)

    if fetch_disorder_flag:
        print(f"\nfetching disorder_fraction from AlphaFold DB (roughly 90-100 min at ~0.3s/protein) ...")
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
    assert n_len > 15_000, (
        f"FAIL: UniProt Swiss-Prot ({UNIPROT_URL}) returned only {n_len:,} "
        f"proteins with a parsed length, expected 15,000+. Either the TSV "
        f"format changed or the response was empty/truncated. Check "
        f"{raw_path} by hand before re-running."
    )
    assert (df["protein_length"].dropna() > 0).all(), "non-positive protein length found"
    print(f"  genes with protein_length:  {n_len:,} / {len(df):,}")
    print(f"  length range: [{int(df['protein_length'].min())} .. {int(df['protein_length'].max())}] aa")
    print(f"  median length: {df['protein_length'].median():.0f} aa")

    if fetch_disorder_flag:
        n_dis = df["disorder_fraction"].notna().sum()
        coverage = n_dis / len(df)
        print(f"  genes with disorder_fraction: {n_dis:,} / {len(df):,} ({coverage:.1%})")
        assert coverage >= MIN_DISORDER_COVERAGE, (
            f"FAIL: disorder_fraction coverage {coverage:.1%} is below the "
            f"{MIN_DISORDER_COVERAGE:.0%} floor (observed coverage historically "
            f"is 98.2%). Refusing to write a parquet file that would get "
            f"silently median-imputed into a near-constant column downstream. "
            f"Check the AlphaFold DB prediction API is up: "
            f"python3 ml/check_endpoints.py"
        )
        print(f"  median disorder_fraction: {df['disorder_fraction'].median():.3f}")

    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  rows: {len(df):,}  columns: {df.columns.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--disorder", action="store_true",
        help="fetch pLDDT-based disorder_fraction from AlphaFold DB (roughly 90-100 min, ~20k requests)"
    )
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()
    build(fetch_disorder_flag=args.disorder, cache_dir=args.cache_dir)
