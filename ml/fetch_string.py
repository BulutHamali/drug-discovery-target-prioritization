"""
Download STRING PPI network features for human genes and cache to ml/cache/.

Source: STRING v12.0, human (taxon 9606)
URLs:
  protein.links: https://stringdb-downloads.org/download/protein.links.v12.0/
                 9606.protein.links.v12.0.txt.gz  (~83 MB compressed)
  protein.info:  https://stringdb-downloads.org/download/protein.info.v12.0/
                 9606.protein.info.v12.0.txt.gz   (~2 MB compressed)

Features produced:
    ppi_degree          number of high-confidence interaction partners
    ppi_betweenness     approximate betweenness centrality in the PPI network

Confidence threshold:
    combined_score >= 700  (STRING "high confidence" tier)
    Below this, interactions are weakly supported and noise dominates signal.
    Lowering to 400 ("medium confidence") quadruples edge count without adding
    much signal and makes betweenness computation slow.

STUDY-BIAS CONFOUND -- read before interpreting feature importances:
    PPI degree and betweenness are heavily confounded by study bias.
    Well-studied genes have been co-immunoprecipitated, yeast-two-hybrid'd,
    and literature-mined more often than poorly studied genes, so they
    accumulate more STRING edges regardless of their true interactome size.
    These features are included because they carry real biology signal, but
    they must NOT be used as evidence that the model is not riding gene fame.
    The study-bias check in train_eval.py (publication count correlation) is
    the right diagnostic. See DESIGN.md section 6.2.

Betweenness computation:
    Exact betweenness is O(VE) which is slow on a full PPI graph. We use
    networkx approximate betweenness with k=500 pivot nodes (random_state=42).
    At k=500 the Pearson correlation with exact betweenness is > 0.99 on
    human PPI networks of this size (Brandes et al. 2007). Runtime: ~2-5 min.

Label independence:
    PPI degree and betweenness are derived from experimentally measured
    protein interactions and computational predictions. They do not contain
    drug, clinical trial, or approval status. Safe to use as features.

Join key: gene symbol (STRING preferred_name field, curated per protein).

Run:  python3 ml/fetch_string.py
Output: ml/cache/string_features.parquet
"""

import os
import sys
import urllib.request

import networkx as nx
import pandas as pd

RANDOM_SEED = 42

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")
OUT_FILE  = os.path.join(CACHE_DIR, "string_features.parquet")

STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)

LINKS_RAW = os.path.join(CACHE_DIR, "string_links_v12.txt.gz")
INFO_RAW  = os.path.join(CACHE_DIR, "string_info_v12.txt.gz")

# STRING combined_score threshold for "high confidence".
# Score is an integer on [0, 1000] where 1000 = maximum confidence.
SCORE_THRESHOLD = 700

# Number of pivot nodes for approximate betweenness. k=500 gives
# Pearson r > 0.99 with exact betweenness on human-sized PPI graphs.
BETWEENNESS_K = 500


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


def load_info(path):
    """Return a dict mapping STRING protein id -> gene symbol."""
    df = pd.read_csv(
        path, sep="\t", compression="gzip",
        usecols=["#string_protein_id", "preferred_name"],
        dtype=str, low_memory=False,
    )
    df = df.rename(columns={
        "#string_protein_id": "string_id",
        "preferred_name":     "symbol",
    })
    return dict(zip(df["string_id"], df["symbol"]))


def load_links(path, score_threshold):
    """Return edge DataFrame filtered to high-confidence interactions."""
    df = pd.read_csv(
        path, sep=" ", compression="gzip",
        dtype={"protein1": str, "protein2": str, "combined_score": int},
        low_memory=False,
    )
    before = len(df)
    df = df[df["combined_score"] >= score_threshold].copy()
    print(f"  loaded {before:,} edges, kept {len(df):,} with score >= {score_threshold}")
    return df


def build(cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)

    links_raw = os.path.join(cache_dir, "string_links_v12.txt.gz")
    info_raw  = os.path.join(cache_dir, "string_info_v12.txt.gz")
    out_path  = os.path.join(cache_dir, "string_features.parquet")

    download(STRING_INFO_URL,  info_raw,  "STRING protein.info (~2 MB)")
    download(STRING_LINKS_URL, links_raw, "STRING protein.links (~83 MB)")

    print("\nloading protein info (STRING id -> gene symbol) ...")
    id_to_sym = load_info(info_raw)
    print(f"  {len(id_to_sym):,} STRING proteins mapped to gene symbols")

    print("\nloading interaction links ...")
    links = load_links(links_raw, SCORE_THRESHOLD)

    # Map STRING IDs to gene symbols. Drop edges where either endpoint
    # has no gene symbol (e.g. non-coding or poorly annotated proteins).
    links["sym1"] = links["protein1"].map(id_to_sym)
    links["sym2"] = links["protein2"].map(id_to_sym)
    before = len(links)
    links = links.dropna(subset=["sym1", "sym2"])
    print(f"  {before - len(links):,} edges dropped (unmapped protein IDs)")
    print(f"  {len(links):,} edges retained after symbol mapping")

    # Build undirected graph. STRING edges are already bidirectional in the
    # file; networkx deduplicates parallel edges automatically with Graph().
    print("\nbuilding PPI graph ...")
    G = nx.Graph()
    G.add_edges_from(zip(links["sym1"], links["sym2"]))
    print(f"  nodes: {G.number_of_nodes():,}  edges: {G.number_of_edges():,}")

    # Degree: simple edge count per gene. Fast O(V).
    print("computing degree ...")
    degree_dict = dict(G.degree())

    # Approximate betweenness centrality using k random pivot nodes.
    # Exact betweenness is O(VE); k=500 approximation runs in ~2-5 min
    # and correlates > 0.99 with exact values at this network size.
    print(f"computing approximate betweenness (k={BETWEENNESS_K}, seed={RANDOM_SEED}) ...")
    print("  this takes ~2-5 minutes ...")
    betweenness_dict = nx.betweenness_centrality(
        G, k=BETWEENNESS_K, normalized=True, seed=RANDOM_SEED
    )
    print("  done.")

    out = pd.DataFrame({
        "symbol":          list(degree_dict.keys()),
        "ppi_degree":      list(degree_dict.values()),
        "ppi_betweenness": [betweenness_dict[s] for s in degree_dict],
    })

    print(f"\nSanity checks:")
    n_genes = out["symbol"].nunique()
    assert n_genes == len(out), "duplicate gene symbols in output"
    print(f"  unique genes: {n_genes:,}")
    print(f"  ppi_degree    range: [{out['ppi_degree'].min()}, {out['ppi_degree'].max()}]  "
          f"median: {out['ppi_degree'].median():.0f}")
    print(f"  ppi_betweenness range: [{out['ppi_betweenness'].min():.6f}, "
          f"{out['ppi_betweenness'].max():.6f}]  "
          f"median: {out['ppi_betweenness'].median():.6f}")

    # Top 10 by degree as a spot check -- these should be well-known hubs.
    top10 = out.nlargest(10, "ppi_degree")[["symbol", "ppi_degree", "ppi_betweenness"]]
    print(f"\nTop 10 genes by PPI degree (spot check -- expect known hubs):")
    print(top10.to_string(index=False))

    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  rows: {len(out):,}  columns: {out.columns.tolist()}")

    print(
        "\nSTUDY-BIAS NOTE: ppi_degree and ppi_betweenness are confounded by "
        "characterisation depth.\n"
        "  Well-studied genes accumulate more measured interactions regardless of "
        "true interactome size.\n"
        "  Interpret feature importances for these columns with caution. "
        "See train_eval.py study-bias check."
    )


if __name__ == "__main__":
    build()
