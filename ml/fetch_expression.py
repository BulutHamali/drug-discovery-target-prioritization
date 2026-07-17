"""
Download GTEx tissue expression and DepMap CRISPR essentiality data and
cache per-gene features to ml/cache/.

Sources:
  GTEx v8 median gene TPM by tissue (54 tissues, ~6.9 MB gzipped):
    https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/
    GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz

  DepMap 24Q4 Public CRISPR gene effect (Chronos score, ~429 MB):
    https://ndownloader.figshare.com/files/51064667  (CRISPRGeneEffect.csv)
    One row per cell line (~1,150), one column per gene. Figshare article:
    https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248

Features produced:
    tau                 tissue-specificity index (Yanai et al. 2005),
                        range [0, 1]. 0 = uniformly expressed across all
                        GTEx tissues (housekeeping), 1 = expressed in a
                        single tissue only.
    essentiality_score  mean Chronos gene effect score across all DepMap
                        screened cell lines. More negative = more
                        commonly essential for cell survival; near 0 =
                        dispensable in most lines.

Why log2-transform before computing tau:
  Yanai's original tau is computed on linear expression values, but doing
  so on raw TPM lets a handful of extremely highly expressed tissues
  dominate the ratio and pushes tau toward 1 for genes that are not truly
  tissue-restricted. Kryuchkova-Mostacci & Robinson-Rechavi (2016,
  Briefings in Bioinformatics) recommend log2(TPM + 1) before computing
  tau for exactly this reason; that is what we do here.

Why mean gene effect (not a binary "is essential" flag):
  DepMap ships curated reference lists (AchillesCommonEssentialControls,
  AchillesNonessentialControls) but those are categorical control sets
  for QC, not a per-gene continuous score. Averaging the Chronos gene
  effect across all screened cell lines gives a continuous essentiality
  signal that preserves genes that are essential in only a subset of
  lineages (context-specific essentiality), which a binary flag would
  erase.

Join key:
  Gene symbol.
    GTEx:   the GCT "Description" column (curated gene symbol).
    DepMap: parsed from the CSV column header, formatted "SYMBOL (ENTREZID)".
  Consistent with the burden/AlphaFold/STRING joins in build_features.py.

  A gene symbol can occasionally appear more than once in either source
  (e.g. two Ensembl loci sharing a historical symbol in GTEx). We keep the
  entry with the highest total signal (sum of tissue TPM for GTEx, mean of
  duplicate columns for DepMap) as the canonical row, matching the
  dedup convention used in fetch_alphafold.py.

Label independence: tissue specificity and cell-line essentiality are
  intrinsic functional-genomics properties, derived from RNA-seq and
  CRISPR knockout screens. Neither depends on drug history, clinical
  trials, or the label source (knownDrugsAggregated). Safe to use as
  features.

Run:  python3 ml/fetch_expression.py
Output: ml/cache/expression_features.parquet
"""

import gzip
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = os.environ.get("ML_CACHE_DIR", "ml/cache")

GTEX_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)
GTEX_RAW = os.path.join(CACHE_DIR, "gtex_v8_gene_median_tpm.gct.gz")

DEPMAP_URL = "https://ndownloader.figshare.com/files/51064667"
DEPMAP_RAW = os.path.join(CACHE_DIR, "depmap_24q4_crispr_gene_effect.csv")

OUT_FILE = os.path.join(CACHE_DIR, "expression_features.parquet")


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


def load_gtex_tau(path):
    """
    Parse the GTEx GCT file and compute tau per gene.

    GCT format: line 1 is a version tag, line 2 is "<n_genes>\\t<n_tissues>",
    line 3 is the column header (Name, Description, then one column per
    tissue), and each subsequent row is one gene.
    """
    with gzip.open(path, "rt") as f:
        f.readline()  # version tag, e.g. "#1.2"
        f.readline()  # dimensions, e.g. "56200\t54"
        raw = f.read()
    df = pd.read_csv(io.StringIO(raw), sep="\t")
    print(f"  loaded {len(df):,} GTEx genes x "
          f"{len(df.columns) - 2} tissues")

    tissue_cols = [c for c in df.columns if c not in ("Name", "Description")]
    tpm = df[tissue_cols].to_numpy(dtype=float)

    # log2(TPM + 1) before computing tau -- see module docstring WHY.
    log_tpm = np.log2(tpm + 1.0)
    max_per_gene = log_tpm.max(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = log_tpm / max_per_gene[:, None]
    # Genes with zero expression in every tissue give max_per_gene == 0,
    # so ratio is 0/0 = NaN for the whole row -- tau is genuinely undefined
    # for a gene never observed expressed, not zero. np.sum (not nansum)
    # propagates that NaN through to tau, which is the correct behaviour.
    tau = np.sum(1.0 - ratio, axis=1) / (len(tissue_cols) - 1)

    out = pd.DataFrame({
        "symbol": df["Description"],
        "tau": tau,
        "_total_tpm": tpm.sum(axis=1),  # used only for dedup below
    })

    before = len(out)
    out = (
        out.sort_values("_total_tpm", ascending=False)
        .drop_duplicates(subset=["symbol"])
        .drop(columns=["_total_tpm"])
        .reset_index(drop=True)
    )
    print(f"  deduplicated to one entry per gene symbol: {before:,} -> {len(out):,}")
    return out


def load_depmap_essentiality(path):
    """
    Parse DepMap CRISPRGeneEffect.csv (rows = cell lines, columns = genes
    formatted "SYMBOL (ENTREZID)") and return the mean gene effect score
    per gene across all screened cell lines.
    """
    df = pd.read_csv(path, index_col=0, low_memory=False)
    print(f"  loaded {len(df):,} cell lines x {len(df.columns):,} genes")

    mean_effect = df.mean(axis=0, skipna=True)
    symbols = mean_effect.index.str.replace(r"\s*\(\d+\)$", "", regex=True)

    out = pd.DataFrame({
        "symbol": symbols,
        "essentiality_score": mean_effect.to_numpy(),
    })

    before = len(out)
    n_dupe_syms = out["symbol"].duplicated().sum()
    if n_dupe_syms:
        # Two Entrez IDs sharing one symbol -- average the duplicate
        # columns rather than arbitrarily picking one.
        out = out.groupby("symbol", as_index=False)["essentiality_score"].mean()
    print(f"  {n_dupe_syms:,} duplicate gene symbols averaged: {before:,} -> {len(out):,}")
    return out


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)

    download(GTEX_URL, GTEX_RAW, "GTEx v8 median gene TPM (~6.9 MB)")
    print("\nparsing GTEx and computing tau ...")
    gtex = load_gtex_tau(GTEX_RAW)

    download(DEPMAP_URL, DEPMAP_RAW, "DepMap 24Q4 CRISPR gene effect (~429 MB)")
    print("\nparsing DepMap and computing mean gene effect ...")
    depmap = load_depmap_essentiality(DEPMAP_RAW)

    # Outer join: a gene missing from one source still keeps the feature
    # from the other. build_features.py is responsible for imputing
    # whichever column is NaN, same as every other feature source.
    out = gtex.merge(depmap, on="symbol", how="outer")

    print(f"\nSanity checks:")
    n_tau = out["tau"].notna().sum()
    n_ess = out["essentiality_score"].notna().sum()
    print(f"  genes with tau:                {n_tau:,} / {len(out):,}")
    print(f"  genes with essentiality_score: {n_ess:,} / {len(out):,}")

    tau_obs = out["tau"].dropna()
    assert tau_obs.between(0, 1).all(), \
        f"tau out of [0,1] -- check the tau computation. min={tau_obs.min()}, max={tau_obs.max()}"
    print(f"  tau range: [{tau_obs.min():.3f}, {tau_obs.max():.3f}]  median: {tau_obs.median():.3f}")

    ess_obs = out["essentiality_score"].dropna()
    print(f"  essentiality_score range: [{ess_obs.min():.3f}, {ess_obs.max():.3f}]  "
          f"median: {ess_obs.median():.3f}")

    n_dupes = out["symbol"].duplicated().sum()
    assert n_dupes == 0, f"FAIL: {n_dupes} duplicate gene symbols in output"

    out.to_parquet(OUT_FILE, index=False)
    print(f"\nwrote {OUT_FILE}")
    print(f"  rows: {len(out):,}  columns: {out.columns.tolist()}")


if __name__ == "__main__":
    build()
