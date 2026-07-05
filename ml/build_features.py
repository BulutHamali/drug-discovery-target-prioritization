"""
Assemble the training table from all feature and label sources.

Run AFTER:
  python3 ml/gene_families.py    -- produces ml/cache/gene_families.parquet
  python3 ml/fetch_gnomad.py     -- produces ml/cache/gnomad_constraint.parquet
  nextflow run pipeline/main.nf  -- produces results/gene_burden_features.parquet

Output: ml/cache/training_table.parquet

Join strategy (all LEFT joins from the gene universe):
  universe -> label:   Ensembl gene ID  (OT uses ENSG IDs as targetId)
  universe -> gnomAD:  Ensembl gene ID, symbol fallback for HGNC genes that
                       lack an Ensembl mapping
  universe -> burden:  gene symbol  (what the Nextflow COLLECT step outputs)

Missing-feature handling (both explicit and documented):

  burden (n_rare, n_lof):
    Fill with 0. Set has_burden=0 (flag; 1 means chr22 data is present).
    WHY 0: a gene not on chr22 genuinely has zero observed burden from our
    pipeline. This is accurate for the structural validation run, not an
    approximation. When burden is extended genome-wide, has_burden will be
    1 for every gene. The flag lets the ML layer and study-bias analysis
    distinguish chr22 genes from the rest.

  gnomAD (pLI, loeuf, oe_lof, oe_mis):
    Fill with column median computed on observed genes. Set has_gnomad=0.
    WHY median: the ~2.6% of genes missing gnomAD scores are typically
    poorly characterised with very few exome observations. Filling with 0
    would incorrectly imply high LoF tolerance (pLI=0 = fully tolerant).
    Filling with 1 for pLI or 0 for loeuf would imply high constraint.
    Median says "this gene is typical" -- neutral and honest for a
    structural run. These genes are NOT dropped: they are part of the
    protein-coding universe and some are positives.

Label:
  Binary: max(phase) >= 1 per gene.
  Continuous: max(phase) (for regression framing).
  Genes absent from knownDrugsAggregated get label=0 under the open-world
  (positive-unlabeled) assumption from DESIGN.md section 4: absence of a
  drug does not mean undruggable; it may mean understudied.
"""

import glob
import os
import sys

import pandas as pd

RANDOM_SEED = 42

CACHE_DIR   = os.environ.get("ML_CACHE_DIR", "ml/cache")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DATA_DIR    = os.environ.get("DATA_CACHE_DIR", "data/cache")

FAMILIES_FILE = os.path.join(CACHE_DIR, "gene_families.parquet")
GNOMAD_FILE   = os.path.join(CACHE_DIR, "gnomad_constraint.parquet")
BURDEN_FILE   = os.path.join(RESULTS_DIR, "gene_burden_features.parquet")
OUT_FILE      = os.path.join(CACHE_DIR, "training_table.parquet")

GNOMAD_FEAT_COLS = ["pLI", "loeuf", "oe_lof", "oe_mis"]


def find_ot_drug_files():
    """
    Locate knownDrugsAggregated Parquet files for whichever OT release is
    locally cached. Uses the most recent release if multiple are present.
    DESIGN.md targets 24.12; 24.09 is also accepted (same schema).
    """
    dirs = sorted(glob.glob(
        os.path.join(DATA_DIR, "open_targets", "*", "knownDrugsAggregated")
    ))
    if not dirs:
        return [], None
    chosen = dirs[-1]
    release = os.path.basename(os.path.dirname(chosen))
    files = glob.glob(os.path.join(chosen, "*.parquet"))
    return files, release


def check_inputs():
    errors = []
    for path, script in [
        (FAMILIES_FILE, "ml/gene_families.py"),
        (GNOMAD_FILE,   "ml/fetch_gnomad.py"),
        (BURDEN_FILE,   "nextflow run pipeline/main.nf -profile local"),
    ]:
        if not os.path.exists(path):
            errors.append(f"  missing: {path}  ->  run: {script}")

    drug_files, release = find_ot_drug_files()
    if not drug_files:
        errors.append(
            f"  missing: data/cache/open_targets/<release>/knownDrugsAggregated/"
            f"  ->  run: python3 data/fetch_chembl_known_drugs.py"
        )

    if errors:
        print("ERROR: prerequisite files are missing. Run these first:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    return drug_files, release


def load_label(drug_files, release):
    """
    One row per gene (Ensembl ID). label=1 if max(phase) >= 1, else 0.
    The phase=0.5 rows in OT represent preclinical nominations; they do not
    count as positives under the chosen label definition.
    """
    df = pd.concat(
        [pd.read_parquet(p, columns=["targetId", "phase"]) for p in drug_files],
        ignore_index=True,
    )
    per_gene = (
        df.groupby("targetId", sort=False)["phase"]
        .max()
        .reset_index()
        .rename(columns={"targetId": "ensembl_id", "phase": "max_phase"})
    )
    per_gene["label"] = (per_gene["max_phase"] >= 1).astype(int)
    return per_gene


def load_gnomad():
    df = pd.read_parquet(GNOMAD_FILE)
    # Keep feature columns plus the two join keys.
    return df[["gene_id", "gene"] + GNOMAD_FEAT_COLS].copy()


def fill_gnomad_missing(df):
    """
    Compute medians on observed (non-imputed) rows, then fill NaN.
    has_gnomad=1 means real data; has_gnomad=0 means median-imputed.
    The flag column is retained in the training table so the model can
    optionally use it and the analysis can check if missing-data genes
    drive any results.
    """
    df["has_gnomad"] = df["pLI"].notna().astype(int)
    medians = {}
    for col in GNOMAD_FEAT_COLS:
        med = df.loc[df["has_gnomad"] == 1, col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)
    return df, medians


def build():
    drug_files, ot_release = check_inputs()

    # ── Gene universe ─────────────────────────────────────────────────────────
    universe = pd.read_parquet(FAMILIES_FILE)
    n_universe = len(universe)
    print(f"gene universe (protein-coding, HGNC): {n_universe:,} genes")

    # ── Label ─────────────────────────────────────────────────────────────────
    print(f"\nloading label from OT release {ot_release} ...")
    label = load_label(drug_files, ot_release)
    n_ot_pos = label["label"].sum()
    print(f"  {len(label):,} genes with any OT drug record")
    print(f"  positives (max_phase >= 1) in OT: {n_ot_pos:,}")

    # ── gnomAD constraint ─────────────────────────────────────────────────────
    print("\nloading gnomAD constraint ...")
    gnomad = load_gnomad()

    # ── Burden features ───────────────────────────────────────────────────────
    print("loading burden features ...")
    burden = (
        pd.read_parquet(BURDEN_FILE, columns=["gene", "n_rare", "n_lof"])
        .rename(columns={"gene": "symbol"})
    )
    print(f"  {len(burden):,} genes with chr22 burden data")

    # ── Join 1: label onto universe (Ensembl ID) ──────────────────────────────
    df = universe.merge(label, on="ensembl_id", how="left")
    # Open-world: unlabeled genes are treated as negatives.
    df["label"]     = df["label"].fillna(0).astype(int)
    df["max_phase"] = df["max_phase"].fillna(0.0)

    # ── Join 2: gnomAD onto universe (Ensembl ID, symbol fallback) ────────────
    # Primary join on Ensembl ID -- stable and unambiguous.
    gnomad_for_merge = gnomad.rename(columns={"gene_id": "ensembl_id"})[
        ["ensembl_id"] + GNOMAD_FEAT_COLS
    ]
    df = df.merge(gnomad_for_merge, on="ensembl_id", how="left")

    # Fallback: for HGNC genes that have no Ensembl ID mapping (~43 genes),
    # try symbol. Deduplicate the lookup table first because a gene symbol can
    # occasionally appear under two Ensembl IDs in gnomAD (retired/merged IDs);
    # set_index on a non-unique index raises InvalidIndexError in pandas.
    missing_gnomad = df["pLI"].isna()
    if missing_gnomad.any():
        gnomad_by_sym = (
            gnomad.sort_values("pLI", ascending=False, na_position="last")
            .drop_duplicates(subset=["gene"])
            .set_index("gene")[GNOMAD_FEAT_COLS]
        )
        for col in GNOMAD_FEAT_COLS:
            df.loc[missing_gnomad, col] = (
                df.loc[missing_gnomad, "symbol"].map(gnomad_by_sym[col])
            )

    df, gnomad_medians = fill_gnomad_missing(df)

    # ── Join 3: burden onto universe (gene symbol) ────────────────────────────
    df = df.merge(burden, on="symbol", how="left")
    df["has_burden"] = df["n_rare"].notna().astype(int)
    df["n_rare"] = df["n_rare"].fillna(0).astype(int)
    df["n_lof"]  = df["n_lof"].fillna(0).astype(int)

    # ── Required checks ───────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("REQUIRED CHECKS")
    print("=" * 56)

    # Check 1: row count must equal the gene universe exactly.
    # A join explosion means one of the source tables has duplicate keys.
    # A row loss means the join silently dropped genes (impossible with LEFT join,
    # but assert anyway to catch future regressions).
    assert len(df) == n_universe, (
        f"FAIL row count: {len(df)} rows, expected {n_universe}. "
        "Check for duplicate keys in gnomAD or burden source files."
    )
    print(f"[PASS] row count: {len(df):,} == universe size ({n_universe:,})")

    # Check 2: positive rate approximately 7.5%.
    pos_rate = df["label"].mean()
    n_pos = df["label"].sum()
    status = "PASS" if 0.05 <= pos_rate <= 0.12 else "WARN"
    print(f"[{status}] positive rate: {n_pos:,} / {len(df):,} = {pos_rate:.2%}  (expected ~7.5%)")
    if status == "WARN":
        print("       Outside 5-12% band. Check that label join used Ensembl ID correctly.")

    # Check 3: burden coverage (expected: most genes missing, chr22-only).
    n_with_burden = df["has_burden"].sum()
    n_no_burden   = len(df) - n_with_burden
    print(f"[INFO] burden coverage: {n_with_burden:,} genes have chr22 data "
          f"({n_with_burden/len(df):.1%} of universe)")
    print(f"       {n_no_burden:,} genes have no chr22 burden -- filled n_rare=0, n_lof=0, has_burden=0")
    print(f"       (this is expected for the structural validation run)")

    # Check 4: gnomAD coverage.
    n_with_gnomad = df["has_gnomad"].sum()
    n_no_gnomad   = len(df) - n_with_gnomad
    print(f"[INFO] gnomAD coverage: {n_with_gnomad:,} genes have real scores "
          f"({n_with_gnomad/len(df):.1%} of universe)")
    print(f"       {n_no_gnomad:,} genes median-imputed: "
          f"{ {k: round(v, 3) for k, v in gnomad_medians.items()} }")

    # Check 5: no gene appears more than once.
    n_dupes = df["symbol"].duplicated().sum()
    assert n_dupes == 0, f"FAIL: {n_dupes} duplicate gene symbols after join"
    print(f"[PASS] no duplicate gene symbols")

    # ── Write output ──────────────────────────────────────────────────────────
    final_cols = [
        "symbol", "ensembl_id", "group_key",
        "label", "max_phase",
        "pLI", "loeuf", "oe_lof", "oe_mis",
        "n_rare", "n_lof",
        "has_gnomad", "has_burden",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    print()
    print(f"wrote {OUT_FILE}")
    print(f"  rows: {len(df):,}   columns: {df.columns.tolist()}")


if __name__ == "__main__":
    build()
