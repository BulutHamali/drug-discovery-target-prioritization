"""
Assemble the training table from all feature and label sources.

Run AFTER:
  python3 ml/gene_families.py    -- produces ml/cache/gene_families.parquet
  python3 ml/fetch_gnomad.py     -- produces ml/cache/gnomad_constraint.parquet
  python3 ml/fetch_alphafold.py  -- produces ml/cache/alphafold_features.parquet
  python3 ml/fetch_string.py     -- produces ml/cache/string_features.parquet
  python3 ml/fetch_expression.py  -- produces ml/cache/expression_features.parquet
  python3 ml/fetch_publications.py -- produces ml/cache/publication_features.parquet
  nextflow run pipeline/main.nf, once per chromosome (chr22 first as a smoke
  test, then chr1/chr2/chr17/chr19, then the remaining 17 autosomes
  concurrently via --chroms), plus the merge step that produces
  results/gene_burden_features_22chrom.parquet -- all 22 autosomes, X/Y
  out of scope. See the Batch run notes for how these were merged, including
  the one duplicate-symbol fix (CKS1B: kept the chr1 entry matching its real
  HGNC location 1q21.3, dropped a spurious chr5 entry that was almost
  certainly a mislabeled processed pseudogene in the GRCh37 r87 annotation).

Output: ml/cache/training_table.parquet

Join strategy (all LEFT joins from the gene universe):
  universe -> label:       Ensembl gene ID  (OT uses ENSG IDs as targetId)
  universe -> gnomAD:      Ensembl gene ID, symbol fallback for HGNC genes that
                          lack an Ensembl mapping
  universe -> burden:      gene symbol  (what the Nextflow COLLECT step outputs)
  universe -> alphafold:   gene symbol  (UniProt gene_names field)
  universe -> STRING:      gene symbol  (STRING preferred_name field)
  universe -> expression:  gene symbol  (GTEx Description field / DepMap column header)
  universe -> publication: gene symbol  (via Entrez GeneID -> HGNC symbol mapping)

Missing-feature handling (both explicit and documented):

  burden (n_rare, n_lof):
    Fill with 0. Set has_burden=0 (flag; 1 means real burden data from one
    of the 22 processed autosomes is present, currently 86.68% of the
    universe).
    WHY 0: a gene with no burden row genuinely has zero observed rare-variant
    burden from our pipeline, either because it sits on X/Y (never
    processed, out of scope) or because it had no qualifying rare variant
    in this call set even though its autosome was processed. Both are
    honest zeros, not approximations. The flag lets the ML layer and
    study-bias analysis distinguish genes with real burden evidence from
    the rest.

  gnomAD (pLI, loeuf, oe_lof, oe_mis):
    Fill with column median computed on observed genes. Set has_gnomad=0.
    WHY median: the ~2.6% of genes missing gnomAD scores are typically
    poorly characterised with very few exome observations. Filling with 0
    would incorrectly imply high LoF tolerance (pLI=0 = fully tolerant).
    Filling with 1 for pLI or 0 for loeuf would imply high constraint.
    Median says "this gene is typical" -- neutral and honest for a
    structural run. These genes are NOT dropped: they are part of the
    protein-coding universe and some are positives.

  AlphaFold/UniProt (protein_length):
    Fill with column median computed on observed genes. Set has_alphafold=0.
    WHY median: same reasoning as gnomAD -- protein_length=0 is not a
    meaningful value for a gene, so median says "this gene is a typical
    length" rather than implying anything about size.

  STRING (ppi_degree, ppi_betweenness):
    Fill with 0. Set has_string=0 (flag; 1 means the gene has at least one
    high-confidence STRING interaction, combined_score >= 700).
    WHY 0: a gene absent from the high-confidence PPI graph genuinely has
    zero measured high-confidence interactions -- the same "real zero"
    reasoning as burden, not an approximation.

  Expression (tau, essentiality_score):
    Fill each with its own column median computed on observed genes.
    Set has_tau=0 / has_essentiality=0 independently, since tau (GTEx) and
    essentiality_score (DepMap) come from two different sources and a gene
    can be missing from one but not the other.
    WHY median: same reasoning as gnomAD/AlphaFold -- tau=0 would falsely
    imply a perfectly ubiquitous housekeeping gene, and essentiality_score=0
    would falsely imply a specific (dispensable) essentiality reading,
    for genes we simply have no measurement for.

  Publication metadata (pub_count, year_first_described):
    pub_count: fill with 0. Set has_pub_count=0.
    WHY 0: a gene absent from gene2pubmed genuinely has zero curated
    publication associations -- the same "real zero" reasoning as burden,
    not an approximation.
    year_first_described: fill with column median. Set has_year_described=0.
    WHY median: a gene with no publications has no "first described" year
    at all, and 0 is not a meaningful year -- median says "this gene's
    literature history is typical" for the genes we have no record for.
    This is the confounder feature (DESIGN.md section 5): included on
    purpose so the study-bias check in train_eval.py can test whether the
    model is ranking on gene fame rather than biology.

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

FAMILIES_FILE   = os.path.join(CACHE_DIR, "gene_families.parquet")
GNOMAD_FILE     = os.path.join(CACHE_DIR, "gnomad_constraint.parquet")
ALPHAFOLD_FILE  = os.path.join(CACHE_DIR, "alphafold_features.parquet")
STRING_FILE     = os.path.join(CACHE_DIR, "string_features.parquet")
EXPRESSION_FILE  = os.path.join(CACHE_DIR, "expression_features.parquet")
PUBLICATION_FILE = os.path.join(CACHE_DIR, "publication_features.parquet")
BURDEN_FILE      = os.path.join(RESULTS_DIR, "gene_burden_features_22chrom.parquet")
OUT_FILE         = os.path.join(CACHE_DIR, "training_table.parquet")

GNOMAD_FEAT_COLS     = ["pLI", "loeuf", "oe_lof", "oe_mis"]
ALPHAFOLD_FEAT_COLS  = ["protein_length"]
STRING_FEAT_COLS     = ["ppi_degree", "ppi_betweenness"]
EXPRESSION_FEAT_COLS = ["tau", "essentiality_score"]
PUBLICATION_FEAT_COLS = ["pub_count", "year_first_described"]


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
        (FAMILIES_FILE,  "ml/gene_families.py"),
        (GNOMAD_FILE,    "ml/fetch_gnomad.py"),
        (ALPHAFOLD_FILE, "ml/fetch_alphafold.py"),
        (STRING_FILE,    "ml/fetch_string.py"),
        (EXPRESSION_FILE, "ml/fetch_expression.py"),
        (PUBLICATION_FILE, "ml/fetch_publications.py"),
        (BURDEN_FILE,    "nextflow run pipeline/main.nf per chromosome (all 22 autosomes), then merge"),
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


def load_alphafold():
    df = pd.read_parquet(ALPHAFOLD_FILE)
    return df[["symbol"] + ALPHAFOLD_FEAT_COLS].copy()


def load_string():
    df = pd.read_parquet(STRING_FILE)
    return df[["symbol"] + STRING_FEAT_COLS].copy()


def load_expression():
    df = pd.read_parquet(EXPRESSION_FILE)
    return df[["symbol"] + EXPRESSION_FEAT_COLS].copy()


def load_publications():
    df = pd.read_parquet(PUBLICATION_FILE)
    return df[["symbol"] + PUBLICATION_FEAT_COLS].copy()


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


def fill_alphafold_missing(df):
    """
    Same median-fill approach as fill_gnomad_missing: has_alphafold=1 means
    real UniProt data; has_alphafold=0 means median-imputed protein_length.
    """
    df["has_alphafold"] = df["protein_length"].notna().astype(int)
    medians = {}
    for col in ALPHAFOLD_FEAT_COLS:
        med = df.loc[df["has_alphafold"] == 1, col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)
    return df, medians


EXPRESSION_FLAG_COLS = {"tau": "has_tau", "essentiality_score": "has_essentiality"}


def fill_expression_missing(df):
    """
    tau (GTEx) and essentiality_score (DepMap) are independently missing
    since they come from two different source files merged in
    fetch_expression.py -- each gets its own has_* flag and its own
    median, rather than sharing one flag like gnomAD's four columns do.
    """
    medians = {}
    for col in EXPRESSION_FEAT_COLS:
        flag = EXPRESSION_FLAG_COLS[col]
        df[flag] = df[col].notna().astype(int)
        med = df.loc[df[flag] == 1, col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)
    return df, medians


def fill_publication_missing(df):
    """
    pub_count: real zero for genes absent from gene2pubmed (burden-style).
    year_first_described: median-fill, no genuine zero value exists
    (gnomAD/AlphaFold-style).
    """
    df["has_pub_count"] = df["pub_count"].notna().astype(int)
    df["pub_count"] = df["pub_count"].fillna(0).astype(int)

    df["has_year_described"] = df["year_first_described"].notna().astype(int)
    year_median = df.loc[df["has_year_described"] == 1, "year_first_described"].median()
    df["year_first_described"] = df["year_first_described"].fillna(year_median)

    return df, {"year_first_described": year_median}


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

    # ── AlphaFold/UniProt protein length ─────────────────────────────────────
    print("loading AlphaFold/UniProt protein features ...")
    alphafold = load_alphafold()
    print(f"  {len(alphafold):,} genes with protein_length")

    # ── STRING PPI features ──────────────────────────────────────────────────
    print("loading STRING PPI features ...")
    string_feats = load_string()
    print(f"  {len(string_feats):,} genes with PPI degree/betweenness")

    # ── Expression / essentiality features ───────────────────────────────────
    print("loading GTEx/DepMap expression features ...")
    expression = load_expression()
    print(f"  {len(expression):,} genes with tau and/or essentiality_score")

    # ── Publication metadata (the deliberate confounder) ─────────────────────
    print("loading publication metadata ...")
    publications = load_publications()
    print(f"  {len(publications):,} genes with pub_count/year_first_described")

    # ── Burden features ───────────────────────────────────────────────────────
    print("loading burden features ...")
    burden = (
        pd.read_parquet(BURDEN_FILE, columns=["gene", "n_rare", "n_lof"])
        .rename(columns={"gene": "symbol"})
    )
    print(f"  {len(burden):,} rows in the merged 22-autosome burden table "
          f"(all autosomes, X/Y out of scope; includes non-protein-coding "
          f"symbols that will not match the universe join below)")

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

    # ── Join 4: AlphaFold/UniProt protein length onto universe (gene symbol) ──
    df = df.merge(alphafold, on="symbol", how="left")
    df, alphafold_medians = fill_alphafold_missing(df)

    # ── Join 5: STRING PPI features onto universe (gene symbol) ──────────────
    df = df.merge(string_feats, on="symbol", how="left")
    df["has_string"] = df["ppi_degree"].notna().astype(int)
    df["ppi_degree"]      = df["ppi_degree"].fillna(0).astype(int)
    df["ppi_betweenness"] = df["ppi_betweenness"].fillna(0.0)

    # ── Join 6: expression/essentiality onto universe (gene symbol) ──────────
    df = df.merge(expression, on="symbol", how="left")
    df, expression_medians = fill_expression_missing(df)

    # ── Join 7: publication metadata onto universe (gene symbol) ─────────────
    df = df.merge(publications, on="symbol", how="left")
    df, publication_medians = fill_publication_missing(df)

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

    # Check 3: burden coverage (all 22 autosomes processed; X/Y out of scope).
    n_with_burden = df["has_burden"].sum()
    n_no_burden   = len(df) - n_with_burden
    print(f"[INFO] burden coverage: {n_with_burden:,} genes have real burden data "
          f"from all 22 autosomes ({n_with_burden/len(df):.1%} of universe)")
    print(f"       {n_no_burden:,} genes have no burden data -- either on X/Y "
          f"(out of scope) or no qualifying rare variant in this call set -- "
          f"filled n_rare=0, n_lof=0, has_burden=0")

    # Check 4: gnomAD coverage.
    n_with_gnomad = df["has_gnomad"].sum()
    n_no_gnomad   = len(df) - n_with_gnomad
    print(f"[INFO] gnomAD coverage: {n_with_gnomad:,} genes have real scores "
          f"({n_with_gnomad/len(df):.1%} of universe)")
    print(f"       {n_no_gnomad:,} genes median-imputed: "
          f"{ {k: round(v, 3) for k, v in gnomad_medians.items()} }")

    # Check 4b: AlphaFold/UniProt coverage.
    n_with_af = df["has_alphafold"].sum()
    n_no_af   = len(df) - n_with_af
    print(f"[INFO] AlphaFold/UniProt coverage: {n_with_af:,} genes have real protein_length "
          f"({n_with_af/len(df):.1%} of universe)")
    print(f"       {n_no_af:,} genes median-imputed: "
          f"{ {k: round(v, 3) for k, v in alphafold_medians.items()} }")

    # Check 4c: STRING coverage.
    n_with_string = df["has_string"].sum()
    n_no_string   = len(df) - n_with_string
    print(f"[INFO] STRING coverage: {n_with_string:,} genes have high-confidence PPI data "
          f"({n_with_string/len(df):.1%} of universe)")
    print(f"       {n_no_string:,} genes have no high-confidence STRING edges -- "
          f"filled ppi_degree=0, ppi_betweenness=0, has_string=0")

    # Check 4d: expression/essentiality coverage.
    n_with_tau = df["has_tau"].sum()
    n_with_ess = df["has_essentiality"].sum()
    print(f"[INFO] GTEx coverage: {n_with_tau:,} genes have real tau "
          f"({n_with_tau/len(df):.1%} of universe)")
    print(f"[INFO] DepMap coverage: {n_with_ess:,} genes have real essentiality_score "
          f"({n_with_ess/len(df):.1%} of universe)")
    print(f"       median-imputed: "
          f"{ {k: round(v, 3) for k, v in expression_medians.items()} }")

    # Check 4e: publication metadata coverage (the deliberate confounder).
    n_with_pub = df["has_pub_count"].sum()
    n_no_pub   = len(df) - n_with_pub
    print(f"[INFO] publication coverage: {n_with_pub:,} genes have real pub_count "
          f"({n_with_pub/len(df):.1%} of universe)")
    print(f"       {n_no_pub:,} genes have no gene2pubmed record -- filled pub_count=0, "
          f"has_pub_count=0")
    print(f"       year_first_described median-imputed: "
          f"{ {k: round(v, 1) for k, v in publication_medians.items()} }")

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
        "protein_length",
        "ppi_degree", "ppi_betweenness",
        "tau", "essentiality_score",
        "pub_count", "year_first_described",
        "has_gnomad", "has_burden", "has_alphafold", "has_string",
        "has_tau", "has_essentiality", "has_pub_count", "has_year_described",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    print()
    print(f"wrote {OUT_FILE}")
    print(f"  rows: {len(df):,}   columns: {df.columns.tolist()}")


if __name__ == "__main__":
    build()
