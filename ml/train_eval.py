"""
Train a gradient-boosted tree inside the GroupKFold and evaluate with ranking
metrics appropriate for positive-unlabeled data (DESIGN.md sections 6.2, 6.3).

Why ranking metrics instead of accuracy or ROC-AUC:
  Positives are rare (~7.8%). A classifier that labels everything negative is
  92% accurate and useless. ROC-AUC is similarly inflated by the easy-to-classify
  true negatives. PR-AUC forces precision to stay high as recall grows -- the
  right penalty for a system where false positives waste expensive drug programs.
  Precision@k and enrichment factor measure what practitioners actually care about:
  are the top-k predictions enriched for real drug targets?

Why GradientBoostingClassifier:
  Interpretable (tree feature importances, SHAP-ready), handles mixed
  numeric/binary features without scaling, robust to the ~7.8% class imbalance
  (predict_proba is well-calibrated under subsample). A linear model would be
  the next check; tree beats it here because pLI and loeuf have non-linear
  thresholds that matter (e.g. pLI > 0.9 is a known constraint cliff).

Study-bias check (DESIGN.md 6.2):
  The gold standard is correlation of gene score with publication count
  (pub_count, from ml/fetch_publications.py / NCBI gene2pubmed -- now part
  of the v1 feature set, included deliberately as the confounder). We
  report the Spearman correlation between OOS score and log(pub_count + 1),
  and PR-AUC binned by publication-count tercile: if the model still ranks
  well among the least-published (understudied) genes, it is doing real
  biological work rather than just tracking gene fame.

  As a secondary, cheaper proxy: has_gnomad is also checked -- a gene
  absent from gnomAD (has_gnomad=0) is poorly characterised, so if such
  genes score low, the model may be tracking characterisation depth rather
  than true biology. We report the mean predicted score for has_gnomad=0
  vs has_gnomad=1.

Fold reconstruction:
  Reads ml/cache/cv_folds.parquet written by split.py. This guarantees that
  training and evaluation use the IDENTICAL family-safe split without
  re-running the splitter (deterministic GroupKFold output matches because the
  same groups are presented in the same order).

Output:
  ml/cache/oos_predictions.parquet -- symbol, score, label, rank, fold_idx
  Printed per-fold and mean metrics to stdout.

Run:  python3 ml/train_eval.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score

RANDOM_SEED = 42

CACHE_DIR   = os.environ.get("ML_CACHE_DIR", "ml/cache")
TABLE_FILE  = os.path.join(CACHE_DIR, "training_table.parquet")
FOLDS_FILE  = os.path.join(CACHE_DIR, "cv_folds.parquet")
OOS_FILE    = os.path.join(CACHE_DIR, "oos_predictions.parquet")

FEATURE_COLS = [
    "pLI", "loeuf", "oe_lof", "oe_mis", "n_rare", "n_lof",
    "protein_length", "ppi_degree", "ppi_betweenness",
    "tau", "essentiality_score",
    "pub_count", "year_first_described",
    "has_gnomad", "has_burden", "has_alphafold", "has_string",
    "has_tau", "has_essentiality", "has_pub_count", "has_year_described",
]

# Ranking thresholds -- k absolute and percent-of-test for enrichment factor.
K_VALUES   = [100, 500]
EF_PERCENTS = [1, 2, 5]

MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=RANDOM_SEED,
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(y_true, scores, k):
    """Fraction of positives in the top-k ranked genes."""
    top_k = np.argsort(scores)[::-1][:k]
    return y_true[top_k].mean()


def enrichment_factor(y_true, scores, pct):
    """
    EF at top-pct% = (precision in top fraction) / (baseline positive rate).
    A random ranker has EF=1. Higher is better.
    """
    n = len(y_true)
    k = max(1, int(round(n * pct / 100)))
    top_k = np.argsort(scores)[::-1][:k]
    hit_rate = y_true[top_k].mean()
    baseline = y_true.mean()
    return hit_rate / baseline if baseline > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load inputs ───────────────────────────────────────────────────────────
    for path, label in [(TABLE_FILE, "training_table"), (FOLDS_FILE, "cv_folds")]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run the prerequisite scripts first.", file=sys.stderr)
            sys.exit(1)

    df    = pd.read_parquet(TABLE_FILE)
    folds = pd.read_parquet(FOLDS_FILE)

    # Merge fold assignments into training table on symbol (unique key).
    df = df.merge(folds[["symbol", "fold_idx"]], on="symbol", how="left")
    assert df["fold_idx"].notna().all(), "some genes have no fold assignment after merge"

    n_folds = int(df["fold_idx"].max()) + 1
    print(f"loaded {len(df):,} genes, {df['label'].sum():,} positives, {n_folds} folds")
    print(f"baseline positive rate: {df['label'].mean():.3%}  "
          f"(random ranker PR-AUC ~ {df['label'].mean():.3f})\n")

    X      = df[FEATURE_COLS].values
    y      = df["label"].values
    fold_v = df["fold_idx"].values.astype(int)

    # ── Per-fold evaluation ───────────────────────────────────────────────────
    col_w = 10  # column width for numeric fields

    prec_k_header = "  ".join(f"P@{k:,}" for k in K_VALUES)
    ef_header     = "  ".join(f"EF@{p}%" for p in EF_PERCENTS)
    header = (
        f"{'Fold':>4}  {'n_test':>6}  {'n_pos':>5}  {'PR-AUC':>{col_w}}  "
        + prec_k_header + "  " + ef_header
    )
    print(header)
    print("-" * len(header))

    fold_results = []
    oos_rows     = []

    for fold in range(n_folds):
        train_mask = fold_v != fold
        test_mask  = fold_v == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Confirm family-safe split: no group_key from test appears in train.
        # This duplicates the assertion in split.py and serves as a regression guard.
        train_keys = set(df.loc[train_mask, "group_key"])
        test_keys  = set(df.loc[test_mask,  "group_key"])
        overlap    = train_keys & test_keys
        if overlap:
            raise AssertionError(
                f"FATAL LEAKAGE in fold {fold}: {len(overlap)} group_keys appear "
                f"in both train and test. cv_folds.parquet may be stale -- re-run split.py."
            )

        clf = GradientBoostingClassifier(**MODEL_PARAMS)
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, scores)
        pk     = [precision_at_k(y_test, scores, k) for k in K_VALUES]
        ef     = [enrichment_factor(y_test, scores, p) for p in EF_PERCENTS]

        fold_results.append({"fold": fold, "pr_auc": pr_auc, "pk": pk, "ef": ef})

        pk_str = "  ".join(f"{v:{col_w}.3f}" for v in pk)
        ef_str = "  ".join(f"{v:{col_w}.2f}" for v in ef)
        print(
            f"{fold:>4}  {len(y_test):>6,}  {int(y_test.sum()):>5}  "
            f"{pr_auc:{col_w}.4f}  {pk_str}  {ef_str}"
        )

        # Accumulate OOS predictions.
        test_symbols = df.loc[test_mask, "symbol"].values
        oos_rows.append(pd.DataFrame({
            "symbol":   test_symbols,
            "score":    scores,
            "label":    y_test,
            "fold_idx": fold,
        }))

    print("-" * len(header))

    # ── Mean across folds ─────────────────────────────────────────────────────
    mean_pr  = np.mean([r["pr_auc"] for r in fold_results])
    mean_pk  = [np.mean([r["pk"][i] for r in fold_results]) for i in range(len(K_VALUES))]
    mean_ef  = [np.mean([r["ef"][i] for r in fold_results]) for i in range(len(EF_PERCENTS))]

    pk_str = "  ".join(f"{v:{col_w}.3f}" for v in mean_pk)
    ef_str = "  ".join(f"{v:{col_w}.2f}" for v in mean_ef)
    print(
        f"{'mean':>4}  {'':>6}  {'':>5}  "
        f"{mean_pr:{col_w}.4f}  {pk_str}  {ef_str}"
    )

    # ── OOS predictions: rank globally and save ───────────────────────────────
    oos = pd.concat(oos_rows, ignore_index=True)
    oos["rank"] = oos["score"].rank(ascending=False, method="first").astype(int)
    oos = oos.sort_values("rank").reset_index(drop=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    oos.to_parquet(OOS_FILE, index=False)
    print(f"\nwrote {OOS_FILE}  ({len(oos):,} rows)")

    # ── Top predicted targets ─────────────────────────────────────────────────
    print("\nTop 10 predicted targets (OOS scores, ranked globally):")
    top10 = oos.head(10)[["rank", "symbol", "score", "label", "fold_idx"]]
    top10_str = top10.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    print(top10_str)

    # ── Feature importances ───────────────────────────────────────────────────
    # Train once on all data (no leakage risk: importances are structural,
    # not used for evaluation). Reported as a study-bias diagnostic only.
    clf_full = GradientBoostingClassifier(**MODEL_PARAMS)
    clf_full.fit(X, y)
    importances = sorted(
        zip(FEATURE_COLS, clf_full.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )
    print("\nFeature importances (full-data model, for study-bias diagnostic):")
    for feat, imp in importances:
        bar = "#" * int(imp * 60)
        print(f"  {feat:<14}  {imp:.4f}  {bar}")

    # ── Study-bias check ──────────────────────────────────────────────────────
    print("\nSTUDY-BIAS CHECK (DESIGN.md section 6.2)")

    # Proxy: has_gnomad=0 genes are poorly characterised in exome databases.
    # If they score systematically low, the model may be rewarding characterisation
    # depth rather than true biology.
    mean_score_gnomad1 = oos.loc[
        df.set_index("symbol").loc[oos["symbol"], "has_gnomad"].values == 1, "score"
    ].mean()
    mean_score_gnomad0 = oos.loc[
        df.set_index("symbol").loc[oos["symbol"], "has_gnomad"].values == 0, "score"
    ].mean()
    print(
        f"  Mean OOS score, genes WITH gnomAD (has_gnomad=1):    {mean_score_gnomad1:.4f}"
    )
    print(
        f"  Mean OOS score, genes WITHOUT gnomAD (has_gnomad=0): {mean_score_gnomad0:.4f}"
    )
    ratio = mean_score_gnomad1 / mean_score_gnomad0 if mean_score_gnomad0 > 0 else float("inf")
    print(f"  Score ratio (gnomad=1 / gnomad=0):                   {ratio:.2f}x")
    if ratio > 2.0:
        print("  [WARN] Characterisation-depth bias suspected: gnomAD genes score >2x higher.")
        print("         See the publication-count check below for the primary signal.")
    else:
        print("  [OK] Score ratio is below 2x -- no strong characterisation-depth signal.")

    # ── Study-bias check: publication count (DESIGN.md 6.2, the killer result) ─
    # This is the primary defense against the "model just learned which genes
    # are famous" failure mode. pub_count/year_first_described are deliberate
    # confounder features (ml/fetch_publications.py) -- included specifically
    # so we can check whether the model is riding them.
    print("\nSTUDY-BIAS CHECK: publication count (DESIGN.md section 6.2)")

    pub_count = df.set_index("symbol").loc[oos["symbol"], "pub_count"].values
    rho, pval = spearmanr(oos["score"].values, np.log1p(pub_count))
    print(f"  Spearman rho(score, log(pub_count + 1)): {rho:.3f}  (p={pval:.2e})")
    if abs(rho) > 0.5:
        print("  [WARN] |rho| > 0.5 -- the model may be primarily ranking on gene fame.")
    else:
        print("  [OK] |rho| <= 0.5 -- publication count is not the dominant ranking signal.")

    # Bin genes by publication-count tercile and compare PR-AUC across bins.
    # Rank-based (not value-based) quantile cuts because pub_count is heavily
    # right-skewed with many ties at low counts -- ranking guarantees three
    # non-empty, roughly equal-sized bins regardless of that skew.
    pub_rank = pd.Series(pub_count).rank(method="first")
    pub_tercile = pd.qcut(pub_rank, q=3, labels=["low", "medium", "high"])

    print("\n  PR-AUC by publication-count tercile "
          "(comparable PR-AUC across bins means real signal, not just fame):")
    print(f"  {'tercile':>8}  {'n':>6}  {'n_pos':>6}  {'PR-AUC':>8}  {'mean pub_count':>15}")
    for tercile in ["low", "medium", "high"]:
        mask = (pub_tercile == tercile).values
        n_pos_bin = int(oos.loc[mask, "label"].sum())
        if n_pos_bin == 0:
            print(f"  {tercile:>8}  {mask.sum():>6,}  {n_pos_bin:>6}  {'n/a':>8}  "
                  f"{pub_count[mask].mean():>15.1f}  (no positives in bin)")
            continue
        bin_pr = average_precision_score(oos.loc[mask, "label"], oos.loc[mask, "score"])
        print(f"  {tercile:>8}  {mask.sum():>6,}  {n_pos_bin:>6}  {bin_pr:>8.4f}  "
              f"{pub_count[mask].mean():>15.1f}")

    # ── Structural-validation caveat ──────────────────────────────────────────
    print()
    print("=" * 65)
    print("STRUCTURAL-VALIDATION CAVEAT")
    print("=" * 65)
    print(
        "Burden features (n_rare, n_lof) cover chr22 only (388 / 19,296\n"
        "genes, 2.0%). For this run the model is effectively trained on\n"
        "gnomAD constraint features (pLI, loeuf, oe_lof, oe_mis) plus\n"
        "flag columns. PR-AUC and enrichment factor reflect the gnomAD\n"
        "signal only. When burden is extended genome-wide, re-run this\n"
        "script: the ranking will incorporate rare-variant evidence for\n"
        "all genes and metrics are expected to improve."
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
