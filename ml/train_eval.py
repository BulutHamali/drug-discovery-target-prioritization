"""
Train a gradient-boosted tree inside the GroupKFold and evaluate with ranking
metrics appropriate for positive-unlabeled data (DESIGN.md sections 6.2, 6.3).

Why ranking metrics instead of accuracy or ROC-AUC:
  Positives are rare (~7.8%). A classifier that labels everything negative is
  92% accurate and useless. ROC-AUC is similarly inflated by the easy-to-classify
  true negatives. PR-AUC forces precision to stay high as recall grows, the
  right penalty for a system where false positives waste expensive drug programs.
  Precision@k and enrichment factor measure what practitioners actually care about:
  are the top-k predictions enriched for real drug targets?

Why GradientBoostingClassifier:
  Interpretable (tree feature importances, SHAP-ready), handles mixed
  numeric/binary features without scaling, robust to the ~7.8% class imbalance
  (predict_proba is well-calibrated under subsample). A linear model would be
  the next check; tree beats it here because pLI and loeuf have non-linear
  thresholds that matter (e.g. pLI > 0.9 is a known constraint cliff).

Study-bias check (DESIGN.md 6.2, "the killer result"):
  pub_count and year_first_described (from ml/fetch_publications.py / NCBI
  gene2pubmed) exist specifically as a DIAGNOSTIC, per DESIGN.md section 5:
  "Publication count and year-first-described. Included specifically so the
  model can be shown not to be riding it." They are not meant to be load-
  bearing training inputs. The all_features baseline trains with them
  anyway so we have a reference point, but the feature-set ablation below
  (FEATURE_SETS) is how we check whether the model has real signal once the
  confounder is removed.

  As a secondary, cheaper proxy: has_gnomad is also checked, a gene absent
  from gnomAD (has_gnomad=0) is poorly characterised, so if such genes score
  low, the model may be tracking characterisation depth rather than true
  biology. We report the mean predicted score for has_gnomad=0 vs has_gnomad=1.

Feature-set ablation (configurable, not hardcoded):
  FEATURE_SETS below defines named subsets of the full v1 feature list.
  Select one with --feature-set, or run all of them and print one
  comparison table with --compare. See the comments on FEATURE_SETS for the
  reasoning behind each subset.

Fold reconstruction:
  Reads ml/cache/cv_folds.parquet written by split.py. This guarantees that
  training and evaluation use the IDENTICAL family-safe split without
  re-running the splitter (deterministic GroupKFold output matches because the
  same groups are presented in the same order). split.py itself is untouched
  by this file; the feature-set choice only changes which columns of the
  training table are handed to the model, never how genes are grouped or split.

Output (single feature-set run only, not --compare):
  ml/cache/oos_predictions.parquet -- symbol, score, label, rank, fold_idx
  Printed per-fold and mean metrics to stdout.

Run:
  python3 ml/train_eval.py                          # default: all_features
  python3 ml/train_eval.py --feature-set biology_only
  python3 ml/train_eval.py --compare                 # all variants, one table
"""

import argparse
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

# The full v1 feature set (DESIGN.md section 5).
FULL_FEATURE_COLS = [
    "pLI", "loeuf", "oe_lof", "oe_mis", "n_rare", "n_lof",
    "protein_length", "ppi_degree", "ppi_betweenness",
    "tau", "essentiality_score",
    "pub_count", "year_first_described",
    "has_gnomad", "has_burden", "has_alphafold", "has_string",
    "has_tau", "has_essentiality", "has_pub_count", "has_year_described",
]

# Narrow removal: only the raw pub_count number. year_first_described and its
# has_year_described flag are left in on purpose for the no_pubcount and
# no_pubcount_no_string variants, so the comparison table can show whether
# that correlated proxy alone still carries fame signal once pub_count
# itself is gone. biology_only (below) removes the whole block instead.
PUBCOUNT_ONLY = ["pub_count"]

# STRING centrality plus its coverage flag. has_string is dropped alongside
# ppi_degree and ppi_betweenness because "does this gene have any high-
# confidence STRING interaction at all" is itself a fame-adjacent signal:
# well-studied proteins are more likely to be in the network at all, not
# just more central once they are in it.
STRING_BLOCK = ["ppi_degree", "ppi_betweenness", "has_string"]

# The complete publication-metadata block, used only for biology_only, where
# we want zero trace of publication history left in the model, including
# year_first_described (an earlier-described gene has had more decades to
# accumulate literature, so it is correlated with pub_count) and both
# has_* coverage flags.
PUBLICATION_BLOCK_FULL = ["pub_count", "year_first_described", "has_pub_count", "has_year_described"]


def _drop(cols, *blocks):
    drop_set = {c for block in blocks for c in block}
    return [c for c in cols if c not in drop_set]


# Named feature sets for the confounder-ablation run (DESIGN.md section 6.2).
# A dict, not a hardcoded list, so --feature-set/--compare can select among
# them without editing this file.
FEATURE_SETS = {
    "all_features":         FULL_FEATURE_COLS,
    "no_pubcount":          _drop(FULL_FEATURE_COLS, PUBCOUNT_ONLY),
    "no_pubcount_no_string": _drop(FULL_FEATURE_COLS, PUBCOUNT_ONLY, STRING_BLOCK),
    "biology_only":         _drop(FULL_FEATURE_COLS, PUBLICATION_BLOCK_FULL, STRING_BLOCK),
}

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


def pub_count_tercile_pr_auc(oos, pub_count):
    """
    Bin genes by publication-count tercile and compute PR-AUC within each
    bin. Rank-based (not value-based) quantile cuts because pub_count is
    heavily right-skewed with many ties at low counts, ranking guarantees
    three non-empty, roughly equal-sized bins regardless of that skew.

    The baseline for PR-AUC is the positive rate of the population being
    scored, not a fixed global number. Understudied genes are less likely
    to have reached a clinical-phase drug, so the low tercile has a lower
    positive rate than the full 19,296-gene universe. Comparing its PR-AUC
    against the global positive rate judges it against an easier bar than
    the one it actually faces, so each tercile gets its own positive rate
    as its baseline, and lift = PR-AUC / that tercile's own positive rate.

    Returns a dict {"low": entry, "medium": entry, "high": entry} where each
    entry has n, n_pos, pos_rate, mean_pub_count, pr_auc, and lift (pr_auc
    and lift are None when the bin has zero positives).
    """
    pub_rank = pd.Series(pub_count).rank(method="first")
    pub_tercile = pd.qcut(pub_rank, q=3, labels=["low", "medium", "high"])

    out = {}
    for tercile in ["low", "medium", "high"]:
        mask = (pub_tercile == tercile).values
        n_bin = int(mask.sum())
        n_pos_bin = int(oos.loc[mask, "label"].sum())
        pos_rate = n_pos_bin / n_bin if n_bin > 0 else float("nan")
        entry = {
            "n": n_bin,
            "n_pos": n_pos_bin,
            "pos_rate": pos_rate,
            "mean_pub_count": float(pub_count[mask].mean()),
            "pr_auc": None,
            "lift": None,
        }
        if n_pos_bin > 0:
            pr_auc = average_precision_score(oos.loc[mask, "label"], oos.loc[mask, "score"])
            entry["pr_auc"] = pr_auc
            entry["lift"] = pr_auc / pos_rate
        out[tercile] = entry
    return out


# ── Core training/eval for one feature set ──────────────────────────────────

def run_variant(name, feature_cols, df, fold_v, n_folds, verbose=True):
    """
    Fit and evaluate one named feature set across all folds, then fit once
    more on all data for feature importances. Returns a dict of results so
    both the single-run report and the --compare table can reuse the same
    logic instead of duplicating the fold loop.
    """
    X = df[feature_cols].values
    y = df["label"].values

    col_w = 10
    if verbose:
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
                f"in both train and test. cv_folds.parquet may be stale, re-run split.py."
            )

        clf = GradientBoostingClassifier(**MODEL_PARAMS)
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, scores)
        pk     = [precision_at_k(y_test, scores, k) for k in K_VALUES]
        ef     = [enrichment_factor(y_test, scores, p) for p in EF_PERCENTS]

        fold_results.append({"fold": fold, "pr_auc": pr_auc, "pk": pk, "ef": ef})

        if verbose:
            pk_str = "  ".join(f"{v:{col_w}.3f}" for v in pk)
            ef_str = "  ".join(f"{v:{col_w}.2f}" for v in ef)
            print(
                f"{fold:>4}  {len(y_test):>6,}  {int(y_test.sum()):>5}  "
                f"{pr_auc:{col_w}.4f}  {pk_str}  {ef_str}"
            )

        test_symbols = df.loc[test_mask, "symbol"].values
        oos_rows.append(pd.DataFrame({
            "symbol":   test_symbols,
            "score":    scores,
            "label":    y_test,
            "fold_idx": fold,
        }))

    mean_pr = np.mean([r["pr_auc"] for r in fold_results])
    mean_pk = [np.mean([r["pk"][i] for r in fold_results]) for i in range(len(K_VALUES))]
    mean_ef = [np.mean([r["ef"][i] for r in fold_results]) for i in range(len(EF_PERCENTS))]

    if verbose:
        pk_str = "  ".join(f"{v:{col_w}.3f}" for v in mean_pk)
        ef_str = "  ".join(f"{v:{col_w}.2f}" for v in mean_ef)
        print("-" * len(header))
        print(f"{'mean':>4}  {'':>6}  {'':>5}  {mean_pr:{col_w}.4f}  {pk_str}  {ef_str}")

    oos = pd.concat(oos_rows, ignore_index=True)
    oos["rank"] = oos["score"].rank(ascending=False, method="first").astype(int)
    oos = oos.sort_values("rank").reset_index(drop=True)

    # Fit once more on all data for feature importances. No leakage risk:
    # importances are structural, not used for evaluation.
    clf_full = GradientBoostingClassifier(**MODEL_PARAMS)
    clf_full.fit(X, y)
    importances = sorted(
        zip(feature_cols, clf_full.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )

    pub_count = df.set_index("symbol").loc[oos["symbol"], "pub_count"].values
    rho, pval = spearmanr(oos["score"].values, np.log1p(pub_count))
    tercile = pub_count_tercile_pr_auc(oos, pub_count)

    return {
        "name": name,
        "n_features": len(feature_cols),
        "mean_pr_auc": mean_pr,
        "mean_pk": mean_pk,
        "mean_ef": mean_ef,
        "top10_importances": importances[:10],
        "spearman_rho": rho,
        "spearman_p": pval,
        "tercile": tercile,
        "oos": oos,
    }


# ── Comparison table for --compare ───────────────────────────────────────────

def print_comparison_table(results_by_name, baseline_pr):
    print("\n" + "=" * 78)
    print("FEATURE-SET ABLATION (DESIGN.md section 6.2)")
    print("=" * 78)
    print(f"Global random-ranker baseline PR-AUC (all 19,296 genes): {baseline_pr:.4f}")
    print(
        "This global number is only the right baseline for the overall\n"
        "PR-AUC column below. Each publication-count tercile has its own,\n"
        "much lower, positive rate, since understudied genes are less\n"
        "likely to have reached a clinical-phase drug. The tercile columns\n"
        "report lift, PR-AUC divided by that tercile's OWN positive rate,\n"
        "not the global one. Lift > 1.0 means the model beats a random\n"
        "ranker working on that same population.\n"
    )

    header = (
        f"{'variant':>24}  {'n_feat':>6}  {'PR-AUC':>8}  "
        f"{'low lift':>9}  {'mid lift':>9}  {'high lift':>10}  {'rho':>7}  {'EF@1%':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results_by_name.items():
        t = r["tercile"]
        low  = t["low"]["lift"]
        mid  = t["medium"]["lift"]
        high = t["high"]["lift"]
        low_str  = f"{low:.2f}"  if low  is not None else "n/a"
        mid_str  = f"{mid:.2f}"  if mid  is not None else "n/a"
        high_str = f"{high:.2f}" if high is not None else "n/a"
        ef1 = r["mean_ef"][EF_PERCENTS.index(1)]
        print(
            f"{name:>24}  {r['n_features']:>6}  {r['mean_pr_auc']:>8.4f}  "
            f"{low_str:>9}  {mid_str:>9}  {high_str:>10}  "
            f"{r['spearman_rho']:>7.3f}  {ef1:>7.2f}"
        )
    print("-" * len(header))

    print("\nFull tercile breakdown (n, positives, the tercile's own positive")
    print("rate as its baseline, PR-AUC, and lift = PR-AUC / that baseline):")
    detail_header = (
        f"  {'variant':>24}  {'tercile':>7}  {'n':>6}  {'n_pos':>6}  "
        f"{'pos_rate':>9}  {'PR-AUC':>8}  {'lift':>7}"
    )
    print(detail_header)
    print("  " + "-" * (len(detail_header) - 2))
    for name, r in results_by_name.items():
        for tercile in ["low", "medium", "high"]:
            t = r["tercile"][tercile]
            pr_str   = f"{t['pr_auc']:.4f}" if t["pr_auc"] is not None else "n/a"
            lift_str = f"{t['lift']:.2f}"   if t["lift"]   is not None else "n/a"
            print(
                f"  {name:>24}  {tercile:>7}  {t['n']:>6,}  {t['n_pos']:>6}  "
                f"{t['pos_rate']:>9.4f}  {pr_str:>8}  {lift_str:>7}"
            )

    for name, r in results_by_name.items():
        print(f"\nTop 10 feature importances, {name}:")
        for feat, imp in r["top10_importances"]:
            bar = "#" * int(imp * 60)
            print(f"  {feat:<20}  {imp:.4f}  {bar}")

    print("\n" + "=" * 78)
    print("DECISION CRITERION (corrected)")
    print("=" * 78)
    print(
        "In the low-publication tercile, does any variant achieve lift > 1.0\n"
        "over that tercile's OWN positive rate? That is the real question,\n"
        "not comparison against the global 7.8% rate.\n"
    )
    for name, r in results_by_name.items():
        low = r["tercile"]["low"]
        if low["lift"] is None:
            print(f"  {name:>24}: low tercile has no positives, cannot evaluate")
            continue
        verdict = "LIFT > 1.0" if low["lift"] > 1.0 else "lift <= 1.0"
        print(
            f"  {name:>24}: low-tercile PR-AUC = {low['pr_auc']:.4f}, "
            f"tercile positive rate = {low['pos_rate']:.4f}, "
            f"lift = {low['lift']:.2f}  ({verdict})"
        )
    print("=" * 78)


# ── Single-run extras (gnomAD proxy check, top predictions, caveat) ─────────

def print_gnomad_proxy_check(df, oos):
    print("\nSTUDY-BIAS CHECK, secondary proxy (DESIGN.md section 6.2)")
    has_gnomad = df.set_index("symbol").loc[oos["symbol"], "has_gnomad"].values
    mean_score_gnomad1 = oos.loc[has_gnomad == 1, "score"].mean()
    mean_score_gnomad0 = oos.loc[has_gnomad == 0, "score"].mean()
    print(f"  Mean OOS score, genes WITH gnomAD (has_gnomad=1):    {mean_score_gnomad1:.4f}")
    print(f"  Mean OOS score, genes WITHOUT gnomAD (has_gnomad=0): {mean_score_gnomad0:.4f}")
    ratio = mean_score_gnomad1 / mean_score_gnomad0 if mean_score_gnomad0 > 0 else float("inf")
    print(f"  Score ratio (gnomad=1 / gnomad=0):                   {ratio:.2f}x")
    if ratio > 2.0:
        print("  [WARN] Characterisation-depth bias suspected: gnomAD genes score >2x higher.")
    else:
        print("  [OK] Score ratio is below 2x, no strong characterisation-depth signal.")


def print_pubcount_check(result):
    print("\nSTUDY-BIAS CHECK: publication count (DESIGN.md section 6.2)")
    print(f"  Spearman rho(score, log(pub_count + 1)): {result['spearman_rho']:.3f}  "
          f"(p={result['spearman_p']:.2e})")
    if abs(result["spearman_rho"]) > 0.5:
        print("  [WARN] |rho| > 0.5, the model may be primarily ranking on gene fame.")
    else:
        print("  [OK] |rho| <= 0.5, publication count is not the dominant ranking signal.")

    print("\n  Lift by publication-count tercile. Each tercile's baseline is its")
    print("  OWN positive rate, not the global rate, understudied genes have a")
    print("  lower positive rate to begin with. Lift > 1.0 means the model beats")
    print("  a random ranker working on that same tercile:")
    print(f"  {'tercile':>8}  {'n':>6}  {'n_pos':>6}  {'pos_rate':>9}  {'PR-AUC':>8}  {'lift':>7}")
    for tercile in ["low", "medium", "high"]:
        t = result["tercile"][tercile]
        pr_str   = f"{t['pr_auc']:.4f}" if t["pr_auc"] is not None else "n/a"
        lift_str = f"{t['lift']:.2f}"   if t["lift"]   is not None else "n/a"
        note = "  (no positives in bin)" if t["pr_auc"] is None else ""
        print(f"  {tercile:>8}  {t['n']:>6,}  {t['n_pos']:>6}  {t['pos_rate']:>9.4f}  "
              f"{pr_str:>8}  {lift_str:>7}{note}")


def print_structural_caveat():
    print()
    print("=" * 65)
    print("STRUCTURAL-VALIDATION CAVEAT")
    print("=" * 65)
    print(
        "Burden features (n_rare, n_lof) cover chr22 only (388 / 19,296\n"
        "genes, 2.0%). Roughly 98% of genes have constant burden values\n"
        "(n_rare=0, n_lof=0, has_burden=0), so burden carries almost no\n"
        "information outside chr22. This is a known limitation and likely\n"
        "part of why the biology-only signal is weak on understudied genes,\n"
        "not evidence that the biology features themselves are useless.\n"
        "When burden is extended genome-wide, re-run this script: the\n"
        "ranking will incorporate rare-variant evidence for all genes."
    )
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-set", choices=sorted(FEATURE_SETS), default="all_features",
        help="Named feature set to train with (default: all_features).",
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="Run every named feature set and print one comparison table, "
             "then exit. Used for the confounder-ablation check in "
             "DESIGN.md section 6.2.",
    )
    return ap.parse_args()


def load_inputs():
    for path, label in [(TABLE_FILE, "training_table"), (FOLDS_FILE, "cv_folds")]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run the prerequisite scripts first.", file=sys.stderr)
            sys.exit(1)

    df    = pd.read_parquet(TABLE_FILE)
    folds = pd.read_parquet(FOLDS_FILE)

    df = df.merge(folds[["symbol", "fold_idx"]], on="symbol", how="left")
    assert df["fold_idx"].notna().all(), "some genes have no fold assignment after merge"

    n_folds = int(df["fold_idx"].max()) + 1
    fold_v = df["fold_idx"].values.astype(int)
    return df, fold_v, n_folds


def main():
    args = parse_args()
    df, fold_v, n_folds = load_inputs()

    print(f"loaded {len(df):,} genes, {df['label'].sum():,} positives, {n_folds} folds")
    baseline_pr = df["label"].mean()
    print(f"baseline positive rate: {baseline_pr:.3%}  (random ranker PR-AUC ~ {baseline_pr:.3f})\n")

    if args.compare:
        results_by_name = {}
        for name, cols in FEATURE_SETS.items():
            print(f"running variant: {name}  ({len(cols)} features)")
            results_by_name[name] = run_variant(name, cols, df, fold_v, n_folds, verbose=False)
        print_comparison_table(results_by_name, baseline_pr)
        return

    name = args.feature_set
    feature_cols = FEATURE_SETS[name]
    print(f"feature set: {name}  ({len(feature_cols)} features)\n")
    result = run_variant(name, feature_cols, df, fold_v, n_folds, verbose=True)
    oos = result["oos"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    oos.to_parquet(OOS_FILE, index=False)
    print(f"\nwrote {OOS_FILE}  ({len(oos):,} rows)")

    print("\nTop 10 predicted targets (OOS scores, ranked globally):")
    top10 = oos.head(10)[["rank", "symbol", "score", "label", "fold_idx"]]
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nFeature importances ({name}, full-data model, for study-bias diagnostic):")
    for feat, imp in result["top10_importances"]:
        bar = "#" * int(imp * 60)
        print(f"  {feat:<20}  {imp:.4f}  {bar}")

    print_gnomad_proxy_check(df, oos)
    print_pubcount_check(result)
    print_structural_caveat()


if __name__ == "__main__":
    main()
