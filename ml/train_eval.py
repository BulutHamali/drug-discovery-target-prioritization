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
  Interpretable (tree feature importances, plus exact SHAP values via
  shap.TreeExplainer), handles mixed
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
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score

RANDOM_SEED = 42

CACHE_DIR   = os.environ.get("ML_CACHE_DIR", "ml/cache")
TABLE_FILE  = os.path.join(CACHE_DIR, "training_table.parquet")
FOLDS_FILE  = os.path.join(CACHE_DIR, "cv_folds.parquet")
OOS_FILE    = os.path.join(CACHE_DIR, "oos_predictions.parquet")

# The full v1 feature set (DESIGN.md section 5), plus disorder_fraction
# (AlphaFold pLDDT-based disorder, DESIGN.md section 5's "protein-intrinsic"
# group), added once ml/fetch_alphafold.py was run with --disorder. It is a
# biology feature like protein_length, not part of the publication/STRING
# fame-confound story, so it stays in every FEATURE_SETS variant below,
# including biology_only.
FULL_FEATURE_COLS = [
    "pLI", "loeuf", "oe_lof", "oe_mis", "n_rare", "n_lof",
    "protein_length", "disorder_fraction", "ppi_degree", "ppi_betweenness",
    "tau", "essentiality_score",
    "pub_count", "year_first_described",
    "has_gnomad", "has_burden", "has_alphafold", "has_disorder", "has_string",
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


def assign_rank_groups(df, value_col, q, labels):
    """
    Assigns each gene to a rank-based group (tercile, median split, ...)
    from a FIXED ordering of df, not from any variant's out-of-fold
    oos dataframe. This matters: oos is sorted by that variant's own
    predicted score, so its row order differs between variants, and
    rank(method="first") breaks ties by row position. Computing group
    membership from df directly, once, before any variant runs, guarantees
    every variant assigns the exact same genes to the exact same group.
    Without this fix, two variants can disagree on group membership for a
    handful of genes sitting exactly at a quantile boundary, which silently
    invalidates any paired comparison between them.

    Returns a pandas Series indexed by symbol with the group label.
    """
    rank = df[value_col].rank(method="first")
    groups = pd.qcut(rank, q=q, labels=labels)
    return pd.Series(groups.values, index=df["symbol"].values)


def group_pr_auc_lift(oos, group_assignment, group_labels):
    """
    PR-AUC and lift for each group in group_assignment (mapped onto oos by
    symbol, so identical across variants by construction).

    The baseline for PR-AUC is the positive rate of the population being
    scored, not a fixed global number. Understudied genes are less likely
    to have reached a clinical-phase drug, so the low-publication group has
    a lower positive rate than the full 19,296-gene universe. Comparing its
    PR-AUC against the global positive rate judges it against an easier bar
    than the one it actually faces, so each group gets its own positive
    rate as its baseline, and lift = PR-AUC / that group's own positive rate.

    Returns a dict {label: entry}, entry has n, n_pos, pos_rate, pr_auc, and
    lift (pr_auc and lift are None when the group has zero positives).
    """
    groups_for_oos = oos["symbol"].map(group_assignment)
    out = {}
    for label in group_labels:
        mask = (groups_for_oos == label).values
        n_bin = int(mask.sum())
        n_pos_bin = int(oos.loc[mask, "label"].sum())
        pos_rate = n_pos_bin / n_bin if n_bin > 0 else float("nan")
        entry = {"n": n_bin, "n_pos": n_pos_bin, "pos_rate": pos_rate, "pr_auc": None, "lift": None}
        if n_pos_bin > 0:
            pr_auc = average_precision_score(oos.loc[mask, "label"], oos.loc[mask, "score"])
            entry["pr_auc"] = pr_auc
            entry["lift"] = pr_auc / pos_rate
        out[label] = entry
    return out


def per_fold_group_lift(oos, group_assignment, group_label):
    """
    Point-estimate lift computed SEPARATELY within each fold's share of the
    given group, not pooled. The pooled lift number (group_pr_auc_lift
    above) can hide how much this varies fold to fold; with only 60-61
    positives in the low tercile spread across 5 folds, that spread is
    exactly the thing a single pooled number hides.

    Returns {fold_idx: {"n", "n_pos", "pos_rate", "pr_auc", "lift"}}, pr_auc
    and lift are None when that fold has zero positives in this group.
    """
    groups_for_oos = oos["symbol"].map(group_assignment)
    mask = (groups_for_oos == group_label).values
    sub = oos.loc[mask]

    out = {}
    for fold in sorted(sub["fold_idx"].unique()):
        fold_sub = sub[sub["fold_idx"] == fold]
        n = len(fold_sub)
        n_pos = int(fold_sub["label"].sum())
        pos_rate = n_pos / n if n > 0 else float("nan")
        entry = {"n": n, "n_pos": n_pos, "pos_rate": pos_rate, "pr_auc": None, "lift": None}
        if n_pos > 0:
            pr_auc = average_precision_score(fold_sub["label"], fold_sub["score"])
            entry["pr_auc"] = pr_auc
            entry["lift"] = pr_auc / pos_rate
        out[int(fold)] = entry
    return out


def bootstrap_blockfold_lift_ci(oos, group_assignment, group_label, n_boot=1000, seed=RANDOM_SEED):
    """
    Block-bootstrap 95% CI for the pooled lift in one group. Resamples WITH
    replacement independently WITHIN each fold's rows (not pooled across
    folds first), because each fold's OOS scores come from a distinct
    held-out model. Each iteration resamples every fold, pools the
    resampled rows back together, and recomputes lift = PR-AUC / positive
    rate on that pool, both numerator and denominator resampled together
    since lift is a ratio.

    This is the ORIGINAL (within-fold) bootstrap. See
    bootstrap_pooled_lift_ci below for the corrected version: fixing each
    fold's contribution to exactly its own size, as this function does,
    forces every bootstrap draw to include the same handful of
    high-variance small folds every time, which can keep the CI wider than
    a plain pooled resample would. Report both so the difference is
    visible, not just asserted.

    Returns (ci_low, ci_high, bootstrap_mean, n_valid_iterations).
    """
    rng = np.random.default_rng(seed)

    groups_for_oos = oos["symbol"].map(group_assignment)
    mask = (groups_for_oos == group_label).values
    sub = oos.loc[mask].reset_index(drop=True)

    fold_positions = [
        np.where(sub["fold_idx"].values == f)[0]
        for f in sub["fold_idx"].unique()
        if (sub["fold_idx"].values == f).sum() > 0
    ]
    labels = sub["label"].values
    scores = sub["score"].values

    lifts = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True) for pos in fold_positions])
        y_b = labels[idx]
        s_b = scores[idx]
        pos_rate_b = y_b.mean()
        if y_b.sum() == 0 or pos_rate_b <= 0:
            continue
        pr_auc_b = average_precision_score(y_b, s_b)
        lifts.append(pr_auc_b / pos_rate_b)

    if not lifts:
        return None, None, None, 0
    lifts = np.array(lifts)
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5])
    return float(ci_low), float(ci_high), float(lifts.mean()), len(lifts)


def bootstrap_pooled_lift_ci(oos, group_assignment, group_label, n_boot=1000, seed=RANDOM_SEED):
    """
    Pooled (non-blocked) bootstrap 95% CI: resample WITH replacement from
    ALL rows in the group at once, ignoring fold membership. Every gene in
    5-fold CV gets exactly one out-of-fold prediction; once training is
    done, fold identity does not change the fact that each gene contributed
    one (label, score) pair to the group. There is no statistical reason to
    force each bootstrap draw to include exactly the original count from
    every fold, as bootstrap_blockfold_lift_ci above does; that is a
    stricter, more conservative resampling scheme, not a more correct one,
    for a metric that only needs the group's out-of-fold predictions as
    its evaluation set. This is the primary fix: pooling all 60-61
    positives into one bootstrap population instead of resampling ~12 at a
    time within 5 separate small blocks.

    Returns (ci_low, ci_high, bootstrap_mean, n_valid_iterations).
    """
    rng = np.random.default_rng(seed)

    groups_for_oos = oos["symbol"].map(group_assignment)
    mask = (groups_for_oos == group_label).values
    labels = oos.loc[mask, "label"].values
    scores = oos.loc[mask, "score"].values
    n = len(labels)

    lifts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        s_b = scores[idx]
        pos_rate_b = y_b.mean()
        if y_b.sum() == 0 or pos_rate_b <= 0:
            continue
        pr_auc_b = average_precision_score(y_b, s_b)
        lifts.append(pr_auc_b / pos_rate_b)

    if not lifts:
        return None, None, None, 0
    lifts = np.array(lifts)
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5])
    return float(ci_low), float(ci_high), float(lifts.mean()), len(lifts)


def paired_fold_diff(per_fold_a, per_fold_b):
    """
    Per-fold lift(A) - lift(B) for folds where both are defined (same
    folds, same genes per fold across variants by construction, since
    group_assignment is computed once from df and fold_v is shared). This
    is the right comparison between two variants: they are evaluated on
    IDENTICAL folds, so comparing their individually-overlapping CIs
    throws away that pairing and is the wrong test. Also returns the sign
    test tally: how many folds A beats B, B beats A, or tie.

    Returns (diffs, a_wins, b_wins, ties), diffs is a list of
    (fold_idx, difference) tuples.
    """
    diffs = []
    a_wins = b_wins = ties = 0
    for fold in sorted(set(per_fold_a) & set(per_fold_b)):
        la = per_fold_a[fold]["lift"]
        lb = per_fold_b[fold]["lift"]
        if la is None or lb is None:
            continue
        d = la - lb
        diffs.append((fold, d))
        if d > 0:
            a_wins += 1
        elif d < 0:
            b_wins += 1
        else:
            ties += 1
    return diffs, a_wins, b_wins, ties


def bootstrap_paired_diff_ci(diffs, n_boot=1000, seed=RANDOM_SEED):
    """
    Bootstrap 95% CI on the mean per-fold lift difference, resampling
    FOLDS (the natural unit for a paired comparison across identical
    folds), not individual genes. With only 5 folds this is necessarily
    coarse (report alongside the sign test, which does not depend on this
    resampling assumption at all).

    Returns (ci_low, ci_high, mean_diff), or (None, None, None) if diffs is
    empty.
    """
    rng = np.random.default_rng(seed)
    vals = np.array([d for _, d in diffs])
    if len(vals) == 0:
        return None, None, None
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        means[i] = vals[idx].mean()
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return float(ci_low), float(ci_high), float(vals.mean())


def make_group_folds(groups, n_splits, seed):
    """
    Assigns each unique group (gene family) to one of n_splits folds via
    greedy size-balancing, using `seed` to shuffle the tie-breaking order
    among equal-size groups. sklearn's GroupKFold has no random_state: it
    is fully deterministic given the groups array (it sorts groups by
    size internally, independent of input row order), so there is no
    built-in way to get a second, different, family-safe split without
    implementing the assignment directly. This is that direct
    implementation; the family-safe guarantee (every row of a group lands
    in exactly one fold) holds by construction, exactly like GroupKFold's.

    Returns an array of fold indices, one per row, aligned to `groups`.
    """
    rng = np.random.default_rng(seed)
    unique_groups, counts = np.unique(groups, return_counts=True)
    perm = rng.permutation(len(unique_groups))
    unique_groups, counts = unique_groups[perm], counts[perm]

    # Stable sort by size descending: the shuffle above already randomized
    # the order among equal-size groups, and a stable sort preserves that
    # randomized order within each size tier rather than re-sorting it away.
    order = np.argsort(-counts, kind="stable")
    unique_groups, counts = unique_groups[order], counts[order]

    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold = {}
    for g, c in zip(unique_groups, counts):
        f = int(np.argmin(fold_sizes))
        group_to_fold[g] = f
        fold_sizes[f] += c

    return np.array([group_to_fold[g] for g in groups])


def run_repeated_cv(feature_cols, df, group_assignment, group_label, n_repeats=10, n_splits=5, model_seed=RANDOM_SEED):
    """
    Repeats GroupKFold n_repeats times with a DIFFERENT group-to-fold
    assignment each time (make_group_folds, varying only its shuffle
    seed), while the model's own random_state stays fixed at model_seed
    throughout (MODEL_PARAMS). This isolates how much of the low-tercile
    lift's fold-to-fold spread is an artifact of which fold happened to get
    which positives, as opposed to genuine model behavior, since that is a
    different source of variance than the bootstrap CIs above (which hold
    the fold assignment fixed and only resample within it).

    The leakage guarantee is asserted fresh for every fold of every
    repeat, exactly as split.py asserts it for the single canonical split.

    Returns an array of pooled lift values, one per repeat (repeats where
    the group ends up with zero positives, vanishingly unlikely at this
    sample size, are skipped).
    """
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["group_key"].values

    repeat_lifts = []
    for repeat in range(n_repeats):
        fold_v_repeat = make_group_folds(groups, n_splits, seed=1000 + repeat)

        oos_rows = []
        for fold in range(n_splits):
            train_mask = fold_v_repeat != fold
            test_mask  = fold_v_repeat == fold

            train_keys = set(df.loc[train_mask, "group_key"])
            test_keys  = set(df.loc[test_mask, "group_key"])
            overlap = train_keys & test_keys
            assert not overlap, (
                f"FATAL LEAKAGE in repeated-CV repeat {repeat} fold {fold}: "
                f"{len(overlap)} group_keys appear in both train and test."
            )

            clf = GradientBoostingClassifier(**MODEL_PARAMS)
            clf.fit(X[train_mask], y[train_mask])
            scores = clf.predict_proba(X[test_mask])[:, 1]
            oos_rows.append(pd.DataFrame({
                "symbol":   df.loc[test_mask, "symbol"].values,
                "score":    scores,
                "label":    y[test_mask],
                "fold_idx": fold,
            }))

        oos_repeat = pd.concat(oos_rows, ignore_index=True)
        groups_for_oos = oos_repeat["symbol"].map(group_assignment)
        mask = (groups_for_oos == group_label).values
        n_pos = int(oos_repeat.loc[mask, "label"].sum())
        if n_pos == 0:
            continue
        pos_rate = n_pos / mask.sum()
        pr_auc = average_precision_score(oos_repeat.loc[mask, "label"], oos_repeat.loc[mask, "score"])
        repeat_lifts.append(pr_auc / pos_rate)

    return np.array(repeat_lifts)


def bootstrap_stability_selection(feature_cols, df, n_boot=50, top_k=10, seed=RANDOM_SEED):
    """
    DESIGN.md section 6.5, implemented: resample the full training set with
    replacement, refit, and record which features land in the top `top_k` by
    feature_importances_ each time. Reports the fraction of `n_boot` resamples
    each feature was selected in. Features selected in most runs are robust
    signal; features that only show up occasionally are more likely
    resampling noise than stable biology.

    This is deliberately a plain row-level bootstrap (sample with
    replacement, ignore gene-family grouping), NOT GroupKFold. The question
    here is whether a feature's importance is stable across resamples of the
    training set, not leakage-safe generalization to held-out genes, so there
    is no test set and no group-overlap concern the way there is everywhere
    else in this file. The model's own random_state stays fixed at `seed`
    throughout (MODEL_PARAMS); the only source of variation across the
    `n_boot` iterations is which genes happen to be resampled.

    Returns a dict {feature: fraction_selected}, sorted by fraction
    descending is the caller's job (matches the style of the other
    print-facing helpers in this file).
    """
    X = df[feature_cols].values
    y = df["label"].values
    n = len(df)
    rng = np.random.default_rng(seed)

    selection_counts = {f: 0 for f in feature_cols}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        clf = GradientBoostingClassifier(**MODEL_PARAMS)
        clf.fit(X[idx], y[idx])
        top_features = sorted(
            zip(feature_cols, clf.feature_importances_),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]
        for feat, _ in top_features:
            selection_counts[feat] += 1

    return {f: c / n_boot for f, c in selection_counts.items()}


# ── Core training/eval for one feature set ──────────────────────────────────

def run_variant(name, feature_cols, df, fold_v, n_folds, tercile_assignment, median_assignment, verbose=True):
    """
    Fit and evaluate one named feature set across all folds, then fit once
    more on all data for feature importances. Returns a dict of results so
    both the single-run report and the --compare table can reuse the same
    logic instead of duplicating the fold loop.

    tercile_assignment and median_assignment are precomputed ONCE (see
    assign_rank_groups) and shared across every variant's call, so group
    membership is identical for all variants, required for the paired
    comparison between variants to be valid.
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

    fold_results     = []
    oos_rows         = []
    fold_importances = []

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
        fold_importances.append(dict(zip(feature_cols, clf.feature_importances_)))

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

    # SHAP importances on the same full-data model (DESIGN.md 6.2/7, deferred
    # until now). TreeExplainer is exact and fast for gradient-boosted trees,
    # no sampling/approximation needed. Reported as mean |SHAP value| per
    # feature (average magnitude of that feature's contribution to the
    # predicted log-odds, across all 19,296 genes), which is comparable in
    # spirit to feature_importances_ but reflects actual per-gene attribution
    # rather than split-count/impurity-reduction bookkeeping. Kept alongside
    # feature_importances_ rather than replacing it: they usually agree, and
    # a case where they disagree is itself diagnostic (e.g. a feature with
    # high impurity-based importance but low actual prediction impact).
    shap_values = shap.TreeExplainer(clf_full).shap_values(X)
    shap_importances = sorted(
        zip(feature_cols, np.abs(shap_values).mean(axis=0)),
        key=lambda t: t[1],
        reverse=True,
    )

    pub_count = df.set_index("symbol").loc[oos["symbol"], "pub_count"].values
    rho, pval = spearmanr(oos["score"].values, np.log1p(pub_count))
    tercile = group_pr_auc_lift(oos, tercile_assignment, ["low", "medium", "high"])

    # Uncertainty on the low-tercile lift, three complementary views:
    # per-fold point estimates (fold-to-fold spread, visible directly),
    # the original within-fold block bootstrap CI, and the corrected
    # pooled bootstrap CI (point 1 fix: resample all 60-61 positives as
    # one pool, not ~12 at a time within 5 small blocks).
    low_per_fold = per_fold_group_lift(oos, tercile_assignment, "low")
    low_block_lo, low_block_hi, low_block_mean, low_block_n = bootstrap_blockfold_lift_ci(
        oos, tercile_assignment, "low"
    )
    low_pool_lo, low_pool_hi, low_pool_mean, low_pool_n = bootstrap_pooled_lift_ci(
        oos, tercile_assignment, "low"
    )

    # Median split (point 4): bottom half vs top half by pub_count, roughly
    # doubling the positives in the understudied group versus the tercile.
    median = group_pr_auc_lift(oos, median_assignment, ["bottom_half", "top_half"])
    median_ci = {}
    for half in ["bottom_half", "top_half"]:
        lo, hi, mean, n = bootstrap_pooled_lift_ci(oos, median_assignment, half)
        median_ci[half] = {"ci_low": lo, "ci_high": hi, "boot_mean": mean, "n_valid": n}

    # Cross-fold spread on feature importance for the burden features. Not a
    # formal CI, just the mean/std/range of feature_importances_ across the
    # 5 fold-specific models already trained above, essentially free since
    # those models already exist. Useful alongside the low-tercile lift CI
    # for judging how much of the n_rare/n_lof importance trend is signal.
    burden_importance_spread = {}
    for feat in ("n_rare", "n_lof"):
        if feat in feature_cols:
            vals = np.array([fi[feat] for fi in fold_importances])
            burden_importance_spread[feat] = {
                "mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max()),
            }

    return {
        "name": name,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "mean_pr_auc": mean_pr,
        "mean_pk": mean_pk,
        "mean_ef": mean_ef,
        "top10_importances": importances[:10],
        "shap_top10_importances": shap_importances[:10],
        "burden_importance_spread": burden_importance_spread,
        "low_tercile_per_fold": low_per_fold,
        "low_tercile_ci_blockfold": {
            "ci_low": low_block_lo, "ci_high": low_block_hi,
            "boot_mean": low_block_mean, "n_valid": low_block_n,
        },
        "low_tercile_ci_pooled": {
            "ci_low": low_pool_lo, "ci_high": low_pool_hi,
            "boot_mean": low_pool_mean, "n_valid": low_pool_n,
        },
        "median_split": median,
        "median_split_ci": median_ci,
        "spearman_rho": rho,
        "spearman_p": pval,
        "tercile": tercile,
        "oos": oos,
    }


# ── Comparison table for --compare ───────────────────────────────────────────

def print_comparison_table(results_by_name, baseline_pr, repeated_cv_by_name=None, stability_by_name=None):
    print("\n" + "=" * 78)
    print("FEATURE-SET ABLATION (DESIGN.md section 6.2)")
    print("=" * 78)
    print(f"Global random-ranker baseline PR-AUC (all 19,296 genes): {baseline_pr:.4f}")
    print(
        "This global number is only the right baseline for the overall\n"
        "PR-AUC column below. Each stratum (tercile or half) has its own,\n"
        "much lower, positive rate, since understudied genes are less\n"
        "likely to have reached a clinical-phase drug. Lift = PR-AUC divided\n"
        "by that stratum's OWN positive rate, not the global one. Lift > 1.0\n"
        "means the model beats a random ranker working on that population.\n"
    )

    header = (
        f"{'variant':>24}  {'n_feat':>6}  {'PR-AUC':>8}  "
        f"{'bottom-half lift':>16}  {'top-half lift':>14}  {'rho':>7}  {'EF@1%':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results_by_name.items():
        m = r["median_split"]
        bottom = m["bottom_half"]["lift"]
        top    = m["top_half"]["lift"]
        bottom_str = f"{bottom:.2f}" if bottom is not None else "n/a"
        top_str    = f"{top:.2f}"    if top    is not None else "n/a"
        ef1 = r["mean_ef"][EF_PERCENTS.index(1)]
        print(
            f"{name:>24}  {r['n_features']:>6}  {r['mean_pr_auc']:>8.4f}  "
            f"{bottom_str:>16}  {top_str:>14}  "
            f"{r['spearman_rho']:>7.3f}  {ef1:>7.2f}"
        )
    print("-" * len(header))
    print(
        "\nThe summary table above uses the median split (bottom half vs top\n"
        "half by pub_count), the properly powered evaluation, as the headline\n"
        "number. See PRIMARY EVALUATION below for the full detail, and\n"
        "SECONDARY / UNDERPOWERED further down for the original tercile-based\n"
        "numbers this comparison table used to lead with."
    )

    print("\n" + "=" * 78)
    print("PRIMARY EVALUATION: MEDIAN SPLIT (bottom half vs top half by pub_count)")
    print("=" * 78)
    print(
        "This is the headline evaluation. A 50/50 split by publication count\n"
        "puts 159 positives in the understudied (bottom) half versus only\n"
        "60-61 in the low tercile used earlier, properly powered rather than\n"
        "underpowered. Lift = PR-AUC / that half's own positive rate; the\n"
        "95% CI is a pooled bootstrap (resample all rows in the half at once).\n"
    )
    med_header = f"  {'variant':>24}  {'half':>12}  {'n':>6}  {'n_pos':>6}  {'pos_rate':>9}  {'lift':>7}  {'95% CI':>16}"
    print(med_header)
    print("  " + "-" * (len(med_header) - 2))
    for name, r in results_by_name.items():
        for half in ["bottom_half", "top_half"]:
            t = r["median_split"][half]
            ci = r["median_split_ci"][half]
            lift_str = f"{t['lift']:.2f}" if t["lift"] is not None else "n/a"
            ci_str = f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]" if ci["ci_low"] is not None else "n/a"
            print(
                f"  {name:>24}  {half:>12}  {t['n']:>6,}  {t['n_pos']:>6}  "
                f"{t['pos_rate']:>9.4f}  {lift_str:>7}  {ci_str:>16}"
            )
        print()

    print("=" * 78)
    print("PRIMARY DECISION CRITERION: median split")
    print("=" * 78)
    print(
        "Does every variant's bottom-half (understudied) CI sit entirely\n"
        "above 1.0? That is the real answer to whether the model rides study\n"
        "bias or has genuine signal on understudied genes.\n"
    )
    for name, r in results_by_name.items():
        ci = r["median_split_ci"]["bottom_half"]
        t = r["median_split"]["bottom_half"]
        if ci["ci_low"] is None:
            print(f"  {name:>24}: bottom half has no positives, cannot evaluate")
            continue
        verdict = "CI entirely above 1.0" if ci["ci_low"] > 1.0 else "CI includes or is below 1.0"
        print(
            f"  {name:>24}: lift = {t['lift']:.2f}, 95% CI = "
            f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]  ({verdict})"
        )
    print("=" * 78)

    print("\n" + "=" * 78)
    print("SECONDARY / UNDERPOWERED: publication-count TERCILE")
    print("=" * 78)
    print(
        "Everything below uses the original tercile split (only 60-61\n"
        "positives in the low group). An earlier run of this analysis, using\n"
        "only these numbers, concluded coverage-level differences in lift\n"
        "were 'not distinguishable from noise'. That conclusion was a power\n"
        "artifact, not a genuine null: the median split above, with 2.6x the\n"
        "positives, resolves the same question with tight, non-overlapping\n"
        "CIs. Kept here for the fold-level detail and the paired comparison\n"
        "below, not as the headline result.\n"
    )

    print("Full tercile breakdown (n, positives, the tercile's own positive")
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

    print("\nLow-tercile lift uncertainty (per-fold spread, then two bootstrap CIs:")
    print("the original within-fold block CI and the pooled CI, reported together")
    print("so the difference between them is visible, it is small, see step 2 report):")
    fold_header = f"  {'variant':>24}  {'fold':>4}  {'n':>5}  {'n_pos':>5}  {'pos_rate':>9}  {'lift':>7}"
    print(fold_header)
    print("  " + "-" * (len(fold_header) - 2))
    for name, r in results_by_name.items():
        for fold, entry in r["low_tercile_per_fold"].items():
            lift_str = f"{entry['lift']:.2f}" if entry["lift"] is not None else "n/a"
            note = "  (no positives)" if entry["lift"] is None else ""
            print(
                f"  {name:>24}  {fold:>4}  {entry['n']:>5}  {entry['n_pos']:>5}  "
                f"{entry['pos_rate']:>9.4f}  {lift_str:>7}{note}"
            )
        block = r["low_tercile_ci_blockfold"]
        pooled = r["low_tercile_ci_pooled"]
        if block["ci_low"] is not None:
            print(
                f"  {name:>24}  {'':>4}  {'':>5}  {'':>5}  {'block CI:':>9}  "
                f"[{block['ci_low']:.2f}, {block['ci_high']:.2f}]  "
                f"(mean {block['boot_mean']:.2f}, {block['n_valid']}/1000 valid)"
            )
        if pooled["ci_low"] is not None:
            print(
                f"  {name:>24}  {'':>4}  {'':>5}  {'':>5}  {'pooled CI:':>9}  "
                f"[{pooled['ci_low']:.2f}, {pooled['ci_high']:.2f}]  "
                f"(mean {pooled['boot_mean']:.2f}, {pooled['n_valid']}/1000 valid)"
            )
        print()

    for name, r in results_by_name.items():
        print(f"Top 10 feature importances, {name}:")
        for feat, imp in r["top10_importances"]:
            bar = "#" * int(imp * 60)
            print(f"  {feat:<20}  {imp:.4f}  {bar}")
        spread = r["burden_importance_spread"]
        if spread:
            print("  cross-fold spread (5 fold-specific models, not the full-data model above):")
            for feat, s in spread.items():
                print(
                    f"    {feat:<10}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                    f"range=[{s['min']:.4f}, {s['max']:.4f}]"
                )
        shap_max = max((v for _, v in r["shap_top10_importances"]), default=0.0)
        print(f"  Top 10 SHAP importances, {name} (mean |SHAP value|, same full-data model):")
        for feat, imp in r["shap_top10_importances"]:
            bar = "#" * int((imp / shap_max) * 60) if shap_max > 0 else ""
            print(f"    {feat:<20}  {imp:.4f}  {bar}")
        print()

    print("\n" + "=" * 78)
    print("PAIRED COMPARISON BETWEEN VARIANTS (low tercile)")
    print("=" * 78)
    print(
        "All four variants are evaluated on IDENTICAL folds, so comparing\n"
        "their individually overlapping CIs is the wrong test. This computes\n"
        "the per-fold DIFFERENCE in low-tercile lift for each pair (A minus B),\n"
        "bootstraps a CI on that difference (resampling folds, n=5, so this is\n"
        "necessarily coarse), and reports the sign test: in how many of the\n"
        "matched folds does A beat B.\n"
    )
    pairs = [
        ("biology_only", "no_pubcount"),
        ("biology_only", "all_features"),
        ("no_pubcount", "no_pubcount_no_string"),
    ]
    for a, b in pairs:
        if a not in results_by_name or b not in results_by_name:
            continue
        diffs, a_wins, b_wins, ties = paired_fold_diff(
            results_by_name[a]["low_tercile_per_fold"], results_by_name[b]["low_tercile_per_fold"]
        )
        ci_lo, ci_hi, mean_diff = bootstrap_paired_diff_ci(diffs)
        print(f"  {a} vs {b}:")
        print(f"    per-fold diffs (A-B): " + ", ".join(f"{d:+.2f}" for _, d in diffs))
        print(f"    sign test: {a} wins {a_wins}, {b} wins {b_wins}, ties {ties}  (of {len(diffs)} matched folds)")
        if ci_lo is not None:
            excludes_zero = "EXCLUDES zero" if (ci_lo > 0 or ci_hi < 0) else "includes zero"
            print(f"    mean diff = {mean_diff:+.2f}, 95% CI = [{ci_lo:+.2f}, {ci_hi:+.2f}]  ({excludes_zero})")
        print()

    if repeated_cv_by_name:
        print("\n" + "=" * 78)
        print("REPEATED CV: does fold assignment itself drive the spread?")
        print("=" * 78)
        print(
            "Same model seed, 10 different group-to-fold assignments (fold\n"
            "composition varied, leakage guarantee re-asserted every fold of\n"
            "every repeat). Isolates variance from WHICH fold got WHICH\n"
            "positives, separate from the bootstrap CIs above, which hold the\n"
            "fold assignment fixed and only resample within it.\n"
        )
        rcv_header = f"  {'variant':>24}  {'n_repeats':>9}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}"
        print(rcv_header)
        print("  " + "-" * (len(rcv_header) - 2))
        for name, lifts in repeated_cv_by_name.items():
            if len(lifts) == 0:
                print(f"  {name:>24}  no valid repeats")
                continue
            print(
                f"  {name:>24}  {len(lifts):>9}  {lifts.mean():>7.2f}  {lifts.std():>7.2f}  "
                f"{lifts.min():>7.2f}  {lifts.max():>7.2f}"
            )
        print()

    if stability_by_name:
        print("\n" + "=" * 78)
        print("BOOTSTRAP STABILITY SELECTION (DESIGN.md 6.5)")
        print("=" * 78)
        print(
            "50 row-level bootstrap resamples of the full training set (NOT\n"
            "GroupKFold, see bootstrap_stability_selection docstring), refit each\n"
            "time, top-10 features by feature_importances_ recorded per resample.\n"
            "Fraction shown is how often that feature landed in the top 10.\n"
            "Features selected in more than 70% of resamples are the stable,\n"
            "resampling-robust signal; below that, treat the ranking as noisier.\n"
        )
        for name, stability in stability_by_name.items():
            stable = sorted(stability.items(), key=lambda t: t[1], reverse=True)
            stable = [(f, frac) for f, frac in stable if frac > 0][:10]
            print(f"  {name}:")
            for feat, frac in stable:
                tag = "STABLE" if frac > 0.70 else ""
                print(f"    {feat:<20}  {frac:.2f}  {tag}")
            print()

    print("\n" + "=" * 78)
    print("SECONDARY DECISION CRITERION (tercile, underpowered, kept for continuity)")
    print("=" * 78)
    print(
        "In the low-publication tercile, does any variant achieve lift > 1.0\n"
        "over that tercile's OWN positive rate? See PRIMARY DECISION CRITERION\n"
        "above for the properly powered version of this same question.\n"
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


def print_descriptive_correlations(df):
    """
    Spearman correlation between year_first_described and the mechanistic
    biology features already in the model. DESCRIPTIVE ONLY, not a
    decomposition: this shows that discovery timing is entangled with
    biology (older-described genes tend to look different on constraint,
    essentiality, tissue specificity), it does NOT tell us how much of
    year_first_described's contribution in the paired comparison is fame
    versus biology. In particular, biology_only already contains pLI,
    loeuf, oe_lof, oe_mis, essentiality_score, and tau, so whatever
    year_first_described adds on top of biology_only is, by construction,
    not the signal those features already capture; it is some other,
    unmeasured thing correlated with discovery timing (technological
    accessibility, funding history, disease salience, or plain noise),
    not a quantity we can name from a correlation table alone. There is
    deliberately no residualization step here: regressing
    year_first_described on the biology features and calling the residual
    "pure fame" would smuggle in a causal claim ("the model uses only fame
    once biology is removed") this analysis cannot support, since that
    residual still contains all of the unmeasured things above.
    """
    print("\n" + "=" * 78)
    print("DESCRIPTIVE CORRELATIONS: year_first_described vs biology features")
    print("=" * 78)
    print(
        "Spearman correlation only. Descriptive, not a decomposition, see\n"
        "the function docstring for why no residualization step follows.\n"
    )
    features = ["pLI", "loeuf", "oe_mis", "oe_lof", "essentiality_score", "tau", "protein_length"]
    print(f"  {'feature':<20}  {'spearman rho':>12}  {'p-value':>10}")
    print("  " + "-" * 46)
    for feat in features:
        rho, pval = spearmanr(df["year_first_described"], df[feat], nan_policy="omit")
        print(f"  {feat:<20}  {rho:>12.3f}  {pval:>10.2e}")
    print("=" * 78)


def print_structural_caveat():
    print()
    print("=" * 65)
    print("STRUCTURAL-VALIDATION CAVEAT")
    print("=" * 65)
    print(
        "Burden features (n_rare, n_lof) now cover all 22 autosomes: 16,725 /\n"
        "19,296 genes (86.7%) have real burden data, up from 388 (2.0%) at\n"
        "the chr22-only validation run. The remaining 2,571 genes are mostly\n"
        "X/Y (896 genes, out of scope for this pipeline) plus genes with no\n"
        "qualifying rare variant in this call set. n_rare importance climbed\n"
        "monotonically as coverage grew (0.0112 -> 0.0352 -> 0.0714 in\n"
        "biology_only, at 2.0% -> 29.3% -> 86.7% coverage), consistent across\n"
        "all four feature-set variants. n_lof remains a much smaller signal,\n"
        "it only just started appearing in top-10 feature importance lists."
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

    # Computed ONCE, shared by every variant, so tercile/median-split group
    # membership is identical across variants (see assign_rank_groups).
    tercile_assignment = assign_rank_groups(df, "pub_count", q=3, labels=["low", "medium", "high"])
    median_assignment  = assign_rank_groups(df, "pub_count", q=2, labels=["bottom_half", "top_half"])

    if args.compare:
        results_by_name = {}
        for name, cols in FEATURE_SETS.items():
            print(f"running variant: {name}  ({len(cols)} features)")
            results_by_name[name] = run_variant(
                name, cols, df, fold_v, n_folds, tercile_assignment, median_assignment, verbose=False
            )

        print("\nrunning repeated CV (10 repeats x 5 folds x 4 variants, this takes a while)...")
        repeated_cv_by_name = {}
        for name, cols in FEATURE_SETS.items():
            print(f"  repeated CV: {name}")
            repeated_cv_by_name[name] = run_repeated_cv(cols, df, tercile_assignment, "low")

        print("\nrunning bootstrap stability selection (50 resamples x 4 variants, DESIGN.md 6.5)...")
        stability_by_name = {}
        for name, cols in FEATURE_SETS.items():
            print(f"  stability selection: {name}")
            stability_by_name[name] = bootstrap_stability_selection(cols, df)

        print_comparison_table(results_by_name, baseline_pr, repeated_cv_by_name, stability_by_name)
        print_descriptive_correlations(df)
        return

    name = args.feature_set
    feature_cols = FEATURE_SETS[name]
    print(f"feature set: {name}  ({len(feature_cols)} features)\n")
    result = run_variant(name, feature_cols, df, fold_v, n_folds, tercile_assignment, median_assignment, verbose=True)
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

    shap_max = max((v for _, v in result["shap_top10_importances"]), default=0.0)
    print(f"\nSHAP importances ({name}, same full-data model, mean |SHAP value|):")
    for feat, imp in result["shap_top10_importances"]:
        bar = "#" * int((imp / shap_max) * 60) if shap_max > 0 else ""
        print(f"  {feat:<20}  {imp:.4f}  {bar}")

    print_gnomad_proxy_check(df, oos)
    print_pubcount_check(result)
    print_structural_caveat()


if __name__ == "__main__":
    main()
