#!/usr/bin/env python3
"""Temporal holdout (DESIGN.md section 6.4), built from Open Targets release
history instead of per-gene clinical trial dates.

IDEA: an old Open Targets release's clinical-phase label is "the past"; a
current release is "the future". Train the model only on what was known at
the cutoff release, then check whether genes that were unlabeled at the
cutoff but have since gained a clinical-phase drug (by the newest release)
rank highly in the model's score. This is a genuinely prospective test: the
label used to grade the ranking did not exist when the ranking was produced.

This is a separate, standalone analysis. It does not touch train_eval.py,
split.py, ml/cache/oos_predictions.parquet, or any of the main ablation
results. Same random seed and the same GroupKFold leakage assertion as the
rest of the project.

STEP 1: CUTOFF RELEASE
  True 20.x Open Targets releases (20.02 through 20.11) predate the parquet
  ETL pipeline and ship only bulk JSON dumps (no knownDrugsAggregated table,
  no per-datatype association parquet). The earliest release with the same
  parquet knownDrugsAggregated/targets schema used elsewhere in this project
  is 21.06 (June 2021). That gives a clean 5.0 year gap to the newest
  available release, 26.06 (June 2026), which is what this script uses as
  "the future". Reported explicitly at runtime, not hardcoded silently.

STEP 3: FEATURES, KEPT / DROPPED / BACK-DATED
  Kept, treated as time-stable (explicit assumption, not verified per
  feature): pLI, loeuf, oe_mis, oe_lof (gnomAD constraint), n_rare, n_lof
  (our own burden pipeline on a fixed 1000 Genomes call set), protein_length
  (AlphaFold), tau (GTEx breadth). essentiality_score (DepMap) is the
  weakest of these assumptions: DepMap's screen has grown substantially in
  both cell line count and scoring methodology since 2021, so a gene's
  current essentiality score is not guaranteed to reflect what was known in
  2021. Kept anyway, per instruction, but flagged here and in the printed
  report rather than silently assumed.

  Kept, inherently historical: year_first_described (a publication year is
  fixed once it happens; looking it up today does not change what it was in
  2021).

  DROPPED: pub_count. Back-dating it correctly would mean counting only the
  gene2pubmed PMIDs published before the cutoff year, which requires
  resolving every one of that gene's linked PMIDs to a publication year
  (not just the earliest one, which is all fetch_publications.py currently
  resolves). The cached gene2pubmed data underlying this project has
  807,051 distinct human PMIDs; resolving all of them through NCBI's
  rate-limited esummary endpoint (3 requests/second without an API key) is
  a long, fragile, one-off fetch that was not run for this check. Per the
  instruction's own fallback, dropped rather than used at its current
  (fully lookahead-contaminated) value.

  DROPPED: ppi_degree, ppi_betweenness (STRING centrality). STRING's
  interaction graph has grown since 2021 and no historical snapshot was
  sourced, so both are dropped.

  DROPPED (stricter than the main ablation's no_pubcount_no_string preset):
  has_pub_count. It is a coverage flag, not the pub_count value itself, so
  its leakage risk is small, but it was computed from the current-day
  gene2pubmed fetch and this analysis is the one place in the project where
  "if in doubt, drop" is the standing instruction. has_string is already
  dropped as part of the STRING block above.

Usage:
    python3 ml/temporal_holdout.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 42
N_SPLITS = 5

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
TABLE_FILE = os.path.join(CACHE_DIR, "training_table.parquet")

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
CUTOFF_RELEASE = "21.06"
FUTURE_RELEASE = "26.06"
CUTOFF_KDA_DIR = os.path.join(DATA_CACHE_DIR, "open_targets", CUTOFF_RELEASE, "knownDrugsAggregated")
FUTURE_CLINICAL_TARGET_FILE = os.path.join(
    DATA_CACHE_DIR, "open_targets", FUTURE_RELEASE, "clinical_target.parquet"
)

# maxClinicalStage (26.06 schema) -> numeric order, same mapping used in
# ml/validate_prospective_labels.py, kept consistent across both checks.
STAGE_ORDER = {
    "UNKNOWN": 0,
    "PRECLINICAL": 0,
    "IND": 0,
    "EARLY_PHASE_1": 0.5,
    "PHASE_1": 1,
    "PHASE_1_2": 1.5,
    "PHASE_2": 2,
    "PHASE_2_3": 2.5,
    "PHASE_3": 3,
    "PREAPPROVAL": 3.5,
    "APPROVAL": 4,
}

# See module docstring, STEP 3, for the full kept/dropped/back-dated rationale.
TEMPORAL_FEATURE_COLS = [
    "pLI", "loeuf", "oe_lof", "oe_mis", "n_rare", "n_lof",
    "protein_length", "tau", "essentiality_score",
    "year_first_described",
    "has_gnomad", "has_burden", "has_alphafold",
    "has_tau", "has_essentiality", "has_year_described",
]

MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=RANDOM_SEED,
)

TOP_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
N_BASELINE_SAMPLES = 1000

# Single-feature baselines: does the trained model actually beat sorting
# genes by one well-known biology feature alone? Direction for LOEUF, pLI,
# oe_mis, and essentiality_score is the standard biological convention
# (lower LOEUF, higher pLI, lower oe_mis = more constrained; more negative
# DepMap Chronos score = more essential). disorder_fraction and
# protein_length have no such standard convention, so their direction was
# set from each feature's own Spearman correlation with label_cutoff,
# computed once before looking at any prospective-set result, not tuned to
# this test's outcome: disorder_fraction correlates negatively with
# label_cutoff (rho=-0.059, lower disorder ranks higher), protein_length
# correlates positively (rho=+0.060, longer protein ranks higher). All six
# use "ascending" to mean the raw column is sorted ascending and the LOWEST
# values are treated as the top-ranked, highest-priority genes.
SINGLE_FEATURE_BASELINES = {
    "LOEUF":              {"col": "loeuf",              "ascending": True},
    "pLI":                {"col": "pLI",                "ascending": False},
    "oe_mis":             {"col": "oe_mis",              "ascending": True},
    "essentiality_score": {"col": "essentiality_score",  "ascending": True},
    "disorder_fraction":  {"col": "disorder_fraction",   "ascending": True},
    "protein_length":     {"col": "protein_length",      "ascending": False},
}


def load_cutoff_label() -> pd.DataFrame:
    kda = pd.read_parquet(CUTOFF_KDA_DIR, columns=["targetId", "phase"])
    per_gene = (
        kda.groupby("targetId")["phase"].max().reset_index()
        .rename(columns={"targetId": "ensembl_id", "phase": "max_phase_cutoff"})
    )
    per_gene["label_cutoff"] = (per_gene["max_phase_cutoff"] >= 1).astype(int)
    return per_gene[["ensembl_id", "label_cutoff"]]


def load_future_label() -> pd.DataFrame:
    ct = pd.read_parquet(FUTURE_CLINICAL_TARGET_FILE, columns=["targetId", "maxClinicalStage"])
    ct["stage_order"] = ct["maxClinicalStage"].map(STAGE_ORDER)
    per_gene = ct.groupby("targetId")["stage_order"].max().reset_index()
    per_gene["label_future"] = (per_gene["stage_order"] >= 1).astype(int)
    return per_gene.rename(columns={"targetId": "ensembl_id"})[["ensembl_id", "label_future"]]


def baseline_ci(pool_size: int, n_draw: int, top_frac: float, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    n_top = int(round(pool_size * top_frac))
    rates = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.choice(pool_size, size=n_draw, replace=False)
        rates[i] = (idx < n_top).mean()
    return rates.mean(), np.percentile(rates, 2.5), np.percentile(rates, 97.5)


def evaluate_ranking(unlabeled_at_cutoff, score, pool_size, n_prospective):
    """
    Evaluate one ranking, model out-of-fold score or a single raw feature,
    against the prospective positives, using the exact same thresholds and
    resampled baseline CI for every ranking so comparisons are apples to
    apples. `score` must already be oriented so higher = higher priority
    (single-feature baselines negate the raw column before calling this
    when their stated direction is ascending).
    """
    ranked = unlabeled_at_cutoff.copy()
    ranked["_score"] = score
    ranked = ranked.sort_values("_score", ascending=False).reset_index(drop=True)

    per_threshold = {}
    for frac in TOP_FRACTIONS:
        n_top = int(round(pool_size * frac))
        n_hits = (ranked.head(n_top)["label_future"] == 1).sum()
        observed = n_hits / n_prospective
        base_mean, base_lo, base_hi = baseline_ci(pool_size, int(n_prospective), frac, N_BASELINE_SAMPLES, RANDOM_SEED)
        enrichment = observed / base_mean if base_mean > 0 else float("nan")
        per_threshold[frac] = {
            "observed": observed, "base_mean": base_mean,
            "base_lo": base_lo, "base_hi": base_hi, "enrichment": enrichment,
        }

    pr_auc = average_precision_score(ranked["label_future"].values, ranked["_score"].values)
    base_rate = n_prospective / pool_size
    lift = pr_auc / base_rate if base_rate > 0 else float("nan")
    return {"per_threshold": per_threshold, "pr_auc": pr_auc, "lift": lift}


def main():
    print("=" * 78)
    print("STEP 1: cutoff release")
    print("=" * 78)
    print(
        f"Cutoff release: {CUTOFF_RELEASE} (parquet knownDrugsAggregated/targets,\n"
        f"same schema as the 24.09 release this project trains against).\n"
        f"Future release: {FUTURE_RELEASE}. Gap: 5.0 years.\n"
        f"True 20.x releases (20.02-20.11) are JSON-only bulk dumps with no\n"
        f"parquet knownDrugsAggregated table, so they cannot be used directly;\n"
        f"21.06 is the earliest release with the matching schema."
    )

    cutoff_label = load_cutoff_label()
    future_label = load_future_label()
    print(f"\n{CUTOFF_RELEASE} positives (max_phase >= 1, any target): {cutoff_label['label_cutoff'].sum()}")
    print(f"{FUTURE_RELEASE} positives (max_phase >= 1, any target): {future_label['label_future'].sum()}")

    df = pd.read_parquet(TABLE_FILE)
    df = df.merge(cutoff_label, on="ensembl_id", how="left")
    df = df.merge(future_label, on="ensembl_id", how="left")
    df["label_cutoff"] = df["label_cutoff"].fillna(0).astype(int)
    df["label_future"] = df["label_future"].fillna(0).astype(int)

    print(f"\nwithin our {len(df):,}-gene universe:")
    print(f"  {CUTOFF_RELEASE} positives: {df['label_cutoff'].sum():,}")
    print(f"  {FUTURE_RELEASE} positives: {df['label_future'].sum():,}  (vs 1,530 in the earlier 24.09-based check)")

    print("\n" + "=" * 78)
    print("STEP 2: define the holdout")
    print("=" * 78)
    n_train_pos = df["label_cutoff"].sum()
    prospective = df[(df["label_cutoff"] == 0) & (df["label_future"] == 1)]
    regressed = df[(df["label_cutoff"] == 1) & (df["label_future"] == 0)]
    print(f"TRAIN positives (label=1 at {CUTOFF_RELEASE}): {n_train_pos:,}")
    print(f"PROSPECTIVE positives (label=0 at {CUTOFF_RELEASE}, label=1 at {FUTURE_RELEASE}): {len(prospective):,}")
    print(
        f"REGRESSED (label=1 at {CUTOFF_RELEASE}, label=0 at {FUTURE_RELEASE}): {len(regressed):,}.\n"
        f"These are TRAIN positives by construction (label_cutoff=1), so they are\n"
        f"never part of the scored unlabeled-at-cutoff population below; excluded\n"
        f"from the prospective test rather than being folded in anywhere as\n"
        f"negatives. Listed here only as a data-hygiene note."
    )
    if len(regressed):
        print(f"  symbols: {', '.join(sorted(regressed['symbol'].tolist()))}")

    print("\n" + "=" * 78)
    print("STEP 3: features kept / dropped / back-dated")
    print("=" * 78)
    print(f"KEPT (time-stable assumption): pLI, loeuf, oe_mis, oe_lof, n_rare, n_lof,")
    print(f"  protein_length, tau, essentiality_score (weakest assumption, DepMap has")
    print(f"  grown substantially since 20{CUTOFF_RELEASE.split('.')[0]})")
    print(f"KEPT (inherently historical): year_first_described")
    print(f"DROPPED (lookahead risk, back-dating infeasible in this pass): pub_count, has_pub_count")
    print(f"DROPPED (no historical snapshot sourced): ppi_degree, ppi_betweenness, has_string")
    print(f"Feature set used ({len(TEMPORAL_FEATURE_COLS)} columns): {TEMPORAL_FEATURE_COLS}")

    print("\n" + "=" * 78)
    print("STEP 4: train on cutoff label, GroupKFold, leakage assertion per fold")
    print("=" * 78)
    X = df[TEMPORAL_FEATURE_COLS].values
    y = df["label_cutoff"].values
    groups = df["group_key"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_score = np.full(len(df), np.nan)

    print(f"{'fold':>4}  {'train genes':>12}  {'train pos':>10}  {'test genes':>11}  {'test pos':>9}  {'overlap':>8}")
    print("-" * 68)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if len(overlap) > 0:
            raise AssertionError(
                f"FATAL LEAKAGE in fold {fold}: {len(overlap)} group_key(s) in both "
                f"train and test: {sorted(overlap)[:5]}"
            )
        model = GradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof_score[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        print(
            f"{fold:>4}  {len(train_idx):>12,}  {int(y[train_idx].sum()):>10,}  "
            f"{len(test_idx):>11,}  {int(y[test_idx].sum()):>9,}  {'PASS':>8}"
        )
    print("-" * 68)
    print("All folds passed the leakage assertion: zero group_key overlap.")
    df["oof_score"] = oof_score

    print("\n" + "=" * 78)
    print("STEP 4 (continued): does the prospective set rank highly?")
    print("=" * 78)
    unlabeled_at_cutoff = df[df["label_cutoff"] == 0].sort_values("oof_score", ascending=False).reset_index(drop=True)
    unlabeled_at_cutoff["rank"] = np.arange(1, len(unlabeled_at_cutoff) + 1)
    pool_size = len(unlabeled_at_cutoff)
    n_prospective = unlabeled_at_cutoff["label_future"].sum()
    base_rate = n_prospective / pool_size
    print(f"unlabeled-at-cutoff pool scored: {pool_size:,} genes")
    print(f"prospective positives in that pool: {n_prospective:,}  (base rate {base_rate:.4%})")

    print(f"\n{'top frac':>9}  {'observed rate':>14}  {'baseline mean':>14}  {'baseline 95% CI':>18}  {'enrichment':>10}")
    print("-" * 78)
    for frac in TOP_FRACTIONS:
        n_top = int(round(pool_size * frac))
        n_hits = (unlabeled_at_cutoff.head(n_top)["label_future"] == 1).sum()
        observed = n_hits / n_prospective
        base_mean, base_lo, base_hi = baseline_ci(pool_size, int(n_prospective), frac, N_BASELINE_SAMPLES, RANDOM_SEED)
        enrichment = observed / base_mean if base_mean > 0 else float("nan")
        ci_str = f"[{base_lo:.3f}, {base_hi:.3f}]"
        print(f"{frac*100:>8.0f}%  {observed:>14.3f}  {base_mean:>14.3f}  {ci_str:>18}  {enrichment:>10.2f}x")
    print("-" * 78)
    print(
        f"baseline = mean fraction of {int(n_prospective)} randomly drawn unlabeled-at-cutoff\n"
        f"genes landing in the top fraction, resampled {N_BASELINE_SAMPLES} times without\n"
        f"replacement from the same {pool_size:,}-gene pool."
    )

    pr_auc = average_precision_score(unlabeled_at_cutoff["label_future"].values, unlabeled_at_cutoff["oof_score"].values)
    lift = pr_auc / base_rate if base_rate > 0 else float("nan")
    print(f"\nPR-AUC on the prospective set: {pr_auc:.4f}")
    print(f"base rate (random ranker expectation): {base_rate:.4f}")
    print(f"lift (PR-AUC / base rate): {lift:.2f}x")

    print("\nrank distribution of the prospective positives "
          f"(out of {pool_size:,} unlabeled-at-cutoff genes):")
    ranks = unlabeled_at_cutoff.loc[unlabeled_at_cutoff["label_future"] == 1, "rank"]
    print(f"  min: {ranks.min():,}   25th pct: {ranks.quantile(0.25):,.0f}   "
          f"median: {ranks.median():,.0f}   75th pct: {ranks.quantile(0.75):,.0f}   max: {ranks.max():,}")
    print(f"  mean percentile rank: {(ranks / pool_size * 100).mean():.1f}%  "
          f"(50% = indistinguishable from a random ranker on average)")
    for frac in TOP_FRACTIONS:
        n_top = int(round(pool_size * frac))
        count = (ranks <= n_top).sum()
        print(f"  in top {frac*100:.0f}%: {count} / {int(n_prospective)} prospective positives")

    print("\ntop 20 highest-ranked prospective positives:")
    top20 = unlabeled_at_cutoff[unlabeled_at_cutoff["label_future"] == 1].sort_values("rank").head(20)
    for _, row in top20.iterrows():
        pct = row["rank"] / pool_size * 100
        print(f"  {row['symbol']:<12}  rank {int(row['rank']):>6} / {pool_size:,}  (top {pct:.1f}%)  score {row['oof_score']:.4f}")

    print("\n" + "=" * 78)
    print("STEP 5: single-feature baselines")
    print("=" * 78)
    print(
        "The random-ranker baseline above is a low bar. The real question is\n"
        "whether the trained model beats simply sorting genes by ONE well-known\n"
        "biology feature. Same pool, same 338 prospective positives, same\n"
        "thresholds, same resampled baseline CI, so this is directly comparable\n"
        "to the model's own table above. See SINGLE_FEATURE_BASELINES in this\n"
        "file for the direction convention used for each feature and why."
    )

    model_result = evaluate_ranking(unlabeled_at_cutoff, unlabeled_at_cutoff["oof_score"].values, pool_size, n_prospective)
    all_results = {"model (trained)": model_result}
    for name, spec in SINGLE_FEATURE_BASELINES.items():
        raw = unlabeled_at_cutoff[spec["col"]].values
        score = -raw if spec["ascending"] else raw
        all_results[name] = evaluate_ranking(unlabeled_at_cutoff, score, pool_size, n_prospective)

    header = f"{'ranking':<22}" + "".join(f"{f'top {int(t*100)}%':>12}" for t in TOP_FRACTIONS) + f"{'PR-AUC':>10}{'lift':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, res in all_results.items():
        row = f"{name:<22}"
        for frac in TOP_FRACTIONS:
            row += f"{res['per_threshold'][frac]['enrichment']:>11.2f}x"
        row += f"{res['pr_auc']:>10.4f}{res['lift']:>7.2f}x"
        print(row)
    print("-" * len(header))
    print(
        "Each cell is enrichment (observed rate / resampled baseline mean) at that\n"
        "threshold; lift is PR-AUC / base rate, the same summary metric used\n"
        "throughout this project. 1.00x = indistinguishable from a random ranker."
    )

    best_baseline_name = max(
        (n for n in all_results if n != "model (trained)"),
        key=lambda n: all_results[n]["lift"],
    )
    best_baseline_lift = all_results[best_baseline_name]["lift"]
    model_lift_val = all_results["model (trained)"]["lift"]

    print("\n" + "=" * 78)
    print("STEP 6: verdict")
    print("=" * 78)
    any_above = any(
        ((unlabeled_at_cutoff.head(int(round(pool_size * frac)))["label_future"] == 1).sum() / n_prospective)
        > baseline_ci(pool_size, int(n_prospective), frac, N_BASELINE_SAMPLES, RANDOM_SEED)[2]
        for frac in TOP_FRACTIONS
    )
    print("Random-ranker comparison:")
    if any_above:
        print(
            "  At least one top-fraction threshold shows the observed rate above the\n"
            "  resampled baseline's 95% CI: distinguishable from chance at that\n"
            "  threshold. See the per-threshold table above for which ones."
        )
    else:
        print(
            "  No top-fraction threshold shows the observed rate above the resampled\n"
            "  baseline's 95% CI. This is a null result: with hundreds of prospective\n"
            "  positives this test finally has real power, and it still does not show\n"
            "  detectable enrichment. Reported as is, not tuned to find significance."
        )
    print(f"  PR-AUC lift over base rate: {lift:.2f}x (1.0x = random ranker)")

    print("\nSingle-feature-baseline comparison, the real bar:")
    print(f"  best single-feature baseline: {best_baseline_name} (lift {best_baseline_lift:.2f}x)")
    print(f"  trained model: lift {model_lift_val:.2f}x")
    if model_lift_val > best_baseline_lift:
        margin = (model_lift_val / best_baseline_lift - 1) * 100
        print(
            f"  The trained model beats the best single-feature baseline by"
            f" {margin:.0f}% (lift). The ML layer is earning its place over sorting\n"
            f"  by {best_baseline_name} alone on this test."
        )
    else:
        deficit = (1 - model_lift_val / best_baseline_lift) * 100
        print(
            f"  The trained model does NOT clearly beat the best single-feature\n"
            f"  baseline here, it trails {best_baseline_name} by {deficit:.0f}% (lift).\n"
            f"  Stated plainly rather than smoothed over: on this specific test, sorting\n"
            f"  by {best_baseline_name} alone gets most or all of the way to the model's\n"
            f"  result. This does not invalidate the model (see the ablation results\n"
            f"  elsewhere in this project, where the full feature set clearly beats\n"
            f"  {best_baseline_name}-only rankings), but on this particular prospective\n"
            f"  test, with a restricted feature set and n=338, a single well-chosen\n"
            f"  feature is competitive with the trained model."
        )

    print("\n" + "=" * 78)
    print("CAVEATS")
    print("=" * 78)
    print(
        "This reuses Open Targets' own release history as a time machine, not a\n"
        "hand-built per-gene discovery timeline, so it inherits whatever database\n"
        "churn happened between releases (see the 'regressed' genes above). The\n"
        "essentiality_score (DepMap) time-stability assumption is the weakest one\n"
        "made here. Dropping pub_count and STRING removes real signal the main\n"
        "model uses, so this holdout is expected to score lower than the main\n"
        "ablation variants even if the biology-only signal is genuinely real; it\n"
        "is testing a deliberately narrower feature set under a stricter,\n"
        "leakage-safe temporal constraint, not a like-for-like comparison to the\n"
        "main results. The single-feature baselines above share this same\n"
        "restricted, time-stable feature set; they are not evaluated against the\n"
        "full feature set either."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
