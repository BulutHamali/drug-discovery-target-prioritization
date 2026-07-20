#!/usr/bin/env python3
"""Weak orthogonal check: do top-ranked UNLABELED genes carry independent
human genetic disease evidence (Open Targets Genetics associationByDatatypeDirect,
datatypeId == 'genetic_association'), more often than a random unlabeled gene?

This does not add features, retrain, or touch the model in any way. It reads
the existing pooled out-of-fold predictions for the biology_only variant
(ml/cache/oos_predictions.parquet, produced by:
    python3 ml/train_eval.py --feature-set biology_only
) and an Open Targets bulk table fetched fresh for this check only.

Why biology_only: it is the variant with no publication-count or STRING
features, so genetic evidence (itself correlated with study effort, see the
caveat below) cannot have leaked in via those columns. Any enrichment found
here has to come from the mechanistic biology features (gnomAD constraint,
essentiality, expression breadth, protein length, burden).

CAVEAT, stated plainly and repeated in the output: this is a weak orthogonal
check, not validation of drug targets. Genes with independent genetic disease
evidence are more likely to already attract drug discovery programs, so a
positive result here means the model ranks toward genes the field would find
interesting, not that those genes are druggable or that a drug program on
them would succeed.

Usage:
    python3 ml/validate_genetic_evidence.py
"""

import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
OOS_FILE = os.path.join(CACHE_DIR, "oos_predictions.parquet")

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
OT_RELEASE = "24.09"
ASSOC_DIR = os.path.join(DATA_CACHE_DIR, "open_targets", OT_RELEASE, "associationByDatatypeDirect")
TARGETS_DIR = os.path.join(DATA_CACHE_DIR, "open_targets", OT_RELEASE, "targets")

# Threshold chosen from the observed score distribution: 0.5 sits close to
# the median score among genes that have ANY genetic_association evidence at
# all, and splits our 19,296-gene universe roughly in half (52.0%), which
# gives the baseline comparison real statistical power in both directions.
# It is not tuned to produce a particular result; it is fixed once here from
# the score distribution before looking at the top-N genes at all.
EVIDENCE_THRESHOLD = 0.5

TOP_N_LEVELS = [50, 100, 500]
N_BASELINE_SAMPLES = 1000
RANDOM_SEED = 42


def load_gene_level_genetic_evidence() -> pd.DataFrame:
    """Return one row per gene symbol: max genetic_association score across
    all diseases (the strongest independent genetic evidence for that gene),
    0.0 for genes with no genetic_association row at all."""
    assoc = pd.read_parquet(ASSOC_DIR, columns=["targetId", "datatypeId", "score"])
    ga = assoc[assoc["datatypeId"] == "genetic_association"]
    gene_max = (
        ga.groupby("targetId")["score"]
        .max()
        .rename("ga_score")
        .reset_index()
        .rename(columns={"targetId": "id"})
    )

    targets = pd.read_parquet(TARGETS_DIR, columns=["id", "approvedSymbol"])
    gene_max = gene_max.merge(targets, on="id", how="left")
    return gene_max[["approvedSymbol", "ga_score"]].rename(columns={"approvedSymbol": "symbol"})


def baseline_ci(unlabeled: pd.DataFrame, n: int, threshold: float, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    rates = np.empty(n_samples)
    pool = unlabeled["has_evidence"].to_numpy()
    pool_size = len(pool)
    for i in range(n_samples):
        idx = rng.choice(pool_size, size=n, replace=False)
        rates[i] = pool[idx].mean()
    return rates.mean(), np.percentile(rates, 2.5), np.percentile(rates, 97.5)


def main():
    if not os.path.exists(OOS_FILE):
        raise SystemExit(
            f"{OOS_FILE} not found. Run: python3 ml/train_eval.py --feature-set biology_only"
        )

    oos = pd.read_parquet(OOS_FILE)
    print(f"loaded pooled OOS predictions: {len(oos)} genes")

    evidence = load_gene_level_genetic_evidence()
    oos = oos.merge(evidence, on="symbol", how="left")
    oos["ga_score"] = oos["ga_score"].fillna(0.0)
    oos["has_evidence"] = oos["ga_score"] >= EVIDENCE_THRESHOLD

    unlabeled = oos[oos["label"] == 0].sort_values("score", ascending=False).reset_index(drop=True)
    n_unlabeled = len(unlabeled)
    n_with_evidence = unlabeled["has_evidence"].sum()

    print("\n" + "=" * 78)
    print("ORTHOGONAL VALIDATION: genetic evidence for top-ranked unlabeled genes")
    print("=" * 78)
    print(f"variant: biology_only (no pub_count, no STRING; see module docstring for why)")
    print(f"evidence source: Open Targets {OT_RELEASE}, associationByDatatypeDirect,")
    print(f"                 datatypeId == 'genetic_association', gene-level score =")
    print(f"                 max over all diseases for that gene")
    print(f"evidence threshold: ga_score >= {EVIDENCE_THRESHOLD}")
    print(f"unlabeled gene pool: {n_unlabeled} genes, {n_with_evidence} ({n_with_evidence/n_unlabeled*100:.1f}%) at/above threshold")

    print("\n" + "-" * 78)
    print(f"{'top-N':>8}  {'n_with_evidence':>15}  {'top-N rate':>10}  {'baseline mean':>14}  {'baseline 95% CI':>18}  {'enrichment':>10}")
    print("-" * 78)

    results = {}
    for n in TOP_N_LEVELS:
        top = unlabeled.head(n)
        top_rate = top["has_evidence"].mean()
        top_n_with = top["has_evidence"].sum()

        base_mean, base_lo, base_hi = baseline_ci(
            unlabeled, n, EVIDENCE_THRESHOLD, N_BASELINE_SAMPLES, RANDOM_SEED
        )
        enrichment = top_rate / base_mean if base_mean > 0 else float("nan")
        results[n] = (top_rate, base_mean, base_lo, base_hi, enrichment)

        ci_str = f"[{base_lo:.3f}, {base_hi:.3f}]"
        print(f"{n:>8}  {top_n_with:>15}  {top_rate:>10.3f}  {base_mean:>14.3f}  {ci_str:>18}  {enrichment:>10.2f}x")

    print("-" * 78)
    print(
        f"baseline = mean fraction with evidence >= {EVIDENCE_THRESHOLD} across "
        f"{N_BASELINE_SAMPLES} random samples of size N drawn without replacement\n"
        f"from the full unlabeled pool ({n_unlabeled} genes). CI is the 2.5/97.5\n"
        f"percentile of that resampling distribution, not a model-based interval."
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for n in TOP_N_LEVELS:
        top_rate, base_mean, base_lo, base_hi, enrichment = results[n]
        if top_rate > base_hi:
            verdict = "ABOVE baseline 95% CI: enrichment, top genes carry more genetic evidence than chance"
        elif top_rate < base_lo:
            verdict = "BELOW baseline 95% CI: depletion, unexpected, worth a second look"
        else:
            verdict = "WITHIN baseline 95% CI: no detectable enrichment at this N"
        print(f"top-{n}: top-N rate {top_rate:.3f} vs baseline {base_mean:.3f} [{base_lo:.3f}, {base_hi:.3f}] -> {verdict}")

    print("\n" + "=" * 78)
    print("TOP 20 UNLABELED GENES (biology_only score, genetic evidence)")
    print("=" * 78)
    print(f"{'rank':>4}  {'symbol':<12}  {'score':>8}  {'ga_score':>9}  {'evidence >= ' + str(EVIDENCE_THRESHOLD):>14}")
    for i, row in unlabeled.head(20).iterrows():
        print(f"{i+1:>4}  {row['symbol']:<12}  {row['score']:>8.4f}  {row['ga_score']:>9.3f}  {str(bool(row['has_evidence'])):>14}")

    print("\n" + "=" * 78)
    print("CAVEAT")
    print("=" * 78)
    print(
        "This is a weak orthogonal check, not validation of drug targets. Genes\n"
        "with independent genetic disease evidence are more likely to already\n"
        "attract drug discovery programs, so this measures whether the model\n"
        "ranks toward genes the field would find interesting, not whether they\n"
        "are druggable. It does not confirm or refute the study-bias question\n"
        "addressed elsewhere in this project (biology_only vs all_features); it\n"
        "is a separate, independent, and much weaker signal."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
