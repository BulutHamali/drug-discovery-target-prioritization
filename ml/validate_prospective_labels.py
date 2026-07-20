#!/usr/bin/env python3
"""Prospective validation: genes that were UNLABELED under Open Targets 24.09
(the release the model trained and was labeled against) but have since gained
a clinical-phase drug in a newer Open Targets release. Does the model's
biology_only pooled out-of-fold score rank those genes higher than chance,
among the genes that were unlabeled at training time?

This is stronger than a correlational check because the label used here did
not exist when the model was trained or evaluated: it is a genuinely
prospective test, not a re-slicing of the same data. It does not add
features or retrain anything; it reads the existing pooled OOS predictions
for biology_only (ml/cache/oos_predictions.parquet, produced by
    python3 ml/train_eval.py --feature-set biology_only
) and two small Open Targets clinical tables (24.09 and a newer release,
fetched fresh for this check only).

CAVEAT, repeated in the output: the gap between releases, even the largest
gap available on the EBI FTP right now, is short relative to how long it
takes a gene to move from no clinical activity to a clinical-phase drug. Few
genes will have changed status in that window, so this test is underpowered
by construction and a null result is expected and fine. Also, Open Targets
recording a drug at a given release may reflect a program that was already
underway before that release, not a new discovery driven by anything in this
project.

Usage:
    python3 ml/validate_prospective_labels.py
"""

import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
OOS_FILE = os.path.join(CACHE_DIR, "oos_predictions.parquet")
TRAINING_FILE = os.path.join(CACHE_DIR, "training_table.parquet")

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
NEW_RELEASE = "26.06"
CLINICAL_TARGET_FILE = os.path.join(
    DATA_CACHE_DIR, "open_targets", NEW_RELEASE, "clinical_target.parquet"
)

# maxClinicalStage (26.06 schema) -> a numeric order so we can take a
# per-gene max and threshold at >= 1, matching the original label definition
# in ml/build_features.py (max_phase >= 1). EARLY_PHASE_1 is mapped below 1,
# same as the phase=0.5 "preclinical nomination" rows the original label
# definition explicitly excludes.
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

TOP_FRACTIONS = [0.05, 0.10, 0.20]
N_BASELINE_SAMPLES = 1000
RANDOM_SEED = 42

PREVIOUS_TOP20 = [
    "UBL5", "FBN1", "SRSF2", "PCNA", "PHF5A", "BUD31", "PRPF38A", "SRSF3",
    "SNRPA1", "MT-CO3", "SNU13", "RACK1", "PRELID1", "PTEN", "PAFAH1B1",
    "NRXN2", "SRSF1", "KCNMA1", "NUTF2", "MT-CYB",
]


def load_new_release_label() -> pd.DataFrame:
    ct = pd.read_parquet(CLINICAL_TARGET_FILE, columns=["targetId", "maxClinicalStage"])
    ct["stage_order"] = ct["maxClinicalStage"].map(STAGE_ORDER)
    gene_max = ct.groupby("targetId")["stage_order"].max().reset_index()
    gene_max["label_new"] = (gene_max["stage_order"] >= 1).astype(int)
    return gene_max.rename(columns={"targetId": "ensembl_id"})[["ensembl_id", "label_new"]]


def baseline_ci(pool_size: int, n_draw: int, top_frac: float, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    n_top = int(round(pool_size * top_frac))
    ranks = np.arange(pool_size)
    rates = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.choice(pool_size, size=n_draw, replace=False)
        rates[i] = (idx < n_top).mean()
    return rates.mean(), np.percentile(rates, 2.5), np.percentile(rates, 97.5)


def main():
    print("=" * 78)
    print("STEP 1: available Open Targets releases")
    print("=" * 78)
    print(
        "EBI FTP (https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/) lists\n"
        "quarterly releases from 16.04 through 26.06 plus 'latest' and 'master'.\n"
        "Confirmed quarterly releases from 24.09 onward: 24.09, 25.03, 25.06,\n"
        "25.09, 25.12, 26.03, 26.06. 25.03 restructured the layout (snake_case,\n"
        "singular dataset names, single parquet files instead of partitioned\n"
        "directories for most tables). Using 26.06, the newest available, about\n"
        "seven quarterly releases (roughly 1.75 years) after the 24.09 release\n"
        "this project trained and labeled against. This is NOT a marginal gap,\n"
        "so the analysis below proceeds rather than stopping early."
    )

    print("\n" + "=" * 78)
    print("STEP 2: gene-level label from the newer release")
    print("=" * 78)
    label_new = load_new_release_label()
    print(f"clinical_target.parquet ({NEW_RELEASE}): {label_new['ensembl_id'].nunique()} targets with any clinical record")
    print(f"positives (max stage order >= 1, i.e. phase >= 1): {label_new['label_new'].sum()}")

    if not os.path.exists(OOS_FILE):
        raise SystemExit(
            f"{OOS_FILE} not found. Run: python3 ml/train_eval.py --feature-set biology_only"
        )
    oos = pd.read_parquet(OOS_FILE)
    training = pd.read_parquet(TRAINING_FILE, columns=["symbol", "ensembl_id", "label"])
    merged = training.merge(label_new, on="ensembl_id", how="left")
    merged["label_new"] = merged["label_new"].fillna(0).astype(int)

    print("\n" + "=" * 78)
    print("STEP 3: newly labeled genes (label=0 in 24.09, label=1 in 26.06)")
    print("=" * 78)
    newly = merged[(merged["label"] == 0) & (merged["label_new"] == 1)].copy()
    regressed = merged[(merged["label"] == 1) & (merged["label_new"] == 0)]
    print(f"24.09 positives in our universe: {merged['label'].sum()}")
    print(f"26.06 positives in our universe: {merged['label_new'].sum()}")
    print(f"NEWLY LABELED (0 -> 1): {len(newly)}")
    if len(newly) <= 9:
        print("This is a single-digit count. It caps how much this test can tell us")
        print("no matter which way the enrichment result comes out.")
    print(f"REGRESSED (1 -> 0, data-hygiene check, not the main test): {len(regressed)}")
    if len(regressed):
        print(f"  symbols: {', '.join(sorted(regressed['symbol'].tolist()))}")
        print("  Plausible causes: OT redefining/dropping a drug-target mechanism")
        print("  record, gene ID remapping, or evidence pruning between releases.")
        print("  Not investigated further; irrelevant to the newly-labeled test below.")

    print("\n" + "=" * 78)
    print("STEP 4: are newly labeled genes enriched among high-scoring unlabeled genes?")
    print("=" * 78)
    unlabeled = oos[oos["label"] == 0].sort_values("score", ascending=False).reset_index(drop=True)
    unlabeled["rank"] = np.arange(1, len(unlabeled) + 1)
    pool_size = len(unlabeled)
    print(f"unlabeled pool at training time (24.09 label=0): {pool_size} genes")

    newly_ranked = newly.merge(unlabeled[["symbol", "score", "rank"]], on="symbol", how="left")
    missing = newly_ranked["rank"].isna().sum()
    if missing:
        print(f"WARNING: {missing} newly labeled gene(s) not found in the unlabeled OOS pool, dropping")
        newly_ranked = newly_ranked.dropna(subset=["rank"])
    n_newly = len(newly_ranked)

    print(f"\n{'top frac':>10}  {'observed rate':>14}  {'baseline mean':>14}  {'baseline 95% CI':>18}  {'enrichment':>10}")
    print("-" * 78)
    for frac in TOP_FRACTIONS:
        n_top = int(round(pool_size * frac))
        observed = (newly_ranked["rank"] <= n_top).mean()
        base_mean, base_lo, base_hi = baseline_ci(pool_size, n_newly, frac, N_BASELINE_SAMPLES, RANDOM_SEED)
        enrichment = observed / base_mean if base_mean > 0 else float("nan")
        ci_str = f"[{base_lo:.3f}, {base_hi:.3f}]"
        print(f"{frac*100:>9.0f}%  {observed:>14.3f}  {base_mean:>14.3f}  {ci_str:>18}  {enrichment:>10.2f}x")
    print("-" * 78)
    print(
        f"baseline = mean fraction of {n_newly} randomly drawn unlabeled genes\n"
        f"landing in the top fraction, resampled {N_BASELINE_SAMPLES} times without\n"
        f"replacement from the same {pool_size}-gene pool. By definition this should\n"
        f"average to the top fraction itself (5%, 10%, 20%); the CI reflects pure\n"
        f"sampling variance at n={n_newly}, which is wide given how few newly labeled\n"
        f"genes exist."
    )

    print("\n" + "=" * 78)
    print(f"ranks of all {n_newly} newly labeled genes (out of {pool_size} unlabeled genes)")
    print("=" * 78)
    newly_ranked_sorted = newly_ranked.sort_values("rank")
    for _, row in newly_ranked_sorted.iterrows():
        pct = row["rank"] / pool_size * 100
        print(f"  {row['symbol']:<12}  rank {int(row['rank']):>6} / {pool_size}  (top {pct:.1f}%)  score {row['score']:.4f}")

    print("\n" + "=" * 78)
    print("STEP 5: did any previous top-20 unlabeled gene gain a drug in 26.06?")
    print("=" * 78)
    prev_check = merged[merged["symbol"].isin(PREVIOUS_TOP20)][["symbol", "label", "label_new"]]
    gained = prev_check[(prev_check["label"] == 0) & (prev_check["label_new"] == 1)]
    if len(gained):
        print(f"YES: {len(gained)} of the previous top-20 unlabeled genes gained a clinical-phase drug:")
        for s in gained["symbol"]:
            print(f"  {s}")
    else:
        print("No gene from the previous top-20 unlabeled list gained a clinical-phase drug in 26.06.")
    print("\nfull status of the previous top-20 list:")
    for s in PREVIOUS_TOP20:
        row = merged[merged["symbol"] == s]
        if row.empty:
            print(f"  {s:<12}  not found in universe")
            continue
        lbl24, lbl26 = row.iloc[0]["label"], row.iloc[0]["label_new"]
        status = "GAINED A DRUG" if (lbl24 == 0 and lbl26 == 1) else ("still labeled" if lbl24 == 1 else "still unlabeled")
        print(f"  {s:<12}  24.09 label={lbl24}  26.06 label={lbl26}  ({status})")

    print("\n" + "=" * 78)
    print("CAVEATS")
    print("=" * 78)
    print(
        "Even the largest available release gap (24.09 to 26.06, about 1.75\n"
        "years) is short relative to how long it takes a gene to move from no\n"
        "clinical activity to a clinical-phase drug, so this test is underpowered\n"
        "by construction. A null result at any individual top-fraction threshold\n"
        "is expected and does not contradict the rest of this project's findings.\n"
        "Also, a gene appearing with a clinical-phase drug in a newer Open Targets\n"
        "release may reflect a drug program that was already underway well before\n"
        "that release and is simply now indexed, not a new discovery prompted by\n"
        "anything in this project."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
