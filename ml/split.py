"""
GroupKFold split on group_key for leakage-safe cross-validation.

The whole point of this file is to PROVE that no gene family spans a fold
boundary. If any family appears in both train and test, the split leaks
paralog features across the boundary and the evaluation is invalid. The
assertion inside the fold loop is the complete implementation of DESIGN.md
section 6.1.

Why GroupKFold and not random KFold:
  Paralogs share sequence, domain, and constraint features. A random split
  puts paralogs on both sides: the model "sees" a paralog in train and its
  close relative in test, inflating the test score without the model having
  learned anything generalisable. GroupKFold guarantees an entire family
  lands in exactly one fold. Singletons (solo_ keys) each form their own
  group of one, so they split independently -- which is correct because a
  gene with no known family has no paralog to leak through.

Fold size imbalance is expected. The largest HGNC family has 763 members.
Whichever fold it lands in will have more genes than the others. This is a
correct consequence of respecting family boundaries, not a bug.

Output: ml/cache/cv_folds.parquet
  columns: symbol, group_key, label, fold_idx
  fold_idx in [0 .. N_SPLITS-1] marks which fold the gene belongs to as the
  TEST set. train_eval.py reads this file to reconstruct the same splits
  without re-running the splitter.

Run:  python3 ml/split.py
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import GroupKFold

N_SPLITS    = 5
RANDOM_SEED = 42  # GroupKFold is deterministic on sorted groups, but kept for explicitness

CACHE_DIR  = os.environ.get("ML_CACHE_DIR", "ml/cache")
TABLE_FILE = os.path.join(CACHE_DIR, "training_table.parquet")
FOLDS_FILE = os.path.join(CACHE_DIR, "cv_folds.parquet")

# Feature columns: must match build_features.py and train_eval.py.
# Defined here so split.py can pass a correctly shaped X to GroupKFold.split().
FEATURE_COLS = ["pLI", "loeuf", "oe_lof", "oe_mis", "n_rare", "n_lof", "has_gnomad", "has_burden"]


def main():
    if not os.path.exists(TABLE_FILE):
        print(f"ERROR: {TABLE_FILE} not found. Run ml/build_features.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(TABLE_FILE)
    print(f"loaded training table: {len(df):,} genes, {df['label'].sum():,} positives "
          f"({df['label'].mean():.2%})")
    print(f"unique group_keys: {df['group_key'].nunique():,}")

    X      = df[FEATURE_COLS].values   # GroupKFold only uses shape, not values
    y      = df["label"].values
    groups = df["group_key"].values

    gkf = GroupKFold(n_splits=N_SPLITS)

    # fold_idx[i] = which fold gene i belongs to as the TEST set.
    fold_idx_arr = [-1] * len(df)
    fold_stats   = []

    print(f"\nGroupKFold  n_splits={N_SPLITS}")
    print(
        f"{'Fold':>4}  {'Train genes':>12}  {'Train pos':>10}  "
        f"{'Test genes':>11}  {'Test pos':>9}  {'Test pos%':>9}  "
        f"{'Train grps':>11}  {'Test grps':>10}"
    )
    print("-" * 85)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_groups = set(groups[train_idx])
        test_groups  = set(groups[test_idx])

        # THE CRITICAL ASSERTION.
        # GroupKFold is supposed to guarantee zero overlap, but we verify it
        # explicitly because this is the entire point of the file. If this ever
        # fires, it means the split machinery changed or group_key has duplicates
        # that cross fold boundaries through some bug in build_features.py.
        overlap = train_groups & test_groups
        if len(overlap) > 0:
            raise AssertionError(
                f"\nFATAL LEAKAGE detected in fold {fold}.\n"
                f"{len(overlap)} group_key(s) appear in BOTH train and test sets.\n"
                f"Example overlapping keys: {sorted(overlap)[:5]}\n"
                f"The split is not leakage-safe. Check group_key construction in "
                f"gene_families.py and the join in build_features.py."
            )

        for i in test_idx:
            fold_idx_arr[i] = fold

        n_train      = len(train_idx)
        n_test       = len(test_idx)
        train_pos    = int(y[train_idx].sum())
        test_pos     = int(y[test_idx].sum())
        test_pos_pct = test_pos / n_test

        fold_stats.append({
            "fold":          fold,
            "train_genes":   n_train,
            "train_pos":     train_pos,
            "test_genes":    n_test,
            "test_pos":      test_pos,
            "test_pos_pct":  test_pos_pct,
            "train_groups":  len(train_groups),
            "test_groups":   len(test_groups),
            "group_overlap": 0,
        })

        print(
            f"{fold:>4}  {n_train:>12,}  {train_pos:>10,}  "
            f"{n_test:>11,}  {test_pos:>9,}  {test_pos_pct:>8.2%}  "
            f"{len(train_groups):>11,}  {len(test_groups):>10,}"
        )

    print("-" * 85)
    stats = pd.DataFrame(fold_stats)
    m = stats[["train_genes", "train_pos", "test_genes", "test_pos"]].mean()
    print(
        f"{'mean':>4}  {m['train_genes']:>12,.0f}  {m['train_pos']:>10,.0f}  "
        f"{m['test_genes']:>11,.0f}  {m['test_pos']:>9,.0f}"
    )

    # Report fold size imbalance so it is not mistaken for a bug.
    test_sizes = stats["test_genes"]
    if test_sizes.max() / test_sizes.min() > 1.5:
        print(
            f"\n  Note: test fold sizes range from {test_sizes.min():,} to {test_sizes.max():,}. "
            f"This is expected when one family is much larger than the others "
            f"(the largest HGNC family has 763 members). It is a correct consequence "
            f"of respecting family boundaries, not an error."
        )

    print()
    print("GROUP OVERLAP CHECK (zero overlap is required in every fold):")
    for s in fold_stats:
        print(
            f"  fold {s['fold']}: {s['train_groups']:,} train groups, "
            f"{s['test_groups']:,} test groups -- overlap = {s['group_overlap']}  [PASS]"
        )
    print()
    print("All assertions passed: ZERO group_keys appear in both train and test in any fold.")

    # Save fold assignments. train_eval.py uses this file to reconstruct the
    # same splits, so any analysis re-run produces identical train/test splits
    # without calling the splitter again.
    out = df[["symbol", "group_key", "label"]].copy()
    out["fold_idx"] = fold_idx_arr
    assert (out["fold_idx"] == -1).sum() == 0, "some genes were not assigned to any fold"
    out.to_parquet(FOLDS_FILE, index=False)
    print(f"wrote {FOLDS_FILE}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
