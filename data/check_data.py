"""
Diagnostic checks for the Open Targets data layer.

Two questions this answers, both from DESIGN.md section 4:
  1. Is the LABEL shaped correctly once collapsed to the gene level?
     (positives should be a minority of the full gene universe, not a majority)
  2. Is associationByOverallDirect safe to use as a FEATURE?
     (it is not: it bundles known-drug evidence and cannot be cleaned)

Run:  python3 data/inspect.py --release 24.09
"""

import argparse
import os
import sys
import pandas as pd

DEFAULT_RELEASE = "24.09"
CACHE_ROOT = os.environ.get("DATA_CACHE_DIR", "data/cache")


def find_dataset(release, name):
    """Locate a dataset directory in the cache, tolerating layout variation."""
    candidates = [
        os.path.join(CACHE_ROOT, "open_targets", release, name),
        os.path.join(CACHE_ROOT, release, name),
        os.path.join(CACHE_ROOT, name),
    ]
    for c in candidates:
        if os.path.isdir(c) or os.path.isfile(c):
            return c
    return None


def load(release, name):
    path = find_dataset(release, name)
    if path is None:
        print(f"  [!] could not find dataset '{name}' under {CACHE_ROOT}")
        return None
    try:
        df = pd.read_parquet(path)
        print(f"  loaded {name}: {len(df):,} rows, {len(df.columns)} cols  ({path})")
        return df
    except Exception as e:
        print(f"  [!] failed to read {name}: {e}")
        return None


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def check_label(release):
    print("\n=== CHECK 1: label shape ===")

    kd = load(release, "knownDrugsAggregated")
    targets = load(release, "targets")
    if kd is None or targets is None:
        print("  cannot run label check without both datasets.")
        return

    phase_col = pick_column(kd, ["phase", "clinicalPhase", "maxPhaseForIndication"])
    target_col = pick_column(kd, ["targetId", "target_id"])
    if phase_col is None or target_col is None:
        print(f"  [!] columns present: {kd.columns.tolist()}")
        print("  [!] could not identify phase/target columns; inspect manually.")
        return
    print(f"  using phase column '{phase_col}', target column '{target_col}'")

    # Collapse target-disease pairs to one row per gene: max clinical phase.
    gene_max_phase = kd.groupby(target_col)[phase_col].max()
    positives = set(gene_max_phase[gene_max_phase >= 1].index)

    # The gene universe: protein-coding targets. This is the real label space.
    tid = pick_column(targets, ["id", "targetId"])
    biotype = pick_column(targets, ["biotype"])
    if biotype is not None:
        universe = set(targets.loc[targets[biotype] == "protein_coding", tid])
        universe_desc = "protein-coding targets"
    else:
        universe = set(targets[tid])
        universe_desc = "all targets (biotype column not found)"

    negatives = universe - positives
    pos_in_universe = positives & universe

    print(f"\n  gene universe ({universe_desc}): {len(universe):,}")
    print(f"  positives (any drug at phase >= 1): {len(pos_in_universe):,}")
    print(f"  negatives (no known drug):          {len(negatives):,}")
    if universe:
        rate = len(pos_in_universe) / len(universe)
        print(f"  positive rate over universe:        {rate:.1%}")

    print("\n  VERDICT:")
    if not universe:
        print("  [!] empty universe; something is wrong with the targets file.")
    elif len(negatives) < 1000:
        print("  [!] SUSPICIOUS: almost no negatives. You are probably measuring the")
        print("      positive rate inside knownDrugsAggregated only, which contains")
        print("      just genes that already have drugs. Negatives must come from the")
        print("      full gene universe. Re-check the join.")
    elif rate > 0.5:
        print("  [!] SUSPICIOUS: majority of genes are positive. Expected a minority.")
    else:
        print("  OK: positives are a minority of the full gene universe, as expected.")
        print("      (Remember: absence of a drug != undruggable. This is PU learning,")
        print("       per DESIGN.md section 4.)")


def check_association_leakage(release):
    print("\n=== CHECK 2: association-file leakage risk ===")
    a = load(release, "associationByOverallDirect")
    if a is None:
        print("  associationByOverallDirect not present (fine if you dropped it).")
        print("  Per DESIGN.md, Open Targets is LABEL-ONLY; you do not need this file.")
        return

    print(f"  columns: {a.columns.tolist()}")
    has_datatype = "datatypeId" in a.columns
    print("\n  VERDICT:")
    if not has_datatype:
        print("  [!] This is a single bundled 'overall' score with no datatypeId")
        print("      breakdown. It mixes known-drug + genetic + expression evidence")
        print("      and CANNOT be cleaned. Do NOT use it as a feature: doing so")
        print("      reintroduces the circularity DESIGN.md section 4 warns about.")
        print("      If you want genetic evidence as a feature, download")
        print("      associationByDatatypeDirect instead and keep only")
        print("      datatypeId == 'genetic_association' (and drop known_drug).")
    else:
        print("  Has datatypeId, but confirm you filter to genetic_association and")
        print("  explicitly drop known_drug before using any of it as a feature.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    args = ap.parse_args()
    print(f"Open Targets release: {args.release}")
    print(f"Cache root:           {CACHE_ROOT}")
    check_label(args.release)
    check_association_leakage(args.release)
    print("\nDone.")


if __name__ == "__main__":
    main()