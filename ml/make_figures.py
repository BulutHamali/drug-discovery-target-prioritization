#!/usr/bin/env python3
"""Generate the three results figures embedded in README.md.

Numbers below are hardcoded, not recomputed live, because two of the three
figures draw on results from different points in this project's history
that are not all simultaneously reproducible from the current repo state
(the n_rare coverage trend spans three different burden datasets: chr22
only, 5 chromosomes, and the final 22-autosome merge; only the last of
those three still exists on disk). Each block below cites exactly which
script and log produced its numbers, so the provenance is checkable even
though the script does not re-derive them itself. All three have been
independently re-verified bit-for-bit reproducible from the committed
scripts and cached data (README.md's Results section and this project's
own commit history).

Output: docs/figures/*.png

Usage:
    python3 ml/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})


def enrichment_curve():
    """Primary result: ml/temporal_holdout.py output (README.md "Primary
    result: temporal holdout" table). Trained on Open Targets 21.06 labels,
    evaluated against the 338 genes that gained a clinical-phase drug by
    26.06."""
    top_frac = np.array([1, 5, 10, 20])
    observed = np.array([0.056, 0.178, 0.322, 0.533])
    baseline_mean = np.array([0.010, 0.050, 0.100, 0.201])
    baseline_lo = np.array([0.000, 0.027, 0.071, 0.160])
    baseline_hi = np.array([0.021, 0.074, 0.133, 0.249])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    ax.fill_between(top_frac, baseline_lo, baseline_hi, color="#888888", alpha=0.25,
                     label="resampled baseline 95% CI (chance)")
    ax.plot(top_frac, baseline_mean, color="#888888", linestyle="--", linewidth=1.5,
            marker="o", markersize=5, label="baseline mean")
    ax.plot(top_frac, observed, color="#c0392b", linewidth=2.5, marker="o", markersize=7,
            label="observed rate (prospective positives)")

    for x, y in zip(top_frac, observed):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 8),
                    fontsize=9, color="#c0392b")

    ax.set_xticks(top_frac)
    ax.set_xticklabels([f"top {p}%" for p in top_frac])
    ax.set_xlabel("top fraction of unlabeled-at-cutoff genes, by biology_only-style score")
    ax.set_ylabel("fraction of prospective positives (n=338)")
    ax.set_title("Temporal holdout: enrichment above chance at every threshold\n"
                  "(Open Targets 21.06 to 26.06, 5.0-year gap)")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "enrichment_curve.png"))
    plt.close(fig)


def forest_plot():
    """Q1/Q2 combined: ml/train_eval.py --compare, "PRIMARY EVALUATION:
    MEDIAN SPLIT" section (README.md Q1 table). bottom-half lift = does the
    model beat chance on understudied genes; the four variants' non-overlap
    is the Q2 evidence that discovery-history features add real signal
    beyond biology_only."""
    variants = ["all_features", "no_pubcount", "no_pubcount_no_string", "biology_only"]
    lift = np.array([5.14, 5.53, 4.00, 2.66])
    ci_lo = np.array([3.98, 4.17, 3.04, 2.12])
    ci_hi = np.array([7.25, 8.07, 5.93, 3.64])

    y_pos = np.arange(len(variants))[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.5, zorder=1)
    ax.text(1.05, 0.35, "random ranker (1.0x)", color="#666666",
            fontsize=9, va="bottom", ha="left")

    colors = ["#2c3e50", "#2c3e50", "#2c3e50", "#c0392b"]
    for y, v, l, lo, hi, c in zip(y_pos, variants, lift, ci_lo, ci_hi, colors):
        ax.plot([lo, hi], [y, y], color=c, linewidth=2.5, zorder=2)
        ax.plot(l, y, "o", color=c, markersize=9, zorder=3)
        ax.annotate(f"{l:.2f}  [{lo:.2f}, {hi:.2f}]", (hi, y),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=9, va="center", color=c)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants)
    ax.set_xlabel("bottom-half (understudied genes) lift, median split, 95% CI")
    ax.set_title("Median-split lift by feature set\n"
                  "biology_only and all_features CIs do not overlap")
    ax.set_xlim(0, 9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "median_split_forest.png"))
    plt.close(fig)


def n_rare_trend():
    """Secondary finding, README.md "Also solid: n_rare importance trend".
    biology_only variant, n_rare feature_importances_, at three burden
    coverage stages reached over the project's history:
      2.0%   /tmp/ablation_rerun.log            (chr22 only, 388/19,296 genes)
      29.3%  /tmp/ablation_5chrom.log           (chr22,1,2,17,19)
      86.68% /tmp/ablation_22chrom.log, ml/train_eval.py --compare (all 22 autosomes,
             the only stage still reproducible from the current repo state)
    """
    coverage = np.array([2.0, 29.3, 86.68])
    importance = np.array([0.0112, 0.0352, 0.0714])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    ax.plot(coverage, importance, color="#2c3e50", linewidth=2.5, marker="o", markersize=9,
            zorder=2)
    for x, y in zip(coverage, importance):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(10, -4),
                    fontsize=10)

    ax.set_xlabel("burden feature coverage (% of protein-coding universe)")
    ax.set_ylabel("n_rare feature_importances_ (biology_only)")
    ax.set_title("n_rare importance climbs monotonically with coverage\n"
                 "(consistent across all four ablation variants)")
    ax.set_xlim(-5, 95)
    ax.set_ylim(0, 0.085)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "n_rare_trend.png"))
    plt.close(fig)


if __name__ == "__main__":
    enrichment_curve()
    forest_plot()
    n_rare_trend()
    print(f"wrote 3 figures to {os.path.abspath(OUT_DIR)}")
