# Target Prioritization Pipeline: Design Rationale

**Repository:** `drug-discovery-target-prioritization`

A Nextflow-on-AWS pipeline that turns population genetic data into ML-ranked druggable targets, scored against real clinical outcomes. Built to demonstrate cloud-native genomics orchestration, biologically grounded feature engineering, and leakage-safe ML evaluation.

---

## 1. Why this project

Target identification ("which protein should we drug") is the highest-value and highest-failure decision in drug discovery. Most clinical failures trace back to the wrong target, not bad chemistry. The AI-first biotechs (Insitro, Recursion) and the Open Targets consortium (GSK + EMBL-EBI) are organized around exactly this question.

This project tests a specific, falsifiable claim: that mechanistic biology features (genetic constraint, rare-variant burden, tissue specificity, protein structure, cell essentiality) predict which genes go on to reach clinical-phase drug development, independent of how well-studied or well-connected a gene already is. That second part, ruling out study bias as the explanation, is most of the design effort described below.

The project is scoped as a single, complete pipeline (data ingestion through a leakage-safe, prospectively-validated model) rather than several smaller, disconnected pieces. The orchestration layer (Nextflow on AWS Batch) is the delivery mechanism; the hand-coded effort is concentrated in the ML layer, feature engineering, leakage-safe evaluation, and the temporal holdout, since that is where the actual scientific claim is made or broken.

---

## 2. Architecture

Canonical AWS genomics reference architecture, as originally planned:

```
Public data (S3, in-region)
        |
   DataSync / direct S3
        |
   Nextflow on AWS Batch
     - Spot for per-sample parallel annotation steps
     - On-demand for aggregation / join steps
        |
   Glue + Athena (tabular evidence layer)
        |
   ML training (target prioritization)
        |
   Ranked target list + SHAP interpretability
```

**As actually built**, this simplified in three ways, tracked here rather
than left for the reader to discover by diffing the diagram against the
code:

```
Public data (S3, in-region)
        |
   Nextflow on AWS Batch (Spot only, all steps)
        |
   Parquet output (results/), read directly by ml/build_features.py
        |
   ML training (target prioritization)
        |
   Ranked target list (feature_importances_ and SHAP, section 6.2)
```

- **Glue + Athena was never built.** The Nextflow pipeline's Parquet output
  is read directly by `ml/build_features.py` with pandas; there is no Glue
  crawler or Athena query layer.
- **No separate on-demand compute environment exists.** `terraform/batch.tf`
  defines one Batch compute environment, type `SPOT`. Every pipeline run,
  annotation and aggregation alike, ran on Spot.
- **SHAP and bootstrap stability selection are implemented** (`ml/train_eval.py`'s
  `shap.TreeExplainer` call and `bootstrap_stability_selection`, both added
  after the rest of this document was written). See sections 6.2 and 6.5.

### What was actually demonstrated

All 22 human autosomes of 1000 Genomes phase 3 (tens of GB of input, not a large-scale cohort by modern genomics standards) processed concurrently on AWS Batch Spot instances, with measured wall clock and cost: 2h32m for the 17 concurrently-run autosomes versus an estimated ~23h run one at a time, under $1 total spend. See section 9 for the full breakdown. The concurrency speedup (roughly 9x, from raising `max_vcpus` from 4 to 8) and the per-chromosome cost basis are the scaling claims this project can actually support; there is no benchmark or capacity model here for cohort sizes beyond what was run.

### Cost control (near-zero spend)

- 1000 Genomes as an in-region S3 source: no egress, no storage bill.
- Spot instances for embarrassingly parallel steps.
- S3 lifecycle policies on intermediates.
- Open Targets and other public layers pulled once and cached.

### Networking (no NAT Gateway)

Run Batch in public subnets and skip the NAT Gateway entirely. Components:

- Public subnets with an Internet Gateway for outbound (container image pulls, AWS API calls).
- Instances auto-assigned public IPs.
- S3 Gateway VPC Endpoint so the large S3 data traffic never traverses the internet and incurs no data-processing charge.

This designs out the number-one cost killer (idle NAT Gateway, ~$32/month for nothing) rather than managing it. It matches the pattern already proven in the existing repo.

Trade-off: instances have public IPs and are therefore internet-reachable. Mitigated with tight security groups (no inbound rules) and the fact that only public reference data is processed. A production deployment handling non-public data would use private subnets with VPC endpoints instead; public subnets with locked-down security groups are a deliberate, cost-motivated choice for this specific case (public data, no inbound access), not a default assumed to be safe everywhere.

### Infrastructure as code (Terraform)

Provision the durable infra with Terraform: VPC and subnets, Internet Gateway, S3 gateway endpoint, the Batch compute environment and job queues, IAM roles, S3 buckets, ECR. Reproducible, version-controlled infrastructure that tears down and rebuilds cleanly, rather than manually clicked-together console resources that are hard to reason about or reverse.

Kept deliberately minimal: no remote-state backends, workspaces, custom module libraries, or CI integration. Flat, readable config that expresses the infra directly. The point of this project is the ML layer and the leakage-safe evaluation; the infrastructure exists to support that reliably and cheaply, not to demonstrate infrastructure engineering for its own sake.

---

## 3. Data

| Layer | Source | Role |
|-------|--------|------|
| Population variants | 1000 Genomes (public, in-region) | Pipeline input, feeds genetic features |
| Gene metadata | Open Targets `targets` table | ID mapping (Ensembl ID, symbol, biotype) |
| Label | Open Targets `knownDrugsAggregated` | Clinical phase per gene; label only, not a feature |
| Constraint | gnomAD (pLI, LOEUF, o/e) | Genetic features |
| Structure | AlphaFold | Protein-intrinsic features |
| Network | STRING | PPI features |
| Expression / essentiality | GTEx, DepMap | Expression and safety features |

Open Targets is used for the label only. The `knownDrugsAggregated` table provides clinical phase per gene (binary label: clinical phase >= 1; continuous variant: max clinical phase). The `targets` table provides gene metadata for ID mapping. All drug and clinical evidence is excluded from features, as required by section 4.

`associationByOverallDirect` is deliberately not used. It is a bundled overall score with no datatype breakdown, so genetic-association evidence cannot be separated from literature and other evidence types. Using it as a feature would reintroduce the section 4 circularity.

Genetic-association evidence from `associationByDatatypeDirect` (filtered to `genetic_association`, with `known_drug` and `literature` datatypes excluded) is deferred to a possible v2. For v1 the feature set is as listed in section 5: gnomAD constraint already captures the genetic signal that matters, and a clean, defensible feature set is worth more than one additional source at this stage.

Verify the label and gene metadata structure against the pinned Open Targets release (24.09). Schema fields shift between releases.

---

## 4. The label (the trap)

The obvious label, predicting the Open Targets association score, is quietly circular: Open Targets computes that score partly from genetic evidence, so the model would predict its own inputs. This is not a subtle problem, it would be caught by anyone reviewing the design.

**Chosen label:** does this target have a drug at clinical phase >= 1 (or approved)? Pulled from ChEMBL / Open Targets known-drug evidence. All drug and clinical evidence is excluded from features. This predicts a genuine real-world outcome from biology rather than predicting inputs from inputs. A continuous variant (max clinical phase) supports a regression framing.

**Open-world caveat, stated explicitly rather than left implicit:** absence of a drug does not mean undruggable; it may mean understudied. This is really positive-unlabeled (PU) learning, not clean binary classification. This project treats it that way: the open-world assumption is stated explicitly throughout rather than defining hard negatives, and section 6.2's study-bias check exists specifically to test whether the model is exploiting that openness (learning "famous" instead of "biologically promising"). This is the single hardest design call in the project, and the one most likely to be challenged on review.

---

## 5. Feature set

Aggregated per gene / protein, grouped by source.

**Genetic constraint and variant burden.** gnomAD constraint (pLI, LOEUF), rare / LoF variant burden, observed-vs-expected ratios, conservation. This is where the Nextflow pipeline output feeds the model, tying the project together. Among the most predictive and most biologically honest features.

**Protein-intrinsic.** Length and disorder fraction derived from AlphaFold and UniProt, both implemented (see below). Domain families and pocket/structure features were considered and not built. Target-class membership (kinase, GPCR, ion channel) was also considered and deliberately excluded: it is almost too predictive of tractability on its own, and risks acting as a shortcut around genuine biological signal rather than a feature that adds to it.

**Implemented status:** `protein_length` (UniProt Swiss-Prot) and `disorder_fraction` (AlphaFold DB prediction API, fraction of residues with pLDDT < 50) are at 98.9% and 98.2% coverage of the gene universe respectively. `disorder_fraction` turned out to be a top-tier contributor, not a minor one, co-dominant with `tau` in the `biology_only` variant by SHAP (section 12.1). Feature importance is tracked two ways, `feature_importances_` and SHAP, both implemented (section 6.2).

**Network.** PPI centrality from STRING (degree, betweenness), pathway membership. Strong signal but heavily confounded by study bias, since well-studied proteins have more measured interactions.

**Expression and essentiality.** Tissue specificity (tau) from GTEx, cell essentiality from DepMap. Tissue-restricted expression is a real safety and druggability signal.

**The confounder, included on purpose.** Publication count and year-first-described. Included specifically so the model can be shown not to be riding it.

Open Targets association scores are not in this feature set. See section 3 for the reasoning. The five source groups above (gnomAD, AlphaFold, STRING, GTEx/DepMap, publication metadata) are the complete v1 feature set.

---

## 6. Leakage-safe evaluation

Five elements, in priority order.

### 6.1 Group by gene family, not by gene
Gene-level splitting is the baseline answer. The good answer groups paralogs and sequence-similar families together (HGNC gene groups or similarity clustering) and uses GroupKFold on that grouping. Otherwise a gene in train and its paralog in test leaks structure and sequence features across the split.

### 6.2 Study-bias stratification
The naive failure mode of every target-prediction model is learning "is this gene famous." Defended two ways:
- Check via SHAP whether publication count dominates.
- Report performance within bins of similar publication count.

If the model still ranks well among understudied genes, that is evidence it is doing real work rather than riding fame; this is the central result the rest of section 6 and section 12 build on.

**Implemented status:** both checks are implemented. SHAP values are computed via `shap.TreeExplainer` on the same full-data model already fit for `feature_importances_` (`ml/train_eval.py`, printed as "SHAP importances" alongside the existing top-10 `feature_importances_` list in every run), so the two importance rankings can be compared directly; they largely agree, which is itself a useful cross-check. The bin-based check was implemented and substantially extended beyond this original two-line plan: see section 6.6 below for the full method (pooled and paired bootstrap, sign test, repeated CV, median split) and README.md's Results section for what it found.

### 6.3 Ranking metrics, not just AUC
Prioritization is a top-of-list problem, like virtual screening; nobody pursues 8,000 targets. Report:
- PR-AUC (positives are rare).
- Precision@k.
- Enrichment factor at the top 1 to 5 percent.

Enrichment factor speaks pharma's native language and signals understanding of the actual decision.

### 6.4 Temporal holdout
Train on targets that reached the clinic before a cutoff year, then predict which genes reached clinical stage after it, using only pre-cutoff values for time-dependent features (publication counts especially). Even a partial temporal split is the most honest "could this have prospectively found a real target" test, and almost no portfolio project does it.

**Implemented** (`ml/temporal_holdout.py`), using Open Targets release history as the time machine instead of per-gene clinical trial dates: a gene's clinical-phase label in an old Open Targets release is "the past," the same gene's label in a current release is "the future." Cutoff release 21.06 (June 2021), evaluated against 26.06 (June 2026), a clean 5.0-year gap. pub_count and STRING centrality (ppi_degree, ppi_betweenness) were dropped rather than back-dated (see the module docstring for why back-dating pub_count correctly was not feasible in this pass), so this is a stricter, narrower feature set than the main ablation variants, not a like-for-like comparison to them. Trained on the 1,196 genes labeled at 21.06, evaluated against the 338 genes that were unlabeled at 21.06 and gained a clinical-phase drug by 26.06. Result: enrichment above the resampled baseline 95% CI at every threshold tested (5.59x at top 1%, 3.57x at top 5%, 3.22x at top 10%, 2.66x at top 20%), PR-AUC 0.055 against a base rate of 0.0187 (lift 2.95x). This is the strongest single result in the project: real prospective signal, with fame-riding excluded by construction rather than by argument. Full numbers in the README results section.

### 6.5 Bootstrap stability selection
Resample, refit, keep features selected in more than X percent of runs. Reports robust biology and guards against overfitting to the famous-gene signal. SHAP + bootstrap approach drops in cleanly here.

**Implemented status:** implemented (`bootstrap_stability_selection` in `ml/train_eval.py`, run as part of `--compare`). 50 row-level bootstrap resamples of the full training set (deliberately NOT GroupKFold, see the function's docstring: this measures whether a feature's importance is stable under resampling of the training set, not leakage-safe generalization, so grouping by gene family is not the relevant concern here), refit each time, top-10 features by `feature_importances_` recorded per resample. Features selected in more than 70% of resamples are flagged as stable. This is a different axis of robustness than the repeated-CV analysis in section 6.6 (which varies fold assignment across 10 seeded repeats): repeated CV tests whether the *evaluation* is stable to which genes land in which fold, this tests whether *feature selection* is stable to which genes land in the training sample.

### 6.6 Evaluation methodology, as implemented

This is what `ml/train_eval.py` actually does, distilled from working through the statistical corrections in this section's history.

**Split.** Gene-family GroupKFold, 5 folds, groups assigned once in `ml/split.py` and never touched by any downstream analysis. Every fold assignment used anywhere in this document, including the repeated-CV variants below, is checked for zero group-key leakage before its results are trusted.

**Ranking metrics with per-stratum baselines.** PR-AUC alone is not informative across strata of very different positive rates (7.8% overall, under 1% in the low-publication tercile, close to 20% in the high tercile). Every stratum's lift is PR-AUC divided by that stratum's own positive rate, not the global one, so lift greater than 1.0 means the model beats a random ranker working on that same population. This is what makes the low-tercile and bottom-half results interpretable at all: a PR-AUC of 0.03 sounds weak until you see the stratum's own baseline is 0.009.

**Paired bootstrap comparison between variants on identical folds.** When two feature-set variants (say `biology_only` and `no_pubcount`) are trained and evaluated on the exact same GroupKFold split, their per-fold scores are not independent draws, they are correlated because the hard folds are hard for both variants and the easy folds are easy for both. Reporting two separate marginal confidence intervals and checking whether they overlap throws that correlation away and is therefore a systematically underpowered test: two CIs can overlap heavily while the fold-by-fold difference is consistently one-directional. The correct test resamples the paired per-fold differences directly (`bootstrap_paired_diff_ci`), preserving the pairing. This is the right test regardless of what it finds: shared folds make the variants' scores correlated, not independent, so a paired comparison can detect a consistent-direction difference that marginal, unpaired CIs would miss, or, as it did after `disorder_fraction` was added, correctly show that a difference is no longer distinguishable from noise. See section 12.2 for what this test actually found, at two different points in the project as the feature set changed.

**Sign test.** A distribution-free complement to the paired bootstrap CI, since 5 folds is too few points for the bootstrap's normal-approximation intuitions to be very trustworthy on their own. Counts how many of the 5 folds favor variant A versus variant B; a lopsided count (e.g. 1-4) corroborates the bootstrap CI's direction without relying on any distributional assumption.

**Repeated CV across 10 fold assignments.** GroupKFold's own fold assignment is one arbitrary draw; some of the spread seen across runs could be fold-assignment luck rather than genuine model instability. `run_repeated_cv` reruns the full training and evaluation loop 10 times with a fresh, seeded fold assignment each time (`make_group_folds`, sklearn's GroupKFold has no `random_state` so this is a hand-rolled equivalent), holding the model seed fixed, and reports the mean/std/min/max of the resulting lift across those 10 repeats. This isolates fold-assignment variance from the bootstrap's resampling variance and confirms the two are of comparable, modest size, not that one is hiding a much larger true instability.

**Median split promoted to primary.** The original stratification was a publication-count tercile (roughly 60 positives per stratum), which turned out to be underpowered for the low-tercile question specifically. A 50/50 median split of the same ranking (159 positives in the bottom half versus 60 in the low tercile) answers the same "does this work on understudied genes" question with much narrower confidence intervals, and is now the headline metric; the tercile numbers are kept as a secondary, explicitly underpowered cross-check for continuity with earlier runs, not because they carry independent evidentiary weight.

---

## 7. Division of labor

**Orchestration:** Nextflow process definitions, Batch queue config, Spot routing (as built, section 2, there is no separate on-demand routing or Glue/Athena layer).

**Hand-coded, not delegated to a library:**
- Leakage-safe splitting (gene-family GroupKFold). Implemented.
- Negative-set / PU handling. Implemented (open-world assumption, section 4).
- Feature engineering from annotated variants. Implemented.
- Bootstrap stability selection. Implemented (section 6.5).
- SHAP interpretability and the study-bias analysis. Both implemented (section 6.2, section 6.6); SHAP alongside `feature_importances_`, not instead of it.
- Ranking metric implementation (enrichment factor, precision@k). Implemented.

---

## 8. Deliverables

The original plan, with delivery status noted:

1. The repository. Delivered.
2. This design doc, as design rationale. Delivered; restructured partway through (section 12) once the results writeup moved to README.md for readability.
3. An architecture diagram and the scaling / cost math. Delivered: the Mermaid diagram in README.md, cost and scaling numbers in section 9 below.
4. A results writeup: which genes ranked well, performance against the clinical-phase label, the study-bias result. Delivered in README.md's Results section, with the extended discussion here in section 12.

---

## 9. Cost

Target: $30 to $80 for the whole project if disciplined. It can exceed $200 if a few specific things are left running. The difference is discipline, not scale. All figures below this line are orders-of-magnitude estimates, kept for the original planning rationale.

**Measured, full 22-autosome run:** total spend under $1. 2h32m wall clock for all 17 remaining autosomes running concurrently on Batch, versus an estimated ~23h if run one chromosome at a time, roughly a 9x speedup from raising `max_vcpus` from 4 to 8 (terraform/variables.tf). chr8 was the only chromosome whose ANNOTATE step needed more than the flat 4GB (it needed exactly 8GB; see pipeline/nextflow.config's dynamic memory escalation and the chr8 OOM incident); bcftools csq's memory footprint tracks transcript density in the region being annotated, not raw sequence length, which is why chr8 (physically smaller than chr3/4/5, which all succeeded at 4GB) was the outlier.

The reason it is this cheap is the in-region data choice already made: 1000 Genomes lives in the AWS Open Data registry in us-east-1. Run compute in the same region and the source data costs nothing to store and nothing to move, removing what is normally the largest genomics bill.

### Breakdown (live subset run plus iteration)

| Item | Estimate | Notes |
|------|----------|-------|
| Batch compute (Spot) | $20 to $60 | Annotation is cheap per sample; includes re-runs and debugging |
| S3 intermediates | $5 to $20 | A few hundred GB for a few weeks, less with lifecycle rules |
| Glue + Athena | $5 to $15 | Planning estimate only; this layer was never built (section 2), so this line was never actually incurred |
| ML training | $0 to $30 | ~20,000 genes x ~100 features fits in laptop memory; train locally for $0 |
| Egress | $0 | Only if everything stays in-region |

The feature matrix is small enough to train locally or on a tiny EC2 instance. The only thing that grows this line is SageMaker, which leads to the killers below.

### The four silent killers (in order of how often they bite)

1. **NAT Gateway (designed out).** Batch in a private subnet usually pulls one in, billing ~$0.045/hr plus data charges whether used or not, ~$32/month idle. This project avoids it entirely by running in public subnets with an S3 gateway endpoint (see section 2 networking). Listed here because it is the most common genomics-on-AWS cost leak and the reason the public-subnet choice was made.
2. **Idle SageMaker notebook.** Spun up, walked away from, billed hourly all weekend. If used at all, stop instances when done. This project skips SageMaker entirely; the feature matrix is small enough to train locally for $0.
3. **Accidental egress.** Pulling large data out of AWS is ~$0.09/GB. One TB down is ~$90 instantly. Rule: the data never leaves the region, only small results do.
4. **Forgetting Spot.** Running parallel, restartable steps on-demand instead of Spot can 3x to 4x the compute line for no benefit.

### Day-one setup

- **AWS Budgets alert at $50.** Email warning before anything surprising happens. This alone makes the worst cases nearly impossible.
- **Tags on every resource.** Makes spend attributable at a glance. Terraform makes this trivial via default tags.
- **`terraform destroy` between sessions.** With the NAT Gateway designed out, the remaining idle costs are small, but destroy is still the clean discipline: one command tears the whole stack down, `terraform apply` rebuilds it next session. Converts "remember to manually delete every resource" into a single reliable habit and guarantees nothing is left billing.

---

## 10. Timeline (3 to 4 weeks, part-time, original estimate)

| Week | Focus (as planned) |
|------|-------|
| 1 | Infra + data ingestion + one Nextflow process running on Batch |
| 2 | Full annotation pipeline (Glue/Athena evidence layer was planned here; never built, section 2) |
| 3 | Hand-coded ML layer (features, leakage-safe eval, stability selection, SHAP) |
| 4 | Scaling benchmark, cost writeup, docs, polish |

This was the original estimate, not a tracked actual. The repository's commit
history spans roughly three weeks (first commit to the temporal holdout and
final writeup), close to the estimate, though the actual work did not follow
these four weekly buckets in order; the ML layer, evaluation methodology, and
scaling benchmark were built and revised iteratively rather than in four
discrete weekly phases.

---

## 11. Open design calls, as resolved

The original open list from planning, with how each was actually resolved:

- **Negative-set definition / PU strategy** (section 4). Resolved: open-world assumption stated explicitly, no synthetic hard-negative construction; absence of a drug is treated as unlabeled, not negative, throughout.
- **Exact clinical-phase cutoff for the label and for the temporal holdout.** Resolved: clinical phase >= 1 (or approved) for both the main label and the temporal holdout's cutoff-release and future-release labels.
- **Gene-family grouping source.** Resolved: HGNC gene groups (`ml/gene_families.py`), not sequence-similarity clustering.
- **Subset size for the live run.** Resolved: all 22 autosomes, not a smaller subset; there is no separate scaling-math extrapolation beyond what was measured (section 9).
- **Open Targets schema version.** Resolved: 24.09 pinned for the main label and ablation; 21.06 and 26.06 for the temporal holdout's cutoff and future releases respectively (section 6.4).

---

## 12. Results detail

README.md was restructured for readability (headline numbers, one picture,
a short "how it works," everything else collapsed or linked). This section
holds the depth that moved out of it: full methodology discussion, the
descriptive correlation table, and the secondary validation detail. No
numbers here differ from README.md, this is the same results, just the
longer version.

### 12.1 SHAP and bootstrap stability selection, in detail

**SHAP** (`shap.TreeExplainer` on the same full-data model used for
`feature_importances_`): the two importance mechanisms broadly agree, and
where they disagree, the disagreement is informative. In `biology_only`,
`feature_importances_` ranks `tau` first (0.213) and `disorder_fraction`
fourth (0.120); SHAP ranks them essentially tied for first (`tau` 0.373,
`disorder_fraction` 0.371), both well clear of everything else. Both
methods agree `disorder_fraction` is a top-tier contributor, not a minor
one, contrary to what was expected going in (a real but likely secondary
contributor was the working assumption before this run).

**Bootstrap stability selection** (50 row-level resamples of the full
training set, refit each time, tracks how often each feature lands in the
top 10 by `feature_importances_`, features selected in more than 70% of
resamples flagged as stable): the core biology features, gnomAD
constraint (`pLI`, `loeuf`, `oe_mis`, `oe_lof`), burden (`n_rare`),
expression (`tau`, `essentiality_score`), and structure (`protein_length`,
`disorder_fraction`), are selected in 96 to 100% of resamples across all
four variants. `pub_count` and `year_first_described`, where present, are
also stable at 100%, consistent with real (if confounded) predictive
signal rather than noise. `n_lof` and `ppi_betweenness` are the least
stable features, near the bottom of most variants' top 10 or absent from
it in some resamples.

**Fetching `disorder_fraction` required a real fix, not just a flag flip.**
`ml/fetch_alphafold.py`'s `--disorder` flag pointed at a dead AlphaFold DB
URL (every versioned bulk file 404s now); the correct source is AlphaFold
DB's prediction API, which conveniently returns the needed fraction
directly (`fractionPlddtVeryLow`, an exact match for the project's existing
pLDDT < 50 disorder threshold) without needing to fetch or parse
per-residue arrays. Once added, `disorder_fraction` turned out to be a much
bigger contributor than expected, enough to change the Q2 answer below from
a clean "yes" to "no longer clearly yes." Both are the kind of thing a real
implementation pass finds that planning documents don't anticipate.

### 12.2 Q1 and Q2, extended discussion

**Q1.** The model beats a random ranker on understudied genes even with
every publication-history and network-centrality feature removed
(`biology_only`). An earlier tercile-based analysis (60 positives in the
low-publication group) had concluded this was "not distinguishable from
noise." That conclusion was a power artifact of an underpowered
stratification, not a genuine null; the median split (159 positives in the
bottom half) resolves it.

**Q2, before and after `disorder_fraction`.** Paired bootstrap comparison
of `biology_only` vs. `no_pubcount` on identical folds, computed on the
low-publication-tercile lift: before `disorder_fraction` was added, mean
difference -8.33, 95% CI [-16.84, -0.03] (excludes zero), and the
median-split CIs for `biology_only` vs. `all_features` did not overlap
([2.12, 3.64] vs. [3.98, 7.25]), the basis for an earlier "yes, measurably"
answer. After adding `disorder_fraction`: mean difference -8.00, 95% CI
[-17.19, +0.21] (now includes zero), sign test still 1-4 in the same
direction (`no_pubcount` wins 4 of 5 folds), and the median-split CIs now
overlap by a hair (0.01): `biology_only` [2.17, 4.00] vs. `all_features`
[3.99, 7.76].

Adding one more real, time-stable biology feature closed most of the gap
that used to be attributed to discovery-history features. The sign test
direction hasn't changed, so there may still be a real, small effect, but
it is no longer distinguishable from sampling noise at conventional
confidence with the current feature set and fold count. This is reported
as the update it is: the honest answer to Q2 moved from "yes" to "no longer
clearly yes" when the feature set became more complete, itself evidence for
the open question below, that missing biology, not just fame, was likely
responsible for some of what looked like a discovery-history-features
effect.

### 12.3 Open question: descriptive correlations, in detail

Descriptive Spearman correlation between `year_first_described` and the
biology features already in `biology_only`:

| feature | rho | p-value |
|---|---|---|
| loeuf | 0.203 | 6.45e-179 |
| oe_lof | 0.166 | 5.76e-119 |
| tau | 0.159 | 1.91e-109 |
| oe_mis | 0.152 | 1.47e-100 |
| pLI | -0.110 | 7.49e-53 |
| essentiality_score | 0.073 | 2.72e-24 |
| protein_length | -0.066 | 6.64e-20 |

Max |rho| is 0.203. At n=19,296, p-values this small are not informative on
their own; with this many genes, even trivial correlations clear any
conventional significance threshold, so the magnitudes are what matter, and
they are small to moderate. This table is descriptive only, not a
decomposition: it shows discovery timing is entangled with biology, it does
not tell us how much of `year_first_described`'s contribution in Q2 is fame
vs. biology. Critically, `biology_only` already contains `pLI`, `loeuf`,
`oe_mis`, `oe_lof`, and `essentiality_score`, so whatever
`year_first_described` adds on top of `biology_only` is, by construction,
not the signal those features already capture; these correlations cannot
explain that gap away. A residualization scheme (regressing
`year_first_described` on biology features and calling the residual "pure
fame") was deliberately not implemented: that residual would contain
technological accessibility, funding history, disease salience, and noise,
not just fame, and is a causal claim this project cannot support.

`disorder_fraction` was not included in this correlation table (it was
added to the feature set after this table was first produced), but the Q2
update above is directly relevant here: closing most of the `biology_only`
vs. `all_features` gap by adding one more real biology feature is
consistent with a meaningful share of the original gap being missing
biology rather than fame, though it does not prove it, `disorder_fraction`
could itself happen to correlate with discovery timing the same way the
table above's features do. That correlation was not separately checked.

### 12.4 Secondary validations, in detail

Two smaller checks, both against real external evidence, both underpowered
enough that a null result at any given threshold does not contradict the
rest of these results.

**Genetic evidence check** (`ml/validate_genetic_evidence.py`): among
`biology_only`'s top-ranked unlabeled genes, checked for independent human
genetic disease evidence (Open Targets Genetics, `associationByDatatypeDirect`
filtered to `genetic_association`, threshold score >= 0.5). Top 50: 0.540
observed vs. 0.501 baseline, within the baseline 95% CI [0.360, 0.620], not
distinguishable from noise at this N. Top 100: 1.26x enrichment, above the
baseline CI. Top 500: 1.42x enrichment, above the baseline CI. This check is
confounded by design: genetic disease evidence makes a gene more likely to
already attract a drug program, so it measures whether the model ranks
toward genes the field would find interesting, not whether they are
druggable.

**24.09 to 26.06 newly-labeled check** (`ml/validate_prospective_labels.py`):
an earlier, smaller version of the temporal holdout, using the same 24.09
label the main ablation trains against and a shorter gap to 26.06. Only 30
genes moved from unlabeled to labeled in that window, an underpowered
sample by construction. Enrichment ratios (0.68x at top 5%, 2.03x at top
10%, 1.84x at top 20%) were noisy at this N and all fell within their
resampled baseline's 95% CI, an inconclusive aggregate result. One
individual gene is worth naming as an anecdote, not evidence: `KCNMA1`
ranked 51st of 17,789 unlabeled genes in that check (was 18th before
`disorder_fraction` was added, still comfortably top 1%) and has since
gained a clinical-phase drug in release 26.06. Interesting, but n=1.

### 12.5 ML layer output files

- `ml/cache/gene_families.parquet`, gene universe with group keys
- `ml/cache/gnomad_constraint.parquet`, constraint metrics
- `ml/cache/alphafold_features.parquet`, protein length and disorder fraction (UniProt Swiss-Prot, AlphaFold DB)
- `ml/cache/string_features.parquet`, PPI degree and betweenness (STRING v12)
- `ml/cache/expression_features.parquet`, tissue-specificity (tau, GTEx) and essentiality (DepMap)
- `ml/cache/publication_features.parquet`, pub_count and year_first_described (NCBI gene2pubmed)
- `ml/cache/training_table.parquet`, full feature matrix, 19,296 genes. Burden features (n_rare, n_lof) cover 16,725 genes (86.68% of the protein-coding universe, 91.0% of the autosome-eligible universe), the rest zero-filled per the documented missing-data convention.
- `ml/cache/cv_folds.parquet`, fold assignments (GroupKFold, n=5)
- `ml/cache/oos_predictions.parquet`, out-of-sample scores, labels, and ranks for whichever `--feature-set` was last run
