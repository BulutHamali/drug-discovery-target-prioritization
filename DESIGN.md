# Target Prioritization Pipeline: Design Rationale

**Repository:** `drug-discovery-target-prioritization`

A Nextflow-on-AWS pipeline that turns population genetic data into ML-ranked druggable targets, scored against real clinical outcomes. Built to demonstrate large-scale cloud orchestration, biologically grounded feature engineering, and leakage-safe ML evaluation.

---

## 1. Why this project

Target identification ("which protein should we drug") is the highest-value and highest-failure decision in drug discovery. Most clinical failures trace back to the wrong target, not bad chemistry. The AI-first biotechs (Insitro, Recursion) and the Open Targets consortium (GSK + EMBL-EBI) are organized around exactly this question.

This project is deliberately scoped as the single Nextflow-on-AWS-at-100TB showpiece, not a sixth standalone portfolio piece. It does three jobs at once:

1. Closes the large-scale-cloud gap that has cost interviews.
2. Demonstrates the bio + cloud intersection rather than "data scientist transitioning."
3. Showcases biology judgment, which is the real edge over the data-science-only crowd.

The orchestration layer is home turf (Nextflow Ambassador) and should not absorb interview-prep energy. The hand-coded effort goes into the ML layer, which is where interviews probe and where cold read-write fluency gets rebuilt.

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
   Ranked target list (feature-importance interpretability, not SHAP)
```

- **Glue + Athena was never built.** The Nextflow pipeline's Parquet output
  is read directly by `ml/build_features.py` with pandas; there is no Glue
  crawler or Athena query layer. See README.md "Future work" for status.
- **No separate on-demand compute environment exists.** `terraform/batch.tf`
  defines one Batch compute environment, type `SPOT`. Every pipeline run,
  annotation and aggregation alike, ran on Spot.
- **SHAP was never implemented.** Interpretability is
  `GradientBoostingClassifier`'s built-in `feature_importances_`, used
  throughout the study-bias analysis (section 6.2) and the results in
  README.md. See section 6.2, 6.5, and 7 below, and README.md "Future work",
  for the full list of what was planned here but deferred.

### The 100TB story (framed honestly)

Frame as "designed and benchmarked to process 100TB-scale cohort data." Run on a real, meaningful subset, then document the capacity math and Spot-routing economics for full scale. This is the load-testing and capacity-planning muscle already studied, and it is precisely the answer that was missing in the interview that prompted this build. Do not claim a 100TB persistent store; claim a pipeline architected and cost-modeled for it.

### Cost control (near-zero spend)

- 1000 Genomes (or a public UK Biobank-style subset) as an in-region S3 source: no egress, no storage bill.
- Spot instances for embarrassingly parallel steps.
- S3 lifecycle policies on intermediates.
- Open Targets and other public layers pulled once and cached.

### Networking (no NAT Gateway)

Run Batch in public subnets and skip the NAT Gateway entirely. Components:

- Public subnets with an Internet Gateway for outbound (container image pulls, AWS API calls).
- Instances auto-assigned public IPs.
- S3 Gateway VPC Endpoint so the large S3 data traffic never traverses the internet and incurs no data-processing charge.

This designs out the number-one cost killer (idle NAT Gateway, ~$32/month for nothing) rather than managing it. It matches the pattern already proven in the existing repo.

Trade-off, and the interview answer: instances have public IPs and are therefore internet-reachable. Mitigate with tight security groups (no inbound rules) and the fact that only public reference data is processed. Framing: "In production I would use private subnets with VPC endpoints; for a cost-optimized portfolio processing public data, public subnets with locked-down security groups avoid the NAT bill for no real risk." This shows awareness of the trade-off rather than ignorance of it.

### Infrastructure as code (Terraform)

Provision the durable infra with Terraform: VPC and subnets, Internet Gateway, S3 gateway endpoint, the Batch compute environment and job queues, IAM roles, S3 buckets, Glue. This is a core Cloud Engineer hiring signal and reads as production-minded in a way the console does not. It is also cloud-portable, which suits the breadth of target roles better than CDK or CloudFormation.

Keep it deliberately minimal. No remote-state backends, workspaces, custom module libraries, or CI integration for a solo project. Flat, readable config that expresses the infra and tears down cleanly. The point is to show IaC fluency, not to build a reusable enterprise module library. If Terraform refactoring is eating days while the ML layer sits untouched, that is the avoidance tripwire firing; stop and ship the science layer.

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

The obvious label, predicting the Open Targets association score, is quietly circular: Open Targets computes that score partly from genetic evidence, so the model would predict its own inputs. An interviewer catches this immediately.

**Chosen label:** does this target have a drug at clinical phase >= 1 (or approved)? Pulled from ChEMBL / Open Targets known-drug evidence. All drug and clinical evidence is excluded from features. This predicts a genuine real-world outcome from biology rather than predicting inputs from inputs. A continuous variant (max clinical phase) supports a regression framing.

**Open-world caveat to raise proactively:** absence of a drug does not mean undruggable; it may mean understudied. This is really positive-unlabeled (PU) learning, not clean binary classification. Either handle with a PU approach or define hard negatives carefully and state the open-world assumption explicitly. This is the single hardest design call in the project and the one a sharp interviewer will push hardest on. Do not let it slide.

---

## 5. Feature set

Aggregated per gene / protein, grouped by source.

**Genetic constraint and variant burden.** gnomAD constraint (pLI, LOEUF), rare / LoF variant burden, observed-vs-expected ratios, conservation. This is where the Nextflow pipeline output feeds the model, tying the project together. Among the most predictive and most biologically honest features.

**Protein-intrinsic.** Length, domain families, disorder fraction, pocket / structure features derived from AlphaFold. Caution: target-class membership (kinase, GPCR, ion channel) is almost too predictive of tractability and can become a shortcut. Keep it but watch its feature importance.

**Implemented status:** only `protein_length` made it into the actual feature set. `ml/fetch_alphafold.py` has a `--disorder` flag that fetches per-residue pLDDT-based disorder fraction, but it was never run for this project's results, so `disorder_fraction` is not in `training_table.parquet`. Domain families and pocket/structure features were not implemented at all. "Watch its SHAP weight" refers to a tool that also was not implemented; see section 6.2.

**Network.** PPI centrality from STRING (degree, betweenness), pathway membership. Strong signal but heavily confounded by study bias, since well-studied proteins have more measured interactions.

**Expression and essentiality.** Tissue specificity (tau) from GTEx, cell essentiality from DepMap. Tissue-restricted expression is a real safety and druggability signal.

**The confounder, included on purpose.** Publication count and year-first-described. Included specifically so the model can be shown not to be riding it.

Open Targets association scores are not in this feature set. See section 3 for the reasoning. The five source groups above (gnomAD, AlphaFold, STRING, GTEx/DepMap, publication metadata) are the complete v1 feature set.

---

## 6. Leakage-safe evaluation (hand-coded, defendable)

Five elements, in priority order.

### 6.1 Group by gene family, not by gene
Gene-level splitting is the baseline answer. The good answer groups paralogs and sequence-similar families together (HGNC gene groups or similarity clustering) and uses GroupKFold on that grouping. Otherwise a gene in train and its paralog in test leaks structure and sequence features across the split.

### 6.2 Study-bias stratification (the killer result)
The naive failure mode of every target-prediction model is learning "is this gene famous." Defend two ways:
- Check via SHAP whether publication count dominates.
- Report performance within bins of similar publication count.

If the model still ranks well among understudied genes, it is doing real work. Lead the writeup with this.

**Implemented status:** the SHAP check was never built (deferred, see README.md "Future work"). The bin-based check was implemented and substantially extended beyond this original two-line plan: see section 6.6 below for the full method (pooled and paired bootstrap, sign test, repeated CV, median split) and README.md's Results section for what it found. The feature-importance mechanism actually used throughout is `GradientBoostingClassifier`'s built-in `feature_importances_`, not SHAP.

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

**Implemented status:** not implemented, deferred (see README.md "Future work"). The repeated-CV analysis in section 6.6 varies fold assignment across 10 seeded repeats, which is adjacent but not a substitute: it tests whether the *evaluation* is stable to which genes land in which fold, not whether *feature selection* is stable to which genes land in the training sample.

### 6.6 Evaluation methodology, as implemented

This is what `ml/train_eval.py` actually does, distilled from working through the statistical corrections in this section's history.

**Split.** Gene-family GroupKFold, 5 folds, groups assigned once in `ml/split.py` and never touched by any downstream analysis. Every fold assignment used anywhere in this document, including the repeated-CV variants below, is checked for zero group-key leakage before its results are trusted.

**Ranking metrics with per-stratum baselines.** PR-AUC alone is not informative across strata of very different positive rates (7.8% overall, under 1% in the low-publication tercile, close to 20% in the high tercile). Every stratum's lift is PR-AUC divided by that stratum's own positive rate, not the global one, so lift greater than 1.0 means the model beats a random ranker working on that same population. This is what makes the low-tercile and bottom-half results interpretable at all: a PR-AUC of 0.03 sounds weak until you see the stratum's own baseline is 0.009.

**Paired bootstrap comparison between variants on identical folds.** When two feature-set variants (say `biology_only` and `no_pubcount`) are trained and evaluated on the exact same GroupKFold split, their per-fold scores are not independent draws, they are correlated because the hard folds are hard for both variants and the easy folds are easy for both. Reporting two separate marginal confidence intervals and checking whether they overlap throws that correlation away and is therefore a systematically underpowered test: two CIs can overlap heavily while the fold-by-fold difference is consistently one-directional. The correct test resamples the paired per-fold differences directly (`bootstrap_paired_diff_ci`), preserving the pairing. This is why the paired test found statistically meaningful, consistent-direction gaps between variants in cases where the marginal CIs alone looked inconclusive.

**Sign test.** A distribution-free complement to the paired bootstrap CI, since 5 folds is too few points for the bootstrap's normal-approximation intuitions to be very trustworthy on their own. Counts how many of the 5 folds favor variant A versus variant B; a lopsided count (e.g. 1-4) corroborates the bootstrap CI's direction without relying on any distributional assumption.

**Repeated CV across 10 fold assignments.** GroupKFold's own fold assignment is one arbitrary draw; some of the spread seen across runs could be fold-assignment luck rather than genuine model instability. `run_repeated_cv` reruns the full training and evaluation loop 10 times with a fresh, seeded fold assignment each time (`make_group_folds`, sklearn's GroupKFold has no `random_state` so this is a hand-rolled equivalent), holding the model seed fixed, and reports the mean/std/min/max of the resulting lift across those 10 repeats. This isolates fold-assignment variance from the bootstrap's resampling variance and confirms the two are of comparable, modest size, not that one is hiding a much larger true instability.

**Median split promoted to primary.** The original stratification was a publication-count tercile (roughly 60 positives per stratum), which turned out to be underpowered for the low-tercile question specifically. A 50/50 median split of the same ranking (159 positives in the bottom half versus 60 in the low tercile) answers the same "does this work on understudied genes" question with much narrower confidence intervals, and is now the headline metric; the tercile numbers are kept as a secondary, explicitly underpowered cross-check for continuity with earlier runs, not because they carry independent evidentiary weight.

---

## 7. Division of labor

**Let the orchestration be efficient (home turf):** Nextflow process definitions, Batch queue config, Spot/on-demand routing, Glue/Athena setup.

**Hand-code and be ready to defend (interview prep):**
- Leakage-safe splitting (gene-family GroupKFold). Implemented.
- Negative-set / PU handling. Implemented (open-world assumption, section 4).
- Feature engineering from annotated variants. Implemented.
- Bootstrap stability selection. Not implemented, deferred (section 6.5, README.md "Future work").
- SHAP interpretability and the study-bias analysis. SHAP not implemented, deferred; the study-bias analysis is implemented using `feature_importances_` instead (section 6.2, section 6.6).
- Ranking metric implementation (enrichment factor, precision@k). Implemented.

---

## 8. Deliverables

1. The repository.
2. This design doc, committed as design rationale.
3. An architecture doc with diagram and the scaling / cost math.
4. A short results writeup: which targets surfaced, performance against the clinical-phase label, study-bias result, Spot strategy savings. Doubles as a LinkedIn post in practitioner voice.

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
| Glue + Athena | $5 to $15 | Athena over Parquet is pennies; Glue crawlers add a little |
| ML training | $0 to $30 | ~20,000 genes x ~100 features fits in laptop memory; train locally for $0 |
| Egress | $0 | Only if everything stays in-region |

The feature matrix is small enough to train locally or on a tiny EC2 instance. The only thing that grows this line is SageMaker, which leads to the killers below.

### The four silent killers (in order of how often they bite)

1. **NAT Gateway (designed out).** Batch in a private subnet usually pulls one in, billing ~$0.045/hr plus data charges whether used or not, ~$32/month idle. This project avoids it entirely by running in public subnets with an S3 gateway endpoint (see section 2 networking). Listed here because it is the most common genomics-on-AWS cost leak and the reason the public-subnet choice was made.
2. **Idle SageMaker notebook.** Spun up, walked away from, billed hourly all weekend. If used at all, stop instances when done. Better: skip SageMaker for training and use it only if wanted on the resume.
3. **Accidental egress.** Pulling large data out of AWS is ~$0.09/GB. One TB down is ~$90 instantly. Rule: the data never leaves the region, only small results do.
4. **Forgetting Spot.** Running parallel, restartable steps on-demand instead of Spot can 3x to 4x the compute line for no benefit.

### Day-one setup (both already familiar from CCP)

- **AWS Budgets alert at $50.** Email warning before anything surprising happens. This alone makes the worst cases nearly impossible.
- **Tags on every resource.** Makes spend attributable at a glance. Terraform makes this trivial via default tags.
- **`terraform destroy` between sessions.** With the NAT Gateway designed out, the remaining idle costs are small, but destroy is still the clean discipline: one command tears the whole stack down, `terraform apply` rebuilds it next session. Converts "remember to manually delete every resource" into a single reliable habit and guarantees nothing is left billing.

### Framing for the writeup

State it as "ran the live benchmark for under $X, with full-100TB cost modeled at $Y." That is itself a credible FinOps detail interviewers like to see, and it ties back to the capacity-math story in section 2.

---

## 10. Timeline (3 to 4 weeks, part-time)

| Week | Focus |
|------|-------|
| 1 | Infra + data ingestion + one Nextflow process running on Batch |
| 2 | Full annotation pipeline + Glue/Athena evidence layer |
| 3 | Hand-coded ML layer (features, leakage-safe eval, stability selection, SHAP) |
| 4 | Scaling benchmark, cost writeup, docs, polish |

---

## 11. Open design calls to resolve during build

- Negative-set definition / PU strategy (section 4). Highest priority.
- Exact clinical-phase cutoff for the label and for the temporal holdout.
- Gene-family grouping source (HGNC groups vs sequence-similarity clustering).
- Subset size for the live run vs the scaling-math extrapolation.
- Current Open Targets schema version and field mappings.
