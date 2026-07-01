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

Canonical AWS genomics reference architecture:

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
| Target-disease evidence | Open Targets | Public ground-truth backbone |
| Known drugs / clinical phase | ChEMBL / Open Targets known-drug evidence | Label source |
| Constraint | gnomAD (pLI, LOEUF, o/e) | Genetic features |
| Structure | AlphaFold | Protein-intrinsic features |
| Network | STRING | PPI features |
| Expression / essentiality | GTEx, DepMap | Expression and safety features |

Verify all feature and label structure against the current Open Targets data model. Their schema versions shift between releases.

---

## 4. The label (the trap)

The obvious label, predicting the Open Targets association score, is quietly circular: Open Targets computes that score partly from genetic evidence, so the model would predict its own inputs. An interviewer catches this immediately.

**Chosen label:** does this target have a drug at clinical phase >= 1 (or approved)? Pulled from ChEMBL / Open Targets known-drug evidence. All drug and clinical evidence is excluded from features. This predicts a genuine real-world outcome from biology rather than predicting inputs from inputs. A continuous variant (max clinical phase) supports a regression framing.

**Open-world caveat to raise proactively:** absence of a drug does not mean undruggable; it may mean understudied. This is really positive-unlabeled (PU) learning, not clean binary classification. Either handle with a PU approach or define hard negatives carefully and state the open-world assumption explicitly. This is the single hardest design call in the project and the one a sharp interviewer will push hardest on. Do not let it slide.

---

## 5. Feature set

Aggregated per gene / protein, grouped by source.

**Genetic constraint and variant burden.** gnomAD constraint (pLI, LOEUF), rare / LoF variant burden, observed-vs-expected ratios, conservation. This is where the Nextflow pipeline output feeds the model, tying the project together. Among the most predictive and most biologically honest features.

**Protein-intrinsic.** Length, domain families, disorder fraction, pocket / structure features derived from AlphaFold. Caution: target-class membership (kinase, GPCR, ion channel) is almost too predictive of tractability and can become a shortcut. Keep it but watch its SHAP weight.

**Network.** PPI centrality from STRING (degree, betweenness), pathway membership. Strong signal but heavily confounded by study bias, since well-studied proteins have more measured interactions.

**Expression and essentiality.** Tissue specificity (tau) from GTEx, cell essentiality from DepMap. Tissue-restricted expression is a real safety and druggability signal.

**The confounder, included on purpose.** Publication count and year-first-described. Included specifically so the model can be shown not to be riding it.

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

### 6.3 Ranking metrics, not just AUC
Prioritization is a top-of-list problem, like virtual screening; nobody pursues 8,000 targets. Report:
- PR-AUC (positives are rare).
- Precision@k.
- Enrichment factor at the top 1 to 5 percent.

Enrichment factor speaks pharma's native language and signals understanding of the actual decision.

### 6.4 Temporal holdout
Train on targets that reached the clinic before a cutoff year, then predict which genes reached clinical stage after it, using only pre-cutoff values for time-dependent features (publication counts especially). Even a partial temporal split is the most honest "could this have prospectively found a real target" test, and almost no portfolio project does it.

### 6.5 Bootstrap stability selection
Resample, refit, keep features selected in more than X percent of runs. Reports robust biology and guards against overfitting to the famous-gene signal. SHAP + bootstrap approach drops in cleanly here.

---

## 7. Division of labor

**Let the orchestration be efficient (home turf):** Nextflow process definitions, Batch queue config, Spot/on-demand routing, Glue/Athena setup.

**Hand-code and be ready to defend (interview prep):**
- Leakage-safe splitting (gene-family GroupKFold).
- Negative-set / PU handling.
- Feature engineering from annotated variants.
- Bootstrap stability selection.
- SHAP interpretability and the study-bias analysis.
- Ranking metric implementation (enrichment factor, precision@k).

---

## 8. Deliverables

1. The repository.
2. This design doc, committed as design rationale.
3. An architecture doc with diagram and the scaling / cost math.
4. A short results writeup: which targets surfaced, performance against the clinical-phase label, study-bias result, Spot strategy savings. Doubles as a LinkedIn post in practitioner voice.

---

## 9. Cost

Target: $30 to $80 for the whole project if disciplined. It can exceed $200 if a few specific things are left running. The difference is discipline, not scale. All figures are orders of magnitude; AWS pricing drifts.

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
