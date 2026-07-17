# drug-discovery-target-prioritization

An AI-driven pipeline that turns population genetic data into ML-ranked druggable targets, scored against real clinical outcomes. Nextflow on AWS Batch for the pipeline, Terraform for the infrastructure, and a leakage-safe ML layer for target prioritization.

Status: in development. See `DESIGN.md` for the full design rationale.

## What it does

1. Ingests public population variant data (1000 Genomes, AWS Open Data, in-region).
2. Runs a Nextflow pipeline on AWS Batch to produce gene-level genetic evidence.
3. Assembles a per-gene feature matrix (constraint, protein-intrinsic, network, expression).
4. Trains a leakage-safe model to prioritize genes by druggability, using clinical-phase drug existence as the label.
5. Outputs a ranked target list with SHAP interpretability and a study-bias analysis.

## Why it is built this way

Target identification is the highest-value and highest-failure decision in drug discovery. This project predicts a genuine downstream outcome (does a target have a drug at clinical phase >= 1) from biology alone, with evaluation designed to prove the model is not simply learning which genes are famous. Full reasoning in `DESIGN.md`.

## Architecture

```
Public data (S3, in-region)
        |
   Nextflow on AWS Batch
     - Spot for per-sample parallel steps
     - On-demand for aggregation
        |
   Glue + Athena (tabular evidence layer)
        |
   ML training (target prioritization)
        |
   Ranked target list + SHAP
```

Networking uses public subnets with an Internet Gateway and an S3 Gateway VPC Endpoint, deliberately avoiding a NAT Gateway for cost. All infrastructure is provisioned with Terraform.

## Repository layout

```
terraform/        Infrastructure as code (VPC, Batch, IAM, S3 endpoint)
pipeline/         Nextflow pipeline (PREPARE, ANNOTATE, BURDEN, COLLECT)
ml/               Feature engineering, leakage-safe split, training, evaluation
data/             Data-layer build scripts: Open Targets, ChEMBL
DESIGN.md         Full design rationale
README.md         This file
```

## Getting started

### 1. Provision infrastructure

```
cd terraform
terraform init
terraform apply
```

Tear down between work sessions to guarantee nothing is left billing:

```
terraform destroy
```

### 2. Run the pipeline (single-sample validation)

Validates the full DAG on one sample inside the default vCPU limit, before any quota increase.

```bash
# Fetch the chr22 VCF and reference files (run from us-east-1 for zero egress)
bash data/fetch_1000genomes.sh
bash data/fetch_ref.sh

# Build the Python container for the burden and collect steps.
# --platform linux/amd64 is required: the pipeline requests amd64 and an ARM Mac
# otherwise builds an arm64 image that the run cannot find.
docker build --platform linux/amd64 --load -t drug-target-burden:1.0 pipeline/docker/

# Run the local validation (chr22, all samples, AF < 1%)
nextflow run pipeline/main.nf -profile local
```

Results land in `results/gene_burden_features.parquet`.

### 3. Build the ML layer (runs locally, no AWS needed)

Each script is idempotent and caches its output in `ml/cache/`. Run them in
order; each step checks that its inputs exist and exits with a clear error if
a prerequisite is missing.

```bash
# Dependencies (once)
pip install pandas pyarrow scikit-learn networkx

# Step 1: download HGNC protein-coding gene universe and build group keys
# for the family-safe cross-validation split.
python3 ml/gene_families.py

# Step 2: download gnomAD v2.1.1 constraint metrics (pLI, LOEUF, oe_lof, oe_mis).
# ~4.6 MB download, cached to ml/cache/.
python3 ml/fetch_gnomad.py

# Step 3: download UniProt Swiss-Prot protein features (protein_length).
# ~3-5 MB download, cached to ml/cache/.
python3 ml/fetch_alphafold.py

# Step 4: download STRING v12 PPI network and compute per-gene degree and
# approximate betweenness centrality (~85 MB download, ~2-5 min to compute).
python3 ml/fetch_string.py

# Step 5: download GTEx v8 median tissue TPM (compute tau, tissue-specificity
# index) and DepMap 24Q4 CRISPR gene effect (mean essentiality score across
# cell lines). ~7 MB + ~430 MB download; the DepMap download is the long pole.
python3 ml/fetch_expression.py

# Step 6: download NCBI gene2pubmed (~272 MB) and compute publication count
# and first-described year per gene -- the deliberate confounder feature
# (DESIGN.md section 5), used by the study-bias check in step 9.
python3 ml/fetch_publications.py

# Step 7: fetch Open Targets label data (knownDrugsAggregated).
python3 data/fetch_chembl_known_drugs.py

# Step 8: assemble the training table (gene universe + gnomAD + AlphaFold +
# STRING + GTEx/DepMap + publication metadata + burden + label). Requires
# results/gene_burden_features.parquet from step 2 of the pipeline.
python3 ml/build_features.py

# Step 9: GroupKFold split on gene family -- prevents paralog leakage.
# Asserts zero group overlap in every fold.
python3 ml/split.py

# Step 10: train and evaluate. Prints PR-AUC, precision@k, and enrichment
# factor per fold and averaged, plus the study-bias check (score vs.
# publication count, PR-AUC by pub-count tercile). Writes OOS predictions
# to ml/cache/.
python3 ml/train_eval.py
```

Outputs:
- `ml/cache/gene_families.parquet` -- gene universe with group keys
- `ml/cache/gnomad_constraint.parquet` -- constraint metrics
- `ml/cache/alphafold_features.parquet` -- protein length (UniProt Swiss-Prot)
- `ml/cache/string_features.parquet` -- PPI degree and betweenness (STRING v12)
- `ml/cache/expression_features.parquet` -- tissue-specificity (tau, GTEx) and essentiality (DepMap)
- `ml/cache/publication_features.parquet` -- pub_count and year_first_described (NCBI gene2pubmed)
- `ml/cache/training_table.parquet` -- full feature matrix (19,296 genes, 26 columns)
- `ml/cache/cv_folds.parquet` -- fold assignments (GroupKFold, n=5)
- `ml/cache/oos_predictions.parquet` -- out-of-sample scores, labels, and ranks

## Cost

Target $30 to $80 for the full project. Public in-region data, Spot instances, no NAT Gateway, and `terraform destroy` between sessions keep it there. An AWS Budgets alert at $50 is set on day one. See `DESIGN.md` section 9 for the full breakdown.

## Status checklist

- [x] Terraform stack applies and destroys cleanly
- [x] Pipeline validated end to end on one sample
- [x] Data layer (Open Targets, ChEMBL) built and labels defined
- [x] ML layer built and validated locally
- [ ] Scaling benchmark run after quota increase
- [ ] Results writeup and architecture diagram
