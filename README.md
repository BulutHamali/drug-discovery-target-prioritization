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
pipeline/         Nextflow pipeline (to add)
ml/               Feature engineering, training, evaluation (to add)
data/             Data-layer build scripts: Open Targets, ChEMBL (to add)
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

```
# to add
```

## Cost

Target $30 to $80 for the full project. Public in-region data, Spot instances, no NAT Gateway, and `terraform destroy` between sessions keep it there. An AWS Budgets alert at $50 is set on day one. See `DESIGN.md` section 9 for the full breakdown.

## Status checklist

- [ ] Terraform stack applies and destroys cleanly
- [ ] Pipeline validated end to end on one sample
- [ ] Data layer (Open Targets, ChEMBL) built and labels defined
- [ ] ML layer built and validated locally
- [ ] Scaling benchmark run after quota increase
- [ ] Results writeup and architecture diagram
