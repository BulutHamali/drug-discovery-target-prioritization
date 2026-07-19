# ECR repository for the BURDEN/COLLECT container image.
#
# AWS Batch pulls container images from a registry, not the local Docker
# daemon. The image built locally for linux/amd64 (pipeline/docker/Dockerfile)
# has no home in a registry yet, so a Batch job referencing it would fail at
# image pull. This repository is that home; the image still needs to be
# tagged and pushed here after apply.

resource "aws_ecr_repository" "burden" {
  name                 = "drug-target-burden"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # Let terraform destroy remove the repository even if it still holds
  # images, matching the "tears down cleanly" goal in DESIGN.md section 2.
  force_delete = true
}

# ECR repository for the PREPARE/ANNOTATE image: the same bcftools
# biocontainer the `local` profile uses, plus the AWS CLI (see
# pipeline/docker/bcftools-batch/Dockerfile for why). Nextflow's awsbatch
# executor needs the CLI inside every container it runs, not just our own.
resource "aws_ecr_repository" "bcftools_batch" {
  name                 = "bcftools-batch"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  force_delete = true
}
