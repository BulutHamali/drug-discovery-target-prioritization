# S3 work bucket for the Nextflow work directory and pipeline outputs.
# Name is suffixed with the AWS account ID to guarantee global uniqueness
# without a random provider dependency.
#
# Lifecycle rule: expire the work/ prefix after 7 days.
# Pipeline outputs (results/) are not subject to expiry.
# This matches the cost discipline in DESIGN.md section 9.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "work" {
  bucket = "${var.project_name}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.project_name}-work"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "work" {
  bucket = aws_s3_bucket.work.id

  rule {
    id     = "expire-nextflow-work"
    status = "Enabled"

    filter {
      prefix = "work/"
    }

    expiration {
      days = 7
    }
  }
}
