output "vpc_id" {
  description = "VPC ID."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs for Batch."
  value       = aws_subnet.public[*].id
}

output "batch_job_queue" {
  description = "Batch job queue name, referenced by the Nextflow config."
  value       = aws_batch_job_queue.main.name
}

output "batch_job_role_arn" {
  description = "Job role ARN for container permissions."
  value       = aws_iam_role.job.arn
}

output "work_bucket_name" {
  description = "S3 bucket for the Nextflow work directory and pipeline outputs."
  value       = aws_s3_bucket.work.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for the BURDEN/COLLECT image. Tag and push here, then reference it in nextflow.config's awsbatch profile."
  value       = aws_ecr_repository.burden.repository_url
}

output "ecr_bcftools_batch_repository_url" {
  description = "ECR repository URL for the PREPARE/ANNOTATE (bcftools + awscli) image."
  value       = aws_ecr_repository.bcftools_batch.repository_url
}
