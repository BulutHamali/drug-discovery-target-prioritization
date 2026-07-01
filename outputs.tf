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
