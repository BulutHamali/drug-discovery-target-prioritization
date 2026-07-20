variable "region" {
  description = "AWS region. Must match the region hosting the public 1000 Genomes data (us-east-1) to avoid egress."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name, used for tagging and resource naming."
  type        = string
  default     = "drug-discovery-target-prioritization"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for the public subnets, one per AZ."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "availability_zones" {
  description = "Availability zones for the public subnets."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "max_vcpus" {
  description = "Max vCPUs for the Batch compute environment. Matches the account's confirmed Spot vCPU quota in us-east-1 (8), so multiple chromosomes can run concurrently instead of one at a time. min_vcpus stays at 0 (see aws_batch_compute_environment.spot in batch.tf): a warm idle instance costs real money around the clock, while a cold start only costs a few minutes of wall clock per process."
  type        = number
  default     = 8
}

variable "budget_limit" {
  description = "Monthly cost budget in USD. Alerts fire as spend approaches this."
  type        = string
  default     = "50"
}

variable "budget_alert_email" {
  description = "Email address that receives budget alerts. Set in terraform.tfvars (gitignored), not here."
  type        = string
}
