# AWS Batch: Spot compute environment + job queue.
# max_vcpus matches the account's confirmed Spot vCPU quota (8) in us-east-1,
# so multiple chromosomes can run concurrently rather than one at a time.
# min_vcpus stays at 0: a warm idle instance costs money continuously
# (~$0.72/day observed for a single 2-vCPU instance), while scaling from
# zero only costs a few minutes of wall clock per process.

resource "aws_batch_compute_environment" "spot" {
  compute_environment_name = "${var.project_name}-spot"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    max_vcpus           = var.max_vcpus
    min_vcpus           = 0

    instance_role      = aws_iam_instance_profile.ecs_instance.arn
    instance_type      = ["optimal"]
    subnets            = aws_subnet.public[*].id
    security_group_ids = [aws_security_group.batch.id]
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "main" {
  name     = "${var.project_name}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}
