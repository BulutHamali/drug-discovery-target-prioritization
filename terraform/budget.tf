# ---------------------------------------------------------------------------
# AWS Budgets: cost alarm
#
# A monthly cost budget that emails before spend becomes a surprise. This
# guardrail needs to exist before the first resource is created, not after
# the first cloud run, so it is applied in the same terraform apply as
# everything else.
#
# Two alert thresholds:
#   ACTUAL     at 80%  - spend has already crossed $40, early warning.
#   FORECASTED at 100% - AWS projects month-end spend will exceed $50.
# The forecasted alert is the more useful one: it warns based on trend before
# the money is actually gone.
# ---------------------------------------------------------------------------

resource "aws_budgets_budget" "monthly_cost" {
  name         = "${var.project_name}-monthly-cost"
  budget_type  = "COST"
  limit_amount = var.budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Alert when ACTUAL spend reaches 80% of the limit.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  # Alert when FORECASTED month-end spend is projected to exceed 100%.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
