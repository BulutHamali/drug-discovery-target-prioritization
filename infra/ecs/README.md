# ECS Fargate Deployment

## Prerequisites

- AWS CLI configured with appropriate permissions
- ECR repository: `drug-target-api`
- ECS cluster already created
- Cognito user pool and app client configured
- Secrets stored in AWS Secrets Manager

## Build and push the Docker image

```bash
# From the repo root
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Authenticate
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push
docker build -t drug-target-api ./backend
docker tag drug-target-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/drug-target-api:latest
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/drug-target-api:latest
```

## Register task definition

Edit `task-definition.example.json` replacing placeholder values:
- `ACCOUNT_ID` → your AWS account ID
- `REGION` → your AWS region
- `TARGET_CORS_ORIGINS` → your Vercel frontend URL
- Secrets ARNs → your Secrets Manager ARNs

```bash
aws ecs register-task-definition \
  --cli-input-json file://infra/ecs/task-definition.example.json
```

## Create or update service

```bash
aws ecs create-service \
  --cluster your-cluster-name \
  --service-name drug-target-api \
  --task-definition drug-target-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

## Environment variables (TARGET_ prefix)

| Variable | Default | Description |
|---|---|---|
| `TARGET_AUTH_REQUIRED` | `false` | Require Cognito token on admin routes |
| `TARGET_PUBLIC_DEMO_MODE` | `true` | Disable data uploads and job submission |
| `TARGET_COGNITO_ISSUER` | `""` | Cognito user pool issuer URL |
| `TARGET_COGNITO_CLIENT_ID` | `""` | Cognito app client ID |
| `TARGET_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `TARGET_EXECUTION_MODE` | `disabled` | `disabled` or `aws_batch` |
| `TARGET_AWS_REGION` | `us-east-1` | AWS region for Batch/S3 |

## Frontend deployment (Vercel)

The frontend deploys automatically from `frontend/` via Vercel. The `vercel.json` SPA rewrite handles client-side routing.

Set the following Vercel environment variable if you want the frontend to call the real backend:
```
VITE_API_BASE_URL=https://your-alb-or-ecs-endpoint.example.com
```

Without this, the frontend loads `targets.json` from the static bundle and shows the public demo. No backend is required for the demo.
