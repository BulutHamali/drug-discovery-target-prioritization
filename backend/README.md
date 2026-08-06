# Protected execution API

This is the target-prioritization equivalent of Reproducell's protected API
boundary. It validates Cognito access tokens, restricts Admin Research routes
to the `TARGET_ADMIN_GROUP`, persists run state, and exposes an execution seam
for AWS Batch.

Run locally:

```bash
cd backend
python3 -m pip install -e .
TARGET_AUTH_REQUIRED=false uvicorn app.main:app --reload --port 8000
```

For a controlled deployment, configure:

```text
TARGET_AUTH_REQUIRED=true
TARGET_COGNITO_ISSUER=https://cognito-idp.<region>.amazonaws.com/<pool-id>
TARGET_COGNITO_CLIENT_ID=<public-app-client-id>
TARGET_ADMIN_GROUP=target-prioritization-admin
TARGET_EXECUTION_MODE=aws_batch
TARGET_BATCH_JOB_QUEUE=<queue>
TARGET_BATCH_JOB_DEFINITION=<job-definition>
TARGET_ARTIFACT_BUCKET=<private-artifact-bucket>
```

The current runner records an auditable run and fails closed when execution is
disabled. The next adapter submits the allowlisted stages to the existing
Nextflow/AWS Batch pipeline; AWS credentials are never accepted from the UI.
