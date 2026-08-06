# Public demo and admin research modes

The results explorer is intentionally split into two modes:

## Public demo mode

Public mode is read-only. It exposes sanitized, verified result artifacts and
the scientific interpretation of those results. It must not accept sensitive
data, launch AWS jobs, or mutate the published demo artifact.

## Admin research mode

Admin mode is a control-plane UI for authenticated operators. The browser may
configure a run, but a protected backend must perform the work:

```text
Admin browser → authenticated API → AWS Batch / S3 / ECR
                             ↓
                      run manifest + artifact
                             ↓
                 explicit publish-to-demo approval
```

AWS credentials must never be sent to or stored in the browser. The API should
enforce authorization server-side, record an audit event for each launch and
publish action, estimate cost before submission, and expose job status/logs
without exposing private data to public mode.

The current frontend includes the mode switch and admin control-console
surface. The controls remain disabled until the protected execution API is
implemented and configured.
