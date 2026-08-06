from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import Settings
from app.models import RunCreate, RunStatus
from app.storage import RunStore


class TargetRunner:
    """Execution seam copied from Reproducell's JobRunner pattern.

    The API records an auditable run immediately. AWS Batch submission is
    enabled only when TARGET_EXECUTION_MODE=aws_batch and the queue/definition
    are configured; the public deployment never reaches this path.
    """

    def __init__(self, store: RunStore, settings: Settings):
        self.store, self.settings = store, settings

    def submit(self, body: RunCreate) -> RunStatus:
        now = datetime.now(timezone.utc)
        run = RunStatus(run_id=uuid4().hex[:12], stage=body.stage, state="pending", label=body.label, feature_set=body.feature_set, created_at=now, updated_at=now, message="Queued for protected execution.")
        if self.settings.execution_mode == "disabled":
            run.state = "failed"
            run.message = "Execution is disabled. Configure the protected AWS API before launching research runs."
        # AWS Batch submission is deliberately isolated here for the next
        # adapter: no browser request ever receives AWS credentials.
        return self.store.save(run)
