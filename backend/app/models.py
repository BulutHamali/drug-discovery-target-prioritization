from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    stage: Literal["prepare", "feature_assembly", "evaluate", "export"]
    feature_set: Literal["biology_only", "all_features", "no_pubcount", "no_pubcount_no_string"] = "biology_only"
    label: str = Field(min_length=1, max_length=120)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunStatus(BaseModel):
    run_id: str
    stage: str
    state: Literal["pending", "running", "succeeded", "failed"]
    label: str
    feature_set: str
    message: str = ""
    created_at: datetime
    updated_at: datetime
    artifact_uri: str | None = None


class AdminOverview(BaseModel):
    public_demo_mode: bool
    auth_required: bool
    execution_mode: str
    runs: int
    active_runs: int
