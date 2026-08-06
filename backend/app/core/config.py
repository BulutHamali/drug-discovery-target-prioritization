from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TARGET_", env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    public_demo_mode: bool = True
    auth_required: bool = False
    admin_group: str = "target-prioritization-admin"
    cognito_issuer: str = ""
    cognito_client_id: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    execution_mode: Literal["disabled", "aws_batch"] = "disabled"
    aws_region: str = "us-east-1"
    batch_job_queue: str = ""
    batch_job_definition: str = ""
    artifact_bucket: str = ""
    artifact_prefix: str = "target-prioritization"

    @property
    def cognito_jwks_uri(self) -> str:
        return f"{self.cognito_issuer.rstrip('/')}/.well-known/jwks.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    @property
    def runs_dir(self) -> Path:
        path = self.data_dir / "admin_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
