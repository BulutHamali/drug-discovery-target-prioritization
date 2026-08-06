import json
from pathlib import Path

from app.models import RunStatus


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: RunStatus) -> RunStatus:
        (self.root / f"{run.run_id}.json").write_text(run.model_dump_json(indent=2) + "\n")
        return run

    def get(self, run_id: str) -> RunStatus | None:
        path = self.root / f"{run_id}.json"
        return RunStatus.model_validate(json.loads(path.read_text())) if path.exists() else None

    def all(self) -> list[RunStatus]:
        return [run for path in sorted(self.root.glob("*.json")) if (run := self.get(path.stem))]
