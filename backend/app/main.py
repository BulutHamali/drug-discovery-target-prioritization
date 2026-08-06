from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.auth import get_current_user, validate_access_token
from app.api.deps import require_admin
from app.core.config import get_settings
from app.models import AdminOverview, RunCreate, RunStatus
from app.runner import TargetRunner
from app.storage import RunStore

settings = get_settings()
store = RunStore(settings.runs_dir)
runner = TargetRunner(store, settings)
app = FastAPI(title="Drug Target Prioritization API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def cognito_middleware(request, call_next):  # type: ignore[no-untyped-def]
    if settings.auth_required and request.url.path != "/health" and request.method != "OPTIONS":
        try:
            request.state.user = get_current_user(request)
        except Exception as exc:  # HTTPException becomes a stable JSON response.
            return JSONResponse(status_code=getattr(exc, "status_code", 401), content={"detail": str(getattr(exc, "detail", exc))})
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/overview", response_model=AdminOverview, dependencies=[Depends(require_admin)])
def admin_overview() -> AdminOverview:
    runs = store.all()
    return AdminOverview(public_demo_mode=settings.public_demo_mode, auth_required=settings.auth_required, execution_mode=settings.execution_mode, runs=len(runs), active_runs=sum(run.state in {"pending", "running"} for run in runs))


@app.get("/admin/runs", response_model=list[RunStatus], dependencies=[Depends(require_admin)])
def list_runs() -> list[RunStatus]:
    return store.all()


@app.post("/admin/runs", response_model=RunStatus, status_code=202, dependencies=[Depends(require_admin)])
def submit_run(body: RunCreate) -> RunStatus:
    return runner.submit(body)


@app.get("/admin/runs/{run_id}", response_model=RunStatus, dependencies=[Depends(require_admin)])
def get_run(run_id: str) -> RunStatus:
    run = store.get(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run
