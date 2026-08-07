import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth middleware ──────────────────────────────────────────────────────────

@app.middleware("http")
async def cognito_middleware(request, call_next):  # type: ignore[no-untyped-def]
    public_path = request.url.path.startswith("/api/public/")
    if (
        settings.auth_required
        and request.url.path != "/health"
        and not public_path
        and request.method != "OPTIONS"
    ):
        try:
            request.state.user = get_current_user(request)
        except Exception as exc:
            origin = request.headers.get("origin")
            cors_headers = (
                {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
                if origin in settings.cors_origin_list
                else {}
            )
            return JSONResponse(
                status_code=getattr(exc, "status_code", 401),
                content={"detail": str(getattr(exc, "detail", exc))},
                headers=cors_headers,
            )
    return await call_next(request)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ─── Public read-only API ─────────────────────────────────────────────────────

def _load_targets() -> dict:
    """Load targets.json from the data/public_demo directory."""
    candidates = [
        settings.data_dir / "public_demo" / "targets.json",
        Path(__file__).parent.parent / "data" / "public_demo" / "targets.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    raise HTTPException(status_code=503, detail="Public demo data not available.")


@app.get("/api/public/targets")
def public_targets() -> dict:
    """Return the full targets.json payload (synthetic demo)."""
    return _load_targets()


@app.get("/api/public/summary")
def public_summary() -> dict:
    return {
        "n_genes": 500,
        "n_known": 50,
        "enrichment_top1": 5.59,
        "ts_tv_ratio": 2.11,
        "burden_coverage": 0.8668,
        "synthetic_demo": True,
    }


@app.get("/api/public/provenance")
def public_provenance() -> dict:
    return {
        "methodology": "GroupKFold cross-validation with gene-family grouping to prevent leakage. Temporal holdout: model trained on Open Targets 2021 labels, evaluated on genes gaining clinical-phase status by 2026.",
        "feature_sources": [
            "gnomAD genetic constraint (pLI, LOEUF)",
            "AlphaFold protein structure quality",
            "STRING protein-protein interaction network",
            "GTEx tissue expression",
            "DepMap essentiality scores",
            "PubMed publication count",
        ],
        "limitations": [
            "The biology_only model does not clearly beat DepMap essentiality alone (2.95× vs 5.03× at top 1%).",
            "High rank indicates genetic importance, not clinical tractability.",
            "Variant burden is from 1000 Genomes (2,504 individuals) — smaller than gnomAD.",
            "This demo uses a synthetic target file; real ML outputs require running the pipeline.",
        ],
        "data_version": "synthetic-demo-v1",
        "generated_at": "2026-08-07T00:00:00+00:00",
    }


# ─── Admin API (protected) ────────────────────────────────────────────────────

@app.get("/admin/overview", response_model=AdminOverview, dependencies=[Depends(require_admin)])
def admin_overview() -> AdminOverview:
    runs = store.all()
    return AdminOverview(
        public_demo_mode=settings.public_demo_mode,
        auth_required=settings.auth_required,
        execution_mode=settings.execution_mode,
        runs=len(runs),
        active_runs=sum(run.state in {"pending", "running"} for run in runs),
    )


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
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run
