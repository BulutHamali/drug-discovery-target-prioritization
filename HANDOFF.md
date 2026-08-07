# Handoff: Drug Target Prioritization

## What was built

### Landing page (`frontend/src/App.tsx`)

A new `showLanding` boolean state controls whether the app shows the landing page or the sidebar workspace. The landing page matches scRNA Reproducell quality:

- Hero with "5.59×" enrichment visual card (right column) and copy (left column)
- "Explore the targets →" CTA button enters the workspace
- Feature section (id="how-it-works") with 3 cards
- Workflow section (3 pipeline steps)
- CTA section at the bottom
- Theme toggle (dark/light, stored in `localStorage["dt-theme"]`)
- "← Home" button in workspace topbar returns to landing

### Synthetic targets data

- `frontend/public/data/targets.json` — 500 genes (50 known drug targets at ranks 1-50, 276 realistic unlabeled genes, 174 synthetic GENE_XXXXX names)
- `frontend/scripts/generate_synthetic_targets.py` — reproducible generator (seed=42)
- `backend/data/public_demo/targets.json` — same data, served by the backend

### CSS refactor (`frontend/src/styles.css`)

Replaced the minified single-line CSS with clean, readable CSS that:
- Adds CSS custom properties: `--text`, `--bg`, `--panel-bg`, `--border`, `--accent`, `--muted`, `--code-bg`
- Adds dark mode (`html[data-theme="dark"]`) overrides
- Adds landing page styles: `.landing-page`, `.landing-hero`, `.hero-copy`, `.hero-visual`, `.hero-actions`, `.landing-proof`, `.feature-section`, `.feature-grid`, `.feature-card`, `.workflow-section`, `.landing-cta`
- Keeps all existing workspace styles (sidebar, panel, metrics, charts, target table, etc.)
- DM Sans + Manrope fonts unchanged

### FastAPI backend (`backend/`)

Updated `backend/app/main.py` to add:
- `GET /api/public/targets` — returns targets.json payload (no auth)
- `GET /api/public/summary` — key metrics (no auth)
- `GET /api/public/provenance` — methodology and limitations (no auth)
- Middleware passes through `/api/public/` without auth check

Existing admin routes (`/admin/overview`, `/admin/runs`) unchanged.

### Frontend API (`frontend/src/targetApi.ts`)

Added `publicApi` export:
```typescript
export const publicApi = {
  summary: () => fetch(`${API_BASE}/api/public/summary`).then(r => r.json()),
  provenance: () => fetch(`${API_BASE}/api/public/provenance`).then(r => r.json()),
};
```

### Deploy files

- `frontend/vercel.json` — SPA rewrite
- `infra/ecs/task-definition.example.json` — Fargate 2vCPU/4GB task definition
- `infra/ecs/README.md` — deployment instructions

### Playwright smoke tests

- `frontend/playwright.config.ts` — runs against `npm run preview` on port 4173
- `frontend/tests/e2e/smoke.spec.ts` — 10 tests covering:
  - Landing page brand and enrichment number visible
  - "Explore the targets" enters workspace
  - Workspace sidebar visible
  - Home button returns to landing
  - No AWS credentials or S3 paths in DOM
  - No sign-in prompts on landing
  - Theme toggle works

## What's left (for production)

### Real ML outputs

The `frontend/public/data/targets.json` is synthetic. To replace it:

```bash
# Run the full ML pipeline
python3 ml/train_eval.py --feature-set biology_only

# Export to frontend
cd frontend && npm run export-targets
```

### Real AWS Batch backend

Set `TARGET_EXECUTION_MODE=aws_batch` and configure:
- `TARGET_BATCH_JOB_QUEUE`
- `TARGET_BATCH_JOB_DEFINITION`
- `TARGET_ARTIFACT_BUCKET`

### Cognito auth for admin

Set `TARGET_AUTH_REQUIRED=true` and configure:
- `TARGET_COGNITO_ISSUER`
- `TARGET_COGNITO_CLIENT_ID`

Create a Cognito user pool group named `target-prioritization-admin` and add admin users.

## Quick start

```bash
# Frontend dev server
cd frontend && npm install && npm run dev

# Backend dev server (optional — frontend works without it)
cd backend && pip install -e . && uvicorn app.main:app --reload

# Playwright tests
cd frontend && npx playwright install chromium
npm run build && npx playwright test --reporter=line
```
