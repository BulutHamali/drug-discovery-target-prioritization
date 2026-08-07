import { useEffect, useMemo, useState } from "react";
import { authEnabled, currentUser, finishCallback, isAdmin, signIn, signOut } from "./auth";
import { targetApi, type AdminOverview, type AdminRun } from "./targetApi";

type View = "overview" | "targets" | "results" | "admin";
type Target = { symbol: string; score: number; label: number; rank: number; fold_idx: number };
type TargetPayload = { generated_at: string; source: string; feature_set: string; rows: number; targets: Target[] };

const holdout = [
  { label: "1%", observed: 0.056, baseline: 0.010, low: 0.000, high: 0.021, lift: "5.59×" },
  { label: "5%", observed: 0.178, baseline: 0.050, low: 0.027, high: 0.074, lift: "3.57×" },
  { label: "10%", observed: 0.322, baseline: 0.100, low: 0.071, high: 0.133, lift: "3.22×" },
  { label: "20%", observed: 0.533, baseline: 0.201, low: 0.160, high: 0.249, lift: "2.66×" },
];

const variants = [
  ["all_features", 5.26, "3.97–7.65"],
  ["no_pubcount", 5.60, "4.16–8.21"],
  ["no_pubcount_no_string", 4.52, "3.27–7.04"],
  ["biology_only", 2.68, "2.17–3.85"],
];

function Metric({ value, label, tone = "ink" }: { value: string; label: string; tone?: string }) {
  return (
    <div className={`metric ${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function EnrichmentChart() {
  const width = 720, height = 265, left = 48, top = 22, bottom = 42, right = 24;
  const x = (i: number) => left + i * ((width - left - right) / 3);
  const y = (v: number) => top + (0.6 - v) * ((height - top - bottom) / 0.6);
  const points = (key: "observed" | "baseline") => holdout.map((d, i) => `${x(i)},${y(d[key])}`).join(" ");
  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Prospective enrichment curve">
        {[0, 0.2, 0.4, 0.6].map((v) => (
          <g key={v}>
            <line x1={left} x2={width - right} y1={y(v)} y2={y(v)} className="gridline" />
            <text x={left - 10} y={y(v) + 4} textAnchor="end">{Math.round(v * 100)}%</text>
          </g>
        ))}
        <polygon
          points={holdout.map((d, i) => `${x(i)},${y(d.high)}`).concat([...holdout].reverse().map((d, j) => `${x(3 - j)},${y(d.low)}`)).join(" ")}
          className="confidence"
        />
        <polyline points={points("baseline")} className="baseline-line" />
        <polyline points={points("observed")} className="observed-line" />
        {holdout.map((d, i) => (
          <g key={d.label}>
            <circle cx={x(i)} cy={y(d.observed)} r="5" className="observed-dot" />
            <text x={x(i)} y={height - 17} textAnchor="middle">top {d.label}</text>
          </g>
        ))}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-observed" /> observed prospective positives</span>
        <span><i className="legend-baseline" /> resampled baseline</span>
        <span><i className="legend-band" /> baseline 95% CI</span>
      </div>
    </div>
  );
}

function ForestChart() {
  return (
    <div className="forest-chart">
      {variants.map(([name, value, ci]) => (
        <div className="forest-row" key={name}>
          <span className="forest-name">{name}</span>
          <div className="forest-track">
            <i style={{ left: `${Number(value) / 9.5 * 100}%` }} />
            <b style={{ left: `${Number(value) / 9.5 * 100}%` }} />
            <em style={{
              left: `${(Number(String(ci).split("–")[0]) / 9.5) * 100}%`,
              width: `${((Number(String(ci).split("–")[1]) - Number(String(ci).split("–")[0])) / 9.5) * 100}%`,
            }} />
          </div>
          <span className="forest-value">{value}× <small>[{ci}]</small></span>
        </div>
      ))}
    </div>
  );
}

// ─── Landing page ────────────────────────────────────────────────────────────

function LandingPage({
  onEnter,
  theme,
  onToggleTheme,
}: {
  onEnter: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}) {
  return (
    <div className="landing-shell">
      {/* Header */}
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark-landing" aria-hidden="true">↗</div>
          <div>
            <h1>Drug Target Prioritization</h1>
            <p className="tagline">Population genetics meets machine learning</p>
          </div>
        </div>
        <div className="header-actions">
          <button
            className="theme-toggle"
            type="button"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            <span aria-hidden="true">{theme === "light" ? "☾" : "☀"}</span>
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </div>
      </header>

      {/* Landing page body */}
      <main className="landing-page">
        {/* ── Hero ── */}
        <section className="landing-hero">
          <div className="hero-copy">
            <p className="eyebrow">OPEN, REPRODUCIBLE, READY TO RUN</p>
            <h2>Turn rare variant burden into drug target predictions.</h2>
            <p className="hero-lede">
              A leakage-safe, AWS-native pipeline that ranks <strong>19,000 human genes</strong> by
              their genetic evidence for clinical-phase drug development — with a prospective
              validation you can inspect.
            </p>
            <div className="hero-actions">
              <button type="button" onClick={onEnter}>
                Explore the targets →
              </button>
              <a href="#how-it-works">See the evidence</a>
            </div>
            <p className="hero-note">
              <span className="live-dot" />
              All results computed · No data uploaded
            </p>
          </div>

          {/* Right column: hero visual */}
          <div className="hero-visual" aria-hidden="true">
            <div className="visual-topline">
              <span>PROSPECTIVE HOLDOUT · 2021 → 2026</span>
              <span className="status-pill">VALIDATED</span>
            </div>

            <div className="visual-enrichment">
              <span className="enrichment-number">5.59×</span>
              <span className="enrichment-label">enrichment in the top 1%</span>
              <span className="enrichment-sub">prospective temporal holdout · 338 genes</span>
              <div className="enrichment-bar-track">
                <div className="enrichment-bar-fill" />
              </div>
            </div>

            <div className="visual-metrics">
              <div>
                <b>338</b>
                prospective positives
              </div>
              <div>
                <b>86.68%</b>
                burden coverage
              </div>
              <div>
                <b>&lt;$1</b>
                22-chr run
              </div>
            </div>

            <div className="visual-footer">
              <span>biology_only · biology_only</span>
              <strong>All checks pass</strong>
            </div>
          </div>
        </section>

        {/* ── Proof strip ── */}
        <div className="landing-proof">
          <span>Leakage-safe evaluation</span>
          <span>AWS Batch · &lt;$1 cost</span>
          <span>Prospective validation</span>
        </div>

        {/* ── Feature section ── */}
        <section className="feature-section" id="how-it-works">
          <div className="section-heading">
            <p className="eyebrow">HOW IT WORKS</p>
            <h3>Population-scale genomics, ranked and ready to explore.</h3>
          </div>
          <div className="feature-grid">
            <div className="feature-card">
              <span className="feature-index">01</span>
              <h4>Population-scale burden</h4>
              <p>
                Rare and LoF variant counts across all 22 autosomes from 1000 Genomes, processed
                on AWS Batch Spot instances at minimal cost.
              </p>
            </div>
            <div className="feature-card">
              <span className="feature-index">02</span>
              <h4>Biology-first ranking</h4>
              <p>
                Six feature layers — genetic constraint, protein structure, PPI network, expression,
                essentiality, publication count — combined with GroupKFold to prevent gene-family
                leakage.
              </p>
            </div>
            <div className="feature-card">
              <span className="feature-index">03</span>
              <h4>Prospective evidence</h4>
              <p>
                The model was trained on 2021 labels and tested on genes that gained clinical-phase
                status by 2026. 5.59× enrichment at the top 1% — inspect every number here.
              </p>
            </div>
          </div>
        </section>

        {/* ── Workflow section ── */}
        <section className="workflow-section">
          <div>
            <p className="eyebrow">THE PIPELINE</p>
            <h3>Three stages, one reproducible artifact.</h3>
            <p className="section-copy">
              Each stage runs in a container on AWS Batch, writes outputs to S3, and is version-locked
              so any collaborator can reproduce the exact ranking.
            </p>
          </div>
          <div className="workflow-list">
            <div>
              <span>01</span>
              <p>
                <b>Prepare</b>
                <small>VCF burden counting across 22 autosomes</small>
              </p>
            </div>
            <div>
              <span>02</span>
              <p>
                <b>Assemble</b>
                <small>Feature matrix from 6 data sources</small>
              </p>
            </div>
            <div>
              <span>03</span>
              <p>
                <b>Evaluate</b>
                <small>GroupKFold + temporal holdout validation</small>
              </p>
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <div className="landing-cta">
          <div>
            <p className="eyebrow">READY TO EXPLORE</p>
            <h3>Bring your target list into focus.</h3>
          </div>
          <button type="button" onClick={onEnter}>
            Explore the evidence →
          </button>
        </div>

        <footer className="landing-footer">
          Drug target prioritization · Results from committed analysis write-up ·{" "}
          <a href="https://github.com/BulutHamali/drug-discovery-target-prioritization" target="_blank" rel="noreferrer">
            source
          </a>
        </footer>
      </main>
    </div>
  );
}

// ─── Workspace (sidebar layout) ──────────────────────────────────────────────

export default function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [view, setView] = useState<View>("overview");
  const [mode, setMode] = useState<"public" | "admin">("public");
  const [userEmail, setUserEmail] = useState<string | undefined>();
  const [adminUser, setAdminUser] = useState(false);
  const [adminOverview, setAdminOverview] = useState<AdminOverview | null>(null);
  const [adminRuns, setAdminRuns] = useState<AdminRun[]>([]);
  const [adminError, setAdminError] = useState<string | null>(null);

  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = window.localStorage.getItem("dt-theme");
    return saved === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("dt-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "light" ? "dark" : "light"));

  useEffect(() => {
    const callback = window.location.pathname === "/auth/callback";
    const loadUser = async () => {
      const user = callback ? await finishCallback() : await currentUser();
      if (callback) window.history.replaceState({}, "", "/");
      setUserEmail(user?.profile.email as string | undefined);
      setAdminUser(await isAdmin());
    };
    void loadUser();
  }, []);

  useEffect(() => {
    if (mode !== "admin") return;
    void Promise.all([targetApi.overview(), targetApi.runs()])
      .then(([overview, runs]) => {
        setAdminOverview(overview);
        setAdminRuns(runs);
        setAdminError(null);
      })
      .catch((error: Error) => setAdminError(error.message));
  }, [mode]);

  const nav = (next: View) => setView(next);

  const switchMode = (next: "public" | "admin") => {
    if (next === "admin" && authEnabled && !adminUser) {
      void signIn();
      return;
    }
    setMode(next);
    setView(next === "admin" ? "admin" : "overview");
  };

  // Show landing page
  if (showLanding) {
    return (
      <LandingPage
        onEnter={() => setShowLanding(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
    );
  }

  // Workspace
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">↗</div>
          <div>
            <strong>Target<br />Prioritization</strong>
            <small>results explorer</small>
          </div>
        </div>

        <div className="project-switcher">
          <small>PROJECT</small>
          <strong>Drug discovery target study</strong>
          <span>v1 · 26.06 holdout</span>
        </div>

        <div className="mode-switcher">
          <small>MODE</small>
          <button className={mode === "public" ? "active" : ""} onClick={() => switchMode("public")}>
            ◉ Public demo
          </button>
          <button className={mode === "admin" ? "active" : ""} onClick={() => switchMode("admin")}>
            ▣ Admin research
          </button>
        </div>

        {mode === "public" && (
          <nav>
            <small>EXPLORE</small>
            {([["overview", "Overview", "◈"], ["targets", "Ranked targets", "≋"], ["results", "Model results", "⌁"]] as [View, string, string][]).map(
              ([key, label, icon]) => (
                <button
                  key={key}
                  className={view === key ? "active" : ""}
                  onClick={() => nav(key)}
                >
                  <span>{icon}</span>{label}
                </button>
              )
            )}
          </nav>
        )}

        {mode === "admin" && (
          <nav>
            <small>RESEARCH CONSOLE</small>
            <button className={view === "admin" ? "active" : ""} onClick={() => nav("admin")}>
              <span>▣</span>Run pipeline
            </button>
            <button onClick={() => nav("results")}>
              <span>⌁</span>Inspect results
            </button>
          </nav>
        )}

        <div className="sidebar-note">
          <span className="status-dot" />
          {mode === "public" ? "Public demo mode" : "Admin mode"}
          <p>
            {mode === "public"
              ? "Explore verified results. Execution and sensitive data are disabled."
              : authEnabled
                ? adminUser
                  ? `Signed in${userEmail ? ` as ${userEmail}` : ""}.`
                  : "Administrator access required."
                : "Local preview; authentication is disabled."}
          </p>
          {authEnabled && (
            <button
              className="sidebar-auth"
              onClick={() => void (userEmail ? signOut() : signIn())}
            >
              {userEmail ? "Sign out" : "Sign in"}
            </button>
          )}
        </div>

        <a
          className="repo-link"
          href="https://github.com/BulutHamali/drug-discovery-target-prioritization"
          target="_blank"
          rel="noreferrer"
        >
          View repository ↗
        </a>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="kicker">
              {mode === "public" ? "PUBLIC DEMO / EVIDENCE REVIEW" : "ADMIN RESEARCH / CONTROL CONSOLE"}
            </span>
            <h1>
              {view === "overview"
                ? "A ranked view of what the data supports."
                : view === "targets"
                  ? "Explore the ranked target universe."
                  : view === "admin"
                    ? "Run the AWS-native analysis pipeline."
                    : "How much signal survives the checks?"}
            </h1>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <span className={`chip ${mode === "admin" ? "admin-chip" : ""}`}>
              {mode === "admin" ? "Protected execution boundary" : "Prospective holdout · 21.06 → 26.06"}
            </span>
            <button
              className="theme-toggle"
              type="button"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
              style={{ border: "1px solid var(--line)", borderRadius: 6, padding: "6px 10px", fontSize: 11, color: "#738087", background: "none" }}
            >
              {theme === "light" ? "☾" : "☀"}
            </button>
            <button
              type="button"
              onClick={() => setShowLanding(true)}
              style={{ border: 0, background: "none", color: "#738087", fontSize: 11, cursor: "pointer" }}
              title="Back to landing"
            >
              ← Home
            </button>
          </div>
        </header>

        {view === "overview" && <Overview nav={nav} />}
        {view === "targets" && <Targets />}
        {view === "results" && <Results />}
        {view === "admin" && <AdminConsole overview={adminOverview} runs={adminRuns} error={adminError} />}

        <footer>
          Drug discovery target prioritization · Results are from the committed analysis write-up ·{" "}
          <a href="https://github.com/BulutHamali/drug-discovery-target-prioritization" target="_blank" rel="noreferrer">
            source
          </a>
        </footer>
      </main>
    </div>
  );
}

// ─── AdminConsole ─────────────────────────────────────────────────────────────

function AdminConsole({ overview, runs, error }: { overview: AdminOverview | null; runs: AdminRun[]; error: string | null }) {
  const [stage, setStage] = useState("evaluate");
  const [featureSet, setFeatureSet] = useState("biology_only");
  const [label, setLabel] = useState("prospective-holdout");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const submit = async () => {
    setSubmitting(true);
    setMessage(null);
    try {
      const run = await targetApi.submit({ stage, feature_set: featureSet, label });
      setMessage(`${run.run_id}: ${run.message}`);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="admin-console">
      <section className="admin-hero">
        <div>
          <span className="eyebrow">ADMIN RESEARCH MODE</span>
          <h2>Turn the AWS pipeline into a controlled, observable run.</h2>
          <p>This console is the protected control plane for feature assembly, model evaluation, and publishing a verified result artifact to the public demo.</p>
        </div>
        <span className="admin-lock">
          ▣ {overview ? "API CONNECTED" : "API OFFLINE"}
          <small>{overview?.execution_mode ?? "protected API required"}</small>
        </span>
      </section>

      <div className="metrics">
        <Metric value={overview ? String(overview.active_runs) : "—"} label="active runs" tone="amber" />
        <Metric value={overview ? String(overview.runs) : "—"} label="recorded runs" />
        <Metric value={runs[0]?.state ?? "—"} label="latest run" tone="blue" />
        <Metric value={overview?.execution_mode ?? "—"} label="execution mode" tone="green" />
      </div>

      {(error || message) && (
        <div className="callout warning">
          <strong>{error ? "Admin API unavailable" : "Run response"}</strong>
          <span>{error ?? message}</span>
        </div>
      )}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">RUN CONFIGURATION</span>
            <h3>Prepare a reproducible ranking run</h3>
          </div>
          <span className="tag">{overview?.execution_mode ?? "API not connected"}</span>
        </div>
        <div className="admin-form">
          <label>
            Pipeline stage
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              <option value="prepare">prepare</option>
              <option value="feature_assembly">feature assembly</option>
              <option value="evaluate">evaluate</option>
              <option value="export">export targets</option>
            </select>
          </label>
          <label>
            Feature set
            <select value={featureSet} onChange={(e) => setFeatureSet(e.target.value)}>
              <option>biology_only</option>
              <option>all_features</option>
              <option>no_pubcount</option>
              <option>no_pubcount_no_string</option>
            </select>
          </label>
          <label>
            Run label
            <input value={label} onChange={(e) => setLabel(e.target.value)} />
          </label>
        </div>
        <div className="admin-actions">
          <button onClick={() => void submit()} disabled={submitting || !overview}>
            {submitting ? "Submitting…" : "Launch protected run"}
          </button>
          <span>Execution submits to the protected API; AWS credentials never enter the browser.</span>
        </div>
      </section>

      <section className="admin-grid">
        <div className="panel">
          <span className="eyebrow">PIPELINE STAGES</span>
          <h3>Observable from one console</h3>
          <div className="stage-list">
            {[
              ["01", "Prepare", "S3 inputs and reference checks"],
              ["02", "Burden", "AWS Batch Spot chromosome jobs"],
              ["03", "Assemble", "Feature matrix and labels"],
              ["04", "Evaluate", "GroupKFold and temporal holdout"],
              ["05", "Publish", "Approve artifact for public demo"],
            ].map(([number, name, detail]) => (
              <div key={number}>
                <b>{number}</b>
                <span>
                  <strong>{name}</strong>
                  <small>{detail}</small>
                </span>
                <em>{runs.find((r) => r.stage === name.toLowerCase())?.state ?? "ready"}</em>
              </div>
            ))}
          </div>
        </div>
        <div className="panel">
          <span className="eyebrow">SAFETY BOUNDARY</span>
          <h3>What Admin Mode can control</h3>
          <ul className="safety-list">
            <li>Never expose AWS credentials to the browser.</li>
            <li>Require server-side authentication and audit logging.</li>
            <li>Show estimated cost before launching Batch jobs.</li>
            <li>Publish only completed artifacts with provenance.</li>
            <li>Keep the public demo read-only and sanitized.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

// ─── Overview ─────────────────────────────────────────────────────────────────

function Overview({ nav }: { nav: (v: View) => void }) {
  return (
    <>
      <section className="hero-card">
        <div>
          <span className="eyebrow">PRIMARY RESULT</span>
          <h2>The strongest signal is prospective enrichment above chance.</h2>
          <p>Genes ranked highly by the biology-focused score were more likely to gain a clinical-phase drug label in the later Open Targets release.</p>
          <button onClick={() => nav("results")}>Inspect the evidence <span>→</span></button>
        </div>
        <div className="hero-number">
          <strong>5.59×</strong>
          <span>enrichment in the top 1%</span>
          <small>95% CI above the resampled baseline</small>
        </div>
      </section>

      <div className="metrics">
        <Metric value="338" label="prospective positives" tone="blue" />
        <Metric value="2.95×" label="PR-AUC lift · biology_only" />
        <Metric value="86.68%" label="burden coverage" tone="green" />
        <Metric value="<$1" label="full 22-autosome run" tone="amber" />
      </div>

      <section className="panel architecture-panel">
        <div>
          <span className="eyebrow">AWS-NATIVE COMPUTE PATH</span>
          <h3>From population data to ranked targets.</h3>
          <p>Scalable genomic processing runs in AWS; this results explorer is the lightweight presentation layer.</p>
        </div>
        <div className="architecture-flow">
          <span><b>1000 Genomes</b><small>S3 open data</small></span>
          <i>→</i>
          <span><b>AWS Batch</b><small>Spot compute</small></span>
          <i>→</i>
          <span><b>Feature matrix</b><small>Parquet</small></span>
          <i>→</i>
          <span><b>ML ranking</b><small>GroupKFold</small></span>
          <i>→</i>
          <span className="flow-result"><b>Results</b><small>Vercel explorer</small></span>
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">TEMPORAL HOLDOUT</span>
              <h3>Signal at every threshold</h3>
            </div>
            <button className="quiet" onClick={() => nav("results")}>Open results ↗</button>
          </div>
          <EnrichmentChart />
        </div>
        <div className="panel finding">
          <span className="eyebrow">READ THIS FIRST</span>
          <h3>Important context</h3>
          <p>The trained model does not clearly beat DepMap essentiality alone on this holdout: 2.95× lift versus 5.03× for the single-feature baseline.</p>
          <div className="callout warning">
            <strong>Rank is not druggability.</strong>
            <span>High rank means important and understudied, not necessarily tractable.</span>
          </div>
          <button className="text-link" onClick={() => nav("targets")}>See target context →</button>
        </div>
      </section>

      <section className="panel method-strip">
        <div>
          <span className="eyebrow">ANALYSIS CONTRACT</span>
          <h3>Designed to test whether the signal earns its place.</h3>
        </div>
        <div className="method-items">
          <span><b>01</b>Gene-family GroupKFold</span>
          <span><b>02</b>Four feature-set variants</span>
          <span><b>03</b>Separate temporal holdout</span>
        </div>
      </section>
    </>
  );
}

// ─── Targets ──────────────────────────────────────────────────────────────────

function Targets() {
  const [payload, setPayload] = useState<TargetPayload | null>(null);
  const [query, setQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState<"all" | "known" | "unlabeled">("all");
  const [selected, setSelected] = useState<Target | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/data/targets.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: TargetPayload) => {
        setPayload(data);
        setSelected(data.targets[0] ?? null);
      })
      .catch(() => setError(true));
  }, []);

  const filtered = useMemo(
    () =>
      (payload?.targets ?? []).filter(
        (t) =>
          t.symbol.toLowerCase().includes(query.toLowerCase()) &&
          (labelFilter === "all" ||
            (labelFilter === "known" ? t.label === 1 : t.label === 0))
      ),
    [payload, query, labelFilter]
  );

  return (
    <>
      <section className="intro">
        <span className="eyebrow">TARGET EXPLORER</span>
        <h2>Move from a score to a biological question.</h2>
        <p>Search the out-of-sample ranking, separate known clinical labels from unlabeled genes, and inspect the provenance of each row. Scores are prioritization evidence—not druggability claims.</p>
      </section>

      {!payload ? (
        <>
          <div className="target-toolbar target-toolbar-disabled">
            <label><span>Search gene symbol</span><input disabled placeholder="Waiting for ranking artifact" /></label>
            <label><span>Clinical label</span><select disabled><option>All genes</option></select></label>
            <div className="target-count"><strong>—</strong><span>targets available</span></div>
          </div>
          <div className="panel empty-targets">
            {error ? (
              <>
                <div className="empty-icon">≋</div>
                <h3>Ranked predictions are not bundled yet</h3>
                <p>Search and filtering are implemented, but the generated model output is not present in this repository.</p>
                <p>Run <code>python3 ml/train_eval.py --feature-set biology_only</code><br />then <code>cd frontend && npm run export-targets</code>.</p>
              </>
            ) : (
              <>
                <div className="empty-icon">…</div>
                <h3>Loading ranked predictions</h3>
                <p>Checking for the generated target artifact.</p>
              </>
            )}
            <div className="target-contract">
              <span>Expected fields</span>
              <code>symbol · score · label · rank · fold_idx</code>
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="target-toolbar">
            <label>
              <span>Search gene symbol</span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. KCNMA1" />
            </label>
            <label>
              <span>Clinical label</span>
              <select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value as typeof labelFilter)}>
                <option value="all">All genes</option>
                <option value="known">Known clinical label</option>
                <option value="unlabeled">Unlabeled at cutoff</option>
              </select>
            </label>
            <div className="target-count">
              <strong>{filtered.length.toLocaleString()}</strong>
              <span>of {payload.rows.toLocaleString()} targets</span>
            </div>
          </div>

          <section className="target-layout">
            <div className="panel target-table-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">OUT-OF-SAMPLE RANKING</span>
                  <h3>{payload.feature_set} · ranked globally</h3>
                </div>
                <span className="tag">{new Date(payload.generated_at).toLocaleDateString()}</span>
              </div>
              <div className="table-scroll">
                <table className="target-table">
                  <thead>
                    <tr><th>Rank</th><th>Gene</th><th>Score</th><th>Label</th><th>Fold</th></tr>
                  </thead>
                  <tbody>
                    {filtered.slice(0, 250).map((t) => (
                      <tr
                        key={t.symbol}
                        className={selected?.symbol === t.symbol ? "selected-row" : ""}
                        onClick={() => setSelected(t)}
                      >
                        <td>#{t.rank}</td>
                        <td><strong>{t.symbol}</strong></td>
                        <td>{t.score.toFixed(4)}</td>
                        <td>
                          <span className={`label-pill ${t.label ? "known-pill" : "unlabeled-pill"}`}>
                            {t.label ? "Known" : "Unlabeled"}
                          </span>
                        </td>
                        <td>{t.fold_idx}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filtered.length > 250 && <p className="table-note">Showing the first 250 matches. Narrow the search to explore further.</p>}
              {filtered.length === 0 && <p className="empty-row">No genes match these filters.</p>}
            </div>

            <TargetDetail target={selected} />
          </section>
        </>
      )}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">VALIDATION EXAMPLE</span>
            <h3>KCNMA1</h3>
          </div>
          <span className="tag">documented example</span>
        </div>
        <p className="detail-copy">The write-up identifies KCNMA1 as one directionally consistent external-evidence check: it ranked in the top 1% and later gained a clinical-phase drug. This is an anecdote, not a powered validation.</p>
      </section>
    </>
  );
}

// ─── TargetDetail ─────────────────────────────────────────────────────────────

function TargetDetail({ target }: { target: Target | null }) {
  return (
    <aside className="panel target-detail">
      {target ? (
        <>
          <span className="eyebrow">SELECTED TARGET</span>
          <h3>{target.symbol}</h3>
          <div className="detail-rank">
            <strong>#{target.rank}</strong>
            <span>global out-of-sample rank</span>
          </div>
          <div className="detail-grid">
            <Metric value={target.score.toFixed(4)} label="model score" tone="blue" />
            <Metric value={target.label ? "Known" : "Unlabeled"} label="clinical label at cutoff" tone={target.label ? "green" : "amber"} />
            <Metric value={`Fold ${target.fold_idx}`} label="held-out family fold" />
          </div>
          <div className="callout warning">
            <strong>Interpretation boundary</strong>
            <span>This score prioritizes genes for follow-up. It does not measure druggability, clinical success, or causal validity.</span>
          </div>
        </>
      ) : (
        <>
          <h3>Select a target</h3>
          <p className="detail-copy">Choose a row to inspect its score and provenance.</p>
        </>
      )}
    </aside>
  );
}

// ─── Results ──────────────────────────────────────────────────────────────────

function Results() {
  return (
    <>
      <section className="split results-top">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">PROSPECTIVE VALIDATION</span>
              <h3>Observed rate vs. chance</h3>
            </div>
            <span className="tag green-tag">all thresholds clear baseline</span>
          </div>
          <EnrichmentChart />
        </div>
        <div className="panel table-panel">
          <span className="eyebrow">HOLDOUT TABLE</span>
          <h3>Enrichment by top fraction</h3>
          <table>
            <thead>
              <tr><th>Top</th><th>Observed</th><th>Baseline</th><th>Lift</th></tr>
            </thead>
            <tbody>
              {holdout.map((d) => (
                <tr key={d.label}>
                  <td>{d.label}</td>
                  <td>{(d.observed * 100).toFixed(1)}%</td>
                  <td>{(d.baseline * 100).toFixed(1)}%</td>
                  <td className="positive">{d.lift}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="table-note">Baseline intervals: 95% resampled CI. Holdout n = 338.</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">ABLATION</span>
            <h3>Signal among understudied genes</h3>
          </div>
          <span className="tag">median split · 95% CI</span>
        </div>
        <ForestChart />
        <div className="result-note">
          <strong>All four variants clear 1.0×.</strong> Removing publication history and STRING network features does not erase the signal, though the biology-only variant is weaker.
        </div>
      </section>

      <section className="caveat-grid">
        <div className="callout warning">
          <strong>The model does not beat essentiality alone here.</strong>
          <span>DepMap essentiality reaches 5.03× lift on the same temporal holdout. The model's added value is not established by this test.</span>
        </div>
        <div className="callout">
          <strong>Study-bias check remains encouraging.</strong>
          <span>Bottom-half lift CIs remain above 1.0 across the feature-set variants.</span>
        </div>
      </section>
    </>
  );
}
