import { accessToken } from "./auth";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type AdminOverview = {
  public_demo_mode: boolean;
  auth_required: boolean;
  execution_mode: string;
  runs: number;
  active_runs: number;
};

export type AdminRun = {
  run_id: string;
  stage: string;
  state: "pending" | "running" | "succeeded" | "failed";
  label: string;
  feature_set: string;
  message: string;
  created_at: string;
  updated_at: string;
  artifact_uri?: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = await accessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init?.body) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    throw new Error(
      ((await response.json().catch(() => ({}))) as { detail?: string }).detail ?? response.statusText
    );
  }
  return response.json() as Promise<T>;
}

export const targetApi = {
  overview: () => request<AdminOverview>("/admin/overview"),
  runs: () => request<AdminRun[]>("/admin/runs"),
  submit: (body: { stage: string; feature_set: string; label: string; parameters?: Record<string, unknown> }) =>
    request<AdminRun>("/admin/runs", { method: "POST", body: JSON.stringify(body) }),
};

export const publicApi = {
  summary: () => fetch(`${API_BASE}/api/public/summary`).then((r) => r.json()),
  provenance: () => fetch(`${API_BASE}/api/public/provenance`).then((r) => r.json()),
};
