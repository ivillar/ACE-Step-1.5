/** REST client for model metadata and session management. */

import type { ModelMetadata, GenerateRequest, GenerateResponse } from "../types";

const API_BASE = "/api";

export async function fetchModelMetadata(): Promise<ModelMetadata> {
  const res = await fetch(`${API_BASE}/model/metadata`);
  if (!res.ok) throw new Error(`Metadata fetch failed: ${res.status}`);
  return res.json();
}

export async function startGeneration(
  params: GenerateRequest = {}
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Generate failed: ${res.status}`);
  return res.json();
}

export function audioUrl(sessionId: string): string {
  return `${API_BASE}/audio/${sessionId}`;
}
