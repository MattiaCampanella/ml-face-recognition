import type { ClusterParams, ClusterResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function clusterImages(
  files: File[],
  params: ClusterParams,
): Promise<ClusterResponse> {
  const formData = new FormData();

  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("params", JSON.stringify(params));

  const response = await fetch(`${API_URL}/cluster`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new ApiError(
      response.status,
      body.detail || `HTTP ${response.status}`,
    );
  }

  return response.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
