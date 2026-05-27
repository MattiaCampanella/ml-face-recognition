export interface ClusterParams {
  algorithm: "dbscan" | "agglomerative";
  eps: number;
  min_samples: number;
  threshold: number;
  linkage: "average" | "complete" | "single";
}

export interface ClusterGroup {
  cluster_id: number;
  name: string;
  image_indices: number[];
}

export interface ClusterResponse {
  clusters: ClusterGroup[];
  total_images: number;
  noise_count: number;
}

export type AppState = "idle" | "ready" | "loading" | "results" | "error";
