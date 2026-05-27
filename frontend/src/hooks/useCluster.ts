import { useCallback, useState } from "react";
import type { AppState, ClusterParams, ClusterResponse } from "../lib/types";
import { ApiError, clusterImages } from "../lib/api";

const DEFAULT_PARAMS: ClusterParams = {
  algorithm: "dbscan",
  eps: 0.25,
  min_samples: 2,
  threshold: 0.25,
  linkage: "average",
};

export function useCluster() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [params, setParams] = useState<ClusterParams>(DEFAULT_PARAMS);
  const [results, setResults] = useState<ClusterResponse | null>(null);
  const [state, setState] = useState<AppState>("idle");
  const [error, setError] = useState<string | null>(null);

  const addFiles = useCallback((newFiles: File[]) => {
    setFiles((prev) => {
      const combined = [...prev, ...newFiles].slice(0, 50);
      return combined;
    });
    const urls = newFiles.map((f) => URL.createObjectURL(f));
    setPreviews((prev) => [...prev, ...urls].slice(0, 50));
    setState("ready");
    setResults(null);
    setError(null);
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => {
      URL.revokeObjectURL(prev[index]);
      return prev.filter((_, i) => i !== index);
    });
    setResults(null);
  }, []);

  const runClustering = useCallback(async () => {
    if (files.length === 0) return;

    setState("loading");
    setError(null);

    try {
      const response = await clusterImages(files, params);
      setResults(response);
      setState("results");
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Connection failed. Is the server running?";
      setError(message);
      setState("error");
    }
  }, [files, params]);

  const reset = useCallback(() => {
    previews.forEach((url) => URL.revokeObjectURL(url));
    setFiles([]);
    setPreviews([]);
    setResults(null);
    setState("idle");
    setError(null);
  }, [previews]);

  const updateParams = useCallback((partial: Partial<ClusterParams>) => {
    setParams((prev) => ({ ...prev, ...partial }));
  }, []);

  return {
    files,
    previews,
    params,
    results,
    state,
    error,
    addFiles,
    removeFile,
    runClustering,
    reset,
    updateParams,
  };
}
