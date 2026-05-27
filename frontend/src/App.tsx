import { DropZone } from "./components/DropZone";
import { ImageGrid } from "./components/ImageGrid";
import { SettingsPanel } from "./components/SettingsPanel";
import { ClusterResults } from "./components/ClusterResults";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { Examples } from "./components/Examples";
import { useCluster } from "./hooks/useCluster";

const MAX_FILES = 50;

export default function App() {
  const {
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
  } = useCluster();

  return (
    <>
      <LoadingOverlay visible={state === "loading"} />

      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-2xl font-bold tracking-tight text-neutral-100">
            Face Clustering
          </h1>
          <p className="mt-1 text-sm text-neutral-400">
            Upload face images to automatically group them by identity.
          </p>
        </header>

        {/* Main content */}
        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          {/* Left column */}
          <div className="space-y-6">
            {/* Upload area */}
            {state !== "results" && (
              <>
                <DropZone
                  onFiles={addFiles}
                  fileCount={files.length}
                  maxFiles={MAX_FILES}
                />
                {state === "idle" && <Examples />}
                <ImageGrid
                  previews={previews}
                  fileNames={files.map((f) => f.name)}
                  onRemove={removeFile}
                />
              </>
            )}

            {/* Error */}
            {error && (
              <div className="rounded-xl border border-red-800 bg-red-950 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}

            {/* Results */}
            {state === "results" && results && (
              <ClusterResults
                clusters={results.clusters}
                previews={previews}
                fileNames={files.map((f) => f.name)}
                files={files}
                noiseCount={results.noise_count}
              />
            )}
          </div>

          {/* Right column — settings */}
          <aside className="space-y-4">
            <SettingsPanel params={params} onChange={updateParams} />

            {/* Action buttons */}
            <div className="space-y-2">
              {(state === "ready" || state === "error" || state === "results") && (
                <button
                  onClick={runClustering}
                  disabled={files.length === 0}
                  className="w-full rounded-xl bg-white px-4 py-3 text-sm font-medium text-neutral-900 hover:bg-neutral-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {state === "results" ? "Recalculate" : "Run Clustering"}
                </button>
              )}

              {state === "results" && (
                <button
                  onClick={reset}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-800 px-4 py-3 text-sm font-medium text-neutral-200 hover:bg-neutral-700 transition-colors"
                >
                  Start Over
                </button>
              )}
            </div>
          </aside>
        </div>

        {/* Footer */}
        <footer className="mt-12 border-t border-neutral-800 pt-6 text-center text-xs text-neutral-500">
          Face embeddings extracted with ResNet-18 &middot; Metric learning
        </footer>
      </div>
    </>
  );
}
