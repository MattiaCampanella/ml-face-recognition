import { useState } from "react";
import type { ClusterGroup } from "../lib/types";
import { buildZip } from "../lib/zip";

interface ClusterResultsProps {
  clusters: ClusterGroup[];
  previews: string[];
  fileNames: string[];
  files: File[];
  noiseCount: number;
}

export function ClusterResults({
  clusters,
  previews,
  fileNames,
  files,
  noiseCount,
}: ClusterResultsProps) {
  const [downloading, setDownloading] = useState(false);

  const identityClusters = clusters.filter((c) => c.cluster_id >= 0);
  const noiseClusters = clusters.filter((c) => c.cluster_id < 0);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const blob = await buildZip(files, clusters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "clusters.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary bar */}
      <div className="flex items-center justify-between rounded-xl bg-neutral-800 border border-neutral-700 px-5 py-3 text-white">
        <div className="flex gap-6 text-sm">
          <span>
            <strong>{identityClusters.length}</strong> identit
            {identityClusters.length !== 1 ? "ies" : "y"}
          </span>
          {noiseCount > 0 && (
            <span className="text-neutral-400">{noiseCount} unassigned</span>
          )}
        </div>
        <button
          onClick={handleDownload}
          disabled={downloading}
          className="rounded-lg bg-white text-neutral-900 px-4 py-1.5 text-sm font-medium hover:bg-neutral-200 transition-colors disabled:opacity-50"
        >
          {downloading ? "Preparing..." : "Download ZIP"}
        </button>
      </div>

      {/* Cluster groups */}
      {identityClusters.map((group) => (
        <ClusterGroupCard
          key={group.cluster_id}
          group={group}
          previews={previews}
          fileNames={fileNames}
        />
      ))}

      {/* Noise group */}
      {noiseClusters.map((group) => (
        <ClusterGroupCard
          key="noise"
          group={group}
          previews={previews}
          fileNames={fileNames}
          isNoise
        />
      ))}
    </div>
  );
}

function ClusterGroupCard({
  group,
  previews,
  fileNames,
  isNoise = false,
}: {
  group: ClusterGroup;
  previews: string[];
  fileNames: string[];
  isNoise?: boolean;
}) {
  return (
    <div className="rounded-xl border border-neutral-700 bg-neutral-900 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span
          className={`inline-block h-3 w-3 rounded-full ${
            isNoise ? "bg-neutral-500" : "bg-white"
          }`}
        />
        <h4 className="text-sm font-medium text-neutral-200">
          {isNoise ? "Unassigned" : group.name}
        </h4>
        <span className="text-xs text-neutral-500">
          {group.image_indices.length} image
          {group.image_indices.length !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
        {group.image_indices.map((idx) => (
          <div key={idx} className="aspect-square">
            <img
              src={previews[idx]}
              alt={fileNames[idx]}
              title={fileNames[idx]}
              className="h-full w-full rounded-lg object-cover"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
