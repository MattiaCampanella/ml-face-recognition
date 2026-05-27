import JSZip from "jszip";
import type { ClusterGroup } from "./types";

export async function buildZip(
  files: File[],
  clusters: ClusterGroup[],
): Promise<Blob> {
  const zip = new JSZip();

  for (const group of clusters) {
    const folder = zip.folder(group.name)!;
    const nameCounts: Record<string, number> = {};

    for (const idx of group.image_indices) {
      const file = files[idx];
      let name = file.name;

      // Deduplicate names within same cluster
      const count = nameCounts[name] || 0;
      nameCounts[name] = count + 1;
      if (count > 0) {
        const dot = name.lastIndexOf(".");
        const stem = dot > 0 ? name.slice(0, dot) : name;
        const ext = dot > 0 ? name.slice(dot) : "";
        name = `${stem}_${count}${ext}`;
      }

      folder.file(name, file);
    }
  }

  return zip.generateAsync({ type: "blob" });
}
