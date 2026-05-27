import { useCallback, useState } from "react";

interface DropZoneProps {
  onFiles: (files: File[]) => void;
  fileCount: number;
  maxFiles: number;
}

export function DropZone({ onFiles, fileCount, maxFiles }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const items = Array.from(e.dataTransfer.files).filter((f) =>
        f.type.startsWith("image/"),
      );
      if (items.length > 0) onFiles(items);
    },
    [onFiles],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const items = Array.from(e.target.files || []);
      if (items.length > 0) onFiles(items);
      e.target.value = "";
    },
    [onFiles],
  );

  const remaining = maxFiles - fileCount;

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`
        relative flex flex-col items-center justify-center gap-3
        rounded-xl border-2 border-dashed p-8 transition-colors cursor-pointer
        ${
          isDragging
            ? "border-neutral-400 bg-neutral-800"
            : "border-neutral-700 hover:border-neutral-500 bg-neutral-900"
        }
      `}
      onClick={() => document.getElementById("file-input")?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        className="hidden"
        onChange={handleChange}
      />

      <svg
        className="h-10 w-10 text-neutral-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z"
        />
      </svg>

      <div className="text-center">
        <p className="text-sm font-medium text-neutral-300">
          Drop images here or click to upload
        </p>
        <p className="text-xs text-neutral-500 mt-1">
          JPG, PNG, WebP &middot; max {maxFiles} images
          {fileCount > 0 && ` (${remaining} remaining)`}
        </p>
      </div>
    </div>
  );
}
