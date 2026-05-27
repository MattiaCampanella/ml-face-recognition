interface ImageGridProps {
  previews: string[];
  fileNames: string[];
  onRemove: (index: number) => void;
}

export function ImageGrid({ previews, fileNames, onRemove }: ImageGridProps) {
  if (previews.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-300">
          {previews.length} image{previews.length !== 1 ? "s" : ""} selected
        </h3>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
        {previews.map((url, i) => (
          <div key={i} className="group relative aspect-square">
            <img
              src={url}
              alt={fileNames[i]}
              className="h-full w-full rounded-lg object-cover"
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove(i);
              }}
              className="
                absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full
                bg-red-500 text-white text-xs
                flex items-center justify-center
                opacity-0 group-hover:opacity-100 transition-opacity
              "
              aria-label={`Remove ${fileNames[i]}`}
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
