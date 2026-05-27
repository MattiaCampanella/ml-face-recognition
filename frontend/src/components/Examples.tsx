const examples = [
  {
    src: "/examples/correct.jpg",
    label: "Correctly cropped",
    good: true,
  },
  {
    src: "/examples/incorrect.jpg",
    label: "Incorrect crop",
    good: false,
  },
  {
    src: "/examples/group.jpg",
    label: "Avoid group photos",
    good: false,
  },
];

export function Examples() {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-3">
      <h3 className="text-sm font-medium text-neutral-300">
        How to prepare your images
      </h3>
      <p className="text-xs text-neutral-500">
        Upload images with a single cropped face per file for best results.
      </p>
      <div className="grid grid-cols-3 gap-3">
        {examples.map((ex) => (
          <div key={ex.src} className="space-y-1.5">
            <div className="relative aspect-square overflow-hidden rounded-lg border border-neutral-700">
              <img
                src={ex.src}
                alt={ex.label}
                className="h-full w-full object-cover"
              />
            </div>
            <p className="text-xs text-center text-neutral-400">
              <span className={ex.good ? "text-green-400" : "text-red-400"}>
                {ex.good ? "✓" : "✗"}
              </span>{" "}
              {ex.label}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
