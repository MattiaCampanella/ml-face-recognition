import { useState, useRef, useEffect } from "react";
import type { ClusterParams } from "../lib/types";

interface SettingsPanelProps {
  params: ClusterParams;
  onChange: (partial: Partial<ClusterParams>) => void;
}

export function SettingsPanel({ params, onChange }: SettingsPanelProps) {
  return (
    <div className="space-y-5 rounded-xl border border-neutral-700/60 bg-neutral-900 p-5">
      <h3 className="text-sm font-semibold text-neutral-100">Settings</h3>

      {/* Algorithm — segmented control */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-neutral-400">
          Algorithm
        </label>
        <SegmentedControl
          value={params.algorithm}
          options={[
            { value: "dbscan", label: "DBSCAN" },
            { value: "agglomerative", label: "Agglomerative" },
          ]}
          onChange={(v) =>
            onChange({ algorithm: v as ClusterParams["algorithm"] })
          }
        />
      </div>

      {/* DBSCAN params */}
      {params.algorithm === "dbscan" && (
        <div className="space-y-4 pt-1">
          <Slider
            label="Neighborhood radius"
            hint="Lower = tighter clusters"
            value={params.eps}
            min={0.05}
            max={1.0}
            step={0.01}
            onChange={(v) => onChange({ eps: v })}
          />
          <Slider
            label="Min samples"
            hint="Minimum images per identity"
            value={params.min_samples}
            min={1}
            max={10}
            step={1}
            onChange={(v) => onChange({ min_samples: v })}
          />
        </div>
      )}

      {/* Agglomerative params */}
      {params.algorithm === "agglomerative" && (
        <div className="space-y-4 pt-1">
          <Slider
            label="Distance threshold"
            hint="Lower = tighter clusters"
            value={params.threshold}
            min={0.05}
            max={1.0}
            step={0.01}
            onChange={(v) => onChange({ threshold: v })}
          />
          <CustomSelect
            label="Linkage"
            value={params.linkage}
            options={[
              {
                value: "average",
                label: "Average",
                description: "Balanced, good default",
              },
              {
                value: "complete",
                label: "Complete",
                description: "Tighter, compact clusters",
              },
              {
                value: "single",
                label: "Single",
                description: "Loose, chain-like clusters",
              },
            ]}
            onChange={(v) =>
              onChange({ linkage: v as ClusterParams["linkage"] })
            }
          />
        </div>
      )}
    </div>
  );
}

/* ─── Segmented Control ─── */

function SegmentedControl({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex rounded-lg bg-neutral-800 p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`
            flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-all
            ${
              value === opt.value
                ? "bg-neutral-100 text-neutral-900 shadow-sm"
                : "text-neutral-400 hover:text-neutral-200"
            }
          `}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/* ─── Slider ─── */

function Slider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  const percent = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-medium text-neutral-300">{label}</span>
          {hint && (
            <span className="ml-1.5 text-[10px] text-neutral-500">{hint}</span>
          )}
        </div>
        <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[11px] font-mono text-neutral-300 tabular-nums">
          {step < 1 ? value.toFixed(2) : value}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full"
          style={
            {
              "--fill-percent": `${percent}%`,
            } as React.CSSProperties
          }
        />
      </div>
    </div>
  );
}

/* ─── Custom Select ─── */

function CustomSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string; description?: string }[];
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const selected = options.find((o) => o.value === value);

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-neutral-300">{label}</label>
      <div ref={ref} className="relative">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="flex w-full items-center justify-between rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2.5 text-left text-sm text-neutral-100 transition-colors hover:border-neutral-600 focus:outline-none focus:ring-2 focus:ring-neutral-500"
        >
          <div>
            <span>{selected?.label}</span>
            {selected?.description && (
              <span className="ml-2 text-[11px] text-neutral-500">
                — {selected.description}
              </span>
            )}
          </div>
          <svg
            className={`h-4 w-4 text-neutral-400 transition-transform ${open ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m19.5 8.25-7.5 7.5-7.5-7.5"
            />
          </svg>
        </button>

        {open && (
          <div className="absolute z-10 mt-1 w-full rounded-lg border border-neutral-700 bg-neutral-800 py-1 shadow-xl shadow-black/30 animate-in fade-in slide-in-from-top-1">
            {options.map((opt) => (
              <button
                key={opt.value}
                onClick={() => {
                  onChange(opt.value);
                  setOpen(false);
                }}
                className={`flex w-full flex-col px-3 py-2 text-left transition-colors hover:bg-neutral-700 ${
                  opt.value === value ? "bg-neutral-700/50" : ""
                }`}
              >
                <span className="text-sm text-neutral-100">{opt.label}</span>
                {opt.description && (
                  <span className="text-[11px] text-neutral-500">
                    {opt.description}
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
