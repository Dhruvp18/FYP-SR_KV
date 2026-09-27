"use client";

import { useState } from "react";
import { useRunStore, RunMode } from "@/store/useRunStore";
import { useGenerationSocket } from "@/hooks/useGenerationSocket";
import { VARIANT_EXPLAINER } from "@/lib/copy";
import type { Variant } from "@/lib/types";

const VARIANT_OPTIONS: { label: string; value: Variant | null; title?: string }[] = [
  { label: "Surprise me", value: null },
  { label: "Attribution", value: "attribution", title: VARIANT_EXPLAINER.attribution },
  { label: "Aggregation", value: "aggregation", title: VARIANT_EXPLAINER.aggregation },
];

const MODE_OPTIONS: { label: string; value: RunMode }[] = [
  { label: "Single question", value: "single" },
  { label: "Benchmark", value: "batch" },
];

export function TopBar() {
  const [pickedVariant, setPickedVariant] = useState<Variant | null>(null);
  const { startGist, cancel } = useGenerationSocket();
  const mode = useRunStore((s) => s.mode);
  const setMode = useRunStore((s) => s.setMode);
  const phase = useRunStore((s) => s.phase);
  const batchRunning = useRunStore((s) => s.batchRunning);
  const connection = useRunStore((s) => s.connection);
  const advancedOpen = useRunStore((s) => s.advancedOpen);
  const toggleAdvanced = useRunStore((s) => s.toggleAdvanced);
  const errorMessage = useRunStore((s) => s.errorMessage);
  const dismissError = useRunStore((s) => s.dismissError);
  const running = mode === "single" ? phase === "generating" : batchRunning;

  function send() {
    if (running || connection !== "connected") return;
    const { budget, contextLen } = useRunStore.getState();
    startGist(budget, pickedVariant, contextLen);
  }

  return (
    <div className="flex shrink-0 flex-col border-b border-border bg-bg">
      <div className="flex flex-wrap items-center gap-3 px-4 py-2.5">
        <span className="text-[14px] font-bold tracking-tight text-text">Cache Lens</span>

        <div className="flex items-center gap-1 rounded-lg border border-border bg-panel-2 p-0.5">
          {MODE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => !running && setMode(opt.value)}
              disabled={running}
              className="rounded-md px-2.5 py-1 text-[12px] font-medium disabled:opacity-50"
              style={
                mode === opt.value ? { background: "var(--text)", color: "var(--bg)" } : { color: "var(--text-dim)" }
              }
            >
              {opt.label}
            </button>
          ))}
        </div>

        {mode === "single" && (
          <div className="flex items-center gap-1 rounded-lg border border-border bg-panel-2 p-0.5">
            {VARIANT_OPTIONS.map((opt) => (
              <button
                key={opt.label}
                onClick={() => setPickedVariant(opt.value)}
                disabled={running}
                title={opt.title}
                className="rounded-md px-2.5 py-1 text-[12px] font-medium disabled:opacity-50"
                style={
                  pickedVariant === opt.value
                    ? { background: "var(--lane-srkv)", color: "var(--bg)" }
                    : { color: "var(--text-dim)" }
                }
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}

        <div className="ml-auto flex items-center gap-2">
          {mode === "single" &&
            (running ? (
              <button
                onClick={cancel}
                className="shrink-0 rounded-lg border border-border bg-panel-2 px-3.5 py-1.5 text-[12.5px] font-semibold text-text hover:bg-border"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={send}
                disabled={connection !== "connected"}
                className="shrink-0 rounded-lg bg-lane-srkv px-4 py-1.5 text-[12.5px] font-semibold text-bg disabled:opacity-40"
              >
                New question →
              </button>
            ))}

          <button
            onClick={toggleAdvanced}
            className="shrink-0 rounded-lg border border-border px-2.5 py-1.5 text-[11.5px] font-medium text-text-dim hover:bg-panel-2 hover:text-text"
          >
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 px-4 pb-2 text-[11px] text-text-dim">
        <span className="text-text-faint">legend:</span>
        <span className="text-text">kept</span>
        <span className="rounded bg-text-dim/30 px-1 py-0.5 shadow-[inset_0_-2px_0_var(--text-dim)]">folded</span>
        <span className="text-text-faint line-through opacity-60">evicted</span>
      </div>

      {errorMessage && (
        <button
          onClick={dismissError}
          className="mx-4 mb-2 truncate rounded-md bg-danger/15 px-2.5 py-1.5 text-left text-[11.5px] text-danger"
          title="dismiss"
        >
          {errorMessage}
        </button>
      )}
    </div>
  );
}
