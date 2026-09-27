"use client";

import { useState } from "react";
import { useRunStore } from "@/store/useRunStore";
import { useGenerationSocket } from "@/hooks/useGenerationSocket";

export function TopBar() {
  const [prompt, setPrompt] = useState("");
  const { start, cancel } = useGenerationSocket();
  const phase = useRunStore((s) => s.phase);
  const connection = useRunStore((s) => s.connection);
  const advancedOpen = useRunStore((s) => s.advancedOpen);
  const toggleAdvanced = useRunStore((s) => s.toggleAdvanced);
  const errorMessage = useRunStore((s) => s.errorMessage);
  const dismissError = useRunStore((s) => s.dismissError);
  const running = phase === "generating";

  function send() {
    const trimmed = prompt.trim();
    if (!trimmed || running || connection !== "connected") return;
    const { budget, maxNewTokens } = useRunStore.getState();
    start(trimmed, budget, maxNewTokens);
  }

  return (
    <div className="flex shrink-0 flex-col border-b border-border bg-bg">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <span className="text-[14px] font-bold tracking-tight text-text">Cache Lens</span>

        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Type a sentence and press Enter…"
          disabled={running}
          className="min-w-0 flex-1 rounded-lg border border-border bg-panel-2 px-3 py-1.5 text-[13px] text-text placeholder:text-text-faint focus:outline-none focus:ring-1 focus:ring-lane-srkv disabled:opacity-50"
        />

        {running ? (
          <button
            onClick={cancel}
            className="shrink-0 rounded-lg border border-border bg-panel-2 px-3.5 py-1.5 text-[12.5px] font-semibold text-text hover:bg-border"
          >
            Cancel
          </button>
        ) : (
          <button
            onClick={send}
            disabled={connection !== "connected" || !prompt.trim()}
            className="shrink-0 rounded-lg bg-lane-srkv px-3.5 py-1.5 text-[12.5px] font-semibold text-bg disabled:opacity-40"
          >
            Send
          </button>
        )}

        <button
          onClick={toggleAdvanced}
          className="shrink-0 rounded-lg border border-border px-2.5 py-1.5 text-[11.5px] font-medium text-text-dim hover:bg-panel-2 hover:text-text"
        >
          {advancedOpen ? "Hide advanced" : "Advanced"}
        </button>
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
