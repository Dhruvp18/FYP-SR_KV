"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useRunStore } from "@/store/useRunStore";
import { BUDGET_MAX, BUDGET_MIN, GIST_CONTEXT_LEN_MAX, GIST_CONTEXT_LEN_MIN, PACE_MAX_MS, PACE_MIN_MS } from "@/lib/config";
import { SummaryTable } from "./SummaryTable";
import { TeammatePilotTable } from "./TeammatePilotTable";

export function AdvancedPanel() {
  const open = useRunStore((s) => s.advancedOpen);
  const connection = useRunStore((s) => s.connection);
  const budget = useRunStore((s) => s.budget);
  const setBudget = useRunStore((s) => s.setBudget);
  const contextLen = useRunStore((s) => s.contextLen);
  const setContextLen = useRunStore((s) => s.setContextLen);
  const playbackMs = useRunStore((s) => s.playbackMs);
  const setPlaybackMs = useRunStore((s) => s.setPlaybackMs);
  const running = useRunStore((s) => s.phase === "generating");
  const resetTally = useRunStore((s) => s.resetTally);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="overflow-hidden border-b border-border bg-panel"
        >
          <div className="flex flex-col gap-4 px-4 py-4">
            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
              <span className="flex items-center gap-1.5 text-[11px] text-text-dim">
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    background:
                      connection === "connected" ? "var(--good)" : connection === "connecting" ? "var(--lane-full)" : "var(--danger)",
                  }}
                />
                {connection}
              </span>

              <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-text-dim">
                budget
                <input
                  type="range"
                  min={BUDGET_MIN}
                  max={BUDGET_MAX}
                  step={0.05}
                  value={budget}
                  disabled={running}
                  onChange={(e) => setBudget(Number(e.target.value))}
                  className="accent-lane-srkv"
                />
                <span className="w-9 text-text">{Math.round(budget * 100)}%</span>
              </label>

              <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-text-dim">
                passage length
                <input
                  type="range"
                  min={GIST_CONTEXT_LEN_MIN}
                  max={GIST_CONTEXT_LEN_MAX}
                  step={50}
                  value={contextLen}
                  disabled={running}
                  onChange={(e) => setContextLen(Number(e.target.value))}
                  className="accent-lane-srkv"
                />
                <span className="w-10 text-text">{contextLen}</span>
              </label>

              <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-text-dim">
                word reveal pace
                <input
                  type="range"
                  min={PACE_MIN_MS}
                  max={PACE_MAX_MS}
                  step={50}
                  value={playbackMs}
                  onChange={(e) => setPlaybackMs(Number(e.target.value))}
                  className="accent-lane-srkv"
                />
                <span className="w-9 text-text">{playbackMs}</span>
              </label>

              <button
                onClick={resetTally}
                disabled={running}
                className="rounded-md border border-border px-2.5 py-1 text-[11px] font-medium text-text-dim hover:bg-panel-2 hover:text-text disabled:opacity-50"
              >
                Reset accuracy tally
              </button>
            </div>

            <div className="border-t border-border-soft pt-3">
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-faint">
                End-of-run summary
              </h3>
              <SummaryTable />
            </div>

            <div className="border-t border-border-soft pt-3">
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-faint">
                A separate pilot result (unpublished)
              </h3>
              <TeammatePilotTable />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
