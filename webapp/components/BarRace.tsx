"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";
import { useRunStore } from "@/store/useRunStore";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { LANE_METHODS } from "@/lib/types";

export function BarRace() {
  const lanes = useRunStore((s) => s.lanes);
  const step = useRunStore((s) => s.step);
  const promptTokens = useRunStore((s) => s.promptTokens);

  const denom = Math.max(promptTokens + step, 1);

  const rows = useMemo(
    () =>
      [...LANE_METHODS]
        .map((id) => ({ id, cached: lanes[id].stats.n_tokens_cached }))
        .sort((a, b) => b.cached - a.cached),
    [lanes],
  );

  return (
    <div className="flex h-full min-h-0 flex-col justify-center gap-2 rounded-xl border border-border bg-panel p-3">
      <div className="mb-0.5 text-[12px] font-semibold text-text">Current cache size, as a share of the sequence</div>
      {rows.map((r) => {
        const pct = Math.min(100, (100 * r.cached) / denom);
        return (
          <div key={r.id} className="grid grid-cols-[92px_1fr_28px] items-center gap-2">
            <span className="truncate text-[11px] font-medium text-text-dim">{LANE_LABEL[r.id]}</span>
            <div className="h-2.5 overflow-hidden rounded-full bg-panel-2">
              <motion.div
                className="h-full rounded-full"
                style={{ background: LANE_COLOR_VAR[r.id] }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.25, ease: "easeOut" }}
              />
            </div>
            <span className="text-right text-[11px] text-text-dim">{r.cached}</span>
          </div>
        );
      })}
    </div>
  );
}
