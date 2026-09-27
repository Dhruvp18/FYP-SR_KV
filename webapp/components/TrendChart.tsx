"use client";

import { useMemo } from "react";
import { useRunStore } from "@/store/useRunStore";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { LANE_METHODS } from "@/lib/types";

const W = 420;
const H = 200;
const PAD_L = 30;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 20;

export function TrendChart() {
  const lanes = useRunStore((s) => s.lanes);
  const promptTokens = useRunStore((s) => s.promptTokens);
  const budget = useRunStore((s) => s.budget);

  const maxY = useMemo(() => {
    let m = Math.ceil(promptTokens * budget) || 4;
    LANE_METHODS.forEach((id) => {
      lanes[id].history.forEach((v) => {
        if (v > m) m = v;
      });
    });
    return Math.ceil(m * 1.15);
  }, [lanes, promptTokens, budget]);

  const maxX = useMemo(() => {
    let m = 1;
    LANE_METHODS.forEach((id) => {
      m = Math.max(m, lanes[id].history.length - 1);
    });
    return m;
  }, [lanes]);

  const budgetTokens = Math.round(promptTokens * budget) || 0;

  function xy(i: number, v: number): string {
    const x = PAD_L + ((W - PAD_L - PAD_R) * i) / maxX;
    const y = H - PAD_B - ((H - PAD_T - PAD_B) * v) / maxY;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }

  const budgetY = H - PAD_B - ((H - PAD_T - PAD_B) * budgetTokens) / maxY;

  return (
    <div className="flex h-full min-h-0 flex-col rounded-xl border border-border bg-panel p-3">
      <div className="mb-1 text-[12px] font-semibold text-text">Cache size over the run</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="min-h-0 w-full flex-1" preserveAspectRatio="none">
        <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--border)" strokeWidth={1} />
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--border)" strokeWidth={1} />
        {budgetTokens > 0 && (
          <line
            x1={PAD_L}
            y1={budgetY}
            x2={W - PAD_R}
            y2={budgetY}
            stroke="var(--text-faint)"
            strokeWidth={1.2}
            strokeDasharray="4,3"
          />
        )}
        <text x={2} y={PAD_T + 6} fontSize={8} fill="var(--text-dim)">
          {maxY}
        </text>
        <text x={2} y={H - PAD_B} fontSize={8} fill="var(--text-dim)">
          0
        </text>
        {LANE_METHODS.map((id) => {
          const hist = lanes[id].history;
          if (hist.length < 2) return null;
          const points = hist.map((v, i) => xy(i, v)).join(" ");
          return <polyline key={id} points={points} fill="none" stroke={LANE_COLOR_VAR[id]} strokeWidth={2} />;
        })}
      </svg>
      <div className="mt-1.5 flex flex-wrap gap-x-2.5 gap-y-1 text-[10.5px] text-text-dim">
        {LANE_METHODS.map((id) => (
          <span key={id} className="inline-flex items-center gap-1">
            <span className="inline-block h-[2px] w-2.5" style={{ background: LANE_COLOR_VAR[id] }} />
            {LANE_LABEL[id]}
          </span>
        ))}
      </div>
    </div>
  );
}
