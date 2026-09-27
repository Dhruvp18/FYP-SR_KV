"use client";

import { useRunStore } from "@/store/useRunStore";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { BATCH_N_PER_CELL, LANE_METHODS, VARIANTS } from "@/lib/types";

const W = 420;
const H = 200;
const PAD_L = 26;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 20;

const TOTAL_QUESTIONS = VARIANTS.length * BATCH_N_PER_CELL;

// Live accuracy trend for benchmark mode: x = questions answered so far
// (both variants combined, in run order), y = cumulative % accuracy per
// lane. Same raw-SVG-polyline approach as TrendChart, fixed axes instead of
// auto-scaling - the point is watching each line move as the run
// progresses, not a tight fit around whatever's plotted so far.
export function BenchmarkTrendChart() {
  const series = useRunStore((s) => s.batchChartSeries);

  function xy(i: number, v: number): string {
    const x = PAD_L + ((W - PAD_L - PAD_R) * (i + 1)) / TOTAL_QUESTIONS;
    const y = H - PAD_B - ((H - PAD_T - PAD_B) * v) / 100;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }

  return (
    <div className="flex h-full min-h-0 flex-col rounded-xl border border-border bg-panel p-3">
      <div className="mb-1 text-[12px] font-semibold text-text">Accuracy so far, live</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="min-h-0 w-full flex-1" preserveAspectRatio="none">
        <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--border)" strokeWidth={1} />
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--border)" strokeWidth={1} />
        <line
          x1={PAD_L}
          y1={H - PAD_B - (H - PAD_T - PAD_B) / 2}
          x2={W - PAD_R}
          y2={H - PAD_B - (H - PAD_T - PAD_B) / 2}
          stroke="var(--text-faint)"
          strokeWidth={1}
          strokeDasharray="4,3"
        />
        <text x={2} y={PAD_T + 6} fontSize={8} fill="var(--text-dim)">
          100%
        </text>
        <text x={10} y={H - PAD_B - (H - PAD_T - PAD_B) / 2 + 3} fontSize={8} fill="var(--text-dim)">
          50%
        </text>
        <text x={10} y={H - PAD_B} fontSize={8} fill="var(--text-dim)">
          0%
        </text>
        {LANE_METHODS.map((id) => {
          const hist = series[id];
          if (hist.length < 2) return null;
          const points = hist.map((v, i) => xy(i, v)).join(" ");
          return <polyline key={id} points={points} fill="none" stroke={LANE_COLOR_VAR[id]} strokeWidth={2} />;
        })}
      </svg>
      <div className="mt-1 flex items-center justify-between text-[9.5px] text-text-faint">
        <span>question 1</span>
        <span>question {TOTAL_QUESTIONS}</span>
      </div>
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
