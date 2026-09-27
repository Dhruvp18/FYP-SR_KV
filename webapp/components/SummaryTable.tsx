"use client";

import { useRunStore } from "@/store/useRunStore";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";

export function SummaryTable() {
  const summary = useRunStore((s) => s.summary);
  const full = summary?.find((r) => r.lane === "full");

  if (!summary) {
    return <p className="text-[12px] text-text-faint">Run a prompt first to see the final numbers here.</p>;
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-text-faint">
              <th className="pb-2 pr-3 font-medium">Method</th>
              <th className="pb-2 pr-3 font-medium">Cached</th>
              <th className="pb-2 pr-3 font-medium">Evicted</th>
              <th className="pb-2 pr-3 font-medium">Centroids</th>
              <th className="pb-2 pr-3 font-medium">vs baseline</th>
              <th className="pb-2 font-medium">Conservation</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => {
              const vs =
                row.lane === "full" || !full
                  ? "—"
                  : `${(100 * (1 - row.final_stats.n_tokens_cached / Math.max(full.final_stats.n_tokens_cached, 1))).toFixed(0)}% smaller`;
              return (
                <tr key={row.lane} className="border-t border-border-soft">
                  <td className="py-1.5 pr-3">
                    <span className="inline-flex items-center gap-2 font-semibold text-text">
                      <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR_VAR[row.lane] }} />
                      {LANE_LABEL[row.lane]}
                    </span>
                  </td>
                  <td className="py-1.5 pr-3 text-text-dim">{row.final_stats.n_tokens_cached}</td>
                  <td className="py-1.5 pr-3 text-text-dim">{row.final_stats.n_tokens_evicted}</td>
                  <td className="py-1.5 pr-3 text-text-dim">{row.final_stats.n_centroids}</td>
                  <td className="py-1.5 pr-3 text-text-dim">{vs}</td>
                  <td className="py-1.5">
                    <span style={{ color: row.conservation_ok ? "var(--good)" : "var(--danger)" }}>
                      {row.conservation_ok ? "✓ holds" : "✗ broke"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[10.5px] leading-relaxed text-text-faint">
        Methods sharing the same budget will often show matching Evicted/Centroids counts — that count is fixed
        by (sequence length − budget), the same for any method at that budget. It says nothing about which
        method is doing better; it&apos;s <em>which tokens</em> get kept, folded, or dropped that differs, visible
        word-by-word on each method&apos;s own screen.
      </p>
      <p className="mt-1.5 text-[10.5px] leading-relaxed text-text-faint">
        This table is mechanism only — cache size, not output quality. See the closing screen for what this
        project&apos;s own evaluations found on quality.
      </p>
    </div>
  );
}
