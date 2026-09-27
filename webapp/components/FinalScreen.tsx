"use client";

import { motion } from "framer-motion";
import { useRunStore } from "@/store/useRunStore";
import { FINAL_SUMMARY } from "@/lib/copy";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { BarRace } from "./BarRace";
import { TrendChart } from "./TrendChart";

export function FinalScreen() {
  const summary = useRunStore((s) => s.summary);
  const answerLetter = useRunStore((s) => s.answerLetter);
  const tally = useRunStore((s) => s.tally);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
      className="mx-auto flex h-full min-h-0 w-full max-w-4xl flex-col gap-3 px-6 py-3"
    >
      <div className="text-center">
        <h2 className="text-[15px] font-semibold text-text">Same question, five outcomes</h2>
        {answerLetter && <p className="text-[11.5px] text-text-faint">correct answer: {answerLetter}</p>}
      </div>

      {summary && (
        <div className="flex flex-wrap justify-center gap-2">
          {summary.map((row) => {
            const t = tally[row.lane];
            const pct = t.total > 0 ? Math.round((100 * t.correct) / t.total) : null;
            return (
              <span
                key={row.lane}
                className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11.5px]"
                style={{ borderColor: `color-mix(in srgb, ${LANE_COLOR_VAR[row.lane]} 40%, var(--border))` }}
              >
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: LANE_COLOR_VAR[row.lane] }} />
                <span className="font-semibold text-text">{LANE_LABEL[row.lane]}</span>
                <span style={{ color: row.correct ? "var(--good)" : "var(--danger)" }}>
                  {row.chosen_letter ?? "?"} {row.correct ? "✓" : "✗"}
                </span>
                {pct !== null && (
                  <span className="text-text-faint">
                    &middot; {pct}% ({t.correct}/{t.total})
                  </span>
                )}
              </span>
            );
          })}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 md:grid-cols-2">
        <TrendChart />
        <BarRace />
      </div>

      <p className="mx-auto max-w-2xl text-center text-[12.5px] leading-relaxed text-text-dim">{FINAL_SUMMARY}</p>
      <p className="mx-auto max-w-2xl text-center text-[11px] text-text-faint">
        This one question is anecdotal — the project&apos;s own aggregate test (n≈50/cell) found this NOT
        SUPPORTED either way. A teammate&apos;s separate, unpublished pilot on conversational QA accuracy found a
        more mixed picture — see Advanced.
      </p>
    </motion.div>
  );
}
