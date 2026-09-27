"use client";

import { motion } from "framer-motion";
import { useRunStore } from "@/store/useRunStore";
import { useLaneWords } from "@/lib/useLaneWords";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { FULL_STEP_CAPTION, MECHANISM, RISK } from "@/lib/copy";
import { clamp } from "@/lib/format";
import { Word } from "./Word";
import type { LaneId } from "@/lib/types";

const MAX_STAGGERED_WORDS = 40; // beyond this, later words all reveal together - no absurd waits on long outputs

export function StepScreen({ id }: { id: LaneId }) {
  const { words, statuses, lane } = useLaneWords(id);
  const playbackMs = useRunStore((s) => s.playbackMs);
  const color = LANE_COLOR_VAR[id];
  const isFull = id === "full";

  const perWordDelay = clamp(playbackMs / 25000, 0.006, 0.045);

  const caption = isFull ? FULL_STEP_CAPTION : composeCaption(lane, statuses);

  return (
    <motion.div
      key={id}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="mx-auto flex h-full w-full max-w-4xl min-h-0 flex-col items-center justify-center gap-5 px-6"
    >
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="text-[36px] font-extrabold uppercase tracking-tight" style={{ color }}>
          {LANE_LABEL[id]}
        </span>
        <p className="max-w-xl text-[14px] leading-relaxed text-text-dim">{MECHANISM[id]}</p>
      </div>

      <div className="flex flex-col items-center">
        <span className="text-[64px] font-bold leading-none text-text">{lane.stats.n_tokens_cached}</span>
        <span className="text-[12px] text-text-faint">tokens in cache</span>
      </div>

      <div
        className="max-h-[26vh] w-full overflow-y-auto rounded-xl border border-border bg-panel px-5 py-4 leading-[2]"
        style={{ borderColor: `color-mix(in srgb, ${color} 35%, var(--border))` }}
      >
        {words.length === 0 ? (
          <span className="text-[13px] text-text-faint">no tokens yet</span>
        ) : (
          words.map((w, i) => (
            <Word key={i} text={w} status={statuses[i]} color={color} delay={Math.min(i, MAX_STAGGERED_WORDS) * perWordDelay} />
          ))
        )}
      </div>

      <p className="max-w-xl text-center text-[13.5px] text-text">{caption}</p>

      <div
        className="max-w-xl rounded-lg border px-4 py-2.5 text-center text-[12.5px] leading-relaxed text-text-dim"
        style={{ borderColor: `color-mix(in srgb, ${color} 30%, var(--border))` }}
      >
        {RISK[id]}
      </div>
    </motion.div>
  );
}

function composeCaption(lane: ReturnType<typeof useLaneWords>["lane"], statuses: ReturnType<typeof useLaneWords>["statuses"]): string {
  const folded = statuses.filter((s) => s === "folded").length;
  const evicted = statuses.filter((s) => s === "evicted").length;
  const alive = statuses.length - folded - evicted;

  const parts = [`Kept ${alive} token${alive === 1 ? "" : "s"} individually`];
  if (folded > 0) {
    parts.push(`folded ${folded} into ${lane.stats.n_centroids} summary slot${lane.stats.n_centroids === 1 ? "" : "s"}`);
  }
  if (evicted > 0) {
    parts.push(`evicted ${evicted} completely`);
  }
  return parts.join(", ") + ".";
}
