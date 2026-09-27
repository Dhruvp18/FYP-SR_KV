"use client";

import { motion } from "framer-motion";
import { Fragment, useEffect, useRef } from "react";
import { useRunStore } from "@/store/useRunStore";
import { useLaneWords } from "@/lib/useLaneWords";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { MECHANISM, RISK } from "@/lib/copy";
import { clamp } from "@/lib/format";
import { Word } from "./Word";
import type { LaneId } from "@/lib/types";

const MAX_STAGGERED_WORDS = 40; // beyond this, later words all reveal together - no absurd waits on long outputs

export function StepScreen({ id }: { id: LaneId }) {
  const { words, statuses, lane, passageEnd, questionEnd } = useLaneWords(id);
  const playbackMs = useRunStore((s) => s.playbackMs);
  const tally = useRunStore((s) => s.tally[id]);
  const color = LANE_COLOR_VAR[id];

  const perWordDelay = clamp(playbackMs / 25000, 0.006, 0.045);
  const caption = composeCaption(id, lane, statuses);
  const pct = tally.total > 0 ? Math.round((100 * tally.correct) / tally.total) : null;

  const streamRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // The passage/question boundary is the whole point of the two-pass
    // protocol (see decode_loop.py's run_gist docstring) - land on it
    // directly rather than making the viewer scroll a long passage
    // themselves to find it.
    const marker = streamRef.current?.querySelector('[data-divider="passage"]');
    marker?.scrollIntoView({ block: "center" });
  }, [id, passageEnd]);

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
        <div className="flex items-baseline gap-3">
          <span className="text-[64px] font-bold leading-none text-text">{pct !== null ? `${pct}%` : "—"}</span>
          {lane.chosenLetter !== null && (
            <span
              className="rounded-md px-2 py-1 text-[15px] font-bold"
              style={{
                color: lane.correct ? "var(--good)" : "var(--danger)",
                background: `color-mix(in srgb, ${lane.correct ? "var(--good)" : "var(--danger)"} 15%, transparent)`,
              }}
            >
              chose {lane.chosenLetter} {lane.correct ? "✓" : "✗"}
            </span>
          )}
        </div>
        <span className="text-[12px] text-text-faint">
          correct this session ({tally.correct}/{tally.total})
        </span>
        <span className="mt-1 text-[11px] text-text-faint">{lane.stats.n_tokens_cached} tokens in cache</span>
      </div>

      <div
        ref={streamRef}
        className="max-h-[26vh] w-full overflow-y-auto rounded-xl border border-border bg-panel px-5 py-4 leading-[2] scroll-smooth"
        style={{ borderColor: `color-mix(in srgb, ${color} 35%, var(--border))` }}
      >
        {words.length === 0 ? (
          <span className="text-[13px] text-text-faint">no tokens yet</span>
        ) : (
          words.map((w, i) => (
            <Fragment key={i}>
              {i === passageEnd && <Divider label="❓ question asked here" color={color} marker="passage" />}
              {i === questionEnd && questionEnd > passageEnd && (
                <Divider label="the model answers from here" color={color} subtle />
              )}
              <Word text={w} status={statuses[i]} color={color} delay={Math.min(i, MAX_STAGGERED_WORDS) * perWordDelay} />
            </Fragment>
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

function Divider({
  label,
  color,
  subtle,
  marker,
}: {
  label: string;
  color: string;
  subtle?: boolean;
  marker?: string;
}) {
  return (
    <span
      data-divider={marker}
      className="my-1.5 flex items-center gap-2 py-0.5"
      style={{ display: "flex", width: "100%" }}
    >
      <span
        className="h-px flex-1"
        style={{ background: `color-mix(in srgb, ${color} ${subtle ? 25 : 55}%, transparent)` }}
      />
      <span
        className="whitespace-nowrap text-[10.5px] font-semibold uppercase tracking-wide"
        style={{ color: `color-mix(in srgb, ${color} ${subtle ? 60 : 100}%, var(--text-dim))` }}
      >
        {label}
      </span>
      <span
        className="h-px flex-1"
        style={{ background: `color-mix(in srgb, ${color} ${subtle ? 25 : 55}%, transparent)` }}
      />
    </span>
  );
}

function composeCaption(
  id: LaneId,
  lane: ReturnType<typeof useLaneWords>["lane"],
  statuses: ReturnType<typeof useLaneWords>["statuses"],
): string {
  if (id === "full") return "Nothing removed — this is the full cost.";

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
