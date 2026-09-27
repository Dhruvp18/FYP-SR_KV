"use client";

import { useMemo } from "react";
import { useRunStore } from "@/store/useRunStore";
import type { LaneId } from "./types";

export const STREAMING_SINK = 4; // fixed real default (src/caches/streaming_llm.py) - not a UI control

export type WordStatus = "alive" | "sink" | "folded" | "evicted";

export function useLaneWords(id: LaneId) {
  const lane = useRunStore((s) => s.lanes[id]);
  const passageWords = useRunStore((s) => s.passageWords);
  const questionWords = useRunStore((s) => s.questionWords);

  const words = useMemo(
    () => [...passageWords, ...questionWords, ...lane.decodeWords],
    [passageWords, questionWords, lane.decodeWords],
  );

  // Where the passage ends and the question begins, and where the question
  // ends and the model's own generated answer begins - the two-pass boundary
  // the whole point of this task rests on (see server/decode_loop.py's
  // run_gist docstring): everything left of passageEnd was already
  // compressed with zero knowledge the question existed.
  const passageEnd = passageWords.length;
  const questionEnd = passageEnd + questionWords.length;

  const aliveSet = useMemo(() => {
    const s = new Set<number>();
    lane.slotSnapshot.positions.forEach((p, i) => {
      if (!lane.slotSnapshot.is_centroid[i]) s.add(Math.round(p));
    });
    return s;
  }, [lane.slotSnapshot]);

  const foldedSet = useMemo(() => new Set(lane.lastDiff.folded_positions), [lane.lastDiff.folded_positions]);
  const evictedSet = useMemo(
    () => new Set(lane.lastDiff.evicted_positions_cumulative),
    [lane.lastDiff.evicted_positions_cumulative],
  );

  const statuses = useMemo<WordStatus[]>(
    () =>
      words.map((_, i) => {
        if (id === "streaming_llm" && i < STREAMING_SINK && aliveSet.has(i)) return "sink";
        if (aliveSet.has(i)) return "alive";
        if (foldedSet.has(i)) return "folded";
        if (evictedSet.has(i)) return "evicted";
        return "alive";
      }),
    [words, aliveSet, foldedSet, evictedSet, id],
  );

  const counts = useMemo(() => {
    let alive = 0,
      folded = 0,
      evicted = 0;
    statuses.forEach((s) => {
      if (s === "folded") folded++;
      else if (s === "evicted") evicted++;
      else alive++;
    });
    return { alive, folded, evicted, total: words.length };
  }, [statuses, words.length]);

  return { lane, words, statuses, counts, passageEnd, questionEnd };
}
