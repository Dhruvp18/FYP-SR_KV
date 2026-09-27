import type { LaneId } from "./types";

export const LANE_LABEL: Record<LaneId, string> = {
  full: "Full",
  streaming_llm: "StreamingLLM",
  snapkv_unified: "SnapKV",
  centroid_merge: "Centroid-merge",
  sr_kv: "SR-KV",
};

export const LANE_SUBTITLE: Record<LaneId, string> = {
  full: "no budget — keeps everything",
  streaming_llm: "sink + sliding window",
  snapkv_unified: "clustering off · recency off",
  centroid_merge: "clustering on · recency off",
  sr_kv: "clustering on · recency on",
};

// CSS var names defined in app/globals.css, also exposed as Tailwind tokens
// (bg-lane-sr-kv etc. aren't used directly since the id has an underscore -
// components read the hex via var(--lane-x) or this map for inline styles/SVG).
export const LANE_COLOR_VAR: Record<LaneId, string> = {
  full: "var(--lane-full)",
  streaming_llm: "var(--lane-streaming)",
  snapkv_unified: "var(--lane-hard)",
  centroid_merge: "var(--lane-merge)",
  sr_kv: "var(--lane-srkv)",
};
