// UI copy, kept separate from components so the factual claims here can be
// reviewed on their own. The risk lines for centroid_merge and sr_kv are
// deliberately grounded in this project's own measured results (Phase 8,
// `scripts/analyse_phase8.py` against `results/phase8_*.json`): centroid
// merging was significantly WORSE than hard eviction on perplexity at all 6
// tested settings, and showed no significant advantage on retrieval/recall.
// Do not soften or reverse that framing without re-checking the data.

import type { LaneId } from "./types";

export const MECHANISM: Record<LaneId, string> = {
  full: "Every token the model has read stays in memory, individually, forever.",
  streaming_llm:
    "Keeps a handful of fixed tokens from the start (a “sink”) plus a sliding window of the most recent ones. Everything in between is dropped, based only on position — never on content.",
  snapkv_unified:
    "Scores every candidate token by how much attention it drew in a short observation window, then drops the lowest-scoring ones outright.",
  centroid_merge:
    "Instead of dropping low-scoring tokens, it blends several of them into one averaged “summary” slot. Nothing is fully discarded, but the individual detail is gone.",
  sr_kv:
    "The same blending as centroid-merge, but biased to fold in the oldest tokens first, instead of choosing purely by score.",
};

export const RISK: Record<LaneId, string> = {
  full: "The cost: memory grows without bound as the conversation gets longer — nothing here is free.",
  streaming_llm:
    "Anything outside the sink and the recent window is gone for good, however important it was. StreamingLLM never reconsiders what it drops.",
  snapkv_unified:
    "A token that never draws strong attention during that one observation window is dropped for good — even if it turns out to matter later.",
  centroid_merge:
    "This project's own tests measured this as worse than dropping tokens outright on perplexity (all 6 settings tried), with no significant advantage on retrieval tasks. Blending is not a proven improvement here.",
  sr_kv:
    "Blending is still lossy — the exact original content isn't recoverable — and this project found no established advantage for recency-biased blending over plain centroid-merging or hard eviction.",
};

export const FULL_STEP_CAPTION = "Nothing removed — this is the full cost.";

export const FINAL_SUMMARY =
  "What this project's own evaluations actually found: centroid-based merging measured worse than hard eviction on perplexity, across every setting tried, and showed no significant advantage on retrieval or recall tasks. There is no established quality advantage for SR-KV or centroid-merge over hard eviction in this codebase's own experiments — this demo shows how each method decides what to keep, not which one performs best.";

// A teammate's separate pilot - not yet merged into this repo, so it isn't
// part of the Phase 8 analysis above. Numbers transcribed as given; do not
// round further in SR-KV's favor or drop the SnapKV-style loss.
export interface PilotRow {
  label: string;
  pct10: number;
  pct20: number;
}

export const TEAMMATE_PILOT: PilotRow[] = [
  { label: "StreamingLLM", pct10: 27, pct20: 40 },
  { label: "SnapKV-style", pct10: 68, pct20: 85 },
  { label: "Recency-only eviction", pct10: 46, pct20: 62 },
  { label: "SR-KV", pct10: 45, pct20: 73 },
];

export const TEAMMATE_PILOT_FULL_SCORE = 100;

export const TEAMMATE_PILOT_METRIC = "Accuracy across two questions per conversation";

export const TEAMMATE_PILOT_FINDING =
  "SR-KV beat recency-only eviction at the 20% budget by 11 percentage points (a statistically significant difference). It lost to SnapKV-style eviction at both budgets.";

export const TEAMMATE_PILOT_CAVEAT =
  "A teammate's own pilot, not yet merged into this repo, so it's separate from the Phase 8 analysis above. Its own conclusion: this pilot did not establish the application advantage it was testing for.";
