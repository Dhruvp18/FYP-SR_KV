// UI copy, kept separate from components so the factual claims here can be
// reviewed on their own. The risk lines for centroid_merge and sr_kv are
// deliberately grounded in this project's own measured results (Phase 8,
// `scripts/analyse_phase8.py` against `results/phase8_*.json`): centroid
// merging was significantly WORSE than hard eviction on perplexity at all 6
// tested settings, and showed no significant advantage on retrieval/recall.
// Do not soften or reverse that framing without re-checking the data.

import type { LaneId, Variant } from "./types";

// Grounded in eval/gist_mcq.py's own module docstring and the two
// `_build_attribution`/`_build_aggregation` sample builders - keep these in
// sync if that file's task design changes.
export const VARIANT_EXPLAINER: Record<Variant, string> = {
  attribution:
    "One person→team fact, restated 4 times in different wording and scattered through the passage (plus 2 unrelated decoy mentions of other people). Tests whether an entity-attribute link survives compression even though no single restatement looks important on its own — nothing in a short attention window flags it, since the question hasn't been asked yet.",
  aggregation:
    "One topic is mentioned 5 times, two others once each, one never. Tests whether the relative frequency of scattered mentions survives compression, not just whether a topic was mentioned at all — something hard eviction can't represent, since a token either survives whole or is gone.",
};

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

// Benchmark mode ("BenchmarkScreen"): a live, interactive demo built on the
// real Phase 8 H3a/H3b task (PREREGISTRATION.md addendum) and its real
// scoring code, run at a single fixed budget and a fraction of the real
// scale so it fits in minutes on a CPU demo machine. Keep this caveat next
// to every table this mode renders.
export const BENCHMARK_CAVEAT =
  "A small, live demo of the real Phase 8 H3a/H3b task and scoring code, unchanged — but at a single fixed budget (20%) picked for a clear demo, not the pre-registered budgets (10%/15%), and 10 questions per variant here versus 50 per cell in the real test, with a much shorter passage. Read this as \"does the direction look right,\" not a statistically significant result — see Advanced for the real Phase 8 findings at the actual pre-registered budgets.";
