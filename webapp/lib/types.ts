// Hand-mirrored from server/protocol.py. Keep both in sync by hand.

export const LANE_METHODS = [
  "full",
  "streaming_llm",
  "snapkv_unified",
  "centroid_merge",
  "sr_kv",
] as const;

export type LaneId = (typeof LANE_METHODS)[number];

export interface CacheStats {
  n_tokens_cached: number;
  n_tokens_evicted: number;
  n_centroids: number;
  budget_used_pct: number;
}

export interface SlotSnapshot {
  positions: number[];
  weights: number[];
  is_centroid: boolean[];
}

export interface MergeEvent {
  position: number;
  weight: number;
  member_positions: number[];
}

export interface StepDiff {
  merged: MergeEvent[];
  evicted_positions: number[];
  folded_positions: number[];
  evicted_positions_cumulative: number[];
}

export interface RunStartedEvent {
  type: "run_started";
  lanes: LaneId[];
  prompt_tokens: number;
  prompt_token_texts: string[];
  budget: number;
  max_new_tokens: number;
}

export interface StepEvent {
  type: "step";
  lane: LaneId;
  step: number;
  phase: "prefill" | "decode";
  position: number | null;
  token_text: string;
  narration: string;
  stats: CacheStats;
  conservation_ok: boolean;
  slot_snapshot: SlotSnapshot;
  diff: StepDiff;
}

export interface LaneCompleteEvent {
  type: "lane_complete";
  lane: LaneId;
  generated_text: string;
  n_tokens_generated: number;
}

export interface SummaryRow {
  lane: LaneId;
  generated_text: string;
  generated_ids: number[];
  final_stats: CacheStats;
  conservation_ok: boolean;
}

export interface RunCompleteEvent {
  type: "run_complete";
  summary: SummaryRow[];
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type ServerEvent =
  | RunStartedEvent
  | StepEvent
  | LaneCompleteEvent
  | RunCompleteEvent
  | ErrorEvent;

export interface StartRunMessage {
  type: "start_run";
  prompt: string;
  budget: number;
  max_new_tokens: number;
}

export interface CancelRunMessage {
  type: "cancel_run";
}

export type ClientMessage = StartRunMessage | CancelRunMessage;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";
