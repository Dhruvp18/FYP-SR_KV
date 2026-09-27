// Hand-mirrored from server/protocol.py. Keep both in sync by hand.

export const LANE_METHODS = [
  "full",
  "streaming_llm",
  "snapkv_unified",
  "centroid_merge",
  "sr_kv",
] as const;

export type LaneId = (typeof LANE_METHODS)[number];

export const VARIANTS = ["attribution", "aggregation"] as const;
export type Variant = (typeof VARIANTS)[number];

// Matches server/batch.py's BATCH_BUDGET/BATCH_N_PER_CELL_DEFAULT exactly -
// a single fixed demo budget (not PREREGISTRATION.md's pre-registered
// 0.1/0.15), 10 questions per variant.
export const BATCH_BUDGET = 0.2;
export const BATCH_N_PER_CELL = 10;

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

export interface StepDiff {
  evicted_positions: number[];
  folded_positions: number[];
  evicted_positions_cumulative: number[];
}

export interface RunStartedEvent {
  type: "run_started";
  lanes: LaneId[];
  variant: Variant;
  question: string;
  options: [string, string, string, string];
  passage_tokens: number;
  passage_token_texts: string[];
  question_token_texts: string[];
  budget: number;
  max_new_tokens: number;
}

export interface BatchCell {
  variant: Variant;
  budget: number;
  sample_idx: number;
  n_per_cell: number;
}

export interface StepEvent {
  type: "step";
  lane: LaneId;
  step: number;
  phase: "passage" | "question" | "decode";
  position: number | null;
  token_text: string;
  narration: string;
  stats: CacheStats;
  conservation_ok: boolean;
  slot_snapshot: SlotSnapshot;
  diff: StepDiff;
  batch_cell?: BatchCell; // present only during a batch run
}

export interface LaneCompleteEvent {
  type: "lane_complete";
  lane: LaneId;
  generated_text: string;
  n_tokens_generated: number;
  chosen_letter: string | null;
  correct: boolean;
  batch_cell?: BatchCell;
}

export interface SummaryRow {
  lane: LaneId;
  generated_text: string;
  generated_ids: number[];
  final_stats: CacheStats;
  conservation_ok: boolean;
  chosen_letter: string | null;
  correct: boolean;
}

export interface RunCompleteEvent {
  type: "run_complete";
  summary: SummaryRow[];
  answer_letter: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type BatchTallyCell = { correct: number; total: number };
// table[variant][lane]
export type BatchTable = Record<Variant, Record<LaneId, BatchTallyCell>>;

export interface BatchSampleCompleteEvent {
  type: "batch_sample_complete";
  variant: Variant;
  budget: number;
  sample_idx: number;
  n_per_cell: number;
  summary: SummaryRow[];
}

export interface BatchProgressEvent {
  type: "batch_progress";
  done: number;
  total: number;
  table: BatchTable;
}

export interface BatchCompleteEvent {
  type: "batch_complete";
  table: BatchTable;
  cancelled: boolean;
}

export type ServerEvent =
  | RunStartedEvent
  | StepEvent
  | LaneCompleteEvent
  | RunCompleteEvent
  | BatchSampleCompleteEvent
  | BatchProgressEvent
  | BatchCompleteEvent
  | ErrorEvent;

export interface StartGistRunMessage {
  type: "start_gist_run";
  budget: number;
  variant?: Variant | null;
  context_len?: number;
}

export interface CancelRunMessage {
  type: "cancel_run";
}

export type ClientMessage = StartGistRunMessage | CancelRunMessage;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";
