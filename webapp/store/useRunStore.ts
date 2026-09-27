import { create } from "zustand";
import {
  BatchCell,
  BatchTable,
  CacheStats,
  ConnectionStatus,
  LANE_METHODS,
  LaneId,
  ServerEvent,
  SlotSnapshot,
  StepDiff,
  SummaryRow,
  Variant,
} from "@/lib/types";
import { GIST_CONTEXT_LEN_DEFAULT } from "@/lib/config";

const EMPTY_STATS: CacheStats = {
  n_tokens_cached: 0,
  n_tokens_evicted: 0,
  n_centroids: 0,
  budget_used_pct: 0,
};

const EMPTY_SNAPSHOT: SlotSnapshot = { positions: [], weights: [], is_centroid: [] };
const EMPTY_DIFF: StepDiff = {
  evicted_positions: [],
  folded_positions: [],
  evicted_positions_cumulative: [],
};

export interface LaneRuntime {
  id: LaneId;
  stats: CacheStats;
  conservationOk: boolean;
  slotSnapshot: SlotSnapshot;
  lastDiff: StepDiff;
  narration: string;
  decodeWords: string[]; // this lane's own generated words, in order - lanes diverge
  generatedText: string;
  history: number[];
  done: boolean;
  chosenLetter: string | null;
  correct: boolean;
}

function freshLane(id: LaneId): LaneRuntime {
  return {
    id,
    stats: EMPTY_STATS,
    conservationOk: true,
    slotSnapshot: EMPTY_SNAPSHOT,
    lastDiff: EMPTY_DIFF,
    narration: "",
    decodeWords: [],
    generatedText: "",
    history: [],
    done: false,
    chosenLetter: null,
    correct: false,
  };
}

function freshLanes(): Record<LaneId, LaneRuntime> {
  return Object.fromEntries(LANE_METHODS.map((id) => [id, freshLane(id)])) as Record<LaneId, LaneRuntime>;
}

export interface LaneTally {
  correct: number;
  total: number;
}

function freshTally(): Record<LaneId, LaneTally> {
  return Object.fromEntries(LANE_METHODS.map((id) => [id, { correct: 0, total: 0 }])) as Record<LaneId, LaneTally>;
}

function freshChartSeries(): Record<LaneId, number[]> {
  return Object.fromEntries(LANE_METHODS.map((id) => [id, [] as number[]])) as Record<LaneId, number[]>;
}

const DEFAULT_PLAYBACK_MS = 550;
export const FINAL_STEP = LANE_METHODS.length; // one extra step: the closing comparison screen

export type RunPhase = "idle" | "generating" | "done";
export type RunMode = "single" | "batch";

interface RunState {
  connection: ConnectionStatus;
  mode: RunMode;
  phase: RunPhase;
  currentStep: number; // 0..LANE_METHODS.length-1 = a method, LANE_METHODS.length = final comparison
  passageTokens: number;
  passageWords: string[]; // shared across all lanes - the passage compresses before the question exists
  questionWords: string[]; // shared across all lanes - the question itself
  question: string;
  options: [string, string, string, string] | null;
  variant: Variant | null;
  answerLetter: string | null; // set only on run_complete - withheld until the reveal
  budget: number;
  contextLen: number;
  maxNewTokens: number;
  step: number;
  activeLane: LaneId | null;
  lanes: Record<LaneId, LaneRuntime>;
  summary: SummaryRow[] | null;
  advancedOpen: boolean;
  errorMessage: string | null;

  // Benchmark mode: a scaled-down live replica of the real Phase 8 table
  // (see server/batch.py). Reuses `lanes` above for the live per-sample
  // view - a batch sample is just a `run_gist` call, same event shapes,
  // just annotated with `currentBatchCell`.
  batchRunning: boolean;
  batchTable: BatchTable | null;
  batchProgress: { done: number; total: number } | null;
  batchCancelled: boolean;
  currentBatchCell: BatchCell | null;

  // Cumulative accuracy per lane after each question answered so far (both
  // variants combined, in run order) - what BenchmarkTrendChart plots live
  // while running. batchRunningTally is the running correct/total behind it;
  // the table above is split by variant and isn't what this chart needs.
  batchChartSeries: Record<LaneId, number[]>;
  batchRunningTally: Record<LaneId, LaneTally>;

  // How each lane has done across every question asked *this session* - not
  // reset by startGistRun, only by resetTally(). This is what actually
  // differs lane to lane; a single run's token-cache count mostly doesn't,
  // since the budgeted lanes all target the same budget_tokens.
  tally: Record<LaneId, LaneTally>;

  // Not a network throttle - purely how fast a step screen's word-by-word
  // reveal animation plays when you land on it (see StepScreen). Exposed as
  // an advanced control.
  playbackMs: number;

  setConnection: (status: ConnectionStatus) => void;
  setPlaybackMs: (ms: number) => void;
  setBudget: (budget: number) => void;
  setContextLen: (n: number) => void;
  setMode: (mode: RunMode) => void;
  startGistRun: (budget: number, variant: Variant | null, contextLen: number) => void;
  startBatchRun: (contextLen: number) => void;
  applyServerEvent: (event: ServerEvent) => void;
  goNext: () => void;
  goPrev: () => void;
  toggleAdvanced: () => void;
  dismissError: () => void;
  resetTally: () => void;
}

export const useRunStore = create<RunState>((set, get) => ({
  connection: "connecting",
  mode: "single",
  phase: "idle",
  currentStep: 0,
  passageTokens: 0,
  passageWords: [],
  questionWords: [],
  question: "",
  options: null,
  variant: null,
  answerLetter: null,
  budget: 0.2,
  contextLen: GIST_CONTEXT_LEN_DEFAULT,
  maxNewTokens: 12,
  step: 0,
  activeLane: null,
  lanes: freshLanes(),
  summary: null,
  advancedOpen: false,
  errorMessage: null,
  playbackMs: DEFAULT_PLAYBACK_MS,
  tally: freshTally(),
  batchRunning: false,
  batchTable: null,
  batchProgress: null,
  batchCancelled: false,
  currentBatchCell: null,
  batchChartSeries: freshChartSeries(),
  batchRunningTally: freshTally(),

  setConnection: (status) => set({ connection: status }),
  setPlaybackMs: (ms) => set({ playbackMs: ms }),
  setBudget: (budget) => set({ budget }),
  setContextLen: (contextLen) => set({ contextLen }),
  setMode: (mode) => set({ mode }),
  toggleAdvanced: () => set((s) => ({ advancedOpen: !s.advancedOpen })),
  dismissError: () => set({ errorMessage: null }),
  resetTally: () => set({ tally: freshTally() }),

  startGistRun: (budget, variant, contextLen) =>
    set({
      mode: "single",
      phase: "generating",
      currentStep: 0,
      budget,
      contextLen,
      step: 0,
      activeLane: null,
      passageWords: [],
      questionWords: [],
      question: "",
      options: null,
      variant,
      answerLetter: null,
      lanes: freshLanes(),
      summary: null,
      errorMessage: null,
    }),

  startBatchRun: (contextLen) =>
    set({
      mode: "batch",
      contextLen,
      activeLane: null,
      passageWords: [],
      questionWords: [],
      question: "",
      options: null,
      variant: null,
      answerLetter: null,
      lanes: freshLanes(),
      summary: null,
      errorMessage: null,
      batchRunning: true,
      batchTable: null,
      batchProgress: null,
      batchCancelled: false,
      currentBatchCell: null,
      batchChartSeries: freshChartSeries(),
      batchRunningTally: freshTally(),
    }),

  goNext: () => set((s) => ({ currentStep: Math.min(FINAL_STEP, s.currentStep + 1) })),
  goPrev: () => set((s) => ({ currentStep: Math.max(0, s.currentStep - 1) })),

  applyServerEvent: (event) => {
    switch (event.type) {
      case "run_started": {
        // A batch run calls run_gist once per sample, so this fires once per
        // question - reset per-lane state each time so the live view shows
        // this sample, not a stale one from the previous cell.
        set({
          passageTokens: event.passage_tokens,
          passageWords: event.passage_token_texts,
          questionWords: event.question_token_texts,
          question: event.question,
          options: event.options,
          variant: event.variant,
          budget: event.budget,
          maxNewTokens: event.max_new_tokens,
          lanes: freshLanes(),
          activeLane: null,
        });
        return;
      }
      case "step": {
        const lanes = { ...get().lanes };
        const prev = lanes[event.lane];
        lanes[event.lane] = {
          ...prev,
          stats: event.stats,
          conservationOk: event.conservation_ok,
          slotSnapshot: event.slot_snapshot,
          lastDiff: event.diff,
          narration: event.narration,
          decodeWords: event.phase === "decode" ? [...prev.decodeWords, event.token_text] : prev.decodeWords,
          generatedText: event.phase === "decode" ? prev.generatedText + event.token_text : prev.generatedText,
          history: [...prev.history, event.stats.n_tokens_cached],
        };
        set({
          lanes,
          step: Math.max(get().step, event.step),
          activeLane: event.lane,
          currentBatchCell: event.batch_cell ?? get().currentBatchCell,
        });
        return;
      }
      case "lane_complete": {
        const lanes = { ...get().lanes };
        lanes[event.lane] = {
          ...lanes[event.lane],
          done: true,
          generatedText: event.generated_text,
          chosenLetter: event.chosen_letter,
          correct: event.correct,
        };
        set({ lanes, currentBatchCell: event.batch_cell ?? get().currentBatchCell });
        return;
      }
      case "run_complete": {
        const tally = { ...get().tally };
        for (const row of event.summary) {
          const prev = tally[row.lane];
          tally[row.lane] = { correct: prev.correct + (row.correct ? 1 : 0), total: prev.total + 1 };
        }
        set({
          phase: "done",
          summary: event.summary,
          activeLane: null,
          currentStep: 0,
          answerLetter: event.answer_letter,
          tally,
        });
        return;
      }
      // Batch mode: `batch_sample_complete` replaces `run_complete` as the
      // per-sample terminal event. The table (split by variant) arrives
      // fully formed on every `batch_progress`/`batch_complete`, so nothing
      // to accumulate there - but the live trend chart plots cumulative
      // accuracy across *both* variants combined, in run order, which
      // nothing else tracks, so build it up here.
      case "batch_sample_complete": {
        const tally = { ...get().batchRunningTally };
        const chart = { ...get().batchChartSeries };
        for (const row of event.summary) {
          const prev = tally[row.lane];
          const next = { correct: prev.correct + (row.correct ? 1 : 0), total: prev.total + 1 };
          tally[row.lane] = next;
          chart[row.lane] = [...chart[row.lane], Math.round((100 * next.correct) / next.total)];
        }
        set({ batchRunningTally: tally, batchChartSeries: chart });
        return;
      }
      case "batch_progress": {
        set({ batchTable: event.table, batchProgress: { done: event.done, total: event.total } });
        return;
      }
      case "batch_complete": {
        set({ batchTable: event.table, batchRunning: false, batchCancelled: event.cancelled });
        return;
      }
      case "error": {
        set({ phase: "idle", batchRunning: false, errorMessage: event.message });
        return;
      }
    }
  },
}));
