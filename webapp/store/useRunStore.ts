import { create } from "zustand";
import {
  CacheStats,
  ConnectionStatus,
  LANE_METHODS,
  LaneId,
  ServerEvent,
  SlotSnapshot,
  StepDiff,
  SummaryRow,
} from "@/lib/types";

const EMPTY_STATS: CacheStats = {
  n_tokens_cached: 0,
  n_tokens_evicted: 0,
  n_centroids: 0,
  budget_used_pct: 0,
};

const EMPTY_SNAPSHOT: SlotSnapshot = { positions: [], weights: [], is_centroid: [] };
const EMPTY_DIFF: StepDiff = {
  merged: [],
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
  };
}

function freshLanes(): Record<LaneId, LaneRuntime> {
  return Object.fromEntries(LANE_METHODS.map((id) => [id, freshLane(id)])) as Record<LaneId, LaneRuntime>;
}

const DEFAULT_PLAYBACK_MS = 550;
export const FINAL_STEP = LANE_METHODS.length; // one extra step: the closing comparison screen

export type RunPhase = "idle" | "generating" | "done";

interface RunState {
  connection: ConnectionStatus;
  phase: RunPhase;
  currentStep: number; // 0..LANE_METHODS.length-1 = a method, LANE_METHODS.length = final comparison
  promptTokens: number;
  promptWords: string[]; // shared across all lanes - prefill is identical for every method
  budget: number;
  maxNewTokens: number;
  step: number;
  activeLane: LaneId | null;
  lanes: Record<LaneId, LaneRuntime>;
  summary: SummaryRow[] | null;
  advancedOpen: boolean;
  errorMessage: string | null;

  // Not a network throttle - purely how fast a step screen's word-by-word
  // reveal animation plays when you land on it (see StepScreen). Exposed as
  // an advanced control.
  playbackMs: number;

  setConnection: (status: ConnectionStatus) => void;
  setPlaybackMs: (ms: number) => void;
  setBudget: (budget: number) => void;
  setMaxNewTokens: (n: number) => void;
  startRun: (budget: number, maxNewTokens: number) => void;
  applyServerEvent: (event: ServerEvent) => void;
  goNext: () => void;
  goPrev: () => void;
  toggleAdvanced: () => void;
  dismissError: () => void;
}

export const useRunStore = create<RunState>((set, get) => ({
  connection: "connecting",
  phase: "idle",
  currentStep: 0,
  promptTokens: 0,
  promptWords: [],
  budget: 0.3,
  maxNewTokens: 24,
  step: 0,
  activeLane: null,
  lanes: freshLanes(),
  summary: null,
  advancedOpen: false,
  errorMessage: null,
  playbackMs: DEFAULT_PLAYBACK_MS,

  setConnection: (status) => set({ connection: status }),
  setPlaybackMs: (ms) => set({ playbackMs: ms }),
  setBudget: (budget) => set({ budget }),
  setMaxNewTokens: (maxNewTokens) => set({ maxNewTokens }),
  toggleAdvanced: () => set((s) => ({ advancedOpen: !s.advancedOpen })),
  dismissError: () => set({ errorMessage: null }),

  startRun: (budget, maxNewTokens) =>
    set({
      phase: "generating",
      currentStep: 0,
      budget,
      maxNewTokens,
      step: 0,
      activeLane: null,
      promptWords: [],
      lanes: freshLanes(),
      summary: null,
      errorMessage: null,
    }),

  goNext: () => set((s) => ({ currentStep: Math.min(FINAL_STEP, s.currentStep + 1) })),
  goPrev: () => set((s) => ({ currentStep: Math.max(0, s.currentStep - 1) })),

  applyServerEvent: (event) => {
    switch (event.type) {
      case "run_started": {
        set({
          promptTokens: event.prompt_tokens,
          promptWords: event.prompt_token_texts,
          budget: event.budget,
          maxNewTokens: event.max_new_tokens,
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
        set({ lanes, step: Math.max(get().step, event.step), activeLane: event.lane });
        return;
      }
      case "lane_complete": {
        const lanes = { ...get().lanes };
        lanes[event.lane] = { ...lanes[event.lane], done: true, generatedText: event.generated_text };
        set({ lanes });
        return;
      }
      case "run_complete": {
        set({ phase: "done", summary: event.summary, activeLane: null, currentStep: 0 });
        return;
      }
      case "error": {
        set({ phase: "idle", errorMessage: event.message });
        return;
      }
    }
  },
}));
