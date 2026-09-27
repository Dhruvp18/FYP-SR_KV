"use client";

import { useGenerationSocket } from "@/hooks/useGenerationSocket";
import { useRunStore } from "@/store/useRunStore";
import { BENCHMARK_CAVEAT, VARIANT_EXPLAINER } from "@/lib/copy";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { BATCH_BUDGET, BATCH_N_PER_CELL, LANE_METHODS, VARIANTS, Variant } from "@/lib/types";
import { BenchmarkTrendChart } from "./BenchmarkTrendChart";
import { GeneratingScreen } from "./GeneratingScreen";

const VARIANT_LABEL: Record<Variant, string> = {
  attribution: "Attribution",
  aggregation: "Aggregation",
};

const LETTERS = ["A", "B", "C", "D"] as const;

const TOTAL_SAMPLES = VARIANTS.length * BATCH_N_PER_CELL;

function cellLabel(correct: number, total: number): string {
  if (total === 0) return "—";
  return `${correct}/${total} (${Math.round((100 * correct) / total)}%)`;
}

export function BenchmarkScreen() {
  const { startBatch, cancel } = useGenerationSocket();
  const connection = useRunStore((s) => s.connection);
  const contextLen = useRunStore((s) => s.contextLen);
  const batchRunning = useRunStore((s) => s.batchRunning);
  const batchTable = useRunStore((s) => s.batchTable);
  const batchProgress = useRunStore((s) => s.batchProgress);
  const batchCancelled = useRunStore((s) => s.batchCancelled);
  const currentBatchCell = useRunStore((s) => s.currentBatchCell);

  const started = batchRunning || batchTable !== null;

  return (
    <div className="mx-auto flex h-full w-full max-w-6xl min-h-0 flex-col gap-3 overflow-y-auto px-6 py-3">
      {!started && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
          <p className="text-[15px] text-text">
            Budget {Math.round(BATCH_BUDGET * 100)}%, {BATCH_N_PER_CELL} questions per variant, both variants —{" "}
            {TOTAL_SAMPLES} questions total, roughly 8–10 minutes on this machine.
          </p>
          <p className="max-w-md text-[12.5px] text-text-faint">{BENCHMARK_CAVEAT}</p>
          <div className="mt-1 flex max-w-lg flex-col gap-1.5 text-left">
            {VARIANTS.map((v) => (
              <p key={v} className="text-[11.5px] leading-snug text-text-faint">
                <span className="font-semibold text-text-dim">{VARIANT_LABEL[v]}:</span> {VARIANT_EXPLAINER[v]}
              </p>
            ))}
          </div>
          <button
            onClick={() => startBatch(contextLen)}
            disabled={connection !== "connected"}
            className="mt-1 shrink-0 rounded-lg bg-lane-srkv px-4 py-1.5 text-[12.5px] font-semibold text-bg disabled:opacity-40"
          >
            Start benchmark
          </button>
        </div>
      )}

      {started && (
        <>
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
            <div className="text-[12px] text-text-dim">
              {batchRunning ? (
                <>
                  {currentBatchCell
                    ? `${VARIANT_LABEL[currentBatchCell.variant]} · question ${currentBatchCell.sample_idx + 1} of ${currentBatchCell.n_per_cell}`
                    : "starting…"}
                  {batchProgress && ` — ${batchProgress.done}/${batchProgress.total} done`}
                </>
              ) : batchCancelled ? (
                "Cancelled — partial results below."
              ) : (
                "Benchmark complete."
              )}
            </div>
            {batchRunning ? (
              <button
                onClick={cancel}
                className="shrink-0 rounded-lg border border-border bg-panel-2 px-3 py-1 text-[11.5px] font-semibold text-text hover:bg-border"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={() => startBatch(contextLen)}
                disabled={connection !== "connected"}
                className="shrink-0 rounded-lg bg-lane-srkv px-3 py-1 text-[11.5px] font-semibold text-bg disabled:opacity-40"
              >
                Run again
              </button>
            )}
          </div>

          <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 md:grid-cols-2">
            <ReadingPanel />

            <div className="flex min-h-0 flex-col gap-3">
              {/* While running: a live trend chart (the table only makes
                  sense once cells have real n behind them). Once done: the
                  final table, in its place. */}
              {batchRunning && (
                <div className="h-55 shrink-0">
                  <BenchmarkTrendChart />
                </div>
              )}

              {!batchRunning && batchTable && (
                <div className="shrink-0 rounded-lg border border-border">
                  <table className="w-full text-[11.5px]">
                    <thead>
                      <tr className="text-text-faint">
                        <th className="px-3 py-1.5 text-left font-medium">method</th>
                        {VARIANTS.map((variant) => (
                          <th key={variant} className="px-3 py-1.5 text-right font-medium">
                            {VARIANT_LABEL[variant]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {LANE_METHODS.map((method) => (
                        <tr key={method} className="border-t border-border/60">
                          <td className="px-3 py-1.5 font-medium" style={{ color: LANE_COLOR_VAR[method] }}>
                            {LANE_LABEL[method]}
                          </td>
                          {VARIANTS.map((variant) => {
                            const cell = batchTable[variant][method];
                            return (
                              <td key={variant} className="px-3 py-1.5 text-right text-text">
                                {cellLabel(cell.correct, cell.total)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="min-h-0 flex-1 overflow-y-auto">
                <GeneratingScreen />
              </div>
            </div>
          </div>

          {!batchRunning && batchTable && (
            <p className="shrink-0 text-[11px] text-text-faint">{BENCHMARK_CAVEAT}</p>
          )}
        </>
      )}
    </div>
  );
}

// The current sample's reading material, shown live next to the running
// lanes - the same `passageWords`/`question`/`options` the single-question
// mode's `QuestionBanner`/`StepScreen` already populate from `run_started`,
// which fires once per sample during a batch too (see useRunStore's
// `applyServerEvent` "run_started" case).
function ReadingPanel() {
  const variant = useRunStore((s) => s.variant);
  const passageWords = useRunStore((s) => s.passageWords);
  const question = useRunStore((s) => s.question);
  const options = useRunStore((s) => s.options);

  const passageText = passageWords.join("");

  return (
    <div className="flex min-h-0 flex-col gap-2 rounded-lg border border-border bg-panel-2 p-3">
      {variant && (
        <div className="shrink-0">
          <span className="text-[10.5px] font-semibold uppercase tracking-wide text-text-faint">
            {variant} question
          </span>
          <p className="mt-0.5 text-[10.5px] leading-snug text-text-faint">{VARIANT_EXPLAINER[variant]}</p>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-border/60 bg-panel px-3 py-2 text-[12px] leading-relaxed text-text-dim">
        {passageText || "waiting for passage…"}
      </div>

      {question && options && (
        <div className="shrink-0 rounded-md border border-border/60 bg-panel px-3 py-2">
          <p className="text-[12px] text-text">{question}</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {options.map((opt, i) => (
              <span
                key={LETTERS[i]}
                className="rounded border border-border px-1.5 py-0.5 text-[11px] text-text-dim"
              >
                {LETTERS[i]}) {opt}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
