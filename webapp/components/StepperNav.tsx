"use client";

import { useRunStore, FINAL_STEP } from "@/store/useRunStore";
import { LANE_COLOR_VAR } from "@/lib/palette";
import { LANE_METHODS } from "@/lib/types";

export function StepperNav() {
  const currentStep = useRunStore((s) => s.currentStep);
  const goNext = useRunStore((s) => s.goNext);
  const goPrev = useRunStore((s) => s.goPrev);

  const steps = [...LANE_METHODS.map((id) => LANE_COLOR_VAR[id]), "var(--text-dim)"]; // last dot = final comparison

  return (
    <div className="flex shrink-0 items-center justify-center gap-4 border-t border-border bg-panel px-4 py-3">
      <button
        onClick={goPrev}
        disabled={currentStep === 0}
        className="rounded-lg border border-border px-3.5 py-1.5 text-[12.5px] font-semibold text-text disabled:opacity-30"
      >
        &larr; Back
      </button>

      <div className="flex items-center gap-2">
        {steps.map((color, i) => (
          <span
            key={i}
            className="h-2 w-2 rounded-full transition-all"
            style={{
              background: color,
              opacity: i === currentStep ? 1 : 0.25,
              transform: i === currentStep ? "scale(1.3)" : "scale(1)",
            }}
          />
        ))}
      </div>

      <button
        onClick={goNext}
        disabled={currentStep === FINAL_STEP}
        className="rounded-lg bg-lane-srkv px-3.5 py-1.5 text-[12.5px] font-semibold text-bg disabled:opacity-30"
      >
        {currentStep === FINAL_STEP - 1 ? "See the comparison →" : "Next →"}
      </button>
    </div>
  );
}
