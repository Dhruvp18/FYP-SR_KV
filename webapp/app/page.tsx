"use client";

import { AnimatePresence } from "framer-motion";
import { useRunStore, FINAL_STEP } from "@/store/useRunStore";
import { AdvancedPanel } from "@/components/AdvancedPanel";
import { BenchmarkScreen } from "@/components/BenchmarkScreen";
import { FinalScreen } from "@/components/FinalScreen";
import { GeneratingScreen } from "@/components/GeneratingScreen";
import { QuestionBanner } from "@/components/QuestionBanner";
import { StepperNav } from "@/components/StepperNav";
import { StepScreen } from "@/components/StepScreen";
import { TopBar } from "@/components/TopBar";
import { LANE_METHODS } from "@/lib/types";

export default function Home() {
  const mode = useRunStore((s) => s.mode);
  const phase = useRunStore((s) => s.phase);
  const currentStep = useRunStore((s) => s.currentStep);
  // Hidden on the final comparison screen only - it's already dense (chart +
  // bar race + summary text) and FinalScreen shows its own answer reveal.
  // Also hidden in benchmark mode - BenchmarkScreen shows its own per-sample
  // question context inline instead of a fixed banner.
  const showQuestionBanner =
    mode === "single" && phase !== "idle" && !(phase === "done" && currentStep === FINAL_STEP);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar />
      <AdvancedPanel />
      {showQuestionBanner && <QuestionBanner />}

      <div className="min-h-0 flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {mode === "batch" && <BenchmarkScreen key="benchmark" />}
          {mode === "single" && phase === "idle" && (
            <div key="idle" className="flex h-full flex-col items-center justify-center gap-2 text-center">
              <p className="text-[15px] text-text">
                Hit &ldquo;New question&rdquo; above to watch five real cache strategies answer the same real eval question.
              </p>
              <p className="max-w-md text-[12.5px] text-text-faint">
                Each one runs the project&apos;s actual eviction/merging code, two-pass, against a real model —
                the passage compresses before the question ever exists.
              </p>
            </div>
          )}
          {mode === "single" && phase === "generating" && <GeneratingScreen key="generating" />}
          {mode === "single" && phase === "done" && currentStep < FINAL_STEP && (
            <StepScreen key={LANE_METHODS[currentStep]} id={LANE_METHODS[currentStep]} />
          )}
          {mode === "single" && phase === "done" && currentStep === FINAL_STEP && <FinalScreen key="final" />}
        </AnimatePresence>
      </div>

      {mode === "single" && phase === "done" && <StepperNav />}
    </div>
  );
}
