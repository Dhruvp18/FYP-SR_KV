"use client";

import { AnimatePresence } from "framer-motion";
import { useRunStore, FINAL_STEP } from "@/store/useRunStore";
import { AdvancedPanel } from "@/components/AdvancedPanel";
import { FinalScreen } from "@/components/FinalScreen";
import { GeneratingScreen } from "@/components/GeneratingScreen";
import { StepperNav } from "@/components/StepperNav";
import { StepScreen } from "@/components/StepScreen";
import { TopBar } from "@/components/TopBar";
import { LANE_METHODS } from "@/lib/types";

export default function Home() {
  const phase = useRunStore((s) => s.phase);
  const currentStep = useRunStore((s) => s.currentStep);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar />
      <AdvancedPanel />

      <div className="min-h-0 flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {phase === "idle" && (
            <div key="idle" className="flex h-full flex-col items-center justify-center gap-2 text-center">
              <p className="text-[15px] text-text">Type a sentence above to watch five real cache strategies decide what to remember.</p>
              <p className="max-w-md text-[12.5px] text-text-faint">
                Each one runs the project&apos;s actual eviction/merging code against a real model — nothing here is scripted.
              </p>
            </div>
          )}
          {phase === "generating" && <GeneratingScreen key="generating" />}
          {phase === "done" && currentStep < FINAL_STEP && <StepScreen key={LANE_METHODS[currentStep]} id={LANE_METHODS[currentStep]} />}
          {phase === "done" && currentStep === FINAL_STEP && <FinalScreen key="final" />}
        </AnimatePresence>
      </div>

      {phase === "done" && <StepperNav />}
    </div>
  );
}
