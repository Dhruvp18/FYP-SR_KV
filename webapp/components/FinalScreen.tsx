"use client";

import { motion } from "framer-motion";
import { FINAL_SUMMARY } from "@/lib/copy";
import { BarRace } from "./BarRace";
import { TrendChart } from "./TrendChart";

export function FinalScreen() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
      className="mx-auto flex h-full min-h-0 w-full max-w-4xl flex-col gap-4 px-6 py-4"
    >
      <div className="text-center">
        <h2 className="text-[15px] font-semibold text-text">Same sentence, five outcomes</h2>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 md:grid-cols-2">
        <TrendChart />
        <BarRace />
      </div>

      <p className="mx-auto max-w-2xl text-center text-[12.5px] leading-relaxed text-text-dim">{FINAL_SUMMARY}</p>
      <p className="mx-auto max-w-2xl text-center text-[11px] text-text-faint">
        A teammate&apos;s separate, unpublished pilot on conversational QA accuracy found a more mixed picture —
        see Advanced.
      </p>
    </motion.div>
  );
}
