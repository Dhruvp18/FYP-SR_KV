"use client";

import { motion } from "framer-motion";
import type { WordStatus } from "@/lib/useLaneWords";

export function Word({
  text,
  status,
  color,
  delay,
}: {
  text: string;
  status: WordStatus;
  color: string;
  delay: number;
}) {
  const title =
    status === "evicted"
      ? "evicted — no longer individually in the cache"
      : status === "folded"
        ? "folded into a shared summary (centroid) slot"
        : status === "sink"
          ? "a permanently pinned sink token — never evicted"
          : "still individually in the cache";

  return (
    <motion.span
      layout
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, delay }}
      title={title}
      className={
        "mr-1.5 inline-block rounded px-0.5 font-mono text-[14px] " +
        (status === "evicted" ? "text-text-faint line-through decoration-1 opacity-50" : "text-text")
      }
      style={
        status === "folded"
          ? { background: `color-mix(in srgb, ${color} 24%, transparent)`, boxShadow: `inset 0 -2px 0 ${color}` }
          : status === "sink"
            ? { boxShadow: `inset 0 -2px 0 ${color}`, fontWeight: 600 }
            : undefined
      }
    >
      {text}
    </motion.span>
  );
}
