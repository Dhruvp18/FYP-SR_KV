"use client";

import { useRunStore } from "@/store/useRunStore";
import { LANE_COLOR_VAR, LANE_LABEL } from "@/lib/palette";
import { LANE_METHODS } from "@/lib/types";

export function GeneratingScreen() {
  const lanes = useRunStore((s) => s.lanes);
  const step = useRunStore((s) => s.step);
  const maxNewTokens = useRunStore((s) => s.maxNewTokens);
  const activeLane = useRunStore((s) => s.activeLane);

  return (
    <div className="mx-auto flex h-full w-full max-w-2xl min-h-0 flex-col justify-center gap-3 px-6">
      <div className="mb-1 text-center">
        <p className="text-[14px] text-text">Running all five methods on your sentence, live…</p>
        <p className="text-[11.5px] text-text-faint">
          step {step} of up to {maxNewTokens}
        </p>
      </div>

      {LANE_METHODS.map((id) => {
        const lane = lanes[id];
        const color = LANE_COLOR_VAR[id];
        const active = activeLane === id && !lane.done;
        return (
          <div
            key={id}
            className="rounded-lg border px-3.5 py-2.5 transition-colors"
            style={{ borderColor: active ? color : "var(--border)" }}
          >
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-2.5 w-2.5 shrink-0 items-center justify-center">
                <span className="h-2 w-2 rounded-full" style={{ background: color }} />
                {active && (
                  <span className="pulse-dot absolute inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                )}
              </span>
              <span className="text-[15px] font-bold" style={{ color }}>
                {LANE_LABEL[id]}
              </span>
              <span className="ml-auto text-[12px] text-text-dim">
                {lane.stats.n_tokens_cached} cached &middot; {lane.decodeWords.length} generated
                {lane.done && <span style={{ color: "var(--good)" }}> &middot; done</span>}
              </span>
            </div>
            <p className="mt-1 truncate pl-5 text-[11.5px] text-text-faint">{lane.narration || "waiting…"}</p>
          </div>
        );
      })}
    </div>
  );
}
