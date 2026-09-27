"use client";

import { useRunStore } from "@/store/useRunStore";
import { VARIANT_EXPLAINER } from "@/lib/copy";

const LETTERS = ["A", "B", "C", "D"] as const;

export function QuestionBanner() {
  const question = useRunStore((s) => s.question);
  const options = useRunStore((s) => s.options);
  const variant = useRunStore((s) => s.variant);
  const answerLetter = useRunStore((s) => s.answerLetter);

  if (!question || !options) return null;

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-2 text-center">
      {variant && (
        <>
          <span className="text-[10.5px] font-semibold uppercase tracking-wide text-text-faint">
            {variant} question
          </span>
          <p className="mx-auto mt-0.5 max-w-xl text-[10.5px] leading-snug text-text-faint">
            {VARIANT_EXPLAINER[variant]}
          </p>
        </>
      )}
      <p className="mt-1 text-[13.5px] text-text">{question}</p>
      <div className="mt-2 flex flex-wrap justify-center gap-2">
        {options.map((opt, i) => {
          const letter = LETTERS[i];
          const revealed = answerLetter !== null;
          const isAnswer = revealed && letter === answerLetter;
          return (
            <span
              key={letter}
              className="rounded-md border px-2.5 py-1 text-[12px]"
              style={{
                borderColor: isAnswer ? "var(--good)" : "var(--border)",
                color: isAnswer ? "var(--good)" : "var(--text-dim)",
                background: isAnswer ? "color-mix(in srgb, var(--good) 14%, transparent)" : "transparent",
                fontWeight: isAnswer ? 700 : 400,
              }}
            >
              {letter}) {opt}
              {isAnswer && " ✓"}
            </span>
          );
        })}
      </div>
    </div>
  );
}
