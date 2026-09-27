# Conversation-state pilot, version 1

Specified before running new GPU examples, 2026-09-26. This exploratory pilot
does not replace or modify the Phase 8 preregistered hypotheses or results.

Question: at the same cache budget, does full SR-KV retain updated delivery
destinations AND unchanged early delivery requirements better than recency-only
hard eviction? Other baselines: full cache, StreamingLLM, SnapKV-style eviction.

Data: controlled synthetic support transcripts, approximately 4,096 tokens.
One corrected destination and one persistent constraint per conversation.
Each of the two four-choice questions is evaluated with a fresh cache, after
compressing the identical transcript WITHOUT seeing the question. Original and
corrected destinations are randomly assigned, with the original among the
answer options. Correction depths cycle through 60/75/90%; unchanged constraint
depths cycle through 5/15/25% of filler. Report actual token lengths and retain
the exact dataset and hashes. Repeated templated filler limits external validity.

Sanity set: 5 conversations, seed 260926, 90 answers total. All methods and budgets
are checked. Continue only if full-cache accuracy is at least 9/10 overall,
at least 4/5 for each question type, every record is error-free, cache accounting
holds, and compressed pre-question caches actually meet their token budgets.
If this fails, stop and report why; do not tune on the 50-conversation pilot.

Pilot set: 50 distinct conversations, seed 260927, 900 answers total. Four
compressed methods at 0.1/0.2 budget; full cache once at budget 1.0. Qwen2.5-1.5B
Instruct, explicit fp16 on T4, greedy generation, at most 12 answer tokens.
Use repository frozen policy defaults, with no tuning or injected rank noise.

Primary metric: accuracy averaged equally across the two questions within each
conversation. Paired bootstrap resamples CONVERSATIONS (not the correlated
questions), 10,000 draws, 95% intervals. Primary comparison: sr_kv minus
recency_hard_evict at each budget. Report both question types, both-correct
accuracy, all other baselines, generated answers, memory and elapsed time.
Exploratory encouragement requires a positive 95% interval at BOTH budgets,
with no measured decrease on either question-type mean relative to recency-only.
This is not a confirmatory or general superiority claim, even if it succeeds.

No cherry-picked depth subgroup, extra sampling to chase significance, or
parameter changes after seeing pilot answers. A flat or negative result is
reported. A positive pilot needs fresh real conversations and stronger baselines.

Timing includes transcript prefill, compression, question processing and answer
generation, not model loading. Report this as end-to-end answer latency, not
decode-only throughput. GPU peak allocation includes model weights and temporary
tensors. Pre-question allocated bytes and cache slot counts are also recorded;
neither is to be mislabeled as isolated KV tensor bytes.
