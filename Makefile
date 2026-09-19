# SR-KV — one target per phase, and the single source of truth for what each
# phase actually runs. Kaggle notebooks (scripts/kaggle_kernel.py) call these
# targets rather than re-spelling the commands, so there is nothing to drift.
#
# Everything is resumable: re-running a target after a killed session continues
# from results/*.jsonl instead of starting over.

PY      ?= python
MODEL   ?= qwen2.5-1.5b
# qwen2.5-3b, not llama3.2-3b: Llama-3.2 is gated on HF and blocks on manual
# access; Qwen2.5-3B is ungated, its RoPE is length-independent (safe for
# centroid re-rotation, unlike most other long-context checkpoints), and it
# natively covers the 16k context the sweeps need. Trade-off: Phase 6 then
# shows generalization across scale within the Qwen family rather than across
# architectures. Set MODEL3B=llama3.2-3b to restore the original comparison
# once HF access lands.
MODEL3B ?= qwen2.5-3b
BUDGET  ?= 0.3
SAMPLES ?= 3
RESULTS ?= results
FIGURES ?= figures
SHARD   ?= 0
NSHARDS ?= 1

.PHONY: help test configs phase8-perplexity phase8-tight-longbench phase8-analyse \
        phase1 phase1-4bit phase2 phase3 phase4 phase4-scan freeze-rope \
        phase5 phase5-longbench phase5-recompress-probe phase6-sweep phase6-3b-scan phase6-3b phase7 \
        gate1 gate2 gate3 gate4 gate5 gate6 gate7 \
        check-complete check-ablation plots report_artifacts clean-figures

help:
	@echo "Local (no GPU):"
	@echo "  make test              - full CPU test suite, no downloads"
	@echo "  make configs           - regenerate configs/*.yaml"
	@echo ""
	@echo "GPU phases (run on Kaggle):"
	@echo "  make phase1            - harness sanity on a real model   -> make gate1"
	@echo "  make phase1-4bit       - exercise the 4-bit fallback path"
	@echo "  make phase2            - StreamingLLM + SnapKV baselines  -> make gate2"
	@echo "  make phase3            - unified SRKVCache at 8k          -> make gate3"
	@echo "  make phase4            - RoPE position ablation           -> make gate4"
	@echo "  make freeze-rope       - write the Phase 4 winner into configs/defaults.yaml"
	@echo "  make phase5            - four-condition factorial NIAH    -> make gate5"
	@echo "  make phase5-longbench  - four-condition LongBench subset"
	@echo "  make phase6-sweep      - alpha/beta sweep on 1.5B"
	@echo "  make phase6-3b         - transfer the frozen config to 3B -> make gate6"
	@echo "  make phase7            - regenerate every figure          -> make gate7"
	@echo "  make phase8-perplexity - E1: does merging win on perplexity?"
	@echo "  make phase8-tight-longbench - E2: does merging win at budget 0.05-0.15?"
	@echo "  make phase8-analyse    - the pre-registered test (PREREGISTRATION.md)"
	@echo ""
	@echo "Each gateN exits non-zero if that phase's pass condition is not met."

test:
	$(PY) -m pytest -q

configs:
	$(PY) scripts/gen_configs.py

# --- Phase 1: is the harness itself trustworthy? ---------------------------
# A 1.5B instruct model retrieves a magic number from 512 tokens without
# difficulty, so anything below the gate means the harness is broken, not the
# model.
phase1:
	$(PY) eval/run.py --method full --model $(MODEL) --task niah \
	  --context_len 512 --depths 50 --n_samples 10 --max_new_tokens 16 \
	  --output $(RESULTS)/phase1_sanity.json

phase1-4bit:
	$(PY) eval/run.py --method full --model $(MODEL3B) --precision 4bit --task niah \
	  --context_len 512 --depths 50 --n_samples 3 --max_new_tokens 16 \
	  --output $(RESULTS)/phase1_4bit.json

gate1:
	$(PY) scripts/check_results.py gate --phase 1 --model $(MODEL)

# --- Phase 2: do the baselines fail the way the papers say they do? --------
phase2:
	$(PY) eval/run.py --method full,streaming_llm,snapkv --model $(MODEL) \
	  --task niah --context_len 4096 --depths 0,25,50,75,100 --budget $(BUDGET) \
	  --n_samples 5 --output $(RESULTS)/phase2_baselines.json

gate2:
	$(PY) scripts/check_results.py gate --phase 2 --model $(MODEL)

# --- Phase 3: the unified class survives a full-length run -----------------
phase3:
	$(PY) eval/run.py --method sr_kv,centroid_merge,snapkv_unified \
	  --model $(MODEL) --task niah --context_len 8192 \
	  --depths 0,50,100 --budget $(BUDGET) --n_samples $(SAMPLES) \
	  --output $(RESULTS)/phase3_8k.json

gate3:
	$(PY) scripts/check_results.py gate --phase 3 --model $(MODEL)

# --- Phase 4: choose the centroid RoPE convention by experiment ------------
# The three literal modes belong here and nowhere else: this target IS the
# ablation. Every later phase reads the winner from configs/defaults.yaml.
#
# SAMPLES4 is separate from SAMPLES on purpose. This comparison needs enough
# records per mode to separate the conventions from sampling noise: measured
# at n=25/mode the three modes came out 7/25, 6/25, 7/25 (Fisher exact
# p=1.000), where one flipped record moves a mean by 0.04 - larger than any
# difference worth reporting. Raise it, do not reuse the sweep default.
SAMPLES4 ?= 5

phase4:
	for mode in latest earliest attn_weighted; do \
	  $(PY) eval/run.py --method sr_kv --model $(MODEL) --task niah \
	    --context_len 8192 --budget $(BUDGET) --n_samples $(SAMPLES4) \
	    --depths 0,25,50,75,100 --rope_position_mode $$mode \
	    --output $(RESULTS)/phase4_rope_$$mode.json || exit 1; \
	done
	$(PY) scripts/make_plots.py --only rope

# Diagnostic, not a gated phase: locate a budget where NIAH accuracy is
# mid-range, so the comparison above has something it can separate. Measured
# so far: budget=0.3 saturates at 1.000 for all three modes, budget=0.1 floors
# at ~0.25 carried entirely by depth=100 (needle inside the observation
# window). A sweep at either extreme cannot distinguish the conventions at any
# sample size, so the budget has to be found before spending quota on power.
# One mode is enough to locate it - the modes are what we are trying to tell
# apart, not what sets the difficulty.
#
# Output goes to results/diagnostics/, not results/, on purpose. A scan cell
# and a real-sweep cell can land on the identical (method, budget, mode,
# depth, sample_idx) task_id/run_key - and did: the first phase4 run at
# budget=0.2 collided with this scan's own budget=0.2 probe on 15/100 cells,
# and load_records' directory-wide dedup silently let the n=3 scan value
# overwrite the n=20 sweep value on one of them (alphabetical file order, no
# recency check) - moving accuracy from 76/100 to 77/100 by chance rather than
# by measurement. load_records globs non-recursively, so a subdirectory is
# enough to keep a diagnostic run out of anything a gate aggregates.
phase4-scan:
	mkdir -p $(RESULTS)/diagnostics
	for b in 0.15 0.20 0.25; do \
	  $(PY) eval/run.py --method sr_kv --model $(MODEL) --task niah \
	    --context_len 8192 --budget $$b --n_samples 3 \
	    --depths 0,25,50,75,100 --rope_position_mode attn_weighted \
	    --output $(RESULTS)/diagnostics/phase4_scan_b$$b.json || exit 1; \
	done

# Phase 4 varied ONLY rope_position_mode; these are the scoring knobs it held
# fixed (defaults.yaml's values at the time it ran). Pinning them here keeps
# Phase 6's alpha/beta sweep - same model, same budget, same default rope mode,
# same results/ directory - from being counted as extra Phase 4 samples.
ALPHA4 ?= 1.0
BETA4  ?= 0.3
LAM4   ?= 0.001

gate4:
	$(PY) scripts/check_results.py gate --phase 4 --model $(MODEL) --budget $(BUDGET)  --alpha $(ALPHA4) --beta $(BETA4) --lam $(LAM4)

freeze-rope:
	$(PY) scripts/freeze_rope_mode.py --model $(MODEL) --budget $(BUDGET)  --alpha $(ALPHA4) --beta $(BETA4) --lam $(LAM4) --apply

# --- Phase 5: the factorial matrix -----------------------------------------
phase5:
	$(PY) eval/run.py --method streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL) --task niah \
	  --context_len 2048,4096,8192 --depths 0,25,50,75,100 \
	  --budget $(BUDGET) --n_samples $(SAMPLES) \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase5_niah_$(MODEL).json
	$(PY) eval/run.py --method full --model $(MODEL) --task niah \
	  --context_len 2048,4096,8192 --depths 0,25,50,75,100 \
	  --budget 1.0 --n_samples $(SAMPLES) --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase5_niah_full_$(MODEL).json

# One process per method, not one process for all five. A single process
# handling all ~500 tasks OOM'd twice on a real run - once at task ~81
# (fixed by expandable_segments below), then again at task ~280 requesting
# just 1.15 GiB with the allocator fix already active, so it isn't purely
# fragmentation: something in the generate/cache stack is not fully
# releasing GPU memory between tasks, and defragmenting a leak only delays
# it. All five invocations still append to the same --output file (that's
# how resumability already works across separate pushes), so this costs
# four extra model loads (~seconds each) in exchange for a fresh CUDA
# context - and therefore a real zeroed allocator - per method. Same
# per-item-loop pattern phase4/phase6-sweep already use for the same reason.
# RECOMPRESS_SLACK=16, not the class default of 0: measured on a real run,
# slack=0 (recompress on every single decode token) causes sr_kv/
# centroid_merge to degenerate into repetition loops on long generations
# (40% of gov_report samples) because every one of those hundreds of
# recompressions fully re-clusters every centroid from scratch. A targeted
# probe (results/diagnostics/phase5_recompress_slack{16,32}_*.json) confirmed
# slack=16 fixes it (0/10 degenerate, both values tested equally well; 16
# chosen as the more conservative of the two). Applied to every method, not
# just the two that showed the bug: recompress_slack also changes eviction
# timing for streaming_llm/snapkv_unified, and this project's whole ablation
# design rests on changing exactly one thing at a time between conditions -
# silently running the baselines at slack=0 while sr_kv/centroid_merge use
# slack=16 would confound the comparison instead of fixing it.
RECOMPRESS_SLACK ?= 16

phase5-longbench:
	for method in full streaming_llm snapkv_unified centroid_merge sr_kv; do \
	  $(PY) eval/run.py --method $$method \
	    --model $(MODEL) --task longbench --budget $(BUDGET) --n_samples 25 \
	    --recompress_slack $(RECOMPRESS_SLACK) \
	    --shard $(SHARD) --num_shards $(NSHARDS) \
	    --output $(RESULTS)/phase5_longbench_$(MODEL).json || exit 1; \
	done

# Diagnostic, not a gated phase: real gov_report run measured sr_kv/
# centroid_merge degenerating into repetition loops on 10/25 samples each
# (0/25 for the non-clustering methods) - 512-token generations against a
# tight budget mean the cache exceeds budget on nearly every decode step, and
# recompress_slack=0 (the only value ever tried) means every one of those
# steps fully re-clusters every centroid from scratch (fresh k-means init,
# not continuing the previous step's assignment), rather than the model ever
# seeing a stable KV history between compressions. Only the two clustering
# methods (use_clustering=True) are affected; only long generations expose
# it (0/25 at 32 tokens, up to 10/25 at 512). This probes whether batching
# compression with recompress_slack fixes it, on the one task and the two
# methods that showed the problem, at reduced samples for a fast answer
# before committing the full sweep to a slack value.
phase5-recompress-probe:
	for slack in 16 32; do \
	  for method in centroid_merge sr_kv; do \
	    $(PY) eval/run.py --method $$method --model $(MODEL) --task longbench \
	      --longbench_tasks gov_report --budget $(BUDGET) --n_samples 10 \
	      --recompress_slack $$slack \
	      --output $(RESULTS)/diagnostics/phase5_recompress_slack$${slack}_$${method}.json \
	      || exit 1; \
	  done; \
	done

gate5:
	$(PY) scripts/check_results.py gate --phase 5 --model $(MODEL) --budget $(BUDGET) \
	  --n-samples $(SAMPLES)

# --- Phase 6: hyperparameters on 1.5B, then transfer unchanged to 3B -------
phase6-sweep:
	for a in 0.5 1.0 2.0; do \
	  for b in 0.0 0.3 0.6; do \
	    $(PY) eval/run.py --method sr_kv --model $(MODEL) --task niah \
	      --context_len 4096,8192 --depths 0,25,50,75,100 --budget $(BUDGET) \
	      --alpha $$a --beta $$b --n_samples $(SAMPLES) \
	      --shard $(SHARD) --num_shards $(NSHARDS) \
	      --output $(RESULTS)/phase6_sweep_a$${a}_b$${b}.json || exit 1; \
	  done; \
	done

# Diagnostic, not a gated phase: `full` (uncompressed) on qwen2.5-1.5b at
# 8192 tokens already used 11.9-13.5 GiB on a 14.56 GiB T4 for LongBench -
# essentially no margin. qwen2.5-3b is a bigger model (more weight memory,
# larger activations at the same sequence length), so `full` at the same
# 8192-token ceiling that phase6-3b tests is a real, foreseeable OOM risk.
# Two samples at the longest context, cheap, before committing the full
# 5-method x 3-context x 5-depth x n_samples sweep to a precision that might
# not fit. choose_precision()'s "auto" only estimates KV-cache size, not
# activation/logits memory - it under-estimated for LongBench (picked bf16
# when bf16 didn't actually fit), so do not trust it uncritically here either.
phase6-3b-scan:
	$(PY) eval/run.py --method full --model $(MODEL3B) --precision $(PRECISION6) \
	  --task niah --context_len 8192 --depths 50 --n_samples 2 \
	  --output $(RESULTS)/diagnostics/phase6_3b_scan_$(PRECISION6).json

# No re-sweep on 3B on purpose: the question is whether the 1.5B-tuned config
# transfers, and re-tuning would answer a different question.
# Two budgets on purpose. 0.2 is where the 1.5B sweep ran and where Phase 4
# measured mid-range accuracy, so it can actually separate the methods; 0.3 is
# the Makefile default that Phase 3 showed saturates to 1.000 for every method
# at 8k. Running both gives the transfer question a discriminating point and
# keeps the saturated one for comparison, instead of betting the whole 3B run
# on a budget that may prove nothing.
BUDGETS6 ?= 0.2,0.3
COMMA := ,

# bf16, and only 2048/4096. The route here matters, because both halves of
# this are forced by hardware, not chosen for convenience.
#
# bf16 at 8192 does not fit: phase6-3b-scan OOM'd on a real T4 inside SDPA,
# asking for 4.05 GiB with 3.91 GiB free and 10.65 GiB already in use.
# choose_precision()'s auto mode had said bf16 was fine, because it sizes
# weights + uncompressed KV + 2.5 GB headroom and never models the attention
# activation - the fourth OOM in this project from that blind spot.
#
# 4-bit does fit (11.96 GiB peak, 1.000 accuracy, ~32s/task in the scan) but
# hangs the GPU. Three runs stalled after 31, 37 and 4 tasks with no error, no
# log and no OOM, at different methods, contexts and memory levels, and the
# wedged process survived SIGKILL. Phases 4 and 5 ran 300 and 500 tasks on
# this same harness in bf16 without a single stall, so bitsandbytes is the
# variable, not the harness.
#
# That leaves bf16 at 2048/4096 as the configuration this hardware can
# actually measure. It is a real scope reduction and is recorded as one: the
# transfer claim is "the 1.5B-tuned config transfers to 3B at 2k-4k", with no
# 8192 evidence on 3B. Recorded, not silently dropped - PHASE6_CONTEXTS in
# check_results.py carries the same note, so the gate cannot quietly pass a
# grid that is missing a column nobody remembers removing.
PRECISION6 ?= bf16
CONTEXTS6  ?= 2048,4096

# `full` is uncompressed, so a budget is meaningless for it - make_cache
# ignores the value. It gets its own invocation at --budget 1.0 because that
# is the label check_completeness expects for an uncompressed reference, and
# because folding it into the sweep above would otherwise run the single most
# expensive method once per budget for identical work.
phase6-3b:
	$(PY) eval/run.py --method streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL3B) --precision $(PRECISION6) --task niah \
	  --context_len $(CONTEXTS6) --depths 0,25,50,75,100 \
	  --budget $(BUDGETS6) --n_samples $(SAMPLES) \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase6_niah_$(MODEL3B).json
	$(PY) eval/run.py --method full --model $(MODEL3B) --precision $(PRECISION6) \
	  --task niah \
	  --context_len $(CONTEXTS6) --depths 0,25,50,75,100 \
	  --budget 1.0 --n_samples $(SAMPLES) \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase6_niah_full_$(MODEL3B).json

gate6:
	$(PY) scripts/check_results.py gate --phase 6 --model $(MODEL3B) \
	  $(foreach b,$(subst $(COMMA), ,$(BUDGETS6)),--budget $(b)) --n-samples $(SAMPLES)

# --- Phase 8: does centroid-merging help anywhere? -------------------------
# Pre-registered in PREREGISTRATION.md BEFORE any of this was run. The success
# criterion (paired bootstrap, 95% CI excluding zero in the predicted
# direction, n>=50, holding at >=2 settings) is fixed there and must not move.
#
# Phases 5/6 gave a clean null for clustering. Two reasons that may be about
# what was measured rather than the mechanism: every metric so far scores
# n-gram overlap, which a centroid cannot produce by construction, and
# compression has been so cheap (hard eviction -2.3%) that there is nothing to
# recover. E1 changes the metric, E2 changes the pressure.
PPL_SAMPLES ?= 50
PPL_CONTEXTS ?= 2048,4096,8192
PPL_BUDGETS ?= 0.2,0.3

phase8-perplexity:
	$(PY) eval/run.py --method streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL) --task perplexity \
	  --context_len $(PPL_CONTEXTS) --budget $(PPL_BUDGETS) \
	  --n_samples $(PPL_SAMPLES) --precision bf16 \
	  --output $(RESULTS)/phase8_perplexity_$(MODEL).json
	$(PY) eval/run.py --method full \
	  --model $(MODEL) --task perplexity \
	  --context_len $(PPL_CONTEXTS) --budget 1.0 \
	  --n_samples $(PPL_SAMPLES) --precision bf16 \
	  --output $(RESULTS)/phase8_perplexity_full_$(MODEL).json

# E2: the same LongBench grid Phase 5 ran, at budgets where hard eviction
# should actually start losing information rather than redundancy.
TIGHT_BUDGETS ?= 0.05,0.1,0.15

# Settings match phase5-longbench exactly (n_samples 25, same
# recompress_slack, same four tasks) apart from the budget. Anything else
# differing would confound "tighter budget" with "different setup".
phase8-tight-longbench:
	for method in streaming_llm snapkv_unified centroid_merge sr_kv; do \
	  $(PY) eval/run.py --method $$method \
	    --model $(MODEL) --task longbench --budget $(TIGHT_BUDGETS) --n_samples 25 \
	    --recompress_slack $(RECOMPRESS_SLACK) \
	    --shard $(SHARD) --num_shards $(NSHARDS) \
	    --output $(RESULTS)/phase8_tight_longbench_$(MODEL).json || exit 1; \
	done

phase8-analyse:
	$(PY) scripts/analyse_phase8.py

# --- Phase 7: every figure, one command ------------------------------------
phase7 plots report_artifacts:
	$(PY) scripts/make_plots.py --results-dir $(RESULTS) --figures-dir $(FIGURES)

gate7:
	$(PY) scripts/check_results.py gate --phase 7 --figures-dir $(FIGURES)

check-complete:
	$(PY) scripts/check_results.py completeness --model $(MODEL) --budget $(BUDGET) \
	  --n-samples $(SAMPLES) --skip-longbench

check-ablation:
	$(PY) scripts/check_results.py ablation --model $(MODEL)

clean-figures:
	rm -rf $(FIGURES)
