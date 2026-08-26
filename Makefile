# SR-KV — one target per phase, and the single source of truth for what each
# phase actually runs. Kaggle notebooks (scripts/kaggle_kernel.py) call these
# targets rather than re-spelling the commands, so there is nothing to drift.
#
# Everything is resumable: re-running a target after a killed session continues
# from results/*.jsonl instead of starting over.

PY      ?= python
MODEL   ?= qwen2.5-1.5b
MODEL3B ?= llama3.2-3b
BUDGET  ?= 0.3
SAMPLES ?= 3
RESULTS ?= results
FIGURES ?= figures
SHARD   ?= 0
NSHARDS ?= 1

.PHONY: help test configs \
        phase1 phase1-4bit phase2 phase3 phase4 freeze-rope \
        phase5 phase5-longbench phase6-sweep phase6-3b phase7 \
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
phase4:
	for mode in latest earliest attn_weighted; do \
	  $(PY) eval/run.py --method sr_kv --model $(MODEL) --task niah \
	    --context_len 8192 --budget $(BUDGET) --n_samples 5 \
	    --depths 0,25,50,75,100 --rope_position_mode $$mode \
	    --output $(RESULTS)/phase4_rope_$$mode.json || exit 1; \
	done
	$(PY) scripts/make_plots.py --only rope

gate4:
	$(PY) scripts/check_results.py gate --phase 4 --model $(MODEL)

freeze-rope:
	$(PY) scripts/freeze_rope_mode.py --model $(MODEL) --apply

# --- Phase 5: the factorial matrix -----------------------------------------
phase5:
	$(PY) eval/run.py --method streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL) --task niah \
	  --context_len 2048,4096,8192,16384 --depths 0,25,50,75,100 \
	  --budget $(BUDGET) --n_samples $(SAMPLES) \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase5_niah_$(MODEL).json
	$(PY) eval/run.py --method full --model $(MODEL) --task niah \
	  --context_len 2048,4096,8192,16384 --depths 0,25,50,75,100 \
	  --n_samples $(SAMPLES) --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase5_niah_full_$(MODEL).json

phase5-longbench:
	$(PY) eval/run.py --method full,streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL) --task longbench --budget $(BUDGET) --n_samples 25 \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase5_longbench_$(MODEL).json

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

# No re-sweep on 3B on purpose: the question is whether the 1.5B-tuned config
# transfers, and re-tuning would answer a different question.
phase6-3b:
	$(PY) eval/run.py --method full,streaming_llm,snapkv_unified,centroid_merge,sr_kv \
	  --model $(MODEL3B) --task niah \
	  --context_len 2048,4096,8192 --depths 0,25,50,75,100 \
	  --budget $(BUDGET) --n_samples $(SAMPLES) \
	  --shard $(SHARD) --num_shards $(NSHARDS) \
	  --output $(RESULTS)/phase6_niah_$(MODEL3B).json

gate6:
	$(PY) scripts/check_results.py gate --phase 6 --model $(MODEL3B) --budget $(BUDGET)

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
