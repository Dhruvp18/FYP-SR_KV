# Experiment status

## Complete — verified 2026-09-27

The local backup captured all 900 unique pilot answers and the completed paired
analysis at 2026-09-26 16:24:53 UTC (21:54:53 IST). Verified zero error records
and zero cache-accounting failures. Every method/budget has 100 answers from
50 conversations. The remote kernel is no longer reachable, but the complete
results and datasets are available locally in `results/conversation_pilot_v1`.

| Method | 10% cache budget | 20% cache budget |
|---|---:|---:|
| StreamingLLM | 27% | 40% |
| SnapKV-style | 68% | 85% |
| Recency-only eviction | 46% | 62% |
| SR-KV | 45% | 73% |

The uncompressed reference scored 100%. SR-KV exceeds recency-only eviction
at 20% budget by 11 percentage points (paired 95% CI: +3 to +19), but not at
10% (-1 point, CI: -6 to +4). It is below SnapKV-style eviction at both budgets.
The predefined exploratory encouragement criterion was not met. This synthetic
pilot does not establish SR-KV as the best technique for this application.

The sections below retain the launch and recovery history.

## Launch — 2026-09-26

The pilot was launched in the user-supplied live Kaggle session. It uses one
Tesla T4, Qwen2.5-1.5B-Instruct at fp16, and an isolated Transformers 5.17.0
installation. The notebook's original Python kernel and packages were left
alone. The original untracked `eval/conv_state.py` draft was not modified.

- Kaggle validation: all 189 tests passed.
- Sanity: all 90 answers completed; accounting and budget checks passed.
- Full-cache sanity accuracy: 10/10 (5/5 on each question type).
- Main pilot: running at the last check, with real answers saved. It targets
  900 answers over 50 new conversations and automatically writes paired analysis.
- Measured sanity duration was about 5 minutes; estimate roughly 50–60 minutes
  for the main pilot. This is an estimate, not a completion guarantee.

Remote working directory: `/kaggle/working/srkv_conversation_pilot_v1`.
Remote outputs: `results/conversation_pilot_v1/` within that directory.
The local directory of the same name contains a downloaded progress snapshot,
including complete sanity results, not a live mirror. Check the remote
`pipeline_status.json` and refresh downloads before interpreting pilot results.

No conclusion about SR-KV's advantage has been drawn from the small sanity set.
The session must remain running for the detached process to finish. The runner
supports resuming completed answers if interrupted, provided the result files
are preserved and the exact same code, dataset and settings are used.

## Recovery in a fresh session — 2026-09-26

The original proxy became inaccessible. The replacement session had an empty
working directory and no mounted model files, so no additional old-session
answers could be recovered there. Restored the complete sanity checkpoint and
22 pilot answers from the local snapshot, with the same source and datasets.
The pipeline revalidates the environment, skips completed answers with matching
run keys, and resumes the remaining work. A local backup monitor now copies new
results approximately every minute; `results/conversation_pilot_v1/backup_status.json`
records the last successful backup and answer count. Backup snapshots are not a
guarantee that an inaccessible remote job is still running.

See `README_conversation_pilot.md` and `conversation_pilot_protocol.md` for
commands and the fixed comparison. No credential-bearing session URL is stored
in this repository.
