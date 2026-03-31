# WebArena Recorded-Action Replay Plan

## Layout Note

The replay directory layout has since been split:

- raw replay batches now land under
  `web/agentlab_results_replay/to_verify/`
- accepted per-folder outputs now live under
  `web/agentlab_results_replay/final/`
- archived bad or replaced trajectories live under
  `web/agentlab_results_replay/failed_runs/`

For the current end-to-end workflow, use:

- [webarena_hybrid_replay_manual_recovery.md](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/docs/webarena_hybrid_replay_manual_recovery.md)

## Goal

Regenerate WebArena screenshots with a taller viewport while keeping the
original BrowserGym action sequence fixed.

This is a best-effort screenshot regeneration workflow, not an exact trajectory
reproduction guarantee. Layout, bid visibility, and app state can still drift.

## Implemented Harness Support

Replay support now lives in two places:

- `src/agentlab/agents/replay_action_agent.py`
  - `ReplayActionAgentArgs`
  - `ReplayActionAgent`
  - helpers for loading recorded actions from an existing AgentLab experiment
- `main_webarena_recorded_action_replay.py`
  - replays one source experiment directory or one whole source study directory
  - writes a fresh AgentLab-style replay batch with screenshots, step pickles,
    `result_df.csv`, `summary_df.csv`, `error_report.md`, and
    `replay_manifest.json`
  - uses the exact source study directory basename under
    `web/agentlab_results_replay/to_verify/` by default
  - deduplicates retry directories by `task_name` before replay
  - uses Ray plus the WebArena task dependency graph for safe parallelism
- `run_webarena_recorded_action_replay.sh`
  - convenience wrapper for replaying one source study directory batch
  - configures WebArena URLs and replay defaults

## Recommended Replay Viewport

Use `1280x1000` for the first replay pass.

Reasoning:

- keeps the BrowserGym default width of `1280`, reducing layout drift risk
- increases vertical context substantially over the default `1280x720`
- moves the aspect ratio closer to the imported OAgent visual trajectories,
  which were mostly `1200x1000`

Do not use browser zoom as a substitute. Bid-based click behavior is less
reliable under zoom than under a plain viewport override.

## Manual Restart Policy

For the first pass:

- restart WebArena manually once before replaying each top-level source study
  directory under `web/agentlab_results/`
- do not restart between tasks inside the same source study directory

This is the intended batching unit for the current replay script.

If a specific source study shows too much drift, fall back to smaller batches or
per-task restarts for that study only.

## Output Naming

When replaying a whole source study directory, the replay batch lands at:

```text
web/agentlab_results_replay/to_verify/<same-source-study-dir-name>
```

Example:

```text
web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

replays into:

```text
web/agentlab_results_replay/to_verify/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

The replay script now refuses to reuse an existing output directory. Remove or
rename the old replay batch before rerunning the same source batch.

## Parallelism Policy

Replay does not involve fresh LLM calls, so the harness now defaults to a
higher worker count.

- default backend: `ray`
- default worker count: `12`
- dependencies inside the selected replay batch are still respected
- dependency edges pointing to tasks outside the selected batch are dropped with
  a warning

This gives parallelism where it is safe, without ignoring WebArena's known
task-sequence coupling.

## Operational Workflow

1. Start WebArena normally.
2. Pick one source study directory from `web/agentlab_results/`.
3. Manually restart the WebArena backend once.
4. Run replay for that one source study directory.
5. Inspect the replay batch outputs.
6. If the replay quality is acceptable, move to the next source study directory
   and repeat.

## Suggested Pilot

Start with one dated source study directory and replay only a small subset:

- 5 to 10 tasks
- mixed sites if possible
- at least one reddit task, since it is the most timing-sensitive

The replay script supports `--offset` and `--limit` for this purpose.

## Example Commands

Replay one whole study directory directly:

```bash
cd enterprise/AgentLab
python main_webarena_recorded_action_replay.py \
  --source-study-dir ../../web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena \
  --viewport-width 1280 \
  --viewport-height 1000 \
  --n-jobs 12 \
  --parallel-backend ray
```

Replay one whole study directory through the wrapper script:

```bash
cd enterprise/AgentLab
./run_webarena_recorded_action_replay.sh 10.0.0.12 \
  ../../web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

Dry-run a batch first:

```bash
cd enterprise/AgentLab
python main_webarena_recorded_action_replay.py \
  --source-study-dir ../../web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena \
  --limit 10 \
  --dry-run
```

Replay a single source experiment directory:

```bash
cd enterprise/AgentLab
python main_webarena_recorded_action_replay.py \
  --source-exp-dir ../../web/agentlab_results/2026-03-04_09-31-38_genericagent-gpt-4-1-mini-2025-04-14-on-webarena/2026-03-04_12-35-57_GenericAgent-gpt-4.1-mini-2025-04-14_on_webarena.204_10 \
  --viewport-width 1280 \
  --viewport-height 1000
```

## What To Check After Each Batch

- `summary_df.csv`
  - whether most runs finish without hard errors
- `error_report.md`
  - whether failures are concentrated on a particular site or action type
- `replay_manifest.json`
  - which duplicate source directories were skipped
  - which dependency edges were dropped because they pointed outside the batch
- replay screenshots
  - whether the taller viewport actually improves reviewer-visible context
- per-task drift
  - unexpected URL changes
  - repeated `last_action_error`
  - early truncation or obviously wrong pages

## Brief Inspection Workflow

Use this as the post-batch go/no-go check before moving to the next replay
batch. Keep it short enough that it can be repeated every time.

1. Screenshot inventory.

- count replay trajectory directories
- count `screenshot_step_*.png` files
- record screenshot count per trajectory: min, median, max
- record screenshot size distribution
- pass if every replay trajectory has screenshots and the size distribution is
  dominated by the intended viewport size

2. Manual sample.

- pick `6` replay trajectories
- select `2` from the shortest trajectories by screenshot count
- select `2` around the median screenshot count
- select `2` from the longest trajectories by screenshot count
- if the batch spans multiple sites, swap samples as needed so each site is
  represented at least once

3. For each sampled trajectory, inspect only three screenshots.

- `screenshot_step_0.png`
- one middle screenshot
- the final screenshot

4. During manual inspection, check only the high-signal failure modes.

- viewport visibly shows more useful page context than the original
- no obvious clipping of the critical page region
- no obviously wrong page, blank page, login page, or repeated error page
- final screenshot is plausibly aligned with the task outcome

5. Quick metadata sanity for the same sample.

- open `summary_info.json`
- confirm no `err_msg`
- if needed, inspect the last `step_*.pkl.gz` for final `obs["url"]` and
  `obs["last_action_error"]`

6. Record one short inspection note in this run book.

- batch name
- screenshot inventory numbers
- sampled trajectories
- proceed or hold decision

Proceed to the next batch if the screenshot inventory is clean and the sampled
trajectories do not show obvious visual or navigation drift.

## Expected Failure Modes

- recorded bid click refers to an element that is no longer visible
- app state drift within a study directory accumulates enough to break later
  tasks
- reddit timing differences require a larger `pre_observation_delay`
- task succeeds but screenshot content differs enough that it is not suitable as
  a drop-in replacement

## Current Recommendation

Proceed study-by-study with:

- one manual restart per source study directory
- `1280x1000` viewport
- a pilot subset before each larger batch

If replay quality is strong on the pilot, reuse the same restart policy for the
full directory.

## Run Log

### 2026-03-29 to 2026-03-30: First Full Replay Batch

Source batch:

```text
web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

Final replay output:

```text
web/agentlab_results_replay/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

Final state:

- `90` replay directories
- `90` `summary_info.json` files
- `0` task-level errors
- replay subdirectory names normalized to exactly match the source trajectory
  directory basenames
- aggregate reports rebuilt from the final normalized layout:
  - `result_df.csv`
  - `summary_df.csv`
  - `error_report.md`
  - `summary_df.csv`

### Issues Encountered

1. Playwright browser binary missing in the `workarena` environment.

- Symptom: all replay tasks failed immediately at browser startup.
- Resolution: install the Chromium binary in the `workarena` env and rerun the
  batch.

2. Aggregate report generation failed after the first batch completed.

- Symptom: all task summaries existed, but `result_df.csv`, `summary_df.csv`,
  and `error_report.md` were missing.
- Root cause: replay metadata introduced unhashable values, especially
  list-valued columns such as recorded action arrays.
- Resolution: `main_webarena_recorded_action_replay.py` now normalizes
  unhashable replay metadata before reloading the study dataframe and writing
  aggregate reports.

3. `23` replay tasks failed in validation with a deprecated OpenAI model.

- Symptom: the replay batch finished with `67` done and `23` error, and every
  error was the same `404` on `gpt-4-1106-preview`.
- Scope: these were `string_match` tasks, concentrated on `shopping` and
  `shopping_admin`.
- Root cause: WebArena string evaluation hard-coded `gpt-4-1106-preview` in
  `webarena/evaluation_harness/helper_functions.py`.
- Resolution:
  - add a repo-local string-evaluator patch in
    `src/agentlab/experiments/webarena_eval_patch.py`
  - default the evaluation model to `gpt-5.2` through
    `WEBARENA_EVAL_LLM_MODEL`
  - call the OpenAI chat completion API with `gpt-5`-compatible parameters
  - install the patch both in the main replay entrypoint and inside
    `ExpArgs.run()` so Ray worker processes also pick it up

4. Renaming replay directories to exact source names cannot be done reliably
   during creation.

- Root cause: AgentLab experiment directories are generated from the current
  timestamp plus `exp_name`.
- Resolution: add a deterministic post-processing renamer driven by
  `replay_manifest.json`.

### Implemented Fixes

- `src/agentlab/experiments/webarena_eval_patch.py`
  - added `install_webarena_string_evaluator_patch()`
  - added `WEBARENA_EVAL_LLM_MODEL` support
  - default evaluation model is `gpt-5.2`
- `src/agentlab/experiments/loop.py`
  - added `_install_runtime_patches_for_task()`
  - installs WebArena evaluator patches inside `ExpArgs.run()` before the
    environment is created, which is required for Ray workers
- `main_webarena_recorded_action_replay.py`
  - installs both HTML and string evaluator patches
  - retains the replay-report fallback for unhashable replay metadata
- `main_webarena_generic_subset_eval.py`
  - also installs the string evaluator patch, so normal WebArena eval runs stay
    consistent with replay validation
- `rename_replay_experiment_dirs_to_source_names.py`
  - renames replay directories to the exact source experiment basenames
  - rewrites `exp_args.pkl.exp_dir`
  - regenerates `result_df.csv`, `summary_df.csv`, and `error_report.md`

### Rerun Procedure Used For The 23 Failed Tasks

1. Identify replay task directories whose `summary_info.json` contains the
   deprecated-model failure.
2. Map each failed replay task back to its `source_exp_dir` using
   `replay_manifest.json`.
3. Move the failed replay directories into a backup directory:

```text
web/agentlab_results_replay/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena.rerun_backup_gpt4_validation_<timestamp>
```

4. Create a temporary source-study directory containing only the failed source
   experiment directories.
5. Replay those `23` tasks into a temporary output root:

```text
web/agentlab_results_replay_rerun_tmp/
```

6. Verify the rerun study completes with `23/23` summaries and `0` errors.
7. Move the completed rerun directories back into the main replay study.
8. Restore the original full-batch `replay_manifest.json`.
9. Rename all replay directories to exact source names with
   `rename_replay_experiment_dirs_to_source_names.py`.
10. Rebuild aggregate reports from the final merged layout.
11. Remove scratch outputs and keep only the main replay study directory under
    `web/agentlab_results_replay/`.

### Final Naming Policy

Study directory:

- keep the replay batch under the same source study basename:
  `web/agentlab_results_replay/<source-study-name>`

Per-trajectory directory:

- rename after replay to match the exact source experiment directory basename
- do not rely on AgentLab's default timestamp-based directory naming for the
  final stored layout

Post-processing command:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd enterprise/AgentLab
python rename_replay_experiment_dirs_to_source_names.py \
  --study-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/<batch-name>
```

### Quick Inspection Snapshot For The First Batch

Batch:

```text
web/agentlab_results_replay/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

Screenshot inventory:

- `90` replay trajectory directories
- `90/90` trajectories have screenshot files
- `1027` total `screenshot_step_*.png` files
- screenshot size distribution: `1027` images at `1280x1000`
- screenshot count per trajectory: min `2`, median `6`, max `51`

Suggested six-trajectory manual sample for this batch:

- shortest: `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.163_10`
- shortest: `2026-02-16_18-16-45_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.194_4`
- median: `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.13_23`
- median: `2026-02-16_18-16-45_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.77_4`
- longest: `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.107_14`
- longest: `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.111_6`

Status:

- automated screenshot inventory check passed
- manual visual sample completed on `2026-03-30`
- proceed decision: yes, with the caveat that final URL equality is too strict
  to use as the acceptance gate for replayed WebArena tasks

Manual sample notes:

- `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.163_10`
  - same final URL in source and replay
  - replay final screenshot showed the product page content clearly
  - source final screenshot was much less informative in the original shorter
    crop
- `2026-02-16_18-16-45_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.194_4`
  - replay matched source closely across both screenshots
- `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.13_23`
  - replay matched source closely through the final step
- `2026-02-16_18-16-45_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.77_4`
  - replay matched source closely through the final step
- `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.107_14`
  - replay drifted during a long admin filtering sequence
  - this was not a new regression: the source run was also truncated with
    `cum_reward = 0.0`
- `2026-02-16_18-16-44_GenericAgent-gpt-5-mini-2025-08-07-re-high_on_webarena.111_6`
  - replay drifted during a long admin filtering sequence and ended on a
    different final URL
  - this was also not a new regression: the source run was already truncated
    with `cum_reward = 0.0`

Batch-level comparison against the source run:

- `88/90` tasks kept the same `cum_reward`
- `2/90` tasks improved from `0.0` to `1.0`
- `0/90` tasks regressed on `cum_reward`
- termination/truncation matched the source batch exactly: `81` terminated,
  `9` truncated
- replay had `5` non-empty final `last_action_error` values versus `3` in the
  source batch
- final URL equality only held for `45/90` tasks, so URL equality should be
  treated as a weak signal rather than a batch acceptance requirement

### Notes For Future Batches

- Always export or inherit `WEBARENA_EVAL_LLM_MODEL=gpt-5.2` unless there is a
  reason to change the evaluator model.
- When replaying against locally forwarded WebArena ports, use `localhost` as
  the replay host, not `127.0.0.1`.
  BrowserGym's WebArena task validates open tabs by exact `netloc`, so a batch
  launched with `WA_*` URLs on `127.0.0.1` will treat tabs opened on
  `localhost` as unauthorized and terminate tasks immediately.
- If a batch is interrupted, it is safe to rerun only the failed subset into a
  temporary output root and merge afterward.
- Keep `replay_manifest.json`; it is the authoritative mapping between replay
  tasks and source directories.
- Run the renaming script after the batch is final, because report rewriting is
  built into the script.

### 2026-03-30: 2026-02-17 Batch Relaunch Diagnosis

Bad replay output:

```text
web/agentlab_results_replay/2026-02-17_09-24-08_genericagent-gpt-5-mini-2025-08-07-on-webarena
```

Observed failure pattern:

- `271/271` replay summaries had `cum_reward = 0.0`
- every replayed task stopped after the first action
- replay `step_1.pkl.gz` files contained
  `task_info = {"error": "Unauthorized url, terminating task"}`

Root cause:

- the bad batch was launched against `127.0.0.1`
- the live pages themselves were on `localhost`
- BrowserGym's WebArena validator checks each open tab against the exact
  allowed `netloc` values from `WA_*`
- `127.0.0.1:PORT` and `localhost:PORT` do not match under that check

Implication:

- do not use the bad `2026-02-17` replay batch for inspection or downstream
  artifacts
- remove it and rerun the same source study with `localhost`
