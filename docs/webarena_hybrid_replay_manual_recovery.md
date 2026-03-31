# WebArena Hybrid Replay and Manual Recovery Workflow

## Purpose

Recover higher-quality WebArena screenshot trajectories for LongMemEval in a
way that is operationally realistic and preserves as much of the original
trajectory set as possible.

This workflow is the current recommended replacement for a replay-only
strategy.

The unit of work is one source study folder at a time:

1. replay the whole source study folder once with the recorded-action replay
   harness
2. keep only the replayed trajectories that are already good enough
3. manually rerun the remaining bad trajectories from that same folder
4. finish the folder completely before moving to the next folder

## Why This Exists

Replay-only regeneration works for some WebArena folders, but not for all of
them.

What we observed in practice:

- some folders replay well enough to salvage a meaningful subset directly
- other folders drift badly even after fixing obvious harness issues
- most bad replay cases are not total environment failures; they are action
  fidelity failures
- the most common bad replay pattern is bid or element mapping drift
  - the recorded action string still parses
  - but the bid now points to a different element, a hidden element, or a
    non-input
- a smaller number of cases are state transition drift
  - the action lands on roughly the right page
  - but the resulting page state or selected record is wrong

That failure pattern makes a hybrid approach attractive:

- use replay to cheaply recover the easy cases
- use manual reruns only for the hard cases
- during manual reruns, the operator is not solving the task from scratch
  blindly
- instead, the operator uses the source trajectory as a guide and steers the
  environment back toward the intended state

## Scope

This document is for WebArena trajectory recovery under:

- source root:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results`
- replay output root:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay`
- raw replay batches to inspect:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/to_verify`
- accepted per-folder outputs:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/final`
- archived bad or replaced trajectories:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/failed_runs`
- harness root:
  `/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab`

This is not a document for WorkArena reruns. WorkArena already has its own
manual rerun workflow.

For this document, a `selected trajectory` means the single canonical source
trajectory chosen for a task after replay-side deduplication. The final replay
folder should contain at most one converted trajectory per selected source
trajectory.

## Current Harness Status

Already implemented:

- recorded-action replay agent:
  [replay_action_agent.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/src/agentlab/agents/replay_action_agent.py)
- replay batch CLI:
  [main_webarena_recorded_action_replay.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/main_webarena_recorded_action_replay.py)
- replay wrapper:
  [run_webarena_recorded_action_replay.sh](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/run_webarena_recorded_action_replay.sh)
- replay watcher:
  [watch_webarena_recorded_action_replay.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/watch_webarena_recorded_action_replay.py)
- replay dir normalization and report rewrite:
  [rename_replay_experiment_dirs_to_source_names.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/rename_replay_experiment_dirs_to_source_names.py)
- interactive manual agent class:
  [manual_action_agent.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/src/agentlab/agents/manual_action_agent.py)
- interactive WorkArena launcher pattern:
  [main_workarena_manual.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/main_workarena_manual.py)
- interactive WebArena launcher:
  [main_webarena_manual.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/main_webarena_manual.py)
- WebArena manual wrapper:
  [run_webarena_manual.sh](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/run_webarena_manual.sh)
- manual-recovery replacement helper:
  [install_manual_recovery_into_replay_study.py](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/install_manual_recovery_into_replay_study.py)

Important replay-side fixes already learned:

- use `localhost`, not `127.0.0.1`, for WebArena URLs
- keep the WebArena string evaluator patched to a live model such as `gpt-5.2`
- rename replay dirs to match the exact source trajectory basenames after a
  replay batch finishes

Previously missing but now added:

- a checked-in WebArena-specific interactive launcher:
  `main_webarena_manual.py`
- a checked-in helper to archive bad replay dirs, install accepted manual
  recoveries, and rewrite reports:
  `install_manual_recovery_into_replay_study.py`

## Core Principles

1. Work folder by folder.

- Reset WebArena once before starting each source study folder.
- Do not mix trajectories from different source folders in one recovery pass.

2. Keep one final destination folder per source study.

- Final output for one source study should live at:
  `web/agentlab_results_replay/final/<same-source-study-dir-name>`
- Raw replay batches should land first at:
  `web/agentlab_results_replay/to_verify/<same-source-study-dir-name>`
- The final folder should contain only accepted converted trajectories.
- Known-bad trajectories should be moved to a backup folder rather than left in
  the final destination.
- There should be at most one converted trajectory per selected source
  trajectory.

3. Preserve source trajectory names.

- Each final trajectory directory in the replay folder should have the exact
  same basename as the corresponding source trajectory directory.

4. Prefer replay when it is already good enough.

- Manual recovery is only for trajectories that replay did not recover
  adequately.
- For source-success trajectories, a replay is only automatically acceptable if
  it also ends with reward `1`.
- For source-failure trajectories, replay acceptance requires a more careful
  human judgment about whether the failure semantics were preserved.

5. Manual recovery is guided recovery, not blind task solving.

- The operator is allowed to inspect the original source trajectory and use it
  as a guide.
- The operator should still act through the live WebArena UI.
- Do not use hidden backend shortcuts, database lookups, or direct deep links
  that bypass the interface.
- For manual recovery, the goal is to preserve the original navigation and
  interaction pattern first, not to optimize for success.

## Folder-Level Deliverable

For each source study folder, the final deliverable is:

- one raw replay study folder under `web/agentlab_results_replay/to_verify/`
- one accepted final study folder under `web/agentlab_results_replay/final/`
- accepted automated replays copied from `to_verify` into the final study folder
- bad automated replay trajectories moved to backup storage
- successful manual recoveries copied into the final study folder as
  replacements for the bad automated replay trajectories
- normalized trajectory directory names matching source basenames
- regenerated aggregate reports:
  - `result_df.csv`
  - `summary_df.csv`
  - `error_report.md`
- a short run-log note recording what was kept automatically and what was
  recovered manually

## Recommended Per-Folder Workflow

### 0. Preflight

Before starting a source study folder:

- ensure the WebArena stack is up and ports are forwarded locally
- ensure `localhost` URLs are used
- activate the `workarena` conda env
- confirm `OPENAI_API_KEY` is present
- export `WEBARENA_EVAL_LLM_MODEL=gpt-5.2`
- manually reset the WebArena backend once
- keep that single reset for the whole folder workflow, including manual
  recovery for that folder
- because there is only one shared WebArena stack, the manual recovery phase
  should run in single-agent mode
- inspection and queue construction can still be prepared in batches, but live
  manual reruns should be executed serially by one agent

### 1. Run Automated Replay For The Whole Folder

Recommended first-pass replay settings:

- viewport: `1280x1000`
- host: `localhost`
- `WEBARENA_REPLAY_N_JOBS=12`
- replay one whole source folder at a time

Example:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd /home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab

export WEBARENA_EVAL_LLM_MODEL=gpt-5.2
export WEBARENA_REPLAY_N_JOBS=12
export WEBARENA_REPLAY_VIEWPORT_WIDTH=1280
export WEBARENA_REPLAY_VIEWPORT_HEIGHT=1000

./run_webarena_recorded_action_replay.sh localhost \
  ../../web/agentlab_results/<source-study-dir-name>
```

Monitor:

```bash
python /home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/watch_webarena_recorded_action_replay.py \
  --study-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/to_verify/<source-study-dir-name> \
  --interval-seconds 30
```

### 2. Normalize Names And Rewrite Reports

After the replay batch finishes:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd /home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab

python rename_replay_experiment_dirs_to_source_names.py \
  --study-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/to_verify/<source-study-dir-name>
```

This does two things:

- renames replay dirs to their exact source trajectory basenames
- rewrites `result_df.csv`, `summary_df.csv`, and `error_report.md`

### 3. Triage The Replay Results

Split the replayed trajectories into:

- `keep`
  - no obvious reward regression versus source
  - no hard runtime error
  - screenshots are at the intended size
  - visually plausible final state
- `manual_recover`
  - reward regressed versus source
  - obvious bid drift, wrong page, wrong selected record, or premature
    truncation
  - screenshot looks wrong even if the reward did not regress
- `archive_or_skip`
  - source trajectory itself is unusable or non-replayable
  - runtime crashes or environment corruption make the task not worth manual
    recovery

Recommended conservative keep rule:

- for source-success trajectories, keep only if the replay also ends with
  reward `1`
- for source-failure trajectories, keep only after manual inspection suggests
  the same failure semantics were preserved
- inspect all trajectories, not only source-success trajectories
- preserve semantics for both source-success and source-failure trajectories
- if unsure, send it to manual recovery instead of keeping it automatically

Practical source-failure acceptance guide:

- prefer replays that visit the same set of major pages as the source
- prefer replays that perform a similar sequence of interaction types
  - for example click, click, fill, submit
- prefer replays whose failure mode looks similar to the original
- a good default rule is:
  - same major page set
  - same final page family
  - same final failed action type
- do not accept a source-failure replay just because its reward drifted upward
  unless the interaction semantics still look faithful enough to the source

### 4. Build The Manual Recovery Queue

The manual recovery queue should be built from the `manual_recover` set inside
that same source study folder.

The accepted automated replay queue should be built from the `keep` set in the
same `to_verify/<source-study-dir-name>` folder so those trajectories can be
copied into `final/<source-study-dir-name>`.

For each queued trajectory, record:

- source trajectory basename
- task id
- source reward and replay reward
- source final URL and replay final URL
- first obvious drift symptom
  - wrong click target
  - element not visible
  - fill on non-input
  - wrong record selected
  - same page family but wrong final state

Dependency rule:

- if a target task depends on earlier tasks in the same folder, recover the
  dependency chain in order
- do not manually recover a dependent task against obviously broken predecessor
  state

### 5. Run Manual Recovery For The Queued Trajectories

Manual recovery should be done one task, or one dependency chain, at a time.
The current policy is one max-effort manual attempt per queued recovery.

Recommended operator setup:

- terminal 1: live interactive manual run
- terminal 2: source trajectory artifacts for reference
- terminal 3: replayed bad trajectory for comparison

The operator should inspect:

- source screenshots
- source action list from `step_*.pkl.gz`
- source final URL and reward
- bad replay screenshots and last action errors

Recommended manual launcher:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd /home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab

PYTHONPATH=src \
./run_webarena_manual.sh localhost \
  --source-exp-dir ../../web/agentlab_results/<source-study-dir-name>/<source-trajectory-dir-name> \
  --max-steps 100 \
  --headless \
  --dump-screenshot \
  --state-view both
```

The operator should then steer the live environment toward the intended source
state using the UI.

### 6. Manual Recovery Rules

Allowed:

- use the source trajectory as a guide
- use screenshots, AXTree, and pruned HTML from the live manual harness
- use normal BrowserGym UI actions such as `click`, `fill`, `select_option`,
  `press`, `scroll`, `go_back`, and benign top-level `goto`
- take a cleaner path than the original source run if the goal is clearly the
  same and the path remains UI-grounded

Disallowed:

- backend queries, database reads, or hidden admin APIs
- deep-link `goto(...)` with hidden ids or crafted params that bypass the UI
- using private implementation details that are not visible from the source
  trajectory or the live browser state
- editing output artifacts by hand to fake a successful run

The goal is a valid UI-driven replacement trajectory that preserves the source
interaction semantics as closely as practical. Literal action-by-action cloning
is not required, but preserving the original navigation and action pattern is
more important than opportunistically improving reward.

### 7. Accept Or Reject Each Manual Recovery

Accept a manual recovery when all of the following are true:

- the run finishes without a harness crash
- for source-success trajectories, the rerun ends with reward `1`
- for source-failure trajectories, the rerun preserves the original navigation
  and interaction pattern closely enough that the failure semantics still look
  faithful
- only after that interaction-fidelity check, a better reward than the source
  is acceptable
- the screenshots are visually sane and useful
- the path is not hacky

Reject a manual recovery if:

- it still drifts badly
- it relies on hidden shortcuts
- it ends in a worse state than the source

### 8. Install Accepted Trajectories Into The Final Folder

The `final/<source-study-dir-name>` folder is the canonical output. Populate it
incrementally from:

- accepted automated replay trajectories from `to_verify`
- accepted manual recovery trajectories from `web/agentlab_results_manual`

When an accepted trajectory is installed into the final folder:

1. archive the bad automated replay trajectory
2. copy the accepted trajectory into the final study folder
3. rename the copied manual trajectory directory to the exact source trajectory
   basename
4. confirm `exp_args.pkl` points at the new path if needed
5. rewrite aggregate reports

Recommended archive location:

- `web/agentlab_results_replay/failed_runs/<source-study-dir-name>/`

The `final/<source-study-dir-name>` folder should remain the single canonical
per-folder output location.

Recommended replacement helper:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd /home/diwu/ralm/LongMemEval-V2-Workspace

python enterprise/AgentLab/install_manual_recovery_into_replay_study.py \
  --study-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/final/<source-study-dir-name> \
  --source-exp-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results/<source-study-dir-name>/<source-trajectory-dir-name> \
  --accepted-exp-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_manual/<manual-study-dir>/<manual-trajectory-dir>
```

For an accepted automated replay from `to_verify`, pass that accepted replay
trajectory dir as `--accepted-exp-dir`.

### 9. Regenerate Aggregate Reports After Manual Replacement

After one or more manual replacements are copied into the replay folder, rewrite
the aggregate reports again:

```bash
source /local/diwu/miniconda3/etc/profile.d/conda.sh
conda activate workarena
cd /home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab

python rename_replay_experiment_dirs_to_source_names.py \
  --study-dir /home/diwu/ralm/LongMemEval-V2-Workspace/web/agentlab_results_replay/final/<source-study-dir-name>
```

This command is safe to rerun even when directory names already match the
source names; it can still be used as the report rewrite step.

### 10. Close The Folder Before Moving On

Do not start the next source study folder until the current folder has:

- finished automated replay
- finished triage
- completed the manual recovery decisions for that folder
- rewritten aggregate reports
- recorded a short folder-level note in the run log

## Practical Notes On What Usually Fails

The most common replay failures we have already seen are:

- same recorded bid now points at a different element
- same recorded bid is now hidden or not clickable
- same recorded fill now lands on a non-input or wrong field
- action lands on the correct page family but selects the wrong record
- source and replay reach similar URLs but different semantic state

Why manual recovery is promising:

- these failures are often easy for a human operator to recognize quickly
- the operator can usually recover by selecting the intended visible element
  directly
- the operator already has a source trajectory to aim for

## Current Recommendation

Use this workflow:

1. one source study folder at a time
2. one backend reset before the folder
3. full-folder recorded-action replay into `to_verify`
4. inspect every replayed trajectory in `to_verify`, regardless of whether the source reward
   was success or failure
5. copy accepted trajectories into `final`
6. targeted manual recovery for the remaining trajectories in that same folder
7. install accepted manual recoveries into `final`
8. move known-bad automated replays to backup storage and keep one canonical
   final replay folder per source study

This is more labor than replay-only, but it is much more realistic than trying
to force blind replay to work for every WebArena folder.

## Relationship To Existing Docs

Use this document as the primary operating procedure for hybrid recovery.

The replay-only run book remains useful as background and for known replay
pitfalls:

- [webarena_recorded_action_replay_plan.md](/home/diwu/ralm/LongMemEval-V2-Workspace/enterprise/AgentLab/docs/webarena_recorded_action_replay_plan.md)

## Resolved Design Decisions

- the final replay folder should contain only accepted converted trajectories
- initial replay batch runs should land under `web/agentlab_results_replay/to_verify`
- accepted trajectories should accumulate under
  `web/agentlab_results_replay/final`
- known-bad trajectories should be moved to backup storage
- inspect replay semantics for both source-success and source-failure
  trajectories
- manual recovery is allowed for both source-success and source-failure
  trajectories if replay did not preserve the original semantics
- a manual recovery may be accepted even if it achieves a better reward than the
  source trajectory, but only after interaction fidelity is judged acceptable
- use one backend reset per source study folder
- allow one max-effort manual recovery attempt per queued trajectory
- add a dedicated `main_webarena_manual.py` entrypoint for convenience
- accepted manual recoveries should replace the bad replay trajectories in the
  final replay folder
- there is only one shared WebArena backend, so the manual recovery phase runs
  in single-agent mode, batch by batch
- source-success trajectories require reward `1` on accepted reruns
- source-failure trajectories require a human check that the same failure
  semantics were preserved
- the default source-failure acceptance rule is:
  - same major page set
  - same final page family
  - same final failed action type

## Remaining Questions

- what exact rule defines `semantics preserved` for a replayed or manual
  trajectory
  - especially for source-failure trajectories, how similar must the visited
    pages and action sequence be before we consider the failure reason preserved
