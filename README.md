# TechJam 2026 — KuaiRand-Pure ML Research Agent

This repository is being built in acceptance-gated phases. Phase 1 establishes a
reproducible official baseline and submission-validation floor before any agent
or operator-graph code is added.

## Authoritative task definition

The problem statement contains conflicting metric descriptions. The starter
kit's unmodified `evaluate.py` is authoritative: the positive label is
`long_view`; the metrics are within-user GAUC and nDCG@5; and primary is their
mean. Recall@50 and `click` are not used for scoring by the supplied evaluator.

`eval/official.py` dynamically imports that file and only normalizes result key
names. It does not reimplement the metrics.

## Phase 1 setup

Use Python 3.13.7, then create the environment and install the pinned direct
dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Apple Silicon, LightGBM also needs the OpenMP runtime:

```bash
brew install libomp
```

Place the downloaded assets at:

```text
data/starter-kit/evaluate.py
data/starter-kit/baseline.py
data/starter-kit/submit.py
data/starter-kit/data.py
data/kuairand-pure/data/log_standard_4_08_to_4_21_pure.csv
data/kuairand-pure/data/log_standard_4_22_to_5_08_pure.csv
data/kuairand-pure/data/log_random_4_22_to_5_08_pure.csv
data/kuairand-pure/data/user_features_pure.csv
data/kuairand-pure/data/video_features_basic_pure.csv
data/kuairand-pure/data/video_features_statistic_pure.csv
```

Run the Phase 1 acceptance gate with one command:

```bash
make phase1-gate
```

It uses the official loader to verify train/validation/test row counts, runs
`baseline.py --model fm` for seeds 0–4, requires the published five-seed mean at
four decimal places, creates a test submission, and calls the kit's
`submit.py --check` on it. The passing artifact is written to
`runs/phase1/submission.csv`.

The checker rejects a wrong header, a row-count mismatch, non-contiguous
`row_id`, user/video misalignment, non-numeric scores, and NaN/Inf. Later
pipeline execution must delegate every candidate submission to this checker.

The starter kit's published validation baseline (GAUC 0.6674, nDCG@5 0.5357,
primary 0.6016) is the five-seed mean. On the M4 Pro/NumPy 2.5.2 environment,
the default seed-0 command is deterministic at 0.6671 / 0.5358 / 0.6015; seeds
0–4 average to the published values exactly at the kit's four-decimal reporting
precision. The gate preserves this distinction instead of treating seed noise as
a setup failure.

Run the fast contract tests with:

```bash
make test
```

Archive and starter-kit checksums recorded during the passing run are in
`data/README.md`.

## Phase 2 operator graph

Pipelines are JSON DAGs whose nodes and edges are validated before any model
compute. `pipeline/registry.py` declares the input signature, output type, and
callable for every supported node. Validation rejects duplicate IDs, unknown
types, missing inputs, cycles, arity errors, and edge-type mismatches.

The implemented node surface is:

- `data.load`
- `features.raw_categorical`, `features.user_history`,
  `features.item_popularity`, `features.user_category_affinity`,
  `features.video_duration`, `features.temporal`
- `model.fm_baseline`, `model.lightgbm_binary`, `model.lightgbm_rank`,
  `model.torch_deepfm`, `model.torch_multitask`
- `ensemble.rank_average`, `ensemble.seed_bag`
- `submit.rank`

Historical feature builders follow the fixed
`build(train_df, target_df, ctx)` contract and use only training rows with dates
strictly earlier than each target date. `submit.rank` writes the candidate and
delegates validation to the starter kit's unmodified `submit.py --check`.

Run the Phase 2 acceptance gate with:

```bash
make phase2-gate
```

It executes the three hand-written `configs/phase2_*.json` graphs and requires
three distinct official validation scores plus three checker-valid submissions.
The FM-equivalent starting point for later agent phases is
`configs/pipeline_seed.json`. Results are written to
`runs/phase2/results.json`.

## Phase 3 guards

Every feature column now carries explicit source and temporal lineage. Before a
model node can run, `guards/leakage.py` rejects:

- same-row use of `is_click`, `play_time_ms`, `is_like`, `is_follow`,
  `is_comment`, `is_forward`, `long_view`, and the other impression outcomes;
- generated feature columns with missing lineage;
- randomized-exposure features that cannot prove their source rows end on or
  before the training cutoff (`20220421`).

Prior-date aggregates of those outcomes remain valid, and the multi-task model
accesses them through the explicitly checked auxiliary-label path rather than as
features. Validation primary above 0.80 triggers a separate empirical leakage
rejection. Every rejection is appended to the run JSONL before raising.

`guards/sandbox.py` captures combined stdout/stderr, enforces a wall-clock
timeout (900 seconds by default), and caps memory with `RLIMIT_AS` on Linux or
supervised-process RSS monitoring on macOS. A structured result is written beside the
stdout log for both success and failure.

Candidate acceptance uses a strict 2σ threshold of 0.0016. A smaller raw delta
can pass only after exactly three seed runs whose mean improves on the incumbent
and where at least two seeds improve.

Run the Phase 3 gate with:

```bash
make phase3-gate
```

It proves that a deliberately leaky graph is rejected before its model callback,
and that a deliberately broken graph is contained and logged by the sandbox.

## Phase 4 autonomous loop

The harness and the scored run are separate. The production entry point is:

```bash
python -m agent.run --config configs/run.yaml --run-id final_01
```

It initializes the seed graph, caches node-level ablations by topology, and uses
the highest-impact ablated component as an outer-loop target. The default live
config explores four schema-constrained variants of that component before
moving to another target; convergence advances only once per completed outer
group. Each candidate is validated and executed in the Phase 3 sandbox, receives
at most three bounded repair attempts, and is accepted or reverted under the
2σ/three-seed significance rule. The proposer receives only the graph, ablation
evidence, ten recent outcomes, and rejected hypothesis IDs—never raw competition
data.

Graph validation also requires every node output to reach the single terminal
submission, preventing successful-looking runs with discarded feature or model
branches.

Live proposals use OpenAI Responses structured outputs with the strict Pydantic
`Hypothesis` schema. `configs/run.yaml` uses `gpt-5.6-terra` at low reasoning for
proposals and the cheaper `gpt-5.6-luna` for repairs. The API key is read only
from `OPENAI_API_KEY`; it is never stored in configuration. Exact input/output
token usage is copied from each API response into the iteration and cumulative
logs.

Free-form generation is limited to `register_feature`. Its source must define
exactly `build(train_df, target_df, ctx)`, while its declared source columns and
temporal scope become mandatory Phase 3 lineage. All other proposals are typed
parameter or topology mutations: `replace_params`, `add_node`, `remove_node`,
and `rewire`.

Use the deterministic acceptance gate to exercise the orchestration without an
API call or model training:

```bash
make phase4-gate
```

It runs five consecutive unattended iterations with canned hypotheses ordered
by expected effect size. One hypothesis deliberately fails and is recovered on
the first repair attempt; the final two small deltas are rejected after their
three-seed checks. The gate requires zero interventions, zero tokens, a complete
JSONL record for every iteration, and the four per-iteration artifacts. Its
result is written under `runs/phase4_gate/` (gitignored).

`--dry-run` swaps live proposal and repair calls for canned hypotheses while
retaining the evaluator selected by the config. Use `configs/phase4_gate.yaml`
when both proposal and evaluation must be synthetic. Synthetic metrics are
marked and exist only to verify loop control; they are never competition results.

During a scored run, append every manual action to the run's
`interventions.jsonl` with a timestamp, action, and reason. The runner creates
the file but never silently claims that later human actions did not happen.
