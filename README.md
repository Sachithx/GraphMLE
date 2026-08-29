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
