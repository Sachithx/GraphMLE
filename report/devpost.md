# Devpost submission — Track 2: Autonomous ML Research Agent

## Inspiration

The task asks for an agent that runs the full ML research loop by itself. The
obvious way to build one is to let a language model rewrite a training script.
That approach fails in a specific, boring way: most edits an LLM makes to a
monolithic script are syntactically valid and semantically broken, and nothing
in the loop can tell the difference until a run has burned its budget and
returned a plausible number.

So the design question was not "how do I get a model to write ML code" but "what
representation makes a bad mutation impossible to express, and a good one cheap
to verify". Everything else followed from that.

## What it does

The agent improves a recommender pipeline for KuaiRand-Pure, ranking each user's
logged impressions by whether they will be long-viewed. It reproduces the
official Factorisation Machine baseline exactly, then iterates on it without
supervision: it ablates its own pipeline to find the load-bearing component,
proposes a schema-constrained change targeting it, validates and executes that
change in a sandbox, tests the result for significance, and either keeps or
reverts it — logging a full record of each iteration.

**Result on the required benchmark (KuaiRand-Pure), validation split:**

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.669639 | 0.6674 | **+0.002239** |
| nDCG@5 | 0.536636 | 0.5357 | **+0.000936** |
| primary | 0.603138 | 0.6016 | **+0.001538** |

Scored quantity per the judging formula, `mean over m of delta(m)`: **+0.001588**.

The run converged under the organisers' own rule (ε = 0.002, N = 3) in **four
iterations**, using **0.96 h of the 6 h ceiling**, **19,211 LLM tokens**, **no
GPU**, and — the number I care about most — **zero manual interventions**.

## How I built it

**A typed operator graph instead of a script.** A pipeline is a JSON DAG whose
nodes declare input and output types in a registry. `graph.validate()` rejects
unknown node types, arity errors, type mismatches, cycles, and any node whose
output cannot reach the terminal submission — all before a single row is loaded.
An invalid proposal costs zero compute and is logged as a cheap failure. The
agent's action space is topology and parameter edits over 21 registered
operators; free-form code generation is confined to one surface, a feature
builder with a fixed `build(train_df, target_df, ctx)` signature.

That last check earned its place. An earlier run proposed a feature that returned
a delta of exactly 0.000000 — bit-identical to the incumbent. A real feature that
adds nothing returns a *slightly* different number; an identical one means the
feature was computed and silently discarded. The model operator was accepting
feature edges and then rebuilding its matrix from the raw splits, ignoring them.
The fix made the operator consume every declared input, and graph validation now
rejects unreachable outputs outright, so that class of silent no-op is
unrepresentable rather than merely unlikely.

**Leakage guards, in two layers.** KuaiRand ships twelve feedback signals.
`is_click`, `is_like`, `is_follow` and the rest are outcomes of the *same*
impression as `long_view`; using them as same-row features produces a beautiful
validation score and a worthless model. A static checker tracks per-column source
and temporal lineage and refuses any feature deriving from a forbidden column at
the same row. An empirical tripwire flags any validation primary above 0.80,
since perfect ranking on this task reaches only 0.8645. Prior-date aggregates of
those same signals remain legal, and multi-task models reach them through a
separately checked auxiliary-label path. The randomised-exposure log overlaps the
evaluation window, so anything derived from it must prove its source rows end on
or before the training cutoff.

**Significance testing, because 50 iterations against a 124,909-row validation
split will manufacture improvements.** The baseline's five-seed standard
deviation is 0.0008. Acceptance requires beating the incumbent by more than 2σ
(0.0016), or surviving a three-seed re-run. This costs a little on the
leaderboard and protects the hidden-test score, which is what actually ranks.

**Ablation as the targeting mechanism.** Before each proposal round the harness
neutralises one node at a time and re-scores. On the converged graph, removing
the model costs 0.1190 and removing the raw categorical bundle costs 0.0726,
while the side-feature bundle contributes 0.0011. That table goes to the proposer
as evidence, which is what turns "the agent tried things" into "the agent chose a
target and can say why".

## Challenges

**The convergence rule binds before the compute budget does.** Every run stopped
after three to five iterations, having used under a sixth of the wall-clock
ceiling. ε = 0.002 is 2.5σ of the baseline's seed noise, but the genuine gains
available here are about 0.001 — the same order. An agent making real progress
trips the convergence test before those gains compound. Recognising that this is
a property of the task rather than a bug in the search changed how I read every
subsequent run.

**A proposer that only knew how to replace things.** In one run all four
proposals swapped the incumbent model wholesale — LambdaRank, a setwise
transformer, multi-task, DeepFM — and all four lost, the worst by 0.089. The
per-iteration logs made the pattern unmissable. The proposer's own guidance was
the cause: it told the agent to prioritise large architectural moves, and
mentioned ensembling only in a buried clause. Rewriting that guidance between
runs, using the agent's own measured history as the evidence, made every
subsequent proposal additive and produced the scored result. The before-and-after
is Figure 1 in the report.

**Assuming a negative result generalises.** I tested the side-feature bundle
against LightGBM, measured +0.000084, and concluded the features were dead. The
agent later paired the same bundle with the Factorisation Machine and gained
+0.001220 — its single best accepted move, and fifteen times what I had measured.
The lesson is now encoded in the proposer's guidance: a feature set that does not
help one model family can still help another.

## What I learned

The structural insight that most shaped the work: both scored metrics rank
*within* a user, over roughly five impressions. Any feature that is constant
across a user's rows cannot change their ordering, no matter how predictive it
looks in aggregate. Measuring this directly, every user-profile column varies
within a user for 0.0% of users, while video duration varies for 81.3%. That
single fact explains why a large block of side features contributes almost
nothing as main effects, and why the feature that did work — a user-by-author
historical affinity — was one that varies within a user by construction.

The second lesson is that variance reduction outperformed novelty. Across
extensive offline experiments the most reliable gain available was not a new
architecture but averaging several seeds of the incumbent, worth about +0.0012 —
more than any single architectural change I tested.

## Built with

- **Languages / runtime:** Python 3.13
- **Libraries:** NumPy, pandas, LightGBM, PyTorch, Pydantic, PyYAML, pytest, matplotlib
- **APIs:** OpenAI Responses API with structured outputs — `gpt-5.6-terra` for
  hypothesis proposal, `gpt-5.6-luna` for error repair
- **Dataset:** KuaiRand-Pure (27K users × 7.6K items, 1.4M interactions), with the
  organisers' fixed date-based splits — train 1,141,112 / validation 124,909 /
  test 170,588 rows
- **Assets:** the official KuaiRand starter kit. `evaluate.py`, `data.py`,
  `baseline.py` and `submit.py` are imported unmodified, so scoring and
  submission validation run through the organisers' own code rather than a
  reimplementation
- **Development tools:** VS Code, tmux, git; runs executed unattended on a Linux
  host with 2× RTX 6000 Ada (the converged pipeline needs no GPU)

## Try it

```bash
.venv-phase5/bin/python -m pytest -q                          # 60 tests
bash scripts/run_scored.sh final_06 configs/run_final06.yaml 1
```

Full reproduction steps, the per-iteration logs for all four scored runs, and an
honest account of the limitations are in the repository README.
