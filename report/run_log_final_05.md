# Run & iteration log — `final_05`

Generated from run artifacts by `report/make_deliverables.py`. Every metric here is the official evaluator's output on the **validation** split; the hidden test set is never read during development.

## Iteration 0 — seed (baseline reproduction)

Graph: `data.load`, `features.raw_categorical`, `model.fm_baseline`, `submit.rank`, `features.static_side`

| metric | seed | official baseline | delta |
|---|---:|---:|---:|
| primary | 0.602689 | 0.601469 | +0.001220 |


## Autonomous iterations

| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |
|---|---|---:|---:|---:|---:|---:|---|
| 001 | `replace_fm_with_lambdarank` | +0.0200 | 0.597717 | 0.661777 | 0.533657 | -0.003752 | no |
| 002 | `replace_fm_with_setwise_rank` | +0.0120 | 0.596662 | 0.661385 | 0.531939 | -0.004807 | no |
| 003 | `replace_fm_with_multitask_auxiliary_learning` | +0.0120 | 0.546201 | 0.587078 | 0.505325 | -0.055268 | no |
| 004 | `replace_fm_with_deepfm` | +0.0080 | 0.513335 | 0.542108 | 0.484561 | -0.088134 | no |

### Per-iteration detail

#### Iteration 001 — `replace_fm_with_lambdarank` (no)

Graph change:

```diff
-    },
-    {
-      "id": "model",
-      "inputs": [
-        "raw",
-        "static_side"
-      ],
-      "params": {
-        "seed": 0
-      },
-      "type": "model.fm_baseline"
+    },
```

Result: validation primary **0.597717** (-0.003752 vs official baseline).

#### Iteration 002 — `replace_fm_with_setwise_rank` (no)

Graph change:

```diff
-    },
-    {
-      "id": "model",
-      "inputs": [
-        "raw",
-        "static_side"
-      ],
-      "params": {
-        "seed": 0
-      },
-      "type": "model.fm_baseline"
+    },
```

Result: validation primary **0.596662** (-0.004807 vs official baseline).

#### Iteration 003 — `replace_fm_with_multitask_auxiliary_learning` (no)

Graph change:

```diff
-    },
-    {
-      "id": "model",
-      "inputs": [
-        "raw",
-        "static_side"
-      ],
-      "params": {
-        "seed": 0
-      },
-      "type": "model.fm_baseline"
+    },
```

Result: validation primary **0.546201** (-0.055268 vs official baseline).

#### Iteration 004 — `replace_fm_with_deepfm` (no)

Graph change:

```diff
-    },
-    {
-      "id": "model",
-      "inputs": [
-        "raw",
-        "static_side"
-      ],
-      "params": {
-        "seed": 0
-      },
-      "type": "model.fm_baseline"
+    },
```

Result: validation primary **0.513335** (-0.088134 vs official baseline).


## Manual interventions

**Count: 0**

No manual interventions occurred during this run.


## Resource usage

| quantity | value |
|---|---:|
| LLM tokens in | 16,123 |
| LLM tokens out | 1,944 |
| LLM tokens total | 18,067 |
| Agent wall-clock | 0.82 h (2,966 s) |
| Iterations used | 4 of 50 |
| Rejected proposals | 0 |
| Stop reason | `converged` |
| Manual interventions | 0 |


## Converged result (validation)

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.668809 | 0.6674 | +0.001409 |
| nDCG@5 | 0.536568 | 0.5357 | +0.000868 |
| primary | 0.602689 | 0.6015 | +0.001220 |

