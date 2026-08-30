# Run & iteration log — `final_03`

Generated from run artifacts by `report/make_deliverables.py`. Every metric here is the official evaluator's output on the **validation** split; the hidden test set is never read during development.

## Iteration 0 — seed (baseline reproduction)

Graph: `data.load`, `features.raw_categorical`, `features.item_popularity`, `model.fm_baseline`, `model.lightgbm_rank`, `ensemble.rank_average`, `submit.rank`

| metric | seed | official baseline | delta |
|---|---:|---:|---:|
| primary | 0.602609 | 0.601469 | +0.001141 |


## Autonomous iterations

| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |
|---|---|---:|---:|---:|---:|---:|---|
| 001 | `add_strict_user_history_features` | +0.0060 | 0.599641 | 0.664425 | 0.534858 | -0.001828 | no |
| 002 | `repair_multitask_auxiliary_targets` | +0.0000 | 0.541595 | 0.581828 | 0.501362 | -0.059874 | no |
| 003 | `add_user_category_affinity_feature` | +0.0030 | 0.600986 | 0.666923 | 0.535050 | -0.000483 | no |
| 004 | `add_video_duration_feature_to_blended_rankers` | +0.0025 | 0.602443 | 0.668322 | 0.536564 | +0.000974 | no |
| 005 | `tune_popularity_smoothing_for_sparse_item_signal` | +0.0005 | 0.601959 | 0.667647 | 0.536270 | +0.000490 | no |

### Per-iteration detail

#### Iteration 001 — `add_strict_user_history_features` (no)

Graph change:

```diff
-        "pop"
+        "pop",
+        "user_history"
-        "pop"
+        "pop",
+        "user_history"
+    },
+    {
+      "id": "user_history",
+      "inputs": [
+        "load"
+      ],
```

Result: validation primary **0.599641** (-0.001828 vs official baseline).

#### Iteration 002 — `repair_multitask_auxiliary_targets` (no)

Graph change:

```diff
-    },
-    {
-      "id": "fm",
-      "inputs": [
-        "raw",
-        "pop"
-      ],
-      "params": {
-        "numeric_bins": 32,
-        "seed": 0
-      },
-      "type": "model.fm_baseline"
```

Result: validation primary **0.541595** (-0.059874 vs official baseline).

Error / recovery events:

- `leakage_rejection` on `auxiliary_targets`: unsupported auxiliary targets: ['comment', 'follow', 'forward', 'hate', 'like']

#### Iteration 003 — `add_user_category_affinity_feature` (no)

Graph change:

```diff
-        "pop"
+        "pop",
+        "affinity"
-        "pop"
+        "pop",
+        "affinity"
+    },
+    {
+      "id": "affinity",
+      "inputs": [
+        "load"
+      ],
```

Result: validation primary **0.600986** (-0.000483 vs official baseline).

#### Iteration 004 — `add_video_duration_feature_to_blended_rankers` (no)

Graph change:

```diff
-        "pop"
+        "pop",
+        "duration"
-        "pop"
+        "pop",
+        "duration"
+    },
+    {
+      "id": "duration",
+      "inputs": [
+        "load"
+      ],
```

Result: validation primary **0.602443** (+0.000974 vs official baseline).

#### Iteration 005 — `tune_popularity_smoothing_for_sparse_item_signal` (no)

Graph change:

```diff
-        "smoothing": 20
+        "smoothing": 10
```

Result: validation primary **0.601959** (+0.000490 vs official baseline).


## Manual interventions

**Count: 1**

- `2026-08-29T08:38:30Z` — post-launch development for a future run

  Implemented and tested the nested inner-refinement scheduler and wrote research/results documentation after final_03 launched. The active agent had already loaded agent/run.py and agent/propose.py; configs/run_final03.yaml and all pipeline model/feature execution code used by final_03 were not changed after launch. Read-only status monitoring is excluded.


## Resource usage

| quantity | value |
|---|---:|
| LLM tokens in | 23,121 |
| LLM tokens out | 2,339 |
| LLM tokens total | 25,460 |
| Agent wall-clock | 1.51 h (5,442 s) |
| Iterations used | 5 of 50 |
| Rejected proposals | 0 |
| Stop reason | `converged` |
| Manual interventions | 1 |


## Converged result (validation)

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.668465 | 0.6674 | +0.001065 |
| nDCG@5 | 0.536754 | 0.5357 | +0.001054 |
| primary | 0.602609 | 0.6015 | +0.001141 |

