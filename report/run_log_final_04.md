# Run & iteration log — `final_04`

Generated from run artifacts by `report/make_deliverables.py`. Every metric here is the official evaluator's output on the **validation** split; the hidden test set is never read during development.

## Iteration 0 — seed (baseline reproduction)

Graph: `data.load`, `features.raw_categorical`, `model.fm_baseline`, `submit.rank`

| metric | seed | official baseline | delta |
|---|---:|---:|---:|
| primary | 0.601469 | 0.601469 | +0.000000 |


## Autonomous iterations

| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |
|---|---|---:|---:|---:|---:|---:|---|
| 001 | `add_strict_history_features` | +0.0250 | 0.598027 | 0.662456 | 0.533599 | -0.003441 | no |
| 002 | `add_item_popularity_to_fm` | +0.0060 | 0.601401 | 0.666727 | 0.536075 | -0.000068 | no |
| 003 | `add_static_side_features_to_fm` | +0.0060 | 0.602689 | 0.668809 | 0.536568 | +0.001220 | yes |

### Per-iteration detail

#### Iteration 001 — `add_strict_history_features` (no)

Graph change:

```diff
-        "raw"
+        "raw",
+        "history"
+    },
+    {
+      "id": "history",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "windows": [
+          1,
```

Result: validation primary **0.598027** (-0.003441 vs official baseline).

#### Iteration 002 — `add_item_popularity_to_fm` (no)

Graph change:

```diff
-        "raw"
+        "raw",
+        "item_popularity"
+    },
+    {
+      "id": "item_popularity",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "smoothing": 20
+      },
```

Result: validation primary **0.601401** (-0.000068 vs official baseline).

#### Iteration 003 — `add_static_side_features_to_fm` (yes)

Graph change:

```diff
-        "raw"
+        "raw",
+        "static_side"
+    },
+    {
+      "id": "static_side",
+      "inputs": [
+        "load"
+      ],
+      "params": {},
+      "type": "features.static_side"
```

Result: validation primary **0.602689** (+0.001220 vs official baseline).


## Manual interventions

**Count: 0**

No manual interventions occurred during this run.


## Resource usage

| quantity | value |
|---|---:|
| LLM tokens in | 11,722 |
| LLM tokens out | 900 |
| LLM tokens total | 12,622 |
| Agent wall-clock | 0.45 h (1,604 s) |
| Iterations used | 3 of 50 |
| Rejected proposals | 0 |
| Stop reason | `converged` |
| Manual interventions | 0 |


## Converged result (validation)

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.668809 | 0.6674 | +0.001409 |
| nDCG@5 | 0.536568 | 0.5357 | +0.000868 |
| primary | 0.602689 | 0.6015 | +0.001220 |

