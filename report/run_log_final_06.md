# Run & iteration log — `final_06`

Generated from run artifacts by `report/make_deliverables.py`. Every metric here is the official evaluator's output on the **validation** split; the hidden test set is never read during development.

## Iteration 0 — seed (baseline reproduction)

Graph: `data.load`, `features.raw_categorical`, `model.fm_baseline`, `submit.rank`, `features.static_side`

| metric | seed | official baseline | delta |
|---|---:|---:|---:|
| primary | 0.602689 | 0.601469 | +0.001220 |


## Autonomous iterations

| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |
|---|---|---:|---:|---:|---:|---:|---|
| 001 | `add_strict_user_history_to_fm` | +0.0030 | 0.601706 | 0.666787 | 0.536626 | +0.000237 | no |
| 002 | `add_item_popularity_to_fm` | +0.0015 | 0.602293 | 0.668421 | 0.536165 | +0.000824 | no |
| 003 | `add_user_category_affinity_to_fm` | +0.0015 | 0.601913 | 0.667876 | 0.535950 | +0.000444 | no |
| 004 | `add_author_affinity_to_fm` | +0.0006 | 0.603138 | 0.669639 | 0.536636 | +0.001669 | yes |

### Per-iteration detail

#### Iteration 001 — `add_strict_user_history_to_fm` (no)

Graph change:

```diff
-        "static_side"
+        "static_side",
+        "user_history"
+    },
+    {
+      "id": "user_history",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "windows": [
+          1,
```

Result: validation primary **0.601706** (+0.000237 vs official baseline).

#### Iteration 002 — `add_item_popularity_to_fm` (no)

Graph change:

```diff
-        "static_side"
+        "static_side",
+        "item_popularity"
+    },
+    {
+      "id": "item_popularity",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "smoothing": 20.0
+      },
```

Result: validation primary **0.602293** (+0.000824 vs official baseline).

#### Iteration 003 — `add_user_category_affinity_to_fm` (no)

Graph change:

```diff
-        "static_side"
+        "static_side",
+        "user_category_affinity"
+    },
+    {
+      "id": "user_category_affinity",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "smoothing": 20
+      },
```

Result: validation primary **0.601913** (+0.000444 vs official baseline).

#### Iteration 004 — `add_author_affinity_to_fm` (yes)

Graph change:

```diff
-        "static_side"
+        "static_side",
+        "author_affinity"
+    },
+    {
+      "id": "author_affinity",
+      "inputs": [
+        "load"
+      ],
+      "params": {
+        "attribute": "author_id",
+        "smoothing": 20
```

Result: validation primary **0.603138** (+0.001669 vs official baseline).


## Manual interventions

**Count: 0**

No manual interventions occurred during this run.


## Resource usage

| quantity | value |
|---|---:|
| LLM tokens in | 17,063 |
| LLM tokens out | 2,148 |
| LLM tokens total | 19,211 |
| Agent wall-clock | 0.96 h (3,448 s) |
| Iterations used | 4 of 50 |
| Rejected proposals | 0 |
| Stop reason | `converged` |
| Manual interventions | 0 |


## Converged result (validation)

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.669639 | 0.6674 | +0.002239 |
| nDCG@5 | 0.536636 | 0.5357 | +0.000936 |
| primary | 0.603138 | 0.6015 | +0.001669 |

