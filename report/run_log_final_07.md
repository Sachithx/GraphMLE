# Run & iteration log — `final_07`

Generated from run artifacts by `report/make_deliverables.py`. Every metric here is the official evaluator's output on the **validation** split; the hidden test set is never read during development.

## Iteration 0 — seed (baseline reproduction)

Graph: `data.load`, `features.raw_categorical`, `model.fm_baseline`, `submit.rank`, `features.static_side`, `features.user_attribute_affinity`

| metric | seed | official baseline | delta |
|---|---:|---:|---:|
| primary | 0.603138 | 0.601469 | +0.001669 |


## Autonomous iterations

| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |
|---|---|---:|---:|---:|---:|---:|---|
| 001 | `seed_bag_incumbent_fm` | +0.0006 | 0.605150 | 0.672167 | 0.538134 | +0.003682 | yes |
| 002 | `repair_timeout_reduce_ranker_estimators_v2` | +0.0001 | 0.604906 | 0.671733 | 0.538080 | +0.003438 | no |
| 003 | `add_pairwise_fm_to_seed_bag` | +0.0003 | 0.604888 | 0.671539 | 0.538238 | +0.003420 | no |
| 004 | `add_decayed_author_affinity_to_fm_bag` | +0.0003 | 0.604161 | 0.670577 | 0.537746 | +0.002692 | no |

### Per-iteration detail

#### Iteration 001 — `seed_bag_incumbent_fm` (yes)

Graph change:

```diff
-        "model"
+        "model_bag"
+    },
+    {
+      "id": "model_s1",
+      "inputs": [
+        "raw",
+        "static_side",
+        "author_affinity"
+      ],
+      "params": {
+        "seed": 1
```

Result: validation primary **0.605150** (+0.003682 vs official baseline).

#### Iteration 002 — `repair_timeout_reduce_ranker_estimators_v2` (no)

Graph change:

```diff
-        "model_s4"
+        "model_s4",
+        "ranker_raw_complement"
+    },
+    {
+      "id": "ranker_raw_complement",
+      "inputs": [
+        "raw",
+        "static_side",
+        "author_affinity"
+      ],
+      "params": {
```

Result: validation primary **0.604906** (+0.003438 vs official baseline).

#### Iteration 003 — `add_pairwise_fm_to_seed_bag` (no)

Graph change:

```diff
-        "model_s4"
+        "model_s4",
+        "model_pairwise"
+    },
+    {
+      "id": "model_pairwise",
+      "inputs": [
+        "raw",
+        "static_side",
+        "author_affinity"
+      ],
+      "params": {
```

Result: validation primary **0.604888** (+0.003420 vs official baseline).

#### Iteration 004 — `add_decayed_author_affinity_to_fm_bag` (no)

Graph change:

```diff
-        "author_affinity"
+        "author_affinity",
+        "decayed_author_affinity"
-        "author_affinity"
+        "author_affinity",
+        "decayed_author_affinity"
-        "author_affinity"
+        "author_affinity",
+        "decayed_author_affinity"
-        "author_affinity"
+        "author_affinity",
+        "decayed_author_affinity"
```

Result: validation primary **0.604161** (+0.002692 vs official baseline).


## Manual interventions

**Count: 0**

No manual interventions occurred during this run.


## Resource usage

| quantity | value |
|---|---:|
| LLM tokens in | 29,878 |
| LLM tokens out | 2,951 |
| LLM tokens total | 32,829 |
| Agent wall-clock | 6.14 h (22,099 s) |
| Iterations used | 4 of 50 |
| Rejected proposals | 0 |
| Stop reason | `wall_clock_cap` |
| Manual interventions | 0 |


## Converged result (validation)

| metric | agent | official baseline | absolute delta |
|---|---:|---:|---:|
| GAUC | 0.672167 | 0.6674 | +0.004767 |
| nDCG@5 | 0.538134 | 0.5357 | +0.002434 |
| primary | 0.605150 | 0.6015 | +0.003682 |

