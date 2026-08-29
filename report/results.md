# Scored-run results and challenges

All Phase 5 model selection below uses the validation split only. Hidden-test labels
are excluded from the scored-run metric path.

## `final_02`

`final_02` completed three executed iterations with zero validation failures and no
human interventions. The incumbent remained the seed FM at validation primary
`0.601468756` (GAUC `0.667132632`, nDCG@5 `0.535804881`). Registered LambdaRank
scored `0.597615969`; the three-epoch hard-sharing multitask model scored
`0.548627572`; and an item-popularity proposal returned an exact zero delta.

The zero delta exposed an execution-contract bug rather than a failed feature:
`model.fm_baseline` accepted feature-bundle edges but reconstructed its matrix only
from the starter-kit split tuples. Consequently, every engineered feature was
discarded. The fix makes the FM encode every explicit input bundle, preserves exact
starter-kit encoding for the raw-only graph, and quantile-tokenizes continuous fields.
Graph validation now also rejects any node whose output cannot reach the single
terminal node. This turns silent discarded computation into a pre-compute rejection.

The run is retained as guard and post-mortem evidence under `runs/final_02/`.

## `final_03`

The seed is a feature-aware FM and LambdaRank pair over raw categorical and strictly
historical item-popularity features, followed by a within-user rank average. A
controlled validation sweep measured:

| Candidate | Validation primary |
|---|---:|
| Feature-aware FM | 0.601400821 |
| LambdaRank | 0.600602263 |
| 50/50 rank blend | 0.602022839 |
| 65/35 FM/LambdaRank rank blend | **0.602609425** |

The component prediction correlation was `0.797082`, providing a concrete reason for
the ensemble gain. The `final_03` scored service was launched only after the 54-test
suite and production preflight passed. Its autonomous output is written under
`runs/final_03/`; the table above is seed-selection evidence, not a completed-run
claim.
