# Research-backed improvement plan

## Finding

The main ceiling is information usage and refinement depth, not a lack of registered
model names. KuaiRand-Pure provides 30 user fields and 11 basic video fields with
complete coverage of this training set. The current pipeline uses none of the user
profile file and only a small subset of basic video metadata. The monthly video
statistics file is deliberately excluded because its outcome aggregates span the
evaluation period and create a temporal-leakage risk.

The existing multitask result is also not a fair negative result. It uses three fixed
epochs, no validation checkpointing, equal loss treatment, and a single hard-shared
representation. In the training data, `is_click` has correlation `0.7605` with
`long_view`, while follow/forward/hate labels each occur around 0.1% or less. Treating
all of these tasks equally invites negative transfer.

## Highest-value next model

Build a side-information interaction branch and retain the proven diverse ensemble:

1. Add leakage-safe `features.user_profile` and `features.video_metadata` nodes.
2. Feed raw IDs, those static fields, duration, and strictly historical aggregates to
   a properly trained DeepFM or low-rank DCN-V2 branch with validation early stopping.
3. For multitask supervision, begin with `long_view + is_click`, use a task-specific
   gate (MMoE/PLE) or at minimum explicit primary/auxiliary loss weights, and add rarer
   tasks only after ablation evidence.
4. Rank-average this nonlinear branch with the sparse FM and LambdaRank branches.
5. Select blend weights and model variants in an inner loop, then use only the best
   inner result for each outer-loop convergence observation.

This ordering first exploits 100%-covered information already on disk, then adds
higher-order interactions, then controls task conflict. It is lower risk than jumping
directly to a large sequential or counterfactual model before the static feature
baseline is exhausted.

## Primary research anchors

- MLE-STAR, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/a9619dd0f0d54a5cf7734add1dc38cd1-Abstract-Conference.html
- KuaiRand paper and official data documentation: https://arxiv.org/abs/2208.08696 and https://github.com/chongminggao/KuaiRand
- DeepFM: https://www.ijcai.org/Proceedings/2017/239
- DCN-V2: https://arxiv.org/abs/2008.13535
- FinalMLP: https://arxiv.org/abs/2304.00902
- MMoE: https://doi.org/10.1145/3219819.3220007
- PLE: https://doi.org/10.1145/3383313.3412258
- Counterfactual Watch Time, KDD 2024: https://arxiv.org/abs/2406.07932
