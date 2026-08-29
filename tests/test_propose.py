from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.propose import Hypothesis, apply_hypothesis, topology_signature
from pipeline.graph import GraphValidationError, OperatorGraph
from pipeline.registry import default_registry


def seed_graph() -> OperatorGraph:
    return OperatorGraph.from_path("configs/pipeline_seed.json")


def test_hypothesis_schema_is_strict_and_patch_is_discriminated() -> None:
    hypothesis = Hypothesis.model_validate(
        {
            "id": "h_001",
            "target_node": "model",
            "rationale": "Match the within-user ranking metric.",
            "method_source": "listwise ranking, LambdaRank",
            "expected_delta": 0.012,
            "expected_cost_minutes": 8,
            "patch": {
                "op": "replace_params",
                "node": "model",
                "params": {"seed": 0, "iterations": 50},
            },
        }
    )
    assert hypothesis.patch.op == "replace_params"

    with pytest.raises(ValidationError):
        Hypothesis.model_validate(
            {
                **hypothesis.model_dump(),
                "unreviewed_field": True,
            }
        )
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(
            {
                **hypothesis.model_dump(),
                "patch": {"op": "execute_shell", "command": "anything"},
            }
        )


def test_param_patch_preserves_topology_and_emits_reviewable_diff() -> None:
    original = seed_graph()
    hypothesis = Hypothesis.model_validate(
        {
            "id": "h_002",
            "target_node": "model",
            "rationale": "Test a bounded parameter change.",
            "method_source": "controlled experiment",
            "expected_delta": 0.004,
            "expected_cost_minutes": 1,
            "patch": {
                "op": "replace_params",
                "node": "model",
                "params": {"seed": 7},
            },
        }
    )
    mutation = apply_hypothesis(original, hypothesis, default_registry())

    mutation.graph.validate(default_registry())
    assert topology_signature(mutation.graph) == topology_signature(original)
    assert not mutation.topology_changed
    assert '"seed": 7' in mutation.diff
    assert mutation.graph.meta["hypothesis_id"] == "h_002"


def test_registered_feature_cannot_read_same_row_target_outcome(tmp_path) -> None:
    hypothesis = Hypothesis.model_validate(
        {
            "id": "h_leak",
            "target_node": "raw",
            "rationale": "deliberately invalid leakage test",
            "method_source": "unit test",
            "expected_delta": 0.5,
            "expected_cost_minutes": 1,
            "patch": {
                "op": "register_feature",
                "node": {
                    "id": "leak",
                    "type": "features.generated",
                    "inputs": ["load"],
                },
                "code": (
                    "def build(train_df, target_df, ctx):\n"
                    "    return target_df[['row_id', 'long_view']].copy()\n"
                ),
                "sources": ["long_view"],
                "temporal_scope": "strictly_earlier",
            },
        }
    )
    with pytest.raises(GraphValidationError, match="forbidden same-row"):
        apply_hypothesis(
            seed_graph(),
            hypothesis,
            default_registry(),
            generated_feature_dir=tmp_path,
        )
