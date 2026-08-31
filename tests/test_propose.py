from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent.llm import StructuredResult, TokenUsage
from agent.propose import (
    Hypothesis,
    LiveHypothesisProposer,
    apply_hypothesis,
    topology_signature,
)
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


def test_add_node_can_atomically_replace_a_model_without_changing_its_id() -> None:
    hypothesis = Hypothesis.model_validate(
        {
            "id": "h_rank",
            "target_node": "model",
            "rationale": "Switch objective in one valid candidate.",
            "method_source": "LambdaRank",
            "expected_delta": 0.01,
            "expected_cost_minutes": 8,
            "patch": {
                "op": "add_node",
                "replace_node": "model",
                "node": {
                        "id": "model",
                        "type": "model.lightgbm_rank",
                        "params": {"objective": "lambdarank"},
                        "inputs": ["raw", "duration", "temporal"],
                    },
                },
        }
    )
    mutation = apply_hypothesis(seed_graph(), hypothesis, default_registry())
    mutation.graph.validate(default_registry())
    model = next(node for node in mutation.graph.nodes if node.id == "model")
    assert model.type == "model.lightgbm_rank"
    assert mutation.topology_changed


def test_live_proposer_injects_registry_contracts_and_two_valid_graphs() -> None:
    class CapturingClient:
        instructions = ""
        input_text = ""

        def parse(self, schema, *, instructions, input_text):
            self.instructions = instructions
            self.input_text = input_text
            return StructuredResult(
                schema.model_validate(
                    {
                        "id": "h_catalog",
                        "target_node": "model",
                        "rationale": "exercise the catalog",
                        "method_source": "unit test",
                        "expected_delta": 0.001,
                        "expected_cost_minutes": 1,
                        "patch": {
                            "op": "replace_params",
                            "node": "model",
                            "params": [{"name": "seed", "value": 7}],
                        },
                    }
                ),
                TokenUsage(),
            )

    client = CapturingClient()
    registry = default_registry()
    proposer = LiveHypothesisProposer(client, registry)
    proposer.propose(
        graph=seed_graph(),
        ablation_table={},
        recent_outcomes=[],
        rejected_hypotheses=[],
        proposal_feedback=["node model expects features input"],
    )
    payload = json.loads(client.input_text)
    rank = next(
        item for item in payload["operator_catalog"]
        if item["type"] == "model.lightgbm_rank"
    )
    assert rank["inputs"] == ["features"]
    assert rank["variadic"] is True
    assert "objective" in rank["parameters"]
    assert len(payload["valid_graph_examples"]) == 2
    assert payload["validation_feedback_for_reproposal"]
    assert "Respect every input/output signature" in client.instructions


def test_bag_node_produces_a_valid_graph_in_one_patch() -> None:
    """Seed bagging must be expressible as a single, always-valid mutation.

    Building a bag from separate add_node steps is impossible: every
    intermediate graph leaves a model node whose output reaches no terminal, and
    validation rejects it. This asserts the atomic version both applies and
    validates, which is what makes the largest measured gain reachable at all.
    """
    from agent.propose import Hypothesis, apply_hypothesis
    from pipeline.graph import OperatorGraph
    from pipeline.registry import default_registry

    registry = default_registry()
    graph = OperatorGraph.from_dict({
        "nodes": [
            {"id": "load", "type": "data.load", "params": {}, "inputs": []},
            {"id": "raw", "type": "features.raw_categorical", "params": {}, "inputs": ["load"]},
            {"id": "model", "type": "model.fm_baseline", "params": {"seed": 0}, "inputs": ["raw"]},
            {"id": "out", "type": "submit.rank", "params": {}, "inputs": ["model"]},
        ],
        "meta": {"name": "t", "hypothesis_id": "h0"},
    })
    graph.validate(registry)

    hypothesis = Hypothesis(
        id="h_bag", target_node="model", rationale="seed variance reduction",
        method_source="bagging", expected_delta=0.001, expected_cost_minutes=5.0,
        patch={"op": "bag_node", "node": "model", "seeds": [0, 1, 2],
               "ensemble_type": "ensemble.seed_bag"},
    )
    result = apply_hypothesis(graph, hypothesis, registry)
    mutated = result.graph
    mutated.validate(registry)

    ids = {node.id for node in mutated.nodes}
    assert {"model", "model_s1", "model_s2", "model_bag"} <= ids
    by_id = {node.id: node for node in mutated.nodes}
    # The ensemble consumes every replica, and the terminal reads the ensemble.
    assert set(by_id["model_bag"].inputs) == {"model", "model_s1", "model_s2"}
    assert tuple(by_id["out"].inputs) == ("model_bag",)
    # Replicas differ only by seed.
    assert {by_id[n].params["seed"] for n in ("model", "model_s1", "model_s2")} == {0, 1, 2}
    assert tuple(by_id["model_s1"].inputs) == ("raw",)


def test_bag_node_rejects_a_non_model_target() -> None:
    from agent.propose import Hypothesis, apply_hypothesis
    from pipeline.graph import GraphValidationError, OperatorGraph
    from pipeline.registry import default_registry

    registry = default_registry()
    graph = OperatorGraph.from_dict({
        "nodes": [
            {"id": "load", "type": "data.load", "params": {}, "inputs": []},
            {"id": "raw", "type": "features.raw_categorical", "params": {}, "inputs": ["load"]},
            {"id": "model", "type": "model.fm_baseline", "params": {"seed": 0}, "inputs": ["raw"]},
            {"id": "out", "type": "submit.rank", "params": {}, "inputs": ["model"]},
        ],
        "meta": {"name": "t", "hypothesis_id": "h0"},
    })
    hypothesis = Hypothesis(
        id="h_bad", target_node="raw", rationale="x", method_source="y",
        expected_delta=0.0, expected_cost_minutes=1.0,
        patch={"op": "bag_node", "node": "raw", "seeds": [0, 1],
               "ensemble_type": "ensemble.seed_bag"},
    )
    import pytest
    with pytest.raises(GraphValidationError, match="must be a model node"):
        apply_hypothesis(graph, hypothesis, registry)
