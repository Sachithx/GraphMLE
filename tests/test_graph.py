from __future__ import annotations

import pytest

from pipeline.graph import GraphValidationError, OperatorGraph
from pipeline.registry import OperatorRegistry, OperatorSpec, default_registry
from pipeline.types import ValueType


def tiny_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec("source", (), ValueType.DATA, lambda inputs, params, ctx: "data")
    )
    registry.register(
        OperatorSpec(
            "feature",
            (ValueType.DATA,),
            ValueType.FEATURES,
            lambda inputs, params, ctx: "features",
        )
    )
    registry.register(
        OperatorSpec(
            "model",
            (ValueType.FEATURES,),
            ValueType.PREDICTIONS,
            lambda inputs, params, ctx: "predictions",
            variadic=True,
        )
    )
    registry.register(
        OperatorSpec(
            "submit",
            (ValueType.PREDICTIONS,),
            ValueType.SUBMISSION,
            lambda inputs, params, ctx: "submission",
        )
    )
    return registry


def valid_graph() -> dict[str, object]:
    return {
        "nodes": [
            {"id": "load", "type": "source", "params": {}},
            {"id": "features", "type": "feature", "inputs": ["load"]},
            {"id": "model", "type": "model", "inputs": ["features"]},
            {"id": "out", "type": "submit", "inputs": ["model"]},
        ],
        "meta": {"hypothesis_id": "handwritten"},
    }


def test_valid_graph_has_stable_topological_order() -> None:
    graph = OperatorGraph.from_dict(valid_graph())
    assert [node.id for node in graph.validate(tiny_registry())] == [
        "load",
        "features",
        "model",
        "out",
    ]


def test_unused_output_fails_before_execution() -> None:
    raw = valid_graph()
    raw["nodes"].insert(
        2,
        {"id": "discarded", "type": "feature", "inputs": ["load"]},
    )
    with pytest.raises(GraphValidationError, match="unused node outputs.*discarded"):
        OperatorGraph.from_dict(raw).validate(tiny_registry())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["nodes"].append(raw["nodes"][0].copy()), "duplicate"),
        (lambda raw: raw["nodes"][1].update({"type": "unknown"}), "unknown"),
        (lambda raw: raw["nodes"][1].update({"inputs": ["missing"]}), "missing"),
        (lambda raw: raw["nodes"][0].update({"inputs": ["out"]}), "cycle"),
        (lambda raw: raw["nodes"][3].update({"inputs": ["load"]}), "expects"),
    ],
)
def test_invalid_graphs_fail_before_execution(mutate, message: str) -> None:
    raw = valid_graph()
    mutate(raw)
    with pytest.raises(GraphValidationError, match=message):
        OperatorGraph.from_dict(raw).validate(tiny_registry())


def test_default_registry_exposes_every_phase2_node_type() -> None:
    expected = {
        "data.load",
        "features.raw_categorical",
        "features.user_history",
        "features.item_popularity",
        "features.user_category_affinity",
        "features.video_duration",
        "features.temporal",
        "features.generated",
        "features.ablation_constant",
        "model.fm_baseline",
        "model.lightgbm_binary",
        "model.lightgbm_rank",
        "model.torch_deepfm",
        "model.torch_multitask",
        "model.ablation_constant",
        "ensemble.rank_average",
        "ensemble.seed_bag",
        "submit.rank",
    }
    assert set(default_registry().names()) == expected
