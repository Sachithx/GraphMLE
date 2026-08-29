from __future__ import annotations

from pathlib import Path

from agent.ablate import Ablator
from agent.propose import Hypothesis, apply_hypothesis
from pipeline.graph import OperatorGraph
from pipeline.registry import default_registry


def test_ablation_is_cached_until_topology_changes(tmp_path: Path) -> None:
    graph = OperatorGraph.from_path("configs/pipeline_seed.json")
    calls: list[str] = []

    def score_without(_graph: OperatorGraph, node_id: str) -> float:
        calls.append(node_id)
        return 0.60 - len(node_id) / 10_000

    ablator = Ablator(tmp_path / "ablation_cache.json")
    first = ablator.run(graph, 0.601, score_without)
    second = ablator.run(graph, 0.602, score_without)
    assert second == first
    assert len(calls) == len(graph.nodes)

    add = Hypothesis.model_validate(
        {
            "id": "h_add",
            "target_node": "raw",
            "rationale": "Add a second safe feature view.",
            "method_source": "ablation",
            "expected_delta": 0.003,
            "expected_cost_minutes": 1,
                "patch": {
                    "op": "add_node",
                    "node": {
                    "id": "history",
                    "type": "features.user_history",
                    "params": {"windows": [1, 7]},
                        "inputs": ["load"],
                    },
                    "consumers": ["model"],
                    "consumer_mode": "append",
                },
            }
        )
    changed = apply_hypothesis(graph, add, default_registry()).graph
    ablator.run(changed, 0.602, score_without)
    assert len(calls) == len(graph.nodes) + len(changed.nodes)
