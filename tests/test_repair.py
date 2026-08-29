from __future__ import annotations

from agent.repair import CannedRepairProvider, RepairManager
from pipeline.graph import OperatorGraph
from pipeline.registry import default_registry


def test_repair_loop_recovers_induced_failure() -> None:
    raw = OperatorGraph.from_path("configs/pipeline_seed.json").to_dict()
    raw["nodes"][2]["params"]["induce_failure"] = True
    graph = OperatorGraph.from_dict(raw)

    def evaluate(candidate: OperatorGraph) -> dict[str, float]:
        model = next(node for node in candidate.nodes if node.id == "model")
        if model.params.get("induce_failure"):
            raise RuntimeError("induced failure")
        return {"gauc": 0.67, "ndcg5": 0.54, "primary": 0.605}

    outcome = RepairManager(CannedRepairProvider(), max_attempts=3).recover(
        graph, RuntimeError("induced failure"), evaluate
    )
    assert outcome.recovered
    assert outcome.metrics is not None
    assert outcome.events[0]["attempt"] == 1
    assert outcome.events[0]["status"] == "recovered"


def test_repair_loop_retries_graph_validation_errors() -> None:
    registry = default_registry()
    seed = OperatorGraph.from_path("configs/pipeline_seed.json")

    class ValidationRepairProvider:
        def repair(self, graph, error, attempt):
            del error
            if attempt == 1:
                raw = graph.to_dict()
                raw["nodes"][0]["type"] = "data.not_registered"
                invalid = OperatorGraph.from_dict(raw)
                invalid.validate(registry)
            graph.validate(registry)
            return graph

    outcome = RepairManager(ValidationRepairProvider(), max_attempts=2).recover(
        seed,
        ValueError("initial graph validation failure"),
        lambda graph: {"gauc": 0.67, "ndcg5": 0.54, "primary": 0.605},
    )
    assert outcome.recovered
    assert [event["status"] for event in outcome.events] == ["failed", "recovered"]
