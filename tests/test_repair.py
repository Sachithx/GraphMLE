from __future__ import annotations

from agent.repair import CannedRepairProvider, RepairManager
from pipeline.graph import OperatorGraph


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

