from __future__ import annotations

from agent.run import with_model_seed
from pipeline.graph import OperatorGraph


def test_significance_rerun_overrides_explicit_model_seed() -> None:
    graph = OperatorGraph.from_path("configs/pipeline_seed.json")
    seeded = with_model_seed(graph, 29)
    model = next(node for node in seeded.nodes if node.id == "model")
    assert model.params["seed"] == 29
    assert seeded.meta["significance_seed"] == 29

