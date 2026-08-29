from __future__ import annotations

from pathlib import Path

from pipeline.graph import OperatorGraph
from pipeline.registry import default_registry


ROOT = Path(__file__).resolve().parents[1]


def test_three_handwritten_graphs_validate() -> None:
    registry = default_registry()
    paths = sorted((ROOT / "configs").glob("phase2_*.json"))
    assert len(paths) == 3
    for path in paths:
        graph = OperatorGraph.from_path(path)
        assert graph.validate(registry)
