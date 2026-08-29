from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pipeline.graph import OperatorGraph

from .propose import topology_signature


AblationScorer = Callable[[OperatorGraph, str], float]


class Ablator:
    """Caches node removal evidence by graph topology, as required by the loop."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict[str, dict[str, float | str]]] = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text())

    def run(
        self,
        graph: OperatorGraph,
        incumbent_primary: float,
        score_without: AblationScorer,
    ) -> dict[str, dict[str, float | str]]:
        signature = topology_signature(graph)
        if signature in self._cache:
            return self._cache[signature]
        table: dict[str, dict[str, float | str]] = {}
        for node in graph.nodes:
            try:
                score = float(score_without(graph, node.id))
                table[node.id] = {
                    "score_without": score,
                    "removal_cost": float(incumbent_primary - score),
                    "status": "scored",
                }
            except Exception as exc:
                table[node.id] = {
                    "score_without": float(incumbent_primary),
                    "removal_cost": 0.0,
                    "status": f"not_ablatable:{type(exc).__name__}",
                }
        self._cache[signature] = table
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        return table

    @staticmethod
    def limiting_node(table: dict[str, dict[str, float | str]]) -> str:
        if not table:
            raise ValueError("ablation table is empty")
        return max(table, key=lambda node: float(table[node].get("removal_cost", 0.0)))

