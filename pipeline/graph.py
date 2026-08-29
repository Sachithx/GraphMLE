from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .registry import OperatorRegistry


class GraphValidationError(ValueError):
    """Raised before compute when an operator graph is not well typed."""


@dataclass(frozen=True)
class OperatorNode:
    id: str
    type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorGraph:
    nodes: tuple[OperatorNode, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OperatorGraph":
        if not isinstance(raw, Mapping):
            raise GraphValidationError("graph must be a JSON object")
        raw_nodes = raw.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise GraphValidationError("graph.nodes must be a non-empty list")
        nodes: list[OperatorNode] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_nodes):
            if not isinstance(item, Mapping):
                raise GraphValidationError(f"node {index} must be an object")
            node_id = item.get("id")
            node_type = item.get("type")
            if not isinstance(node_id, str) or not node_id:
                raise GraphValidationError(f"node {index} has an invalid id")
            if node_id in seen:
                raise GraphValidationError(f"duplicate node id: {node_id}")
            seen.add(node_id)
            if not isinstance(node_type, str) or not node_type:
                raise GraphValidationError(f"node {node_id} has an invalid type")
            params = item.get("params", {})
            inputs = item.get("inputs", [])
            if not isinstance(params, Mapping):
                raise GraphValidationError(f"node {node_id} params must be an object")
            if not isinstance(inputs, list) or not all(isinstance(x, str) for x in inputs):
                raise GraphValidationError(f"node {node_id} inputs must be a list of ids")
            nodes.append(OperatorNode(node_id, node_type, dict(params), tuple(inputs)))
        meta = raw.get("meta", {})
        if not isinstance(meta, Mapping):
            raise GraphValidationError("graph.meta must be an object")
        return cls(tuple(nodes), dict(meta))

    @classmethod
    def from_path(cls, path: str | Path) -> "OperatorGraph":
        path = Path(path)
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphValidationError(f"could not read graph {path}: {exc}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "params": dict(node.params),
                    "inputs": list(node.inputs),
                }
                for node in self.nodes
            ],
            "meta": dict(self.meta),
        }

    def validate(self, registry: OperatorRegistry) -> tuple[OperatorNode, ...]:
        by_id = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if node.type not in registry:
                raise GraphValidationError(f"unknown node type {node.type!r} at {node.id}")
            for input_id in node.inputs:
                if input_id not in by_id:
                    raise GraphValidationError(
                        f"node {node.id} references missing input {input_id!r}"
                    )

        order_index = {node.id: index for index, node in enumerate(self.nodes)}
        indegree = {node.id: len(node.inputs) for node in self.nodes}
        consumers: dict[str, list[str]] = {node.id: [] for node in self.nodes}
        for node in self.nodes:
            for input_id in node.inputs:
                consumers[input_id].append(node.id)
        ready = sorted(
            (node.id for node in self.nodes if indegree[node.id] == 0),
            key=order_index.__getitem__,
        )
        ordered: list[OperatorNode] = []
        while ready:
            node_id = ready.pop(0)
            ordered.append(by_id[node_id])
            for consumer in sorted(consumers[node_id], key=order_index.__getitem__):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)
                    ready.sort(key=order_index.__getitem__)
        if len(ordered) != len(self.nodes):
            cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
            raise GraphValidationError(f"cycle detected involving nodes: {cyclic}")

        output_types = {}
        for node in ordered:
            spec = registry[node.type]
            actual = tuple(output_types[input_id] for input_id in node.inputs)
            error = spec.input_error(actual)
            if error:
                raise GraphValidationError(f"node {node.id} {error}")
            output_types[node.id] = spec.output_type
        return tuple(ordered)
