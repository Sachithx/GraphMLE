from __future__ import annotations

import ast
import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from guards.leakage import FORBIDDEN_SAME_ROW
from pipeline.graph import GraphValidationError, OperatorGraph
from pipeline.registry import OperatorRegistry

from .llm import StructuredClient, TokenUsage


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeDefinition(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)


class ReplaceParamsPatch(StrictModel):
    op: Literal["replace_params"]
    node: str = Field(min_length=1)
    params: dict[str, Any]


class AddNodePatch(StrictModel):
    op: Literal["add_node"]
    node: NodeDefinition
    replace_node: str | None = None
    consumers: list[str] = Field(default_factory=list)
    consumer_mode: Literal["append", "replace"] = "append"


class RemoveNodePatch(StrictModel):
    op: Literal["remove_node"]
    node: str = Field(min_length=1)


class RewirePatch(StrictModel):
    op: Literal["rewire"]
    node: str = Field(min_length=1)
    inputs: list[str]


class RegisterFeaturePatch(StrictModel):
    op: Literal["register_feature"]
    node: NodeDefinition
    code: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    temporal_scope: Literal["strictly_earlier", "same_row", "static"]


Patch = Annotated[
    Union[
        ReplaceParamsPatch,
        AddNodePatch,
        RemoveNodePatch,
        RewirePatch,
        RegisterFeaturePatch,
    ],
    Field(discriminator="op"),
]


class Hypothesis(StrictModel):
    id: str = Field(min_length=1)
    target_node: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    method_source: str = Field(min_length=1)
    expected_delta: float
    expected_cost_minutes: PositiveFloat
    patch: Patch


ParameterValue = Union[
    str,
    int,
    float,
    bool,
    list[str],
    list[int],
    list[float],
    list[bool],
    None,
]


class ParameterEntry(StrictModel):
    name: str = Field(min_length=1)
    value: ParameterValue


class LLMNodeDefinition(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    type: str = Field(min_length=1)
    params: list[ParameterEntry]
    inputs: list[str]


class LLMReplaceParamsPatch(StrictModel):
    op: Literal["replace_params"]
    node: str = Field(min_length=1)
    params: list[ParameterEntry]


class LLMAddNodePatch(StrictModel):
    op: Literal["add_node"]
    node: LLMNodeDefinition
    replace_node: str | None
    consumers: list[str]
    consumer_mode: Literal["append", "replace"]


class LLMRemoveNodePatch(StrictModel):
    op: Literal["remove_node"]
    node: str = Field(min_length=1)


class LLMRewirePatch(StrictModel):
    op: Literal["rewire"]
    node: str = Field(min_length=1)
    inputs: list[str]


class LLMRegisterFeaturePatch(StrictModel):
    op: Literal["register_feature"]
    node: LLMNodeDefinition
    code: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    temporal_scope: Literal["strictly_earlier", "same_row", "static"]


LLMPatch = Annotated[
    Union[
        LLMReplaceParamsPatch,
        LLMAddNodePatch,
        LLMRemoveNodePatch,
        LLMRewirePatch,
        LLMRegisterFeaturePatch,
    ],
    Field(discriminator="op"),
]


class LLMHypothesis(StrictModel):
    """API wire schema: maps are encoded as entries to remain strict-JSON compatible."""

    id: str = Field(min_length=1)
    target_node: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    method_source: str = Field(min_length=1)
    expected_delta: float
    expected_cost_minutes: PositiveFloat
    patch: LLMPatch

    def to_runtime(self) -> Hypothesis:
        raw = self.model_dump(mode="json")
        patch = raw["patch"]
        if "params" in patch:
            patch["params"] = _parameter_entries_to_dict(patch["params"])
        if isinstance(patch.get("node"), dict):
            patch["node"]["params"] = _parameter_entries_to_dict(
                patch["node"]["params"]
            )
        return Hypothesis.model_validate(raw)


def _parameter_entries_to_dict(entries: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in entries:
        name = str(entry["name"])
        if name in result:
            raise ValueError(f"duplicate parameter name: {name}")
        result[name] = entry["value"]
    return result


SYSTEM_PROMPT = """You are the hypothesis proposer inside an autonomous ML research harness.
Return exactly one bounded graph mutation using the supplied schema.

Problem framing:
- GAUC and nDCG rank roughly five logged impressions within each user. Pointwise
  calibration is secondary; listwise or pairwise objectives grouped by user align
  with the scored metrics.
- A monotone per-user score transform cannot change within-user ranking.
- KuaiRand has 12 feedback signals. Only long_view is scored; other same-impression
  outcomes are auxiliary targets, never same-row input features.
- About 27.1% of users have no positive label and 9.2% are all-positive. The
  perfect-ranking ceiling is about 0.8645, not 1.0.
- Randomized exposure supports IPS or doubly robust correction, but its overlap
  with the test window means derived features must remain restricted to training dates.
- Every historical aggregate must use dates strictly earlier than its target row.

Prioritize large expected effects such as listwise ranking and multi-task learning
before hyperparameter tuning. Do not request raw data or executable shell access.
"""


@dataclass(frozen=True)
class ProposalResult:
    hypothesis: Hypothesis
    usage: TokenUsage


class LiveHypothesisProposer:
    def __init__(self, client: StructuredClient) -> None:
        self.client = client

    def propose(
        self,
        *,
        graph: OperatorGraph,
        ablation_table: dict[str, dict[str, float | str]],
        recent_outcomes: list[dict[str, Any]],
        rejected_hypotheses: list[str],
        preferred_target: str | None = None,
    ) -> ProposalResult:
        context = {
            "incumbent_graph": graph.to_dict(),
            "ablation_table": ablation_table,
            "last_10_outcomes": recent_outcomes[-10:],
            "already_tried_and_rejected": rejected_hypotheses,
            "scheduler_preferred_target": preferred_target,
        }
        result = self.client.parse(
            LLMHypothesis,
            instructions=SYSTEM_PROMPT,
            input_text=json.dumps(context, sort_keys=True),
        )
        return ProposalResult(result.value.to_runtime(), result.usage)


class CannedHypothesisProposer:
    def __init__(self, hypotheses: list[Hypothesis]) -> None:
        self.hypotheses = sorted(
            hypotheses, key=lambda hypothesis: -hypothesis.expected_delta
        )
        self.index = 0

    def propose(
        self,
        *,
        graph: OperatorGraph,
        ablation_table: dict[str, dict[str, float | str]],
        recent_outcomes: list[dict[str, Any]],
        rejected_hypotheses: list[str],
        preferred_target: str | None = None,
    ) -> ProposalResult:
        del graph, ablation_table, recent_outcomes, rejected_hypotheses, preferred_target
        if self.index >= len(self.hypotheses):
            raise RuntimeError("canned hypothesis list exhausted")
        hypothesis = self.hypotheses[self.index]
        self.index += 1
        return ProposalResult(hypothesis, TokenUsage())


@dataclass(frozen=True)
class MutationResult:
    graph: OperatorGraph
    diff: str
    topology_changed: bool
    generated_file: Path | None = None


def topology_signature(graph: OperatorGraph) -> str:
    topology = [
        {"id": node.id, "type": node.type, "inputs": list(node.inputs)}
        for node in graph.nodes
    ]
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _validate_feature_code(code: str, declared_sources: list[str]) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise GraphValidationError(f"generated feature is not valid Python: {exc}") from exc
    builds = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
    ]
    if len(builds) != 1 or isinstance(builds[0], ast.AsyncFunctionDef):
        raise GraphValidationError("generated feature must define exactly one synchronous build")
    positional = [*builds[0].args.posonlyargs, *builds[0].args.args]
    if [argument.arg for argument in positional] != ["train_df", "target_df", "ctx"]:
        raise GraphValidationError(
            "generated build signature must be build(train_df, target_df, ctx)"
        )
    target_columns: set[str] = set()
    for node in ast.walk(builds[0]):
        if isinstance(node, ast.Subscript) and _root_name(node.value) == "target_df":
            target_columns.update(
                child.value
                for child in ast.walk(node.slice)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
        if (
            isinstance(node, ast.Attribute)
            and _root_name(node.value) == "target_df"
            and not node.attr.startswith("_")
            and (node.attr in FORBIDDEN_SAME_ROW or node.attr in declared_sources)
        ):
            target_columns.add(node.attr)
    forbidden = sorted(target_columns.intersection(FORBIDDEN_SAME_ROW))
    if forbidden:
        raise GraphValidationError(
            f"generated feature reads forbidden same-row target columns: {forbidden}"
        )
    undeclared = sorted(target_columns - set(declared_sources) - {"row_id"})
    if undeclared:
        raise GraphValidationError(
            f"generated feature target columns are missing from sources: {undeclared}"
        )


def graph_diff(before: OperatorGraph, after: OperatorGraph) -> str:
    left = json.dumps(before.to_dict(), indent=2, sort_keys=True).splitlines(keepends=True)
    right = json.dumps(after.to_dict(), indent=2, sort_keys=True).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(left, right, fromfile="incumbent/graph.json", tofile="candidate/graph.json")
    )


def apply_hypothesis(
    graph: OperatorGraph,
    hypothesis: Hypothesis,
    registry: OperatorRegistry,
    *,
    generated_feature_dir: Path | None = None,
) -> MutationResult:
    raw = graph.to_dict()
    nodes = raw["nodes"]
    by_id = {node["id"]: node for node in nodes}
    patch = hypothesis.patch
    generated_file: Path | None = None

    if patch.op == "replace_params":
        if patch.node not in by_id:
            raise GraphValidationError(f"replace_params target does not exist: {patch.node}")
        by_id[patch.node]["params"] = patch.params
    elif patch.op == "add_node":
        if patch.node.id in by_id and patch.replace_node != patch.node.id:
            raise GraphValidationError(f"add_node id already exists: {patch.node.id}")
        if patch.replace_node is not None:
            if patch.replace_node not in by_id:
                raise GraphValidationError(
                    f"add_node replacement does not exist: {patch.replace_node}"
                )
            nodes[:] = [node for node in nodes if node["id"] != patch.replace_node]
            if patch.node.id != patch.replace_node:
                for node in nodes:
                    node["inputs"] = [
                        patch.node.id if value == patch.replace_node else value
                        for value in node["inputs"]
                    ]
        nodes.append(patch.node.model_dump())
        for consumer_id in patch.consumers:
            consumer = next(
                (node for node in nodes if node["id"] == consumer_id), None
            )
            if consumer is None:
                raise GraphValidationError(f"add_node consumer does not exist: {consumer_id}")
            if patch.consumer_mode == "replace":
                consumer["inputs"] = [patch.node.id]
            elif patch.node.id not in consumer["inputs"]:
                consumer["inputs"].append(patch.node.id)
    elif patch.op == "remove_node":
        if patch.node not in by_id:
            raise GraphValidationError(f"remove_node target does not exist: {patch.node}")
        raw["nodes"] = [node for node in nodes if node["id"] != patch.node]
    elif patch.op == "rewire":
        if patch.node not in by_id:
            raise GraphValidationError(f"rewire target does not exist: {patch.node}")
        by_id[patch.node]["inputs"] = patch.inputs
    elif patch.op == "register_feature":
        if generated_feature_dir is None:
            raise GraphValidationError("register_feature requires a generated feature directory")
        if patch.node.id in by_id:
            raise GraphValidationError(f"feature node id already exists: {patch.node.id}")
        if patch.node.type != "features.generated":
            raise GraphValidationError("registered feature node type must be features.generated")
        _validate_feature_code(patch.code, patch.sources)
        generated_feature_dir.mkdir(parents=True, exist_ok=True)
        generated_file = generated_feature_dir / f"{patch.node.id}.py"
        generated_file.write_text(patch.code)
        node = patch.node.model_dump()
        node["params"] = {
            **node["params"],
            "module_path": str(generated_file.resolve()),
            "sources": patch.sources,
            "temporal_scope": patch.temporal_scope,
        }
        nodes.append(node)

    raw["meta"] = {
        **raw.get("meta", {}),
        "parent": graph.meta.get("hypothesis_id"),
        "hypothesis_id": hypothesis.id,
        "expected_delta": hypothesis.expected_delta,
    }
    candidate = OperatorGraph.from_dict(raw)
    candidate.validate(registry)
    return MutationResult(
        candidate,
        graph_diff(graph, candidate),
        topology_signature(graph) != topology_signature(candidate),
        generated_file,
    )
