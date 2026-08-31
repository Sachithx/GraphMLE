from __future__ import annotations

import ast
import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, create_model

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


class BagNodePatch(StrictModel):
    """Atomically seed-bag one model node.

    Seed bagging needs three coordinated edits: replicate the model under new
    seeds, insert an ensemble over the replicas, and rewire the original
    consumers. A one-patch-per-hypothesis action space cannot express that as
    separate steps, because each intermediate graph has an unreachable node and
    is correctly rejected. Making the whole thing one patch keeps every
    intermediate state valid.
    """

    op: Literal["bag_node"]
    node: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=2, max_length=8)
    ensemble_type: Literal["ensemble.seed_bag", "ensemble.rank_average"] = "ensemble.seed_bag"


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
        BagNodePatch,
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


class LLMBagNodePatch(StrictModel):
    op: Literal["bag_node"]
    node: str = Field(min_length=1)
    seeds: list[int]
    ensemble_type: Literal["ensemble.seed_bag", "ensemble.rank_average"]


class LLMRegisterFeaturePatch(StrictModel):
    op: Literal["register_feature"]
    node: LLMNodeDefinition
    code: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    temporal_scope: Literal["strictly_earlier", "same_row", "static"]


LLMPatch = Union[
    LLMReplaceParamsPatch,
    LLMAddNodePatch,
    LLMRemoveNodePatch,
    LLMRewirePatch,
    LLMBagNodePatch,
    LLMRegisterFeaturePatch,
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


def llm_hypothesis_schema(registry: OperatorRegistry) -> type[LLMHypothesis]:
    """Build the API schema from the live registry so unknown types are impossible."""
    operator_type = Literal.__getitem__(registry.keys())
    registry_node = create_model(
        "RegistryNodeDefinition",
        __base__=LLMNodeDefinition,
        __module__=__name__,
        type=(operator_type, ...),
    )
    registry_add = create_model(
        "RegistryAddNodePatch",
        __base__=LLMAddNodePatch,
        __module__=__name__,
        node=(registry_node, ...),
    )
    registry_feature = create_model(
        "RegistryRegisterFeaturePatch",
        __base__=LLMRegisterFeaturePatch,
        __module__=__name__,
        node=(registry_node, ...),
    )
    registry_patch = Union[
        LLMReplaceParamsPatch,
        registry_add,
        LLMRemoveNodePatch,
        LLMRewirePatch,
        LLMBagNodePatch,
        registry_feature,
    ]
    return create_model(
        "RegistryLLMHypothesis",
        __base__=LLMHypothesis,
        __module__=__name__,
        patch=(registry_patch, ...),
    )


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

Use only operator types and parameters in operator_catalog. The schema makes any
unregistered node type invalid. Respect every input/output signature. For an existing
operator, replace_params replaces its complete parameter map. Use add_node with
replace_node to switch an existing operator type. Internal ablation operators are not
candidate operators. Use register_feature, not add_node, for generated Python features.
Every node output must reach the single terminal submission; attach new feature or
model nodes to consumers atomically so the graph has no discarded computation.

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
- The FM consumes every explicit feature bundle. FM and LambdaRank make different
  errors, so weighted rank-average changes are valid targeted ensemble hypotheses.

Search strategy, ordered by observed effect size on this benchmark:
- Additive changes beat wholesale substitution. Replacing an incumbent model that
  already beats the baseline has repeatedly produced large regressions, because the
  replacement starts from scratch while the incumbent is already near a local optimum.
  Keep the incumbent and add to it.
- Seed variance is comparable to the real signal here: the baseline's own five-seed
  std is 0.0008, and genuine improvements are of the same order. Averaging several
  seeds of the incumbent reduces that variance directly and is usually the single
  largest gain available. Use the bag_node patch, which replicates a model node
  across seeds and inserts the ensemble in one atomic mutation; three to five seeds
  is normally enough. Consider it before trying new architectures.
- Adding a feature bundle to a factorisation machine is not free: every extra field
  enters every pairwise interaction, so a weakly informative bundle dilutes the
  strong ones. Measured on this benchmark, the damage from stacking bundles is
  additive. Prefer removing a bundle that ablation shows contributes little over
  adding another one.
- Combining decorrelated models through ensemble.rank_average can beat either
  component, and a model that is weaker on its own can still earn weight if it makes
  different errors. Judge a candidate by what it adds to the incumbent, not by its
  standalone score.
- Feature bundles interact with model families differently. A feature set that does
  not help one model can still help another; pair a promising bundle with more than
  one model family before discarding it.
- Tune hyperparameters last; on this benchmark their effect is typically smaller than
  seed noise.

When scheduler_preferred_target is supplied, keep the hypothesis focused on that
component. Do not request raw data or executable shell access.
"""


EXAMPLE_GRAPHS = [
    {
        "nodes": [
            {"id": "load", "type": "data.load", "params": {}, "inputs": []},
            {
                "id": "raw",
                "type": "features.raw_categorical",
                "params": {},
                "inputs": ["load"],
            },
            {
                "id": "pop",
                "type": "features.item_popularity",
                "params": {"smoothing": 20},
                "inputs": ["load"],
            },
            {
                "id": "model",
                "type": "model.lightgbm_rank",
                "params": {
                    "objective": "lambdarank",
                    "n_estimators": 200,
                    "num_leaves": 31,
                    "lr": 0.05,
                    "seed": 0,
                },
                "inputs": ["raw", "pop"],
            },
            {
                "id": "out",
                "type": "submit.rank",
                "params": {"filename": "submission.csv"},
                "inputs": ["model"],
            },
        ]
    },
    {
        "nodes": [
            {"id": "load", "type": "data.load", "params": {}, "inputs": []},
            {
                "id": "raw",
                "type": "features.raw_categorical",
                "params": {},
                "inputs": ["load"],
            },
            {
                "id": "pop",
                "type": "features.item_popularity",
                "params": {"smoothing": 20},
                "inputs": ["load"],
            },
            {
                "id": "fm",
                "type": "model.fm_baseline",
                "params": {"seed": 0, "numeric_bins": 32},
                "inputs": ["raw", "pop"],
            },
            {
                "id": "ranker",
                "type": "model.lightgbm_rank",
                "params": {
                    "objective": "lambdarank",
                    "n_estimators": 100,
                    "num_leaves": 31,
                    "lr": 0.05,
                    "seed": 0,
                },
                "inputs": ["raw", "pop"],
            },
            {
                "id": "blend",
                "type": "ensemble.rank_average",
                "params": {"weights": [0.65, 0.35]},
                "inputs": ["fm", "ranker"],
            },
            {
                "id": "out",
                "type": "submit.rank",
                "params": {"filename": "submission.csv"},
                "inputs": ["blend"],
            },
        ]
    },
]


@dataclass(frozen=True)
class ProposalResult:
    hypothesis: Hypothesis
    usage: TokenUsage


class LiveHypothesisProposer:
    def __init__(self, client: StructuredClient, registry: OperatorRegistry) -> None:
        self.client = client
        self.registry = registry
        self.schema = llm_hypothesis_schema(registry)

    def propose(
        self,
        *,
        graph: OperatorGraph,
        ablation_table: dict[str, dict[str, float | str]],
        recent_outcomes: list[dict[str, Any]],
        rejected_hypotheses: list[str],
        preferred_target: str | None = None,
        proposal_feedback: list[str] | None = None,
    ) -> ProposalResult:
        context = {
            "incumbent_graph": graph.to_dict(),
            "operator_catalog": self.registry.catalog(),
            "valid_graph_examples": EXAMPLE_GRAPHS,
            "ablation_table": ablation_table,
            "last_10_outcomes": recent_outcomes[-10:],
            "already_tried_and_rejected": rejected_hypotheses,
            "scheduler_preferred_target": preferred_target,
            "validation_feedback_for_reproposal": proposal_feedback or [],
        }
        result = self.client.parse(
            self.schema,
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
        proposal_feedback: list[str] | None = None,
    ) -> ProposalResult:
        del (
            graph,
            ablation_table,
            recent_outcomes,
            rejected_hypotheses,
            preferred_target,
            proposal_feedback,
        )
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
    elif patch.op == "bag_node":
        if patch.node not in by_id:
            raise GraphValidationError(f"bag_node target does not exist: {patch.node}")
        target = by_id[patch.node]
        if not str(target["type"]).startswith("model."):
            raise GraphValidationError(
                f"bag_node target must be a model node, got {target['type']}"
            )
        seeds = list(dict.fromkeys(int(s) for s in patch.seeds))
        if len(seeds) < 2:
            raise GraphValidationError("bag_node needs at least two distinct seeds")
        bag_id = f"{patch.node}_bag"
        if bag_id in by_id:
            raise GraphValidationError(f"bag_node ensemble id already exists: {bag_id}")
        # Consumers of the original node now read the ensemble instead.
        consumers = [n for n in nodes if patch.node in n["inputs"]]
        replicas: list[str] = []
        for index, seed in enumerate(seeds):
            if index == 0:
                target["params"] = {**target["params"], "seed": seed}
                replicas.append(patch.node)
                continue
            replica_id = f"{patch.node}_s{seed}"
            if replica_id in by_id:
                raise GraphValidationError(f"bag_node replica id already exists: {replica_id}")
            nodes.append({
                "id": replica_id,
                "type": target["type"],
                "params": {**target["params"], "seed": seed},
                "inputs": list(target["inputs"]),
            })
            replicas.append(replica_id)
        nodes.append({
            "id": bag_id,
            "type": patch.ensemble_type,
            "params": {},
            "inputs": replicas,
        })
        for consumer in consumers:
            consumer["inputs"] = [
                bag_id if value == patch.node else value for value in consumer["inputs"]
            ]
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
