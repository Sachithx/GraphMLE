from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .types import ExecutionContext, ValueType


OperatorCallable = Callable[[list[Any], dict[str, Any], ExecutionContext], Any]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_types: tuple[ValueType, ...]
    output_type: ValueType
    function: OperatorCallable
    variadic: bool = False
    parameters: Mapping[str, str] = field(default_factory=dict)

    def input_error(self, actual: tuple[ValueType, ...]) -> str | None:
        if self.variadic:
            if len(self.input_types) != 1:
                return "has an invalid variadic registry signature"
            if not actual:
                return f"expects at least one {self.input_types[0].value} input"
            if any(value != self.input_types[0] for value in actual):
                return (
                    f"expects only {self.input_types[0].value} inputs, got "
                    f"{[value.value for value in actual]}"
                )
            return None
        if actual != self.input_types:
            return (
                f"expects {[value.value for value in self.input_types]} inputs, got "
                f"{[value.value for value in actual]}"
            )
        return None


class OperatorRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"operator already registered: {spec.name}")
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> OperatorSpec:
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def keys(self) -> tuple[str, ...]:
        """Return the live operator names used to constrain LLM output schemas."""
        return self.names()

    def catalog(self) -> list[dict[str, Any]]:
        """Return model-facing operator contracts from the live registry."""
        return [
            {
                "type": spec.name,
                "inputs": [value.value for value in spec.input_types],
                "variadic": spec.variadic,
                "output": spec.output_type.value,
                "parameters": dict(spec.parameters),
            }
            for spec in self._specs.values()
        ]


OPERATOR_PARAMETERS: dict[str, dict[str, str]] = {
    "data.load": {},
    "features.raw_categorical": {},
    "features.user_history": {
        "windows": "list[int], default [1, 7], strictly-earlier history windows in days",
    },
    "features.item_popularity": {
        "smoothing": "float, default 0, prior-count smoothing",
    },
    "features.user_category_affinity": {
        "smoothing": "float, default 0, prior-count smoothing",
    },
    "features.video_duration": {
        "buckets": "int, default 10, duration quantile buckets",
    },
    "features.temporal": {},
    "features.generated": {
        "module_path": "string, supplied by register_feature; do not invent",
        "builder_params": "object encoded as parameter entries",
        "sources": "list[str], supplied by register_feature",
        "temporal_scope": "strictly_earlier | same_row | static",
        "source_log": "standard | randomized | side_file, default standard",
        "max_source_date": "int YYYYMMDD; required for randomized sources",
        "categorical_columns": "list[str], default []",
    },
    "features.ablation_constant": {
        "internal": "ablation-only operator; never propose for a candidate",
    },
    "model.fm_baseline": {
        "seed": "int",
        "lr": "float, default 0.001",
        "epochs": "int, default 40",
        "batch_size": "int, default 8192",
        "patience": "int, default 4",
        "k": "int, default 16",
        "l2": "float, default 1e-6",
    },
    "model.lightgbm_binary": {
        "seed": "int",
        "n_estimators": "int, default 200",
        "num_leaves": "int, default 31",
        "lr": "float, default 0.05",
        "min_child_samples": "int, default 20",
        "subsample": "float, default 1",
        "colsample_bytree": "float, default 1",
        "reg_lambda": "float, default 0",
        "n_jobs": "int, default 0",
    },
    "model.lightgbm_rank": {
        "objective": "string, default lambdarank",
        "seed": "int",
        "n_estimators": "int, default 200",
        "num_leaves": "int, default 31",
        "lr": "float, default 0.05",
        "min_child_samples": "int, default 20",
        "subsample": "float, default 1",
        "colsample_bytree": "float, default 1",
        "reg_lambda": "float, default 0",
        "n_jobs": "int, default 0",
    },
    "model.torch_deepfm": {
        "seed": "int",
        "device": "auto | cuda | cpu, default auto",
        "threads": "int, default 4",
        "embedding_dim": "int, default 16",
        "hidden_dim": "int, default 64",
        "dropout": "float, default 0.1",
        "batch_size": "int, default 8192",
        "epochs": "int, default 3",
        "lr": "float, default 0.001",
        "weight_decay": "float, default 1e-6",
        "prediction_batch_size": "int, default 65536",
    },
    "model.torch_multitask": {
        "seed": "int",
        "device": "auto | cuda | cpu, default auto",
        "threads": "int, default 4",
        "embedding_dim": "int, default 16",
        "hidden_dim": "int, default 64",
        "dropout": "float, default 0.1",
        "batch_size": "int, default 8192",
        "epochs": "int, default 3",
        "lr": "float, default 0.001",
        "weight_decay": "float, default 1e-6",
        "prediction_batch_size": "int, default 65536",
        "auxiliary_targets": "list[str] of allowed same-row auxiliary labels",
    },
    "model.ablation_constant": {
        "internal": "ablation-only operator; never propose for a candidate",
    },
    "ensemble.rank_average": {
        "weights": "list[float], default equal weights; length must match inputs",
    },
    "ensemble.seed_bag": {
        "weights": "list[float], default equal weights; length must match inputs",
    },
    "submit.rank": {
        "filename": "basename string, default submission.csv",
        "split": "valid | test, default test",
        "check_timeout_s": "int, default 300",
    },
}


def default_registry() -> OperatorRegistry:
    from .execute import op_submit_rank
    from .ops_ensemble import op_rank_average, op_seed_bag
    from .ops_features import (
        op_data_load,
        op_ablation_constant_feature,
        op_generated_feature,
        op_item_popularity,
        op_raw_categorical,
        op_temporal,
        op_user_category_affinity,
        op_user_history,
        op_video_duration,
    )
    from .ops_models import (
        op_ablation_constant_model,
        op_fm_baseline,
        op_lightgbm_binary,
        op_lightgbm_rank,
        op_torch_deepfm,
        op_torch_multitask,
    )

    registry = OperatorRegistry()
    for name, inputs, output, function, variadic in (
        ("data.load", (), ValueType.DATA, op_data_load, False),
        ("features.raw_categorical", (ValueType.DATA,), ValueType.FEATURES, op_raw_categorical, False),
        ("features.user_history", (ValueType.DATA,), ValueType.FEATURES, op_user_history, False),
        ("features.item_popularity", (ValueType.DATA,), ValueType.FEATURES, op_item_popularity, False),
        ("features.user_category_affinity", (ValueType.DATA,), ValueType.FEATURES, op_user_category_affinity, False),
        ("features.video_duration", (ValueType.DATA,), ValueType.FEATURES, op_video_duration, False),
        ("features.temporal", (ValueType.DATA,), ValueType.FEATURES, op_temporal, False),
        ("features.generated", (ValueType.DATA,), ValueType.FEATURES, op_generated_feature, False),
        ("features.ablation_constant", (ValueType.DATA,), ValueType.FEATURES, op_ablation_constant_feature, False),
        ("model.fm_baseline", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_fm_baseline, True),
        ("model.lightgbm_binary", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_lightgbm_binary, True),
        ("model.lightgbm_rank", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_lightgbm_rank, True),
        ("model.torch_deepfm", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_torch_deepfm, True),
        ("model.torch_multitask", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_torch_multitask, True),
        ("model.ablation_constant", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_ablation_constant_model, True),
        ("ensemble.rank_average", (ValueType.PREDICTIONS,), ValueType.PREDICTIONS, op_rank_average, True),
        ("ensemble.seed_bag", (ValueType.PREDICTIONS,), ValueType.PREDICTIONS, op_seed_bag, True),
        ("submit.rank", (ValueType.PREDICTIONS,), ValueType.SUBMISSION, op_submit_rank, False),
    ):
        registry.register(
            OperatorSpec(
                name,
                inputs,
                output,
                function,
                variadic,
                OPERATOR_PARAMETERS[name],
            )
        )
    return registry
