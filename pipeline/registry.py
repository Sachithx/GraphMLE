from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import ExecutionContext, ValueType


OperatorCallable = Callable[[list[Any], dict[str, Any], ExecutionContext], Any]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_types: tuple[ValueType, ...]
    output_type: ValueType
    function: OperatorCallable
    variadic: bool = False

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


def default_registry() -> OperatorRegistry:
    from .execute import op_submit_rank
    from .ops_ensemble import op_rank_average, op_seed_bag
    from .ops_features import (
        op_data_load,
        op_item_popularity,
        op_raw_categorical,
        op_temporal,
        op_user_category_affinity,
        op_user_history,
        op_video_duration,
    )
    from .ops_models import (
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
        ("model.fm_baseline", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_fm_baseline, True),
        ("model.lightgbm_binary", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_lightgbm_binary, True),
        ("model.lightgbm_rank", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_lightgbm_rank, True),
        ("model.torch_deepfm", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_torch_deepfm, True),
        ("model.torch_multitask", (ValueType.FEATURES,), ValueType.PREDICTIONS, op_torch_multitask, True),
        ("ensemble.rank_average", (ValueType.PREDICTIONS,), ValueType.PREDICTIONS, op_rank_average, True),
        ("ensemble.seed_bag", (ValueType.PREDICTIONS,), ValueType.PREDICTIONS, op_seed_bag, True),
        ("submit.rank", (ValueType.PREDICTIONS,), ValueType.SUBMISSION, op_submit_rank, False),
    ):
        registry.register(OperatorSpec(name, inputs, output, function, variadic))
    return registry
