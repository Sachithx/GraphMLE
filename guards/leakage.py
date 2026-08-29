from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from pipeline.types import FeatureBundle, SourceLog, TemporalScope


TRAIN_END_DATE = 20220421
EMPIRICAL_PRIMARY_TRIPWIRE = 0.80
FORBIDDEN_SAME_ROW = frozenset(
    {
        "click",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time",
        "play_time_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
    }
)
AUXILIARY_TARGETS = frozenset(FORBIDDEN_SAME_ROW - {"long_view"})


class LeakageError(ValueError):
    def __init__(self, message: str, record: Mapping[str, object]) -> None:
        super().__init__(message)
        self.record = dict(record)


class LeakageGuard:
    def __init__(
        self,
        log_path: Path,
        empirical_primary_tripwire: float = EMPIRICAL_PRIMARY_TRIPWIRE,
    ) -> None:
        self.log_path = Path(log_path)
        self.empirical_primary_tripwire = empirical_primary_tripwire

    def _reject(
        self,
        *,
        layer: str,
        node_id: str,
        reason: str,
        feature: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        record: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "leakage_rejection",
            "layer": layer,
            "node_id": node_id,
            "reason": reason,
        }
        if feature is not None:
            record["feature"] = feature
        if details:
            record["details"] = dict(details)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        raise LeakageError(reason, record)

    def check_feature_bundle(self, bundle: FeatureBundle, node_id: str) -> None:
        columns = {
            column
            for frame in bundle.frames.values()
            for column in frame.columns
            if column != "row_id"
        }
        for column in sorted(columns):
            lineage = bundle.provenance.get(column)
            if lineage is None:
                self._reject(
                    layer="static",
                    node_id=node_id,
                    feature=column,
                    reason=f"feature {column!r} is missing lineage metadata",
                )
            forbidden_sources = sorted(lineage.sources.intersection(FORBIDDEN_SAME_ROW))
            if (
                forbidden_sources or column in FORBIDDEN_SAME_ROW
            ) and lineage.temporal_scope != TemporalScope.STRICTLY_EARLIER:
                self._reject(
                    layer="static",
                    node_id=node_id,
                    feature=column,
                    reason=(
                        f"feature {column!r} uses same-row outcome data: "
                        f"{forbidden_sources or [column]}"
                    ),
                    details={"sources": forbidden_sources or [column]},
                )
            if lineage.source_log == SourceLog.RANDOMIZED:
                if (
                    lineage.max_source_date is None
                    or lineage.max_source_date > TRAIN_END_DATE
                ):
                    self._reject(
                        layer="static",
                        node_id=node_id,
                        feature=column,
                        reason=(
                            "randomized-exposure features must prove restriction "
                            f"to training dates through {TRAIN_END_DATE}"
                        ),
                        details={"max_source_date": lineage.max_source_date},
                    )

    def check_auxiliary_targets(self, target_names: Iterable[str]) -> None:
        invalid = sorted(set(target_names) - AUXILIARY_TARGETS)
        if invalid:
            self._reject(
                layer="static",
                node_id="auxiliary_targets",
                reason=f"unsupported auxiliary targets: {invalid}",
                details={"invalid": invalid},
            )

    def check_metrics(self, metrics: Mapping[str, float], node_id: str) -> None:
        primary = float(metrics["primary"])
        if primary > self.empirical_primary_tripwire:
            self._reject(
                layer="empirical",
                node_id=node_id,
                reason=(
                    f"validation primary {primary:.6f} exceeds the 0.80 leakage tripwire"
                ),
                details={
                    "primary": primary,
                    "tripwire": self.empirical_primary_tripwire,
                },
            )
