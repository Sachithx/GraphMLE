from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd

from .types import (
    DataBundle,
    ExecutionContext,
    FeatureBundle,
    FeatureLineage,
    SourceLog,
    TemporalScope,
)


SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
AUXILIARY_TARGETS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)


@dataclass(frozen=True)
class FeatureBuildContext:
    """Context for the fixed feature-builder contract.

    Built-in and generated feature functions receive only historical training
    rows, target rows, and this context. They must return one row per target row.
    """

    params: dict[str, Any] = field(default_factory=dict)
    date_column: str = "date"
    label_column: str = "long_view"


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"required starter-kit file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_data(ctx: ExecutionContext) -> DataBundle:
    cache_key = f"data:{ctx.data_dir}:{ctx.starter_kit_dir}"
    cached = ctx.cache.get(cache_key)
    if isinstance(cached, DataBundle):
        return cached

    loader = _load_module(ctx.starter_kit_dir / "data.py", "techjam_graph_data")
    official_splits = loader.load(str(ctx.data_dir))
    raw_columns = [
        "date",
        "user_id",
        "video_id",
        "hourmin",
        "time_ms",
        *AUXILIARY_TARGETS,
        "long_view",
        "play_time_ms",
        "profile_stay_time",
        "comment_stay_time",
    ]
    raw = pd.concat(
        [
            pd.read_csv(ctx.data_dir / filename, usecols=raw_columns)
            for filename in (
                "log_standard_4_08_to_4_21_pure.csv",
                "log_standard_4_22_to_5_08_pure.csv",
            )
        ],
        ignore_index=True,
    )
    video = pd.read_csv(
        ctx.data_dir / "video_features_basic_pure.csv",
        usecols=["video_id", "tag"],
        dtype={"video_id": "string", "tag": "string"},
    )
    tag_by_video = video.set_index("video_id")["tag"]

    frames: dict[str, pd.DataFrame] = {}
    canonical_columns = [
        "date",
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_ms",
        "long_view",
    ]
    for split, (lo, hi) in SPLITS.items():
        canonical = pd.DataFrame(official_splits[split], columns=canonical_columns)
        selected = raw.loc[raw["date"].between(lo, hi)].reset_index(drop=True)
        if len(canonical) != len(selected):
            raise RuntimeError(
                f"raw/official {split} row mismatch: {len(selected)} vs {len(canonical)}"
            )
        raw_users = selected["user_id"].astype(str).reset_index(drop=True)
        raw_videos = selected["video_id"].astype(str).reset_index(drop=True)
        if not raw_users.equals(canonical["user_id"].astype(str)):
            raise RuntimeError(f"raw/official {split} user ordering mismatch")
        if not raw_videos.equals(canonical["video_id"].astype(str)):
            raise RuntimeError(f"raw/official {split} video ordering mismatch")
        frame = canonical.copy()
        frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
        for column in raw_columns:
            if column not in canonical_columns:
                frame[column] = selected[column].to_numpy()
        frame["tag"] = frame["video_id"].astype("string").map(tag_by_video).fillna("UNK")
        frames[split] = frame

    bundle = DataBundle(
        frames=frames,
        official_splits=official_splits,
        data_dir=ctx.data_dir,
        starter_kit_dir=ctx.starter_kit_dir,
    )
    ctx.cache[cache_key] = bundle
    return bundle


def op_data_load(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> DataBundle:
    del inputs, params
    return _canonical_data(ctx)


def _state_key(values: tuple[Any, ...]) -> Any:
    return values[0] if len(values) == 1 else values


def _expanding_rate(
    history: pd.DataFrame,
    target: pd.DataFrame,
    ctx: FeatureBuildContext,
    keys: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    date_column, label_column = ctx.date_column, ctx.label_column
    smoothing = float(ctx.params.get("smoothing", 0.0))
    daily = (
        history.groupby([date_column, *keys], sort=True, dropna=False)[label_column]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values([date_column, *keys], kind="stable")
        .reset_index(drop=True)
    )
    counts: dict[Any, float] = {}
    positives: dict[Any, float] = {}
    global_count = 0.0
    global_positive = 0.0
    output_count = np.zeros(len(target), dtype=np.float32)
    output_rate = np.zeros(len(target), dtype=np.float32)
    daily_records = list(daily.itertuples(index=False, name=None))
    pointer = 0
    target_dates = target[date_column].to_numpy()
    target_key_values = list(target.loc[:, list(keys)].itertuples(index=False, name=None))

    for date in sorted(pd.unique(target_dates)):
        while pointer < len(daily_records) and daily_records[pointer][0] < date:
            record = daily_records[pointer]
            key = _state_key(tuple(record[1 : 1 + len(keys)]))
            positive, count = float(record[-2]), float(record[-1])
            positives[key] = positives.get(key, 0.0) + positive
            counts[key] = counts.get(key, 0.0) + count
            global_positive += positive
            global_count += count
            pointer += 1
        prior = global_positive / global_count if global_count else 0.0
        positions = np.flatnonzero(target_dates == date)
        for position in positions:
            key = _state_key(tuple(target_key_values[position]))
            count = counts.get(key, 0.0)
            positive = positives.get(key, 0.0)
            denominator = count + smoothing
            output_count[position] = count
            output_rate[position] = (
                (positive + smoothing * prior) / denominator if denominator else prior
            )
    return pd.DataFrame(
        {
            "row_id": target.get("row_id", pd.Series(target.index)).to_numpy(),
            f"{prefix}_impressions": output_count,
            f"{prefix}_rate": output_rate,
        }
    )


def build_item_popularity(
    train_df: pd.DataFrame, target_df: pd.DataFrame, ctx: FeatureBuildContext
) -> pd.DataFrame:
    """Return item statistics derived only from strictly earlier train dates."""
    return _expanding_rate(train_df, target_df, ctx, ("video_id",), "item_pop")


def build_user_category_affinity(
    train_df: pd.DataFrame, target_df: pd.DataFrame, ctx: FeatureBuildContext
) -> pd.DataFrame:
    """Return user/tag affinity derived only from strictly earlier train dates."""
    return _expanding_rate(
        train_df, target_df, ctx, ("user_id", "tag"), "user_category_affinity"
    )


def _windowed_user_history(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ctx: FeatureBuildContext,
    window_days: int,
) -> pd.DataFrame:
    history_dates = pd.to_datetime(train_df[ctx.date_column].astype(str), format="%Y%m%d")
    target_dates = pd.to_datetime(target_df[ctx.date_column].astype(str), format="%Y%m%d")
    users = target_df["user_id"].astype(str)
    counts = np.zeros(len(target_df), dtype=np.float32)
    rates = np.zeros(len(target_df), dtype=np.float32)
    for date in sorted(target_dates.unique()):
        eligible = (history_dates < date) & (
            history_dates >= date - pd.Timedelta(days=window_days)
        )
        grouped = train_df.loc[eligible].groupby("user_id")[ctx.label_column].agg(["sum", "count"])
        positions = np.flatnonzero(target_dates == date)
        selected_users = users.iloc[positions]
        selected_count = selected_users.map(grouped["count"]).fillna(0.0).to_numpy()
        selected_positive = selected_users.map(grouped["sum"]).fillna(0.0).to_numpy()
        counts[positions] = selected_count
        rates[positions] = np.divide(
            selected_positive,
            selected_count,
            out=np.zeros_like(selected_positive, dtype=float),
            where=selected_count > 0,
        )
    return pd.DataFrame(
        {
            "row_id": target_df["row_id"].to_numpy(),
            f"user_history_{window_days}d_impressions": counts,
            f"user_history_{window_days}d_rate": rates,
        }
    )


def build_user_history(
    train_df: pd.DataFrame, target_df: pd.DataFrame, ctx: FeatureBuildContext
) -> pd.DataFrame:
    windows = tuple(int(value) for value in ctx.params.get("windows", [1, 7]))
    result = pd.DataFrame({"row_id": target_df["row_id"].to_numpy()})
    for window in windows:
        built = _windowed_user_history(train_df, target_df, ctx, window)
        result = result.merge(built, on="row_id", validate="one_to_one")
    return result


FeatureBuilder = Callable[[pd.DataFrame, pd.DataFrame, FeatureBuildContext], pd.DataFrame]


def _historical_feature_bundle(
    data: DataBundle,
    name: str,
    params: dict[str, Any],
    builder: FeatureBuilder,
    sources: frozenset[str],
) -> FeatureBundle:
    build_ctx = FeatureBuildContext(params=params)
    history = data.frames["train"]
    frames = {
        split: builder(history, target, build_ctx)
        for split, target in data.frames.items()
    }
    for split, frame in frames.items():
        if len(frame) != len(data.frames[split]) or not frame["row_id"].equals(
            data.frames[split]["row_id"]
        ):
            raise RuntimeError(f"feature {name} violated the row-alignment contract on {split}")
    lineage = FeatureLineage(sources, TemporalScope.STRICTLY_EARLIER)
    provenance = {
        column: lineage
        for column in frames["train"].columns
        if column != "row_id"
    }
    return FeatureBundle(name, frames, (), data, provenance)


def _require_data(inputs: list[Any]) -> DataBundle:
    if len(inputs) != 1 or not isinstance(inputs[0], DataBundle):
        raise TypeError("feature operators require one DataBundle")
    return inputs[0]


def op_raw_categorical(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del params, ctx
    data = _require_data(inputs)
    durations = data.frames["train"]["duration_ms"].to_numpy(dtype=float)
    edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
    columns = ("cat_user_id", "cat_video_id", "cat_author_id", "cat_tab", "cat_dur_bucket")
    frames = {}
    for split, source in data.frames.items():
        frames[split] = pd.DataFrame(
            {
                "row_id": source["row_id"].to_numpy(),
                "cat_user_id": source["user_id"].astype(str).to_numpy(),
                "cat_video_id": source["video_id"].astype(str).to_numpy(),
                "cat_author_id": source["author_id"].astype(str).to_numpy(),
                "cat_tab": source["tab"].astype(str).to_numpy(),
                "cat_dur_bucket": np.searchsorted(
                    edges, source["duration_ms"].to_numpy(dtype=float)
                ).astype(str),
            }
        )
    provenance = {
        "cat_user_id": FeatureLineage(frozenset({"user_id"}), TemporalScope.SAME_ROW),
        "cat_video_id": FeatureLineage(frozenset({"video_id"}), TemporalScope.SAME_ROW),
        "cat_author_id": FeatureLineage(frozenset({"author_id"}), TemporalScope.STATIC),
        "cat_tab": FeatureLineage(frozenset({"tab"}), TemporalScope.SAME_ROW),
        "cat_dur_bucket": FeatureLineage(
            frozenset({"duration_ms"}), TemporalScope.STATIC
        ),
    }
    return FeatureBundle("raw_categorical", frames, columns, data, provenance)


def op_item_popularity(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del ctx
    return _historical_feature_bundle(
        _require_data(inputs),
        "item_popularity",
        params,
        build_item_popularity,
        frozenset({"date", "video_id", "long_view"}),
    )


def op_user_category_affinity(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del ctx
    return _historical_feature_bundle(
        _require_data(inputs),
        "user_category_affinity",
        params,
        build_user_category_affinity,
        frozenset({"date", "user_id", "tag", "long_view"}),
    )


def op_user_history(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del ctx
    return _historical_feature_bundle(
        _require_data(inputs),
        "user_history",
        params,
        build_user_history,
        frozenset({"date", "user_id", "long_view"}),
    )


def op_video_duration(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del ctx
    data = _require_data(inputs)
    buckets = int(params.get("buckets", 10))
    train_duration = data.frames["train"]["duration_ms"].to_numpy(dtype=float)
    edges = np.quantile(train_duration, np.linspace(0, 1, buckets + 1)[1:-1])
    frames = {}
    for split, source in data.frames.items():
        duration = source["duration_ms"].to_numpy(dtype=float)
        frames[split] = pd.DataFrame(
            {
                "row_id": source["row_id"].to_numpy(),
                "video_duration_log_ms": np.log1p(duration).astype(np.float32),
                "cat_video_duration_bucket": np.searchsorted(edges, duration).astype(str),
            }
        )
    provenance = {
        "video_duration_log_ms": FeatureLineage(
            frozenset({"duration_ms"}), TemporalScope.STATIC
        ),
        "cat_video_duration_bucket": FeatureLineage(
            frozenset({"duration_ms"}), TemporalScope.STATIC
        ),
    }
    return FeatureBundle(
        "video_duration", frames, ("cat_video_duration_bucket",), data, provenance
    )


def op_temporal(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del params, ctx
    data = _require_data(inputs)
    train_start = pd.Timestamp("2022-04-08")
    frames = {}
    for split, source in data.frames.items():
        dates = pd.to_datetime(source["date"].astype(str), format="%Y%m%d")
        hourmin = source["hourmin"].to_numpy(dtype=int)
        frames[split] = pd.DataFrame(
            {
                "row_id": source["row_id"].to_numpy(),
                "temporal_day_index": (dates - train_start).dt.days.to_numpy(dtype=np.float32),
                "cat_temporal_weekday": dates.dt.weekday.astype(str).to_numpy(),
                "cat_temporal_hour": (hourmin // 100).astype(str),
            }
        )
    provenance = {
        "temporal_day_index": FeatureLineage(
            frozenset({"date"}), TemporalScope.SAME_ROW
        ),
        "cat_temporal_weekday": FeatureLineage(
            frozenset({"date"}), TemporalScope.SAME_ROW
        ),
        "cat_temporal_hour": FeatureLineage(
            frozenset({"hourmin"}), TemporalScope.SAME_ROW
        ),
    }
    return FeatureBundle(
        "temporal",
        frames,
        ("cat_temporal_weekday", "cat_temporal_hour"),
        data,
        provenance,
    )


def op_generated_feature(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    """Execute the one permitted free-form surface through the fixed build contract."""
    del ctx
    data = _require_data(inputs)
    module_path = Path(str(params["module_path"])).resolve()
    module = _load_module(module_path, f"techjam_generated_{module_path.stem}")
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise TypeError("generated feature module must expose callable build")
    build_ctx = FeatureBuildContext(params=dict(params.get("builder_params", {})))
    history = data.frames["train"]
    frames = {
        split: builder(history, target, build_ctx)
        for split, target in data.frames.items()
    }
    for split, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("generated build must return a pandas DataFrame")
        if "row_id" not in frame or len(frame) != len(data.frames[split]):
            raise RuntimeError(f"generated feature violated row count on {split}")
        if not frame["row_id"].reset_index(drop=True).equals(
            data.frames[split]["row_id"].reset_index(drop=True)
        ):
            raise RuntimeError(f"generated feature violated row alignment on {split}")
    scope = TemporalScope(str(params["temporal_scope"]))
    source_log = SourceLog(str(params.get("source_log", "standard")))
    max_source_date = params.get("max_source_date")
    lineage = FeatureLineage(
        frozenset(str(value) for value in params["sources"]),
        scope,
        source_log,
        int(max_source_date) if max_source_date is not None else None,
    )
    provenance = {
        column: lineage for column in frames["train"].columns if column != "row_id"
    }
    categorical = tuple(str(value) for value in params.get("categorical_columns", []))
    return FeatureBundle(module_path.stem, frames, categorical, data, provenance)


def op_ablation_constant_feature(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    del params, ctx
    data = _require_data(inputs)
    frames = {
        split: pd.DataFrame(
            {
                "row_id": source["row_id"].to_numpy(),
                "ablation_constant": np.zeros(len(source), dtype=np.float32),
            }
        )
        for split, source in data.frames.items()
    }
    provenance = {
        "ablation_constant": FeatureLineage(frozenset(), TemporalScope.STATIC)
    }
    return FeatureBundle("ablation_constant", frames, (), data, provenance)
