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


LONG_VIEW_HINGE_MS = 18_000
"""Knee in the long_view label definition.

The native label behaves as ``play_time >= min(duration, 18s)``: a video at or
under eighteen seconds must be watched essentially to the end, while a longer
one only needs eighteen seconds. Probing the training split, that rule
reproduces the label for 97.81% of rows, and eighteen seconds is a sharp
maximum (fifteen gives 96.09%, twenty gives 96.70%). Quantile duration bins do
not place a boundary there, so the piecewise structure is encoded explicitly.
Only ``duration_ms``, a static video property, is used; ``play_time_ms`` is an
impression outcome and is never read as a feature.
"""


def op_duration_hinge(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    """Piecewise duration features aligned to the long_view definition."""
    del ctx
    data = _require_data(inputs)
    hinge = float(params.get("hinge_ms", LONG_VIEW_HINGE_MS))
    edges_s = [7.0, 12.0, 18.0, 30.0, 60.0]

    frames: dict[str, pd.DataFrame] = {}
    for split, source in data.frames.items():
        duration = source["duration_ms"].to_numpy(dtype=float)
        seconds = duration / 1000.0
        below = np.minimum(duration, hinge) / 1000.0
        above = np.maximum(duration - hinge, 0.0) / 1000.0
        frames[split] = pd.DataFrame(
            {
                "row_id": source["row_id"].to_numpy(),
                # Watch requirement implied by the label, in seconds.
                "dur_hinge_below_s": below.astype(np.float32),
                # Excess length beyond the requirement; compressed, it is heavy tailed.
                "dur_hinge_above_log_s": np.log1p(above).astype(np.float32),
                "dur_hinge_ratio": (below / np.maximum(seconds, 1e-6)).astype(np.float32),
                "cat_dur_regime": np.where(duration <= hinge, "short", "long"),
                "cat_dur_band": np.searchsorted(edges_s, seconds).astype(str),
            }
        )
    lineage = FeatureLineage(frozenset({"duration_ms"}), TemporalScope.STATIC)
    provenance = {
        column: lineage for column in frames["train"].columns if column != "row_id"
    }
    return FeatureBundle(
        "duration_hinge",
        frames,
        ("cat_dur_regime", "cat_dur_band"),
        data,
        provenance,
    )


USER_ATTRIBUTE_CHOICES = (
    "duration_bucket",
    "duration_regime",
    "tab",
    "tag",
    "author_id",
)


def _attribute_column(
    frame: pd.DataFrame, attribute: str, edges: np.ndarray | None
) -> pd.Series:
    if attribute == "duration_bucket":
        assert edges is not None
        return pd.Series(
            np.searchsorted(edges, frame["duration_ms"].to_numpy(dtype=float)),
            index=frame.index,
        ).astype(str)
    if attribute == "duration_regime":
        # Split at the long_view hinge rather than at a quantile: the question is
        # whether this user finishes short clips or stays with long ones.
        duration = frame["duration_ms"].to_numpy(dtype=float)
        return pd.Series(
            np.where(duration <= LONG_VIEW_HINGE_MS, "short", "long"), index=frame.index
        )
    return frame[attribute].astype(str)


def op_user_attribute_affinity(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    """Historical long-view rate for this user against this video's attribute.

    The scored metric ranks within a user, so a feature that is constant across a
    user's rows cannot change their ordering. This operator deliberately builds
    user-by-attribute statistics, which vary within a user because the video
    attribute varies, and are therefore able to move the ranking.
    """
    del ctx
    data = _require_data(inputs)
    attribute = str(params.get("attribute", "duration_bucket"))
    if attribute not in USER_ATTRIBUTE_CHOICES:
        raise ValueError(
            f"unsupported attribute {attribute!r}; choose from {list(USER_ATTRIBUTE_CHOICES)}"
        )
    edges: np.ndarray | None = None
    if attribute == "duration_bucket":
        buckets = int(params.get("buckets", 10))
        train_duration = data.frames["train"]["duration_ms"].to_numpy(dtype=float)
        edges = np.quantile(train_duration, np.linspace(0, 1, buckets + 1)[1:-1])

    prefix = f"user_{attribute}_affinity"

    def builder(
        train_df: pd.DataFrame, target_df: pd.DataFrame, build_ctx: FeatureBuildContext
    ) -> pd.DataFrame:
        history = train_df.copy()
        target = target_df.copy()
        history["_attribute"] = _attribute_column(history, attribute, edges)
        target["_attribute"] = _attribute_column(target, attribute, edges)
        return _expanding_rate(
            history, target, build_ctx, ("user_id", "_attribute"), prefix
        )

    sources = frozenset({"date", "user_id", "long_view"}) | (
        frozenset({"duration_ms"})
        if attribute in ("duration_bucket", "duration_regime")
        else frozenset({attribute})
    )
    return _historical_feature_bundle(
        data, prefix, params, builder, sources
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


STATIC_USER_CATEGORICAL = (
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *(f"onehot_feat{index}" for index in range(18)),
)
STATIC_USER_NUMERIC = (
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
)
STATIC_VIDEO_CATEGORICAL = ("video_type", "upload_type", "music_type", "upload_dt")
STATIC_VIDEO_NUMERIC = ("server_width", "server_height", "video_duration")
# Columns held out on purpose: constant within KuaiRand-Pure (is_lowactive_period,
# visible_status), already supplied by features.raw_categorical (author_id, tag), or
# effectively a row identifier that would memorize rather than generalize (music_id).


def op_static_side_features(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    """Join the static user profile and video metadata side files.

    Both files describe entities rather than interactions, so every column is
    STATIC and carries SIDE_FILE lineage. Nothing here depends on the split
    date, which is what makes the join safe for valid and test rows alike.
    """
    del ctx
    data = _require_data(inputs)
    log_numeric = frozenset(
        {"follow_user_num", "fans_user_num", "friend_user_num", "register_days"}
    )

    users = pd.read_csv(
        data.data_dir / "user_features_pure.csv",
        usecols=["user_id", *STATIC_USER_CATEGORICAL, *STATIC_USER_NUMERIC],
    )
    users["user_id"] = users["user_id"].astype(str)
    videos = pd.read_csv(
        data.data_dir / "video_features_basic_pure.csv",
        usecols=["video_id", *STATIC_VIDEO_CATEGORICAL, *STATIC_VIDEO_NUMERIC],
    )
    videos["video_id"] = videos["video_id"].astype(str)

    user_lookup = users.set_index("user_id")
    video_lookup = videos.set_index("video_id")
    categorical_columns: list[str] = []
    frames: dict[str, pd.DataFrame] = {}

    for split, source in data.frames.items():
        frame = pd.DataFrame({"row_id": source["row_id"].to_numpy()})
        key_users = source["user_id"].astype(str)
        key_videos = source["video_id"].astype(str)
        for column in STATIC_USER_CATEGORICAL:
            name = f"cat_su_{column}"
            frame[name] = (
                key_users.map(user_lookup[column]).astype("string").fillna("UNK").to_numpy()
            )
        for column in STATIC_USER_NUMERIC:
            name = f"su_{column}"
            values = pd.to_numeric(key_users.map(user_lookup[column]), errors="coerce")
            if column in log_numeric:
                # Follower/fan counts are heavy-tailed; compress before the models see them.
                values = np.log1p(values.clip(lower=0))
                name = f"su_log_{column}"
            frame[name] = values.astype(np.float32).to_numpy()
        for column in STATIC_VIDEO_CATEGORICAL:
            name = f"cat_sv_{column}"
            frame[name] = (
                key_videos.map(video_lookup[column]).astype("string").fillna("UNK").to_numpy()
            )
        for column in STATIC_VIDEO_NUMERIC:
            values = pd.to_numeric(key_videos.map(video_lookup[column]), errors="coerce")
            if column == "video_duration":
                frame["sv_log_video_duration"] = np.log1p(
                    values.clip(lower=0)
                ).astype(np.float32).to_numpy()
            else:
                frame[f"sv_{column}"] = values.astype(np.float32).to_numpy()
        width = pd.to_numeric(key_videos.map(video_lookup["server_width"]), errors="coerce")
        height = pd.to_numeric(key_videos.map(video_lookup["server_height"]), errors="coerce")
        frame["sv_aspect_ratio"] = (
            (height / width.replace(0, np.nan)).astype(np.float32).to_numpy()
        )
        frames[split] = frame
        if not categorical_columns:
            categorical_columns = [
                name for name in frame.columns if name.startswith("cat_")
            ]

    lineage_user = FeatureLineage(
        frozenset({"user_features_pure"}), TemporalScope.STATIC, SourceLog.SIDE_FILE
    )
    lineage_video = FeatureLineage(
        frozenset({"video_features_basic_pure"}), TemporalScope.STATIC, SourceLog.SIDE_FILE
    )
    provenance = {
        column: (lineage_user if "su_" in column else lineage_video)
        for column in frames["train"].columns
        if column != "row_id"
    }
    return FeatureBundle(
        "static_side_features",
        frames,
        tuple(categorical_columns),
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
    scope = TemporalScope(str(params["temporal_scope"]))
    frames: dict[str, pd.DataFrame] = {}
    for split, target in data.frames.items():
        if scope == TemporalScope.STRICTLY_EARLIER:
            pieces = []
            for date in sorted(pd.unique(target[build_ctx.date_column])):
                target_piece = target.loc[target[build_ctx.date_column] == date]
                eligible_history = history.loc[history[build_ctx.date_column] < date]
                pieces.append(builder(eligible_history, target_piece, build_ctx))
            built = pd.concat(pieces, ignore_index=True)
            expected_ids = target["row_id"].tolist()
            if "row_id" not in built or built["row_id"].duplicated().any():
                raise RuntimeError(f"generated feature produced invalid row_id on {split}")
            frames[split] = built.set_index("row_id").reindex(expected_ids).reset_index()
        else:
            frames[split] = builder(history, target, build_ctx)
    for split, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("generated build must return a pandas DataFrame")
        if "row_id" not in frame or len(frame) != len(data.frames[split]):
            raise RuntimeError(f"generated feature violated row count on {split}")
        if not frame["row_id"].reset_index(drop=True).equals(
            data.frames[split]["row_id"].reset_index(drop=True)
        ):
            raise RuntimeError(f"generated feature violated row alignment on {split}")
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


DECAY_ATTRIBUTE_CHOICES = (
    "author_id",
    "tag",
    "tab",
    "duration_regime",
    "duration_bucket",
    "video_id",
)


def _decayed_affinity_frame(
    history: pd.DataFrame,
    target: pd.DataFrame,
    *,
    attribute: str,
    edges: np.ndarray | None,
    half_life_days: float,
    alpha: float,
    centered: bool,
    prefix: str,
) -> pd.DataFrame:
    """Exponentially decayed, shrunk user-by-attribute long-view rate.

        A_ua(t) = (sum_{j<t} w(t-t_j) y_j + alpha * p_a) / (sum_{j<t} w(t-t_j) + alpha)
        w(dt)   = 2 ** (-dt / half_life)

    Only rows strictly earlier than the target date contribute, and the decayed
    accumulators are advanced date by date, so every value is causal by
    construction rather than by a later filtering step.

    With ``centered``, the feature becomes a lift against the same user's own
    decayed base rate, in logit space. That matters here because GAUC never
    compares one user against another: what carries signal is whether *this*
    attribute is better than the user's own alternatives.
    """
    date_column, label_column = "date", "long_view"
    hist = history.copy()
    tgt = target.copy()
    hist["_attr"] = _attribute_column(hist, attribute, edges)
    tgt["_attr"] = _attribute_column(tgt, attribute, edges)
    hist["_user"] = hist["user_id"].astype(str)
    tgt["_user"] = tgt["user_id"].astype(str)

    daily = (
        hist.groupby([date_column, "_user", "_attr"], sort=True, dropna=False)[label_column]
        .agg(["sum", "count"]).reset_index()
    )
    daily_attr = (
        hist.groupby([date_column, "_attr"], sort=True, dropna=False)[label_column]
        .agg(["sum", "count"]).reset_index()
    )
    daily_user = (
        hist.groupby([date_column, "_user"], sort=True, dropna=False)[label_column]
        .agg(["sum", "count"]).reset_index()
    )

    def as_day(values):
        # pandas may return second, microsecond, or nanosecond resolution here, so
        # subtract the epoch and take whole days rather than dividing raw integers.
        stamps = pd.to_datetime(values.astype(str), format="%Y%m%d")
        return (stamps - pd.Timestamp("1970-01-01")).dt.days

    for frame in (daily, daily_attr, daily_user):
        frame["_day"] = as_day(frame[date_column])
    target_days = as_day(tgt[date_column]).to_numpy()

    pair_w: dict[tuple, float] = {}
    pair_y: dict[tuple, float] = {}
    attr_w: dict[str, float] = {}
    attr_y: dict[str, float] = {}
    user_w: dict[str, float] = {}
    user_y: dict[str, float] = {}

    rec_pair = daily.sort_values("_day").itertuples(index=False, name=None)
    rec_attr = daily_attr.sort_values("_day").itertuples(index=False, name=None)
    rec_user = daily_user.sort_values("_day").itertuples(index=False, name=None)
    lists = {k: list(v) for k, v in
             (("pair", rec_pair), ("attr", rec_attr), ("user", rec_user))}
    ptr = {"pair": 0, "attr": 0, "user": 0}

    out_rate = np.zeros(len(tgt), dtype=np.float32)
    out_weight = np.zeros(len(tgt), dtype=np.float32)
    current_day = None

    def decay_all(factor: float) -> None:
        for store in (pair_w, pair_y, attr_w, attr_y, user_w, user_y):
            for key in store:
                store[key] *= factor

    for day in sorted(np.unique(target_days)):
        if current_day is None:
            current_day = day
        elif day > current_day:
            decay_all(2.0 ** (-(day - current_day) / half_life_days))
            current_day = day
        # Absorb every history day strictly earlier than the target day.
        for name, (wstore, ystore, keyidx) in (
            ("pair", (pair_w, pair_y, (1, 2))),
            ("attr", (attr_w, attr_y, (1,))),
            ("user", (user_w, user_y, (1,))),
        ):
            records = lists[name]
            while ptr[name] < len(records) and records[ptr[name]][-1] < day:
                rec = records[ptr[name]]
                gap = day - rec[-1]
                weight = 2.0 ** (-gap / half_life_days)
                key = rec[keyidx[0]] if len(keyidx) == 1 else (rec[keyidx[0]], rec[keyidx[1]])
                positives, count = float(rec[-3]), float(rec[-2])
                wstore[key] = wstore.get(key, 0.0) + weight * count
                ystore[key] = ystore.get(key, 0.0) + weight * positives
                ptr[name] += 1

        positions = np.flatnonzero(target_days == day)
        users_at = tgt["_user"].to_numpy()
        attrs_at = tgt["_attr"].to_numpy()
        global_w = sum(attr_w.values()) or 0.0
        global_y = sum(attr_y.values()) or 0.0
        global_rate = global_y / global_w if global_w else 0.0
        for pos in positions:
            u, a = users_at[pos], attrs_at[pos]
            prior_w, prior_y = attr_w.get(a, 0.0), attr_y.get(a, 0.0)
            prior = prior_y / prior_w if prior_w else global_rate
            w = pair_w.get((u, a), 0.0)
            yv = pair_y.get((u, a), 0.0)
            rate = (yv + alpha * prior) / (w + alpha) if (w + alpha) else prior
            if centered:
                uw, uy = user_w.get(u, 0.0), user_y.get(u, 0.0)
                base = (uy + alpha * global_rate) / (uw + alpha) if (uw + alpha) else global_rate
                eps = 1e-6
                rate = float(
                    np.log(np.clip(rate, eps, 1 - eps) / (1 - np.clip(rate, eps, 1 - eps)))
                    - np.log(np.clip(base, eps, 1 - eps) / (1 - np.clip(base, eps, 1 - eps)))
                )
            out_rate[pos] = rate
            out_weight[pos] = w

    suffix = "lift" if centered else "rate"
    return pd.DataFrame({
        "row_id": tgt["row_id"].to_numpy(),
        f"{prefix}_{suffix}": out_rate,
        f"{prefix}_weight": out_weight,
    })


def op_decayed_affinity(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> FeatureBundle:
    """Time-decayed, shrunk, optionally user-centred affinity for one attribute."""
    del ctx
    data = _require_data(inputs)
    attribute = str(params.get("attribute", "author_id"))
    if attribute not in DECAY_ATTRIBUTE_CHOICES:
        raise ValueError(
            f"unsupported attribute {attribute!r}; choose from {list(DECAY_ATTRIBUTE_CHOICES)}"
        )
    half_life = float(params.get("half_life_days", 7.0))
    if half_life <= 0:
        raise ValueError("half_life_days must be positive")
    alpha = float(params.get("alpha", 20.0))
    centered = bool(params.get("centered", True))

    edges: np.ndarray | None = None
    if attribute == "duration_bucket":
        buckets = int(params.get("buckets", 10))
        train_duration = data.frames["train"]["duration_ms"].to_numpy(dtype=float)
        edges = np.quantile(train_duration, np.linspace(0, 1, buckets + 1)[1:-1])

    prefix = f"decay_{attribute}_h{half_life:g}"
    history = data.frames["train"]
    frames = {
        split: _decayed_affinity_frame(
            history, target, attribute=attribute, edges=edges,
            half_life_days=half_life, alpha=alpha, centered=centered, prefix=prefix,
        )
        for split, target in data.frames.items()
    }
    for split, frame in frames.items():
        if not frame["row_id"].equals(data.frames[split]["row_id"]):
            raise RuntimeError(f"decayed affinity violated row alignment on {split}")
    sources = frozenset({"date", "user_id", "long_view"}) | (
        frozenset({"duration_ms"})
        if attribute in ("duration_bucket", "duration_regime")
        else frozenset({attribute})
    )
    lineage = FeatureLineage(sources, TemporalScope.STRICTLY_EARLIER)
    provenance = {c: lineage for c in frames["train"].columns if c != "row_id"}
    return FeatureBundle(prefix, frames, (), data, provenance)
