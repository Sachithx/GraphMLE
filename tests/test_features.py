from __future__ import annotations

import pandas as pd

from pipeline.ops_features import FeatureBuildContext, build_item_popularity


def test_item_popularity_uses_strictly_earlier_dates() -> None:
    history = pd.DataFrame(
        {
            "date": [1, 2],
            "video_id": ["a", "a"],
            "long_view": [1, 0],
        }
    )
    target = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "date": [1, 2, 3],
            "video_id": ["a", "a", "a"],
        }
    )
    ctx = FeatureBuildContext(params={"smoothing": 0.0})

    built = build_item_popularity(history, target, ctx)

    assert built["item_pop_impressions"].tolist() == [0.0, 1.0, 2.0]
    assert built["item_pop_rate"].tolist() == [0.0, 1.0, 0.5]


def test_same_date_labels_do_not_change_features() -> None:
    history = pd.DataFrame(
        {
            "date": [1, 1, 2],
            "video_id": ["a", "a", "a"],
            "long_view": [0, 1, 0],
        }
    )
    target = pd.DataFrame({"row_id": [0], "date": [1], "video_id": ["a"]})
    ctx = FeatureBuildContext(params={"smoothing": 0.0})

    built = build_item_popularity(history, target, ctx)

    assert built.loc[0, "item_pop_impressions"] == 0.0
    assert built.loc[0, "item_pop_rate"] == 0.0


def test_static_side_features_are_row_aligned_and_leakage_clean(tmp_path) -> None:
    """The side-file join must stay static, aligned, and free of outcome columns."""
    import numpy as np

    from guards.leakage import LeakageGuard
    from pipeline.ops_features import op_static_side_features
    from pipeline.types import (
        DataBundle,
        ExecutionContext,
        SourceLog,
        TemporalScope,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "user_id": [0, 1],
            "user_active_degree": ["full_active", "low_active"],
            "is_live_streamer": [1, 0],
            "is_video_author": [1, 0],
            "follow_user_num": [10, 20],
            "follow_user_num_range": ["(0,10]", "500+"],
            "fans_user_num": [1, 2],
            "fans_user_num_range": ["[10,100)", "500+"],
            "friend_user_num": [3, 4],
            "friend_user_num_range": ["[1,5)", "[30,60)"],
            "register_days": [100, 200],
            "register_days_range": ["181-365", "730+"],
            **{f"onehot_feat{index}": [index, index + 1] for index in range(18)},
        }
    ).to_csv(data_dir / "user_features_pure.csv", index=False)
    pd.DataFrame(
        {
            "video_id": [7, 8],
            "video_type": ["NORMAL", "AD"],
            "upload_dt": ["2022-04-10", "2022-04-09"],
            "upload_type": ["Web", "Kmovie"],
            "video_duration": [1000.0, 2000.0],
            "server_width": [720.0, 1080.0],
            "server_height": [1280.0, 1920.0],
            "music_type": [9.0, 4.0],
        }
    ).to_csv(data_dir / "video_features_basic_pure.csv", index=False)

    frames = {
        split: pd.DataFrame(
            {
                "row_id": np.arange(2, dtype=np.int64),
                "user_id": [0, 1],
                "video_id": [7, 8],
                "long_view": [1, 0],
            }
        )
        for split in ("train", "valid", "test")
    }
    data = DataBundle(frames, {}, data_dir, tmp_path / "kit")
    ctx = ExecutionContext(data_dir, tmp_path, tmp_path / "out")

    bundle = op_static_side_features([data], {}, ctx)

    for split, frame in bundle.frames.items():
        assert frame["row_id"].equals(frames[split]["row_id"]), split
    # Every column must be declared static side-file data, never an outcome column.
    assert bundle.provenance
    for lineage in bundle.provenance.values():
        assert lineage.temporal_scope is TemporalScope.STATIC
        assert lineage.source_log is SourceLog.SIDE_FILE
    assert "long_view" not in bundle.frames["train"].columns
    LeakageGuard(tmp_path / "log.jsonl").check_feature_bundle(bundle, "static_side")


def test_static_side_features_survive_unknown_ids(tmp_path) -> None:
    """Ids absent from the side files must fall back, not drop or misalign rows."""
    import numpy as np

    from pipeline.ops_features import op_static_side_features
    from pipeline.types import DataBundle, ExecutionContext

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "user_id": [0],
            "user_active_degree": ["full_active"],
            "is_live_streamer": [1],
            "is_video_author": [1],
            "follow_user_num": [10],
            "follow_user_num_range": ["(0,10]"],
            "fans_user_num": [1],
            "fans_user_num_range": ["[10,100)"],
            "friend_user_num": [3],
            "friend_user_num_range": ["[1,5)"],
            "register_days": [100],
            "register_days_range": ["181-365"],
            **{f"onehot_feat{index}": [index] for index in range(18)},
        }
    ).to_csv(data_dir / "user_features_pure.csv", index=False)
    pd.DataFrame(
        {
            "video_id": [7],
            "video_type": ["NORMAL"],
            "upload_dt": ["2022-04-10"],
            "upload_type": ["Web"],
            "video_duration": [1000.0],
            "server_width": [720.0],
            "server_height": [1280.0],
            "music_type": [9.0],
        }
    ).to_csv(data_dir / "video_features_basic_pure.csv", index=False)

    # user 99 and video 99 appear in the log but not in either side file.
    frames = {
        split: pd.DataFrame(
            {
                "row_id": np.arange(2, dtype=np.int64),
                "user_id": [0, 99],
                "video_id": [7, 99],
                "long_view": [1, 0],
            }
        )
        for split in ("train", "valid", "test")
    }
    data = DataBundle(frames, {}, data_dir, tmp_path / "kit")
    ctx = ExecutionContext(data_dir, tmp_path, tmp_path / "out")

    bundle = op_static_side_features([data], {}, ctx)

    train = bundle.frames["train"]
    assert len(train) == 2
    assert train["cat_su_user_active_degree"].tolist() == ["full_active", "UNK"]
    assert train["cat_sv_video_type"].tolist() == ["NORMAL", "UNK"]


def test_decayed_affinity_is_causal_and_recency_weighted() -> None:
    """Decayed affinity must never see the target day, and must favour recency.

    The day arithmetic is the fragile part: pandas may hand back second,
    microsecond, or nanosecond resolution for the same input, so the day index is
    asserted behaviourally rather than trusted.
    """
    from pipeline.ops_features import _decayed_affinity_frame

    history = pd.DataFrame({
        "date": [20220408, 20220409],
        "user_id": ["u0", "u0"],
        "author_id": ["a", "a"],
        "long_view": [1, 0],
        "duration_ms": [1000, 1000],
    })
    target = pd.DataFrame({
        "row_id": [0, 1, 2],
        "date": [20220408, 20220409, 20220410],
        "user_id": ["u0", "u0", "u0"],
        "author_id": ["a", "a", "a"],
        "duration_ms": [1000, 1000, 1000],
    })

    def build(half_life: float):
        return _decayed_affinity_frame(
            history, target, attribute="author_id", edges=None,
            half_life_days=half_life, alpha=0.0, centered=False, prefix="d",
        )

    slow = build(7.0)
    # The first target day has no strictly-earlier history at all.
    assert slow["d_weight"][0] == 0.0
    # The second sees only the first day, which was a positive.
    assert abs(slow["d_rate"][1] - 1.0) < 1e-9
    # The third sees one positive and one negative, so it must lie strictly between.
    assert 0.0 < slow["d_rate"][2] < 1.0

    # A short half-life discounts the older positive, pulling the rate down.
    fast = build(0.5)
    assert fast["d_rate"][2] < slow["d_rate"][2]


def test_decayed_affinity_centering_cancels_the_user_base_rate() -> None:
    """With one author, the lift against the user's own rate must vanish."""
    from pipeline.ops_features import _decayed_affinity_frame

    history = pd.DataFrame({
        "date": [20220408, 20220409],
        "user_id": ["u0", "u0"],
        "author_id": ["a", "a"],
        "long_view": [1, 0],
        "duration_ms": [1000, 1000],
    })
    target = pd.DataFrame({
        "row_id": [0], "date": [20220410], "user_id": ["u0"],
        "author_id": ["a"], "duration_ms": [1000],
    })
    built = _decayed_affinity_frame(
        history, target, attribute="author_id", edges=None,
        half_life_days=7.0, alpha=5.0, centered=True, prefix="d",
    )
    assert abs(built["d_lift"][0]) < 1e-9
