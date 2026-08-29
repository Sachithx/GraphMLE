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
