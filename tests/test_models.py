from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ops_features import AUXILIARY_TARGETS
from pipeline.ops_models import _encode_fm_features, op_torch_deepfm, op_torch_multitask
from pipeline.types import DataBundle, ExecutionContext, FeatureBundle


def synthetic_bundle(tmp_path: Path) -> tuple[FeatureBundle, ExecutionContext]:
    sizes = {"train": 32, "valid": 8, "test": 8}
    data_frames = {}
    feature_frames = {}
    for split, size in sizes.items():
        labels = np.arange(size) % 2
        data = {
            "row_id": np.arange(size),
            "user_id": (np.arange(size) // 4).astype(str),
            "long_view": labels,
        }
        data.update({column: (labels + index) % 2 for index, column in enumerate(AUXILIARY_TARGETS)})
        data_frames[split] = pd.DataFrame(data)
        feature_frames[split] = pd.DataFrame(
            {
                "row_id": np.arange(size),
                "cat_item": (np.arange(size) % 5).astype(str),
                "numeric": np.linspace(0.0, 1.0, size),
            }
        )
    data_bundle = DataBundle(data_frames, {}, tmp_path, tmp_path)
    feature_bundle = FeatureBundle(
        "synthetic", feature_frames, ("cat_item",), data_bundle
    )
    context = ExecutionContext(tmp_path, tmp_path, tmp_path / "out")
    return feature_bundle, context


def test_torch_model_nodes_return_aligned_finite_scores(tmp_path: Path) -> None:
    bundle, context = synthetic_bundle(tmp_path)
    params = {
        "epochs": 1,
        "batch_size": 8,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "threads": 1,
    }

    deepfm = op_torch_deepfm([bundle], params, context)
    multitask = op_torch_multitask([bundle], params, context)

    for prediction in (deepfm, multitask):
        for split, frame in bundle.frames.items():
            assert prediction.scores[split].shape == (len(frame),)
            assert np.isfinite(prediction.scores[split]).all()


def test_fm_encoding_consumes_categorical_and_numeric_feature_columns(tmp_path: Path) -> None:
    bundle, _ = synthetic_bundle(tmp_path)
    encoded, dimension = _encode_fm_features([bundle], numeric_bins=4)

    assert dimension > 1
    assert encoded["train"][0].shape == (32, 2)
    assert np.unique(encoded["train"][0][:, 0]).size == 5
    assert np.unique(encoded["train"][0][:, 1]).size == 4
    assert encoded["valid"][0].shape == (8, 2)


def _setwise_fixture(tmp_path):
    """Two users, three impressions each, with mixed labels so sets are discriminative."""
    import numpy as np
    import pandas as pd

    from pipeline.types import DataBundle, FeatureBundle, FeatureLineage, TemporalScope

    rows = 6
    frames = {}
    for split in ("train", "valid", "test"):
        frames[split] = pd.DataFrame(
            {
                "row_id": np.arange(rows, dtype=np.int64),
                "user_id": ["u0", "u0", "u0", "u1", "u1", "u1"],
                "video_id": ["a", "b", "c", "a", "b", "c"],
                "date": [20220408] * 3 + [20220408] * 3,
                "long_view": [1, 0, 0, 0, 1, 0],
            }
        )
    data = DataBundle(frames, {}, tmp_path, tmp_path)
    lineage = FeatureLineage(frozenset({"video_id"}), TemporalScope.SAME_ROW)
    feature_frames = {
        split: pd.DataFrame(
            {
                "row_id": np.arange(rows, dtype=np.int64),
                "cat_video_id": ["a", "b", "c", "a", "b", "c"],
                "numeric": np.arange(rows, dtype=np.float32),
            }
        )
        for split in ("train", "valid", "test")
    }
    bundle = FeatureBundle(
        "fixture", feature_frames, ("cat_video_id",), data,
        {"cat_video_id": lineage, "numeric": lineage},
    )
    return data, bundle


def test_setwise_rank_returns_aligned_finite_scores(tmp_path) -> None:
    import numpy as np

    from pipeline.ops_models import op_setwise_rank
    from pipeline.types import ExecutionContext

    data, bundle = _setwise_fixture(tmp_path)
    ctx = ExecutionContext(tmp_path, tmp_path, tmp_path / "out")

    # No evaluate.py in the fixture, so this also exercises the no-checkpoint path.
    pred = op_setwise_rank([bundle], {"epochs": 2, "device": "cpu", "d_model": 16}, ctx)

    for split, frame in data.frames.items():
        assert len(pred.scores[split]) == len(frame)
        assert np.isfinite(pred.scores[split]).all()


def test_setwise_scoring_depends_on_the_rest_of_the_set(tmp_path) -> None:
    """The defining property: an item's score must reflect what it competes against.

    Rows 0 and 3 carry byte-identical features but sit in different user sets. Any
    univariate scorer must give them the same score; a setwise one must not. This is
    what distinguishes this operator from every other model in the registry, so it
    is asserted directly rather than inferred from a metric.
    """
    import numpy as np
    import pandas as pd

    from pipeline.ops_models import op_setwise_rank
    from pipeline.types import (
        DataBundle,
        ExecutionContext,
        FeatureBundle,
        FeatureLineage,
        TemporalScope,
    )

    # u0 sees {a, b, c}; u1 sees {a, d, e}. Item "a" is identical in both.
    frames = {
        split: pd.DataFrame(
            {
                "row_id": np.arange(6, dtype=np.int64),
                "user_id": ["u0", "u0", "u0", "u1", "u1", "u1"],
                "video_id": ["a", "b", "c", "a", "d", "e"],
                "date": [20220408] * 6,
                "long_view": [1, 0, 0, 0, 1, 0],
            }
        )
        for split in ("train", "valid", "test")
    }
    data = DataBundle(frames, {}, tmp_path, tmp_path)
    lineage = FeatureLineage(frozenset({"video_id"}), TemporalScope.SAME_ROW)
    feature_frames = {
        split: pd.DataFrame(
            {
                "row_id": np.arange(6, dtype=np.int64),
                "cat_video_id": ["a", "b", "c", "a", "d", "e"],
                "numeric": np.array([1.0, 2.0, 3.0, 1.0, 9.0, 9.0], dtype=np.float32),
            }
        )
        for split in ("train", "valid", "test")
    }
    bundle = FeatureBundle(
        "fixture", feature_frames, ("cat_video_id",), data,
        {"cat_video_id": lineage, "numeric": lineage},
    )
    ctx = ExecutionContext(tmp_path, tmp_path, tmp_path / "out")

    pred = op_setwise_rank(
        [bundle], {"epochs": 3, "device": "cpu", "d_model": 16, "seed": 0}, ctx
    )
    scores = pred.scores["train"]

    assert np.isfinite(scores).all()
    # Identical features, different competition -> different score.
    assert scores[0] != scores[3]
