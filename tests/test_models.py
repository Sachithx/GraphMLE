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
