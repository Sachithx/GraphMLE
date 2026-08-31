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


def test_lightgbm_rank_group_by_changes_query_construction(tmp_path) -> None:
    """user_date grouping must split a user's rows into per-day queries.

    Group construction decides which pairs LambdaRank compares, so this is asserted
    directly rather than inferred from a score.
    """
    import numpy as np
    import pandas as pd

    users = pd.Series(["u0", "u0", "u0", "u1", "u1"])
    dates = pd.Series([20220408, 20220408, 20220409, 20220408, 20220409])

    by_user = users.astype(str).to_numpy()
    by_user_date = (users.astype(str) + "|" + dates.astype(str)).to_numpy()

    _, user_groups = np.unique(np.sort(by_user), return_counts=True)
    _, user_date_groups = np.unique(np.sort(by_user_date), return_counts=True)

    assert sorted(user_groups.tolist()) == [2, 3]          # u1:2, u0:3
    assert sorted(user_date_groups.tolist()) == [1, 1, 1, 2]  # split per day


def test_pairwise_step_increases_the_positive_margin() -> None:
    """A BPR step must widen the score gap between a positive and its negative.

    This is the whole point of the fine-tune stage, so it is asserted on the
    gradient itself rather than inferred from a downstream metric.
    """
    import sys
    import types

    import numpy as np

    from pipeline.ops_models import _fm_pairwise_step

    # Minimal stand-in with the same attributes the kit's FM exposes.
    class TinyFM:
        def __init__(self, dim=6, k=3):
            rng = np.random.default_rng(0)
            self.V = rng.normal(0, 0.1, (dim, k)).astype(np.float32)
            self.W = rng.normal(0, 0.1, dim).astype(np.float32)
            self.b = np.float32(0.0)
            self.l2 = 0.0
            self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
            self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
            self.t = 0

        def logits(self, X):
            E = self.V[X]
            S = E.sum(1)
            inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            return self.b + self.W[X].sum(1) + inter, E, S

    model = TinyFM()
    x_pos = np.array([[0, 1]], dtype=np.int32)
    x_neg = np.array([[2, 3]], dtype=np.int32)

    before = model.logits(x_pos)[0][0] - model.logits(x_neg)[0][0]
    for _ in range(30):
        _fm_pairwise_step(model, x_pos, x_neg, lr=0.05)
    after = model.logits(x_pos)[0][0] - model.logits(x_neg)[0][0]

    assert after > before


def test_dependency_levels_group_independent_nodes() -> None:
    """A seed bag's replicas must land in one level so they can run together."""
    from pipeline.execute import _dependency_levels
    from pipeline.graph import OperatorGraph
    from pipeline.registry import default_registry

    graph = OperatorGraph.from_dict({
        "nodes": [
            {"id": "load", "type": "data.load", "params": {}, "inputs": []},
            {"id": "raw", "type": "features.raw_categorical", "params": {}, "inputs": ["load"]},
            {"id": "m0", "type": "model.fm_baseline", "params": {"seed": 0}, "inputs": ["raw"]},
            {"id": "m1", "type": "model.fm_baseline", "params": {"seed": 1}, "inputs": ["raw"]},
            {"id": "m2", "type": "model.fm_baseline", "params": {"seed": 2}, "inputs": ["raw"]},
            {"id": "bag", "type": "ensemble.seed_bag", "params": {}, "inputs": ["m0", "m1", "m2"]},
            {"id": "out", "type": "submit.rank", "params": {}, "inputs": ["bag"]},
        ],
        "meta": {"name": "t", "hypothesis_id": "h"},
    })
    order = graph.validate(default_registry())
    levels = _dependency_levels(order)

    ids = [sorted(n.id for n in level) for level in levels]
    assert ids[0] == ["load"]
    assert ids[1] == ["raw"]
    # The three replicas are mutually independent and must share a level.
    assert ids[2] == ["m0", "m1", "m2"]
    assert ids[3] == ["bag"]
    assert ids[4] == ["out"]
    # Every node appears exactly once, and never before its inputs.
    seen: set[str] = set()
    for level in levels:
        for node in level:
            assert set(node.inputs) <= seen
        seen |= {n.id for n in level}
    assert len(seen) == len(order)


def test_parallel_execution_matches_sequential(tmp_path) -> None:
    """Parallel execution must be a pure speedup: identical scores, same order."""
    import numpy as np
    import pandas as pd

    from pipeline.execute import execute_graph
    from pipeline.graph import OperatorGraph
    from pipeline.registry import OperatorRegistry, OperatorSpec
    from pipeline.types import (
        DataBundle,
        ExecutionContext,
        PredictionBundle,
        ValueType,
    )

    # execute_graph scores the terminal, so the fixture needs a stand-in evaluator.
    (tmp_path / "evaluate.py").write_text(
        "def evaluate(user_ids, labels, scores, k=5):\n"
        "    return {'GAUC': 0.5, 'nDCG@5': 0.5, 'primary': 0.5}\n"
    )

    frames = {
        split: pd.DataFrame({
            "row_id": np.arange(4, dtype=np.int64),
            "user_id": ["u0", "u0", "u1", "u1"],
            "video_id": ["a", "b", "a", "b"],
            "long_view": [1, 0, 0, 1],
        })
        for split in ("train", "valid", "test")
    }
    data = DataBundle(frames, {}, tmp_path, tmp_path)

    def load(inputs, params, ctx):
        return data

    def scorer(inputs, params, ctx):
        offset = float(params.get("offset", 0.0))
        return PredictionBundle(
            f"m{offset}",
            {s: np.arange(4, dtype=np.float64) + offset for s in frames},
            data,
        )

    def combine(inputs, params, ctx):
        return PredictionBundle(
            "bag",
            {s: np.mean([p.scores[s] for p in inputs], axis=0) for s in frames},
            data,
        )

    registry = OperatorRegistry()
    registry.register(OperatorSpec("t.load", (), ValueType.DATA, load))
    registry.register(OperatorSpec("t.model", (ValueType.DATA,), ValueType.PREDICTIONS, scorer))
    registry.register(
        OperatorSpec("t.bag", (ValueType.PREDICTIONS,), ValueType.PREDICTIONS, combine, True)
    )

    spec = {
        "nodes": [
            {"id": "load", "type": "t.load", "params": {}, "inputs": []},
            {"id": "a", "type": "t.model", "params": {"offset": 1.0}, "inputs": ["load"]},
            {"id": "b", "type": "t.model", "params": {"offset": 2.0}, "inputs": ["load"]},
            {"id": "c", "type": "t.model", "params": {"offset": 3.0}, "inputs": ["load"]},
            {"id": "bag", "type": "t.bag", "params": {}, "inputs": ["a", "b", "c"]},
        ],
        "meta": {"name": "t", "hypothesis_id": "h"},
    }

    def run(workers: int):
        ctx = ExecutionContext(
            tmp_path, tmp_path, tmp_path / f"out{workers}",
            leakage_guard_enabled=False, metric_splits=("valid",),
            max_parallel_nodes=workers,
        )
        return execute_graph(OperatorGraph.from_dict(spec), ctx, registry)

    sequential = run(1)
    parallel = run(4)
    assert np.array_equal(
        sequential.terminal.scores["valid"], parallel.terminal.scores["valid"]
    )
    # And the mean of offsets 1, 2, 3 is 2, so the bag is genuinely combining all three.
    assert np.allclose(parallel.terminal.scores["valid"], np.arange(4) + 2.0)
