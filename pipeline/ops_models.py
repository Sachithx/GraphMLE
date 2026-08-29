from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eval.official import evaluate_official

from .ops_features import AUXILIARY_TARGETS
from .types import DataBundle, ExecutionContext, FeatureBundle, PredictionBundle


@dataclass(frozen=True)
class PreparedFeatures:
    arrays: dict[str, np.ndarray]
    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    categorical_dims: tuple[int, ...]
    data: DataBundle


def _require_features(inputs: list[Any]) -> list[FeatureBundle]:
    if not inputs or not all(isinstance(value, FeatureBundle) for value in inputs):
        raise TypeError("model operators require one or more FeatureBundle inputs")
    bundles: list[FeatureBundle] = inputs
    data = bundles[0].data
    if any(bundle.data is not data for bundle in bundles[1:]):
        raise ValueError("all model inputs must derive from the same DataBundle")
    return bundles


def op_ablation_constant_model(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    del params, ctx
    bundles = _require_features(inputs)
    data = bundles[0].data
    scores = {
        split: np.zeros(len(frame), dtype=np.float64)
        for split, frame in data.frames.items()
    }
    return PredictionBundle("ablation_constant", scores, data)


def _merge_features(bundles: list[FeatureBundle]) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    categorical: list[str] = []
    merged: dict[str, pd.DataFrame] = {}
    for bundle in bundles:
        categorical.extend(bundle.categorical_columns)
    if len(set(categorical)) != len(categorical):
        raise ValueError("feature bundles contain duplicate categorical column names")
    for split in ("train", "valid", "test"):
        result = pd.DataFrame({"row_id": bundles[0].frames[split]["row_id"].to_numpy()})
        seen = {"row_id"}
        for bundle in bundles:
            frame = bundle.frames[split]
            if not frame["row_id"].equals(result["row_id"]):
                raise ValueError(f"feature bundle {bundle.name} is misaligned on {split}")
            columns = [column for column in frame.columns if column != "row_id"]
            duplicates = seen.intersection(columns)
            if duplicates:
                raise ValueError(f"duplicate feature columns: {sorted(duplicates)}")
            result = pd.concat([result, frame[columns].reset_index(drop=True)], axis=1)
            seen.update(columns)
        merged[split] = result
    return merged, tuple(categorical)


def prepare_features(bundles: list[FeatureBundle]) -> PreparedFeatures:
    frames, categorical_columns = _merge_features(bundles)
    feature_names = tuple(column for column in frames["train"].columns if column != "row_id")
    categorical_set = set(categorical_columns)
    encoded: dict[str, list[np.ndarray]] = {split: [] for split in frames}
    categorical_indices: list[int] = []
    categorical_dims: list[int] = []
    for index, column in enumerate(feature_names):
        train_column = frames["train"][column]
        if column in categorical_set:
            categorical_indices.append(index)
            train_strings = train_column.astype("string").fillna("UNK")
            categories = pd.Index(pd.unique(train_strings))
            categorical_dims.append(len(categories) + 1)
            for split, frame in frames.items():
                codes = pd.Categorical(
                    frame[column].astype("string").fillna("UNK"), categories=categories
                ).codes
                encoded[split].append((codes + 1).astype(np.float32))
        else:
            numeric_train = pd.to_numeric(train_column, errors="coerce")
            median = float(numeric_train.median()) if numeric_train.notna().any() else 0.0
            for split, frame in frames.items():
                values = (
                    pd.to_numeric(frame[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(median)
                    .to_numpy(dtype=np.float32)
                )
                encoded[split].append(values)
    arrays = {
        split: np.column_stack(columns).astype(np.float32, copy=False)
        for split, columns in encoded.items()
    }
    return PreparedFeatures(
        arrays,
        feature_names,
        tuple(categorical_indices),
        tuple(categorical_dims),
        bundles[0].data,
    )


def _load_kit_runtime(ctx: ExecutionContext) -> tuple[Any, Any]:
    cache_key = f"kit-runtime:{ctx.starter_kit_dir}"
    cached = ctx.cache.get(cache_key)
    if cached is not None:
        return cached
    names = ("data", "evaluate", "baseline")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(ctx.starter_kit_dir))
    try:
        kit_data = importlib.import_module("data")
        baseline = importlib.import_module("baseline")
    finally:
        sys.path.pop(0)
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]
    ctx.cache[cache_key] = (kit_data, baseline)
    return kit_data, baseline


def op_fm_baseline(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    bundles = _require_features(inputs)
    data = bundles[0].data
    kit_data, baseline = _load_kit_runtime(ctx)
    encoded, dimension = kit_data.encode(data.official_splits)
    x_train, y_train, _ = encoded["train"]
    x_valid, y_valid, users_valid = encoded["valid"]
    seed = int(params.get("seed", ctx.seed))
    learning_rate = float(params.get("lr", 0.001))
    epochs = int(params.get("epochs", 40))
    batch_size = int(params.get("batch_size", 8192))
    patience = int(params.get("patience", 4))
    model = baseline.FM(
        dimension,
        k=int(params.get("k", 16)),
        lr=learning_rate,
        l2=float(params.get("l2", 1e-6)),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    best = -np.inf
    best_state = None
    bad_epochs = 0
    for _epoch in range(epochs):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            selection = order[start : start + batch_size]
            model.step(x_train[selection], y_train[selection])
        metrics = evaluate_official(
            users_valid,
            y_valid,
            model.predict(x_valid),
            starter_kit_dir=data.starter_kit_dir,
        )
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("FM training did not produce a checkpoint")
    model.V, model.W, model.b = best_state
    scores = {
        split: model.predict(encoded[split][0]).astype(np.float64)
        for split in ("train", "valid", "test")
    }
    return PredictionBundle("fm_baseline", scores, data)


def _lightgbm_common(params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "n_estimators": int(params.get("n_estimators", 200)),
        "num_leaves": int(params.get("num_leaves", 31)),
        "learning_rate": float(params.get("lr", params.get("learning_rate", 0.05))),
        "min_child_samples": int(params.get("min_child_samples", 20)),
        "subsample": float(params.get("subsample", 1.0)),
        "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
        "reg_lambda": float(params.get("reg_lambda", 0.0)),
        "random_state": seed,
        "n_jobs": int(params.get("n_jobs", 0)),
        "verbosity": -1,
    }


def op_lightgbm_binary(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    import lightgbm as lgb

    prepared = prepare_features(_require_features(inputs))
    seed = int(params.get("seed", ctx.seed))
    model = lgb.LGBMClassifier(objective="binary", **_lightgbm_common(params, seed))
    model.fit(
        prepared.arrays["train"],
        prepared.data.frames["train"]["long_view"].to_numpy(dtype=np.int8),
        categorical_feature=list(prepared.categorical_indices),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    scores = {
        split: model.predict_proba(values)[:, 1].astype(np.float64)
        for split, values in prepared.arrays.items()
    }
    return PredictionBundle("lightgbm_binary", scores, prepared.data)


def op_lightgbm_rank(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    import lightgbm as lgb

    prepared = prepare_features(_require_features(inputs))
    seed = int(params.get("seed", ctx.seed))
    users = prepared.data.frames["train"]["user_id"].astype(str).to_numpy()
    order = np.argsort(users, kind="stable")
    _, groups = np.unique(users[order], return_counts=True)
    labels = prepared.data.frames["train"]["long_view"].to_numpy(dtype=np.int8)
    model = lgb.LGBMRanker(
        objective=str(params.get("objective", "lambdarank")),
        metric="ndcg",
        label_gain=[0, 1],
        **_lightgbm_common(params, seed),
    )
    model.fit(
        prepared.arrays["train"][order],
        labels[order],
        group=groups.tolist(),
        categorical_feature=list(prepared.categorical_indices),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    scores = {
        split: model.predict(values).astype(np.float64)
        for split, values in prepared.arrays.items()
    }
    return PredictionBundle("lightgbm_rank", scores, prepared.data)


def _train_torch(
    bundles: list[FeatureBundle],
    params: dict[str, Any],
    ctx: ExecutionContext,
    multitask: bool,
) -> PredictionBundle:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    prepared = prepare_features(bundles)
    seed = int(params.get("seed", ctx.seed))
    torch.manual_seed(seed)
    torch.set_num_threads(int(params.get("threads", 4)))
    device_name = str(params.get("device", "cpu"))
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    categorical_indices = list(prepared.categorical_indices)
    numeric_indices = [
        index for index in range(len(prepared.feature_names)) if index not in categorical_indices
    ]
    embedding_dim = int(params.get("embedding_dim", 16))
    hidden_dim = int(params.get("hidden_dim", 64))
    task_columns = ["long_view"]
    if multitask:
        from guards.leakage import LeakageGuard

        requested = list(params.get("auxiliary_targets", list(AUXILIARY_TARGETS)))
        LeakageGuard(ctx.run_log_path or ctx.output_dir / "run_log.jsonl").check_auxiliary_targets(
            requested
        )
        task_columns.extend(column for column in requested if column in AUXILIARY_TARGETS)

    class TorchRecModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size, embedding_dim) for size in prepared.categorical_dims]
            )
            self.linear_embeddings = nn.ModuleList(
                [nn.Embedding(size, 1) for size in prepared.categorical_dims]
            )
            input_dim = len(categorical_indices) * embedding_dim + len(numeric_indices)
            output_tasks = len(task_columns)
            self.deep = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(params.get("dropout", 0.1))),
                nn.Linear(hidden_dim, output_tasks),
            )
            self.numeric_linear = (
                nn.Linear(len(numeric_indices), output_tasks, bias=False)
                if numeric_indices
                else None
            )
            self.fm_projection = nn.Linear(1, output_tasks, bias=False)

        def forward(self, values: Any) -> Any:
            cats = [
                values[:, column].long().clamp(min=0, max=size - 1)
                for column, size in zip(categorical_indices, prepared.categorical_dims)
            ]
            embeddings = [layer(cat) for layer, cat in zip(self.embeddings, cats)]
            numeric = values[:, numeric_indices] if numeric_indices else values[:, :0]
            deep_input = torch.cat([*embeddings, numeric], dim=1)
            logits = self.deep(deep_input)
            if self.numeric_linear is not None:
                logits = logits + self.numeric_linear(numeric)
            if embeddings:
                stacked = torch.stack(embeddings, dim=1)
                fm = 0.5 * (
                    stacked.sum(dim=1).pow(2) - stacked.pow(2).sum(dim=1)
                ).sum(dim=1, keepdim=True)
                first = torch.stack(
                    [layer(cat) for layer, cat in zip(self.linear_embeddings, cats)], dim=1
                ).sum(dim=1)
                logits = logits + self.fm_projection(fm) + first.expand(-1, len(task_columns))
            return logits

    train_x = torch.from_numpy(prepared.arrays["train"])
    train_y = torch.from_numpy(
        prepared.data.frames["train"][task_columns].to_numpy(dtype=np.float32, copy=True)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(params.get("batch_size", 8192)),
        shuffle=True,
        generator=generator,
    )
    model = TorchRecModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-6)),
    )
    loss_function = nn.BCEWithLogitsLoss()
    model.train()
    for _epoch in range(int(params.get("epochs", 3))):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    scores: dict[str, np.ndarray] = {}
    prediction_batch = int(params.get("prediction_batch_size", 65_536))
    with torch.no_grad():
        for split, values in prepared.arrays.items():
            chunks = []
            for start in range(0, len(values), prediction_batch):
                batch = torch.from_numpy(values[start : start + prediction_batch]).to(device)
                chunks.append(model(batch)[:, 0].cpu().numpy())
            scores[split] = np.concatenate(chunks).astype(np.float64)
    name = "torch_multitask" if multitask else "torch_deepfm"
    return PredictionBundle(name, scores, prepared.data)


def op_torch_deepfm(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    return _train_torch(_require_features(inputs), params, ctx, multitask=False)


def op_torch_multitask(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    return _train_torch(_require_features(inputs), params, ctx, multitask=True)
