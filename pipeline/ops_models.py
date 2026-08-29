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


def _encode_fm_features(
    bundles: list[FeatureBundle],
    *,
    numeric_bins: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[Any]]], int]:
    """Encode every explicit feature input as a sparse FM field.

    The starter FM consumes integer ids for one-hot fields. Categorical columns
    therefore receive a train-fitted vocabulary, while continuous columns are
    converted to train-fitted quantile tokens. There is intentionally no hidden
    fallback to canonical data: graph ablation and usage checks must describe
    the fields that the model actually receives.
    """

    if numeric_bins < 2:
        raise ValueError("FM numeric_bins must be at least 2")
    frames, categorical_columns = _merge_features(bundles)
    feature_columns = [
        column
        for column in frames["train"].columns
        if column != "row_id"
    ]
    if not feature_columns:
        raise ValueError("FM requires at least one feature column")
    categorical_set = set(categorical_columns)
    encoded_columns: dict[str, list[np.ndarray]] = {split: [] for split in frames}
    next_offset = 0

    for column in feature_columns:
        if column in categorical_set:
            train_values = frames["train"][column].astype("string").fillna("UNK")
            categories = pd.Index(pd.unique(train_values))
            field_dimension = len(categories) + 1
            for split, frame in frames.items():
                codes = pd.Categorical(
                    frame[column].astype("string").fillna("UNK"),
                    categories=categories,
                ).codes.astype(np.int64, copy=False)
                codes = np.where(codes < 0, len(categories), codes)
                encoded_columns[split].append(
                    (codes + next_offset).astype(np.int32)
                )
        else:
            train_values = (
                pd.to_numeric(frames["train"][column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            median = float(train_values.median()) if train_values.notna().any() else 0.0
            clean_train = train_values.fillna(median).to_numpy(dtype=np.float64)
            edges = np.unique(
                np.quantile(
                    clean_train,
                    np.linspace(0.0, 1.0, numeric_bins + 1)[1:-1],
                )
            )
            field_dimension = len(edges) + 1
            for split, frame in frames.items():
                values = (
                    pd.to_numeric(frame[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(median)
                    .to_numpy(dtype=np.float64)
                )
                codes = np.searchsorted(edges, values, side="right")
                encoded_columns[split].append(
                    (codes + next_offset).astype(np.int32)
                )
        next_offset += field_dimension

    encoded = {}
    data = bundles[0].data
    for split, columns in encoded_columns.items():
        encoded[split] = (
            np.column_stack(columns).astype(np.int32, copy=False),
            data.frames[split]["long_view"].to_numpy(dtype=np.float32),
            data.frames[split]["user_id"].astype(str).tolist(),
        )
    return encoded, next_offset


def op_fm_baseline(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    bundles = _require_features(inputs)
    data = bundles[0].data
    _kit_data, baseline = _load_kit_runtime(ctx)
    encoded, dimension = _encode_fm_features(
        bundles,
        numeric_bins=int(params.get("numeric_bins", 32)),
    )
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
    device_name = str(params.get("device", "auto"))
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
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
    if multitask:
        # Long-view is the scored task; auxiliaries only regularize the shared trunk.
        weights = [1.0] + [
            float(params.get("auxiliary_weight", 0.3)) for _ in task_columns[1:]
        ]
        task_weights = torch.tensor(weights, device=device).view(1, -1)
    else:
        task_weights = None

    prediction_batch = int(params.get("prediction_batch_size", 65_536))

    def predict(values: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(values), prediction_batch):
                batch = torch.from_numpy(values[start : start + prediction_batch]).to(device)
                chunks.append(model(batch)[:, 0].cpu().numpy())
        return np.concatenate(chunks).astype(np.float64)

    # Mirror the FM operator: checkpoint on validation primary rather than trusting
    # the final epoch, so a longer budget cannot silently overfit. Synthetic fixtures
    # ship no evaluator and no validation rows, so fall back to a plain fixed-epoch
    # loop instead of making the operator unusable without the starter kit.
    valid_frame = prepared.data.frames["valid"]
    checkpointing = (
        len(valid_frame) > 0
        and (prepared.data.starter_kit_dir / "evaluate.py").is_file()
    )
    valid_users = valid_frame["user_id"].astype(str).tolist()
    valid_labels = valid_frame["long_view"].to_numpy()
    patience = int(params.get("patience", 3))
    best_primary = -np.inf
    best_state = None
    bad_epochs = 0
    for _epoch in range(int(params.get("epochs", 3))):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            targets = batch_y.to(device)
            if task_weights is None:
                loss = loss_function(logits, targets)
            else:
                per_task = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                loss = (per_task * task_weights).mean()
            loss.backward()
            optimizer.step()
        if not checkpointing:
            continue
        model.eval()
        metrics = evaluate_official(
            valid_users,
            valid_labels,
            predict(prepared.arrays["valid"]),
            starter_kit_dir=prepared.data.starter_kit_dir,
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    scores: dict[str, np.ndarray] = {
        split: predict(values) for split, values in prepared.arrays.items()
    }
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


def _set_indices(frame: pd.DataFrame, by: str) -> list[np.ndarray]:
    """Return positional index arrays, one per ranking set."""
    if by == "user_date":
        keys = pd.Series(
            list(zip(frame["user_id"].astype(str), frame["date"])), index=frame.index
        )
    elif by == "user":
        keys = frame["user_id"].astype(str)
    else:
        raise ValueError(f"unsupported grouping {by!r}")
    positions = np.arange(len(frame))
    return [group.to_numpy() for _, group in pd.Series(positions).groupby(keys.to_numpy(), sort=False)]


def op_setwise_rank(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    """Permutation-equivariant setwise ranker (SetRank-style, Pang et al. SIGIR 2020).

    Every other model here is a univariate scoring function: each row is scored in
    isolation even though GAUC and nDCG@5 rank a user's impressions against each
    other. LambdaRank changes only the loss and keeps univariate scoring, which is
    consistent with it not beating pointwise FM on this data. This operator changes
    the scoring function instead: a self-attention encoder scores all items in a set
    jointly, so an item's score depends on what it is competing against.

    Training groups are (user, date) co-exposure sets (mean size 5.8); inference
    groups are the per-user impressions the metric actually ranks (mean 5.6 valid,
    7.1 test). Those size distributions match, and attention is permutation
    equivariant, so the encoder transfers across the two groupings. No positional
    encoding is used: a ranking set is unordered by construction.
    """
    import torch
    from torch import nn

    prepared = prepare_features(_require_features(inputs))
    seed = int(params.get("seed", ctx.seed))
    torch.manual_seed(seed)
    torch.set_num_threads(int(params.get("threads", 4)))

    device_name = str(params.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    categorical_indices = list(prepared.categorical_indices)
    numeric_indices = [
        index for index in range(len(prepared.feature_names)) if index not in categorical_indices
    ]
    embedding_dim = int(params.get("embedding_dim", 16))
    d_model = int(params.get("d_model", 64))
    n_heads = int(params.get("n_heads", 4))
    n_layers = int(params.get("n_layers", 2))
    dropout = float(params.get("dropout", 0.1))
    max_set_size = int(params.get("max_set_size", 48))

    class SetwiseRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size, embedding_dim) for size in prepared.categorical_dims]
            )
            input_dim = len(categorical_indices) * embedding_dim + len(numeric_indices)
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.score_head = nn.Linear(d_model, 1)

        def forward(self, values: Any, pad_mask: Any) -> Any:
            # values: (batch, set, feature); pad_mask: (batch, set) True where padded.
            cats = [
                values[:, :, column].long().clamp(min=0, max=size - 1)
                for column, size in zip(categorical_indices, prepared.categorical_dims)
            ]
            parts = [layer(cat) for layer, cat in zip(self.embeddings, cats)]
            if numeric_indices:
                parts.append(values[:, :, numeric_indices])
            hidden = self.input_projection(torch.cat(parts, dim=-1))
            # Cross-item context: each item attends over the others in its set.
            hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
            return self.score_head(hidden).squeeze(-1)

    model = SetwiseRanker().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-6)),
    )

    train_frame = prepared.data.frames["train"]
    train_labels = train_frame["long_view"].to_numpy(dtype=np.float32)
    train_sets = _set_indices(train_frame, str(params.get("train_group", "user_date")))
    # Only mixed-label sets carry ordering information for a listwise loss.
    train_sets = [
        index for index in train_sets
        if 0 < train_labels[index].sum() < len(index)
    ]
    if not train_sets:
        raise RuntimeError("no discriminative training sets available")

    train_x = prepared.arrays["train"]
    rng = np.random.default_rng(seed)
    set_batch = int(params.get("set_batch_size", 256))

    def make_batch(chunk: list[np.ndarray], values: np.ndarray, labels: np.ndarray | None):
        width = max(len(index) for index in chunk)
        batch_x = np.zeros((len(chunk), width, values.shape[1]), dtype=np.float32)
        pad = np.ones((len(chunk), width), dtype=bool)
        batch_y = np.zeros((len(chunk), width), dtype=np.float32) if labels is not None else None
        for row, index in enumerate(chunk):
            batch_x[row, : len(index)] = values[index]
            pad[row, : len(index)] = False
            if batch_y is not None:
                batch_y[row, : len(index)] = labels[index]
        tensors = [torch.from_numpy(batch_x), torch.from_numpy(pad)]
        if batch_y is not None:
            tensors.append(torch.from_numpy(batch_y))
        return tensors

    def predict_split(split: str) -> np.ndarray:
        values = prepared.arrays[split]
        sets = _set_indices(prepared.data.frames[split], "user")
        out = np.zeros(len(values), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            # Group similar sizes together so padding stays cheap.
            for start in range(0, len(sets), set_batch):
                chunk = sets[start : start + set_batch]
                batch_x, pad = make_batch(chunk, values, None)
                logits = model(batch_x.to(device), pad.to(device)).cpu().numpy()
                for row, index in enumerate(chunk):
                    out[index] = logits[row, : len(index)]
        return out

    valid_users = prepared.data.frames["valid"]["user_id"].astype(str).tolist()
    valid_labels = prepared.data.frames["valid"]["long_view"].to_numpy()
    checkpointing = (
        len(valid_labels) > 0 and (prepared.data.starter_kit_dir / "evaluate.py").is_file()
    )
    patience = int(params.get("patience", 3))
    best_primary = -np.inf
    best_state = None
    bad_epochs = 0

    for _epoch in range(int(params.get("epochs", 20))):
        model.train()
        order = rng.permutation(len(train_sets))
        for start in range(0, len(order), set_batch):
            chunk = []
            for position in order[start : start + set_batch]:
                index = train_sets[position]
                if len(index) > max_set_size:
                    # Subsample oversized sets; a fresh draw each epoch acts as augmentation.
                    index = rng.choice(index, size=max_set_size, replace=False)
                chunk.append(index)
            batch_x, pad, batch_y = make_batch(chunk, train_x, train_labels)
            batch_x, pad, batch_y = batch_x.to(device), pad.to(device), batch_y.to(device)
            logits = model(batch_x, pad).masked_fill(pad, float("-inf"))
            # Listwise softmax cross-entropy over each set (ListNet-style).
            target = batch_y.masked_fill(pad, 0.0)
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-9)
            loss = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if not checkpointing:
            continue
        metrics = evaluate_official(
            valid_users, valid_labels, predict_split("valid"),
            starter_kit_dir=prepared.data.starter_kit_dir,
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    scores = {split: predict_split(split) for split in ("train", "valid", "test")}
    return PredictionBundle("setwise_rank", scores, prepared.data)
