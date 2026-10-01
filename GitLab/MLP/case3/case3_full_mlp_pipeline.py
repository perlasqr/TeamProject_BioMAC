#!/usr/bin/env python3
"""
Case 3 full MLP pipeline
========================

One script for:
- all five paper workflows
- primary, weighted, and balanced analyses
- full 90-combination paper grid
- participant-grouped five-fold cross-validation
- resume-safe hyperparameter search
- strict Zennit LRP-epsilon for all workflows
- PNG result visualizations

Input file
----------
All_DBs_Case3_ID.csv

Five workflows
--------------
1. wf1_featurewise
2. wf1_timeserieswise
3. wf2_featurewise
4. wf2_timeserieswise
5. wf3_trialwise

Paper grid
----------
Hidden units:
    32, 64, 128, 256, 512, 1024
Learning rates:
    1e-5, 1e-4, 1e-3
Batch sizes:
    8, 16, 32, 64, 128

Total:
    6 x 3 x 5 = 90 combinations per fold/workflow

Examples
--------
Smoke test:
    python -u case1_full_mlp_pipeline.py \
      --stage train \
      --analysis primary \
      --grid quick \
      --methods wf1_featurewise \
      --folds 1 \
      --max-epochs 5 \
      --out-dir case3_results_new

Full primary analysis:
    python -u case1_full_mlp_pipeline.py \
      --stage train \
      --analysis primary \
      --grid paper \
      --methods all \
      --folds all \
      --out-dir case3_results_new

Weighted sensitivity:
    python -u case1_full_mlp_pipeline.py \
      --stage train \
      --analysis weighted \
      --grid paper \
      --methods all \
      --folds all \
      --out-dir case3_results_new

Balanced sensitivity with a new full grid:
    python -u case1_full_mlp_pipeline.py \
      --stage train \
      --analysis balanced \
      --grid paper \
      --methods all \
      --folds all \
      --out-dir case3_results_new

Balanced sensitivity reusing the primary fold hyperparameters (25 fits):
    python -u case1_full_mlp_pipeline.py \
      --stage train \
      --analysis balanced \
      --reuse-primary-hyperparameters \
      --methods all \
      --folds all \
      --out-dir case3_results_new

Strict Zennit LRP for all completed primary workflows:
    python -u case1_full_mlp_pipeline.py \
      --stage lrp \
      --analysis primary \
      --methods all \
      --folds all \
      --out-dir case3_results_new

Generate PNG files:
    python -u case1_full_mlp_pipeline.py \
      --stage visualize \
      --analysis all \
      --out-dir case3_results_new

Notes
-----
- The script saves hyperparameter_search.csv after every completed combination.
  If interrupted, rerunning the same command resumes the unfinished fold.
- A completed fold is skipped unless --force-fold is supplied.
- WF2 uses the true DB identity to select a DB-specific scaler. It is a
  dataset-bias diagnostic, not deployable preprocessing for an unknown DB.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="DataFrame is highly fragmented",
    category=pd.errors.PerformanceWarning,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


INPUT_FILENAME = "All_DBs_Case3_ID.csv"
KEY_COLUMNS = ["DB", "Participant", "Trial", "Leg_Used"]
OPTIONAL_META = ["Mass_kg", "Height_m"]
N_TIMEPOINTS = 101
DB_LABELS = ["DB1", "DB2", "DB3", "DB4", "DB5"]

METHODS = {
    "wf1_featurewise": ("WF1", "Feature-wise", "Pool then feature-wise scale"),
    "wf1_timeserieswise": (
        "WF1",
        "Time series-wise",
        "Pool then channel-wise trajectory scale",
    ),
    "wf2_featurewise": (
        "WF2",
        "Feature-wise",
        "Scale every DB feature-wise, then pool",
    ),
    "wf2_timeserieswise": (
        "WF2",
        "Time series-wise",
        "Scale every DB channel-wise, then pool",
    ),
    "wf3_trialwise": (
        "WF3",
        "Trial-wise",
        "Scale each trial/channel independently",
    ),
}
METHOD_ORDER = list(METHODS)

METHOD_LABELS = {
    "wf1_featurewise": "WF1 Feature-wise",
    "wf1_timeserieswise": "WF1 Time-series",
    "wf2_featurewise": "WF2 Feature-wise",
    "wf2_timeserieswise": "WF2 Time-series",
    "wf3_trialwise": "WF3 Trial-wise",
}

ANALYSES = ["primary", "weighted", "balanced"]

GRID_PRESETS = {
    "quick": {
        "hidden": [64],
        "lr": [1e-3],
        "batch": [32],
    },
    "paper": {
        "hidden": [32, 64, 128, 256, 512, 1024],
        "lr": [1e-5, 1e-4, 1e-3],
        "batch": [8, 16, 32, 64, 128],
    },
}


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is missing. Activate mlp_env and run:\n"
            "pip install torch pandas numpy scikit-learn matplotlib zennit"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def require_zennit():
    try:
        from zennit.attribution import Gradient
        from zennit.composites import LayerMapComposite
        from zennit.rules import Epsilon, Pass
    except ImportError as exc:
        raise ImportError(
            "Zennit is missing. Activate mlp_env and run:\n"
            "pip install zennit"
        ) from exc
    return Gradient, LayerMapComposite, Epsilon, Pass


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, _, _, _ = require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(name: str):
    torch, _, _, _ = require_torch()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_torch_load(path: Path):
    torch, _, _, _ = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_methods(values: Sequence[str]) -> List[str]:
    if "all" in values:
        return METHOD_ORDER.copy()
    invalid = [value for value in values if value not in METHODS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return list(dict.fromkeys(values))


def parse_folds(values: Sequence[str], n_splits: int) -> List[int]:
    if "all" in values:
        return list(range(1, n_splits + 1))
    folds = sorted(set(int(value) for value in values))
    if any(value < 1 or value > n_splits for value in folds):
        raise ValueError(f"Folds must be in 1..{n_splits}")
    return folds


def parse_analyses(value: str) -> List[str]:
    if value == "all":
        return ANALYSES.copy()
    if value not in ANALYSES:
        raise ValueError(value)
    return [value]


def resolve_csv(data_dir: Path) -> Path:
    direct = data_dir / INPUT_FILENAME
    if direct.exists():
        return direct
    mapping = {path.name.lower(): path for path in data_dir.glob("*.csv")}
    if INPUT_FILENAME.lower() in mapping:
        return mapping[INPUT_FILENAME.lower()]
    raise FileNotFoundError(
        f"{INPUT_FILENAME} not found in {data_dir.resolve()}"
    )


def read_metadata(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    missing = [column for column in KEY_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    usecols = KEY_COLUMNS + [
        column for column in OPTIONAL_META if column in header.columns
    ]
    metadata = pd.read_csv(csv_path, usecols=usecols, low_memory=False).copy()
    metadata = metadata.dropna(subset=KEY_COLUMNS).reset_index(drop=True)
    for column in KEY_COLUMNS:
        metadata[column] = metadata[column].astype(str)
    if metadata.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate DB/Participant/Trial/Leg_Used rows found.")
    return metadata


def make_plan(
    metadata: pd.DataFrame,
    analysis: str,
    seed: int,
) -> pd.DataFrame:
    if analysis in {"primary", "weighted"}:
        plan = metadata.copy()
    elif analysis == "balanced":
        minimum = int(metadata["DB"].value_counts().min())
        parts = []
        for db in sorted(metadata["DB"].unique()):
            part = metadata[metadata["DB"] == db].sample(
                n=minimum,
                replace=False,
                random_state=seed,
            )
            parts.append(part)
        plan = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(analysis)

    plan = plan.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    plan["sample_id"] = np.arange(len(plan), dtype=int)
    plan["group_id"] = (
        plan["DB"].astype(str)
        + "::"
        + plan["Participant"].astype(str)
    )
    return plan


def get_or_create_plan(
    metadata: pd.DataFrame,
    output_root: Path,
    analysis: str,
    seed: int,
) -> pd.DataFrame:
    analysis_dir = output_root / analysis
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "sample_plan.csv"

    if path.exists():
        plan = pd.read_csv(path)
        for column in KEY_COLUMNS + ["group_id"]:
            plan[column] = plan[column].astype(str)
        return plan.sort_values("sample_id").reset_index(drop=True)

    if analysis == "weighted":
        primary_path = output_root / "primary" / "sample_plan.csv"
        if primary_path.exists():
            shutil.copy2(primary_path, path)
            plan = pd.read_csv(path)
            for column in KEY_COLUMNS + ["group_id"]:
                plan[column] = plan[column].astype(str)
            return plan.sort_values("sample_id").reset_index(drop=True)

    plan = make_plan(metadata, analysis, seed)
    plan.to_csv(path, index=False)
    return plan


def get_or_create_outer_folds(
    plan: pd.DataFrame,
    output_root: Path,
    analysis: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    analysis_dir = output_root / analysis
    path = analysis_dir / "outer_fold_assignments.csv"

    if path.exists():
        folds = pd.read_csv(path)
        for column in KEY_COLUMNS + ["group_id"]:
            folds[column] = folds[column].astype(str)
        if int(folds["outer_fold"].nunique()) != n_splits:
            raise ValueError(
                "Existing fold file does not match --n-splits."
            )
        return folds.sort_values("sample_id").reset_index(drop=True)

    if analysis == "weighted":
        primary_path = (
            output_root / "primary" / "outer_fold_assignments.csv"
        )
        if primary_path.exists():
            shutil.copy2(primary_path, path)
            folds = pd.read_csv(path)
            for column in KEY_COLUMNS + ["group_id"]:
                folds[column] = folds[column].astype(str)
            return folds.sort_values("sample_id").reset_index(drop=True)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    y = plan["DB"].astype(str).to_numpy()
    groups = plan["group_id"].astype(str).to_numpy()
    fold_assignment = np.full(len(plan), -1, dtype=int)

    for fold, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(plan)), y, groups),
        start=1,
    ):
        fold_assignment[test_idx] = fold

    if np.any(fold_assignment < 1):
        raise RuntimeError("Some samples have no outer fold.")

    folds = plan.copy()
    folds["outer_fold"] = fold_assignment

    if folds.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Participant leakage in outer folds.")

    folds.to_csv(path, index=False)
    return folds


def inner_split(
    trainval_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )
    relative_train, relative_val = next(
        splitter.split(
            np.zeros(len(trainval_idx)),
            y[trainval_idx],
            groups[trainval_idx],
        )
    )
    train_idx = trainval_idx[relative_train]
    val_idx = trainval_idx[relative_val]

    if set(groups[train_idx]) & set(groups[val_idx]):
        raise RuntimeError("Participant leakage in inner split.")

    return train_idx, val_idx


def detect_channels(
    columns: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    pattern = re.compile(r"^(.*)_(\d+)$")
    mapping: Dict[str, Dict[int, str]] = {}

    for column in columns:
        match = pattern.match(str(column))
        if not match:
            continue
        base = match.group(1)
        index = int(match.group(2))
        if 0 <= index < N_TIMEPOINTS:
            mapping.setdefault(base, {})[index] = str(column)

    complete = {
        base: [indices[index] for index in range(N_TIMEPOINTS)]
        for base, indices in mapping.items()
        if all(index in indices for index in range(N_TIMEPOINTS))
    }

    def channel_order(name: str):
        upper = name.upper()
        if upper.startswith("TRC_"):
            return (0, name)
        if upper.startswith("GRF_"):
            return (1, name)
        return (2, name)

    channels = sorted(complete, key=channel_order)
    if not channels:
        raise ValueError(
            "No complete channel columns ending _0 ... _100 were found."
        )

    return channels, {channel: complete[channel] for channel in channels}


def load_tensor(
    csv_path: Path,
    plan: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    List[str],
    Dict[str, int],
]:
    data = pd.read_csv(csv_path, low_memory=False).copy()
    for column in KEY_COLUMNS:
        data[column] = data[column].astype(str)

    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate sample keys in input CSV.")

    channels, channel_columns = detect_channels(data.columns)
    flat_columns = [
        column
        for channel in channels
        for column in channel_columns[channel]
    ]

    keep_columns = list(
        dict.fromkeys(
            KEY_COLUMNS
            + [column for column in OPTIONAL_META if column in data.columns]
            + flat_columns
        )
    )
    data = data.loc[:, keep_columns].copy()

    selected = plan.merge(
        data,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    ).copy()

    missing = selected[selected["_merge"] != "both"]
    if len(missing):
        raise ValueError(
            f"{len(missing)} planned samples are missing from the CSV."
        )

    selected = (
        selected.drop(columns="_merge")
        .sort_values("sample_id")
        .reset_index(drop=True)
        .copy()
    )

    raw = selected[flat_columns].to_numpy(dtype=np.float32)
    n_samples = len(selected)
    n_channels = len(channels)

    channel_time = raw.reshape(
        n_samples,
        n_channels,
        N_TIMEPOINTS,
    )
    X = np.transpose(channel_time, (0, 2, 1)).astype(np.float32)

    labels = sorted(selected["DB"].astype(str).unique())
    label_to_id = {
        label: index for index, label in enumerate(labels)
    }
    y = (
        selected["DB"]
        .astype(str)
        .map(label_to_id)
        .to_numpy(dtype=np.int64)
    )

    return X, y, selected, channels, label_to_id


def nan_min_max(
    array: np.ndarray,
    axes: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        minimum = np.nanmin(array, axis=axes, keepdims=True)
        maximum = np.nanmax(array, axis=axes, keepdims=True)

    minimum = np.where(np.isfinite(minimum), minimum, 0.0)
    maximum = np.where(
        np.isfinite(maximum),
        maximum,
        minimum + 1.0,
    )
    return minimum.astype(np.float32), maximum.astype(np.float32)


def apply_minmax(
    array: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    denominator = maximum - minimum
    denominator = np.where(
        np.isfinite(denominator) & (np.abs(denominator) >= 1e-8),
        denominator,
        1.0,
    )
    filled = np.where(np.isfinite(array), array, minimum)
    return ((filled - minimum) / denominator).astype(np.float32)


def scale_splits(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = X[train_idx]
    val = X[val_idx]
    test = X[test_idx]

    if method == "wf1_featurewise":
        minimum, maximum = nan_min_max(train, axes=(0,))
        train_scaled = apply_minmax(train, minimum, maximum)
        val_scaled = apply_minmax(val, minimum, maximum)
        test_scaled = apply_minmax(test, minimum, maximum)

    elif method == "wf1_timeserieswise":
        minimum, maximum = nan_min_max(train, axes=(0, 1))
        train_scaled = apply_minmax(train, minimum, maximum)
        val_scaled = apply_minmax(val, minimum, maximum)
        test_scaled = apply_minmax(test, minimum, maximum)

    elif method in {"wf2_featurewise", "wf2_timeserieswise"}:
        train_scaled = np.empty_like(train, dtype=np.float32)
        val_scaled = np.empty_like(val, dtype=np.float32)
        test_scaled = np.empty_like(test, dtype=np.float32)

        train_y = y[train_idx]
        val_y = y[val_idx]
        test_y = y[test_idx]

        axes = (0,) if method == "wf2_featurewise" else (0, 1)

        for class_id in range(n_classes):
            train_mask = train_y == class_id
            val_mask = val_y == class_id
            test_mask = test_y == class_id

            if not np.any(train_mask):
                raise RuntimeError(
                    f"No inner-training samples for class {class_id}."
                )

            minimum, maximum = nan_min_max(
                train[train_mask],
                axes=axes,
            )
            train_scaled[train_mask] = apply_minmax(
                train[train_mask],
                minimum,
                maximum,
            )
            if np.any(val_mask):
                val_scaled[val_mask] = apply_minmax(
                    val[val_mask],
                    minimum,
                    maximum,
                )
            if np.any(test_mask):
                test_scaled[test_mask] = apply_minmax(
                    test[test_mask],
                    minimum,
                    maximum,
                )

    elif method == "wf3_trialwise":
        def scale_trial_part(part: np.ndarray) -> np.ndarray:
            minimum, maximum = nan_min_max(part, axes=(1,))
            return apply_minmax(part, minimum, maximum)

        train_scaled = scale_trial_part(train)
        val_scaled = scale_trial_part(val)
        test_scaled = scale_trial_part(test)

    else:
        raise ValueError(method)

    def flatten_channel_major(part: np.ndarray) -> np.ndarray:
        return np.transpose(part, (0, 2, 1)).reshape(
            len(part),
            -1,
        ).astype(np.float32)

    return (
        flatten_channel_major(train_scaled),
        flatten_channel_major(val_scaled),
        flatten_channel_major(test_scaled),
    )


def build_model(input_size: int, hidden_size: int, n_classes: int):
    torch, nn, _, _ = require_torch()
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_classes),
    )
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    torch, _, DataLoader, TensorDataset = require_torch()
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def class_weights_from_training(
    y_train: np.ndarray,
    n_classes: int,
):
    torch, _, _, _ = require_torch()
    counts = np.bincount(
        y_train,
        minlength=n_classes,
    ).astype(float)
    weights = np.zeros(n_classes, dtype=np.float32)
    valid = counts > 0
    weights[valid] = (
        len(y_train)
        / (n_classes * counts[valid])
    )
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_loss_accuracy(
    model,
    loader,
    criterion,
    device,
) -> Tuple[float, float]:
    torch, _, _, _ = require_torch()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            total_correct += int(
                (logits.argmax(dim=1) == batch_y).sum().item()
            )
            total_count += len(batch_y)

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
    )


def train_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    n_classes: int,
    weighted: bool,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device,
):
    torch, nn, _, _ = require_torch()
    set_seed(seed)

    model = build_model(
        X_train.shape[1],
        hidden_size,
        n_classes,
    ).to(device)

    if weighted:
        weights = class_weights_from_training(
            y_train,
            n_classes,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    best_val_loss = math.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_y)
            train_correct += int(
                (logits.argmax(dim=1) == batch_y).sum().item()
            )
            train_count += len(batch_y)

        train_loss = train_loss_sum / max(train_count, 1)
        train_accuracy = train_correct / max(train_count, 1)

        val_loss, val_accuracy = evaluate_loss_accuracy(
            model,
            val_loader,
            criterion,
            device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_val_loss = history_rows[-1]["val_loss"]
        best_epoch = history_rows[-1]["epoch"]

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_rows)),
        "state_dict": best_state,
        "history": pd.DataFrame(history_rows),
    }


def grid_combinations(grid_name: str) -> List[Dict]:
    grid = GRID_PRESETS[grid_name]
    combinations = []
    for hidden in grid["hidden"]:
        for learning_rate in grid["lr"]:
            for batch_size in grid["batch"]:
                combinations.append(
                    {
                        "hidden_size": int(hidden),
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                    }
                )
    return combinations


def combination_key(row: Dict) -> Tuple[int, float, int]:
    return (
        int(row["hidden_size"]),
        float(row["learning_rate"]),
        int(row["batch_size"]),
    )


def search_fold(
    fold_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    combinations: List[Dict],
    n_classes: int,
    weighted: bool,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device,
) -> Dict:
    torch, _, _, _ = require_torch()

    search_path = fold_dir / "hyperparameter_search.csv"
    best_checkpoint_path = fold_dir / "search_best_model.pt"
    best_history_path = fold_dir / "search_best_history.csv"
    best_config_path = fold_dir / "search_best_config.json"

    if search_path.exists():
        search_df = pd.read_csv(search_path)
    else:
        search_df = pd.DataFrame()

    completed = set()
    if len(search_df):
        for _, row in search_df.iterrows():
            completed.add(combination_key(row))

    best_config = None
    best_loss = math.inf

    if best_config_path.exists() and best_checkpoint_path.exists():
        best_config = json.loads(
            best_config_path.read_text(encoding="utf-8")
        )
        best_loss = float(best_config["best_val_loss"])

    rows = search_df.to_dict("records") if len(search_df) else []

    for position, combination in enumerate(combinations, start=1):
        key = combination_key(combination)
        if key in completed:
            print(
                f"    {position}/{len(combinations)} already done: "
                f"h={combination['hidden_size']} "
                f"lr={combination['learning_rate']} "
                f"bs={combination['batch_size']}",
                flush=True,
            )
            continue

        print(
            f"    {position}/{len(combinations)} "
            f"h={combination['hidden_size']} "
            f"lr={combination['learning_rate']} "
            f"bs={combination['batch_size']}",
            flush=True,
        )

        started = time.time()
        result = train_candidate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_size=combination["hidden_size"],
            learning_rate=combination["learning_rate"],
            batch_size=combination["batch_size"],
            n_classes=n_classes,
            weighted=weighted,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed + position,
            device=device,
        )
        runtime = time.time() - started

        row = {
            **combination,
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "runtime_seconds": runtime,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(search_path, index=False)

        if result["best_val_loss"] < best_loss:
            best_loss = result["best_val_loss"]
            best_config = {
                **combination,
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "epochs_completed": result["epochs_completed"],
            }
            torch.save(
                {
                    "state_dict": result["state_dict"],
                    "input_size": int(X_train.shape[1]),
                    "hidden_size": int(combination["hidden_size"]),
                    "n_classes": int(n_classes),
                },
                best_checkpoint_path,
            )
            result["history"].to_csv(
                best_history_path,
                index=False,
            )
            best_config_path.write_text(
                json.dumps(best_config, indent=2),
                encoding="utf-8",
            )

        print(
            f"      val_loss={result['best_val_loss']:.6f}, "
            f"best_epoch={result['best_epoch']}, "
            f"runtime={runtime/60:.1f} min",
            flush=True,
        )

    if best_config is None:
        raise RuntimeError("No hyperparameter model was completed.")

    return best_config


def predict(
    model,
    X: np.ndarray,
    batch_size: int,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    torch, _, _, _ = require_torch()
    model.eval()
    probabilities = []
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                X[start : start + batch_size].astype(np.float32)
            ).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu().numpy())
            predictions.append(
                probs.argmax(dim=1).cpu().numpy()
            )

    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(predictions, axis=0),
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_split_file(
    selected: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_dir: Path,
) -> None:
    frames = []
    for split, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        columns = [
            column
            for column in (
                KEY_COLUMNS
                + ["sample_id", "group_id", "outer_fold"]
            )
            if column in selected.columns
        ]
        part = selected.iloc[indices][columns].copy()
        part["split"] = split
        frames.append(part)
    pd.concat(frames, ignore_index=True).to_csv(
        fold_dir / "sample_split.csv",
        index=False,
    )


def train_one_fold(
    analysis: str,
    method: str,
    fold: int,
    fold_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    selected: pd.DataFrame,
    channels: List[str],
    label_to_id: Dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    combinations: List[Dict],
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    device,
    force_fold: bool,
    reuse_primary_hyperparameters: bool,
    output_root: Path,
) -> Dict:
    torch, _, _, _ = require_torch()

    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "metrics.csv"
    model_path = fold_dir / "best_model.pt"

    if metrics_path.exists() and model_path.exists() and not force_fold:
        print(
            f"  Fold {fold} completed already; skipping.",
            flush=True,
        )
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    labels = [
        label
        for label, _ in sorted(
            label_to_id.items(),
            key=lambda item: item[1],
        )
    ]
    id_to_label = {
        value: key for key, value in label_to_id.items()
    }

    save_split_file(
        selected,
        train_idx,
        val_idx,
        test_idx,
        fold_dir,
    )

    X_train, X_val, X_test = scale_splits(
        method,
        X,
        y,
        train_idx,
        val_idx,
        test_idx,
        len(labels),
    )
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    if reuse_primary_hyperparameters:
        if analysis == "primary":
            raise ValueError(
                "--reuse-primary-hyperparameters is only for weighted "
                "or balanced sensitivity analyses."
            )

        primary_config_path = (
            output_root
            / "primary"
            / method
            / f"fold_{fold}"
            / "best_hyperparameters.json"
        )
        if not primary_config_path.exists():
            raise FileNotFoundError(
                "Primary hyperparameters are missing for this workflow/fold: "
                f"{primary_config_path}. Complete the primary analysis first."
            )

        primary_config = json.loads(
            primary_config_path.read_text(encoding="utf-8")
        )
        reused_combination = {
            "hidden_size": int(primary_config["hidden_size"]),
            "learning_rate": float(primary_config["learning_rate"]),
            "batch_size": int(primary_config["batch_size"]),
        }

        print(
            "    Reusing primary hyperparameters: "
            f"h={reused_combination['hidden_size']} "
            f"lr={reused_combination['learning_rate']} "
            f"bs={reused_combination['batch_size']}",
            flush=True,
        )

        started = time.time()
        result = train_candidate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_size=reused_combination["hidden_size"],
            learning_rate=reused_combination["learning_rate"],
            batch_size=reused_combination["batch_size"],
            n_classes=len(labels),
            weighted=analysis == "weighted",
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed + fold * 1000,
            device=device,
        )
        runtime = time.time() - started

        best_config = {
            **reused_combination,
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "reused_from_primary": True,
            "primary_config_path": str(primary_config_path),
        }

        torch.save(
            {
                "state_dict": result["state_dict"],
                "input_size": int(X_train.shape[1]),
                "hidden_size": int(reused_combination["hidden_size"]),
                "n_classes": int(len(labels)),
            },
            fold_dir / "search_best_model.pt",
        )
        result["history"].to_csv(
            fold_dir / "search_best_history.csv",
            index=False,
        )
        (fold_dir / "search_best_config.json").write_text(
            json.dumps(best_config, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    **reused_combination,
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "epochs_completed": result["epochs_completed"],
                    "runtime_seconds": runtime,
                    "reused_from_primary": True,
                    "primary_config_path": str(primary_config_path),
                }
            ]
        ).to_csv(
            fold_dir / "hyperparameter_search.csv",
            index=False,
        )

        print(
            f"      sensitivity fit complete: "
            f"val_loss={result['best_val_loss']:.6f}, "
            f"best_epoch={result['best_epoch']}, "
            f"runtime={runtime/60:.1f} min",
            flush=True,
        )

    else:
        best_config = search_fold(
            fold_dir=fold_dir,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            combinations=combinations,
            n_classes=len(labels),
            weighted=analysis == "weighted",
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed + fold * 1000,
            device=device,
        )

    search_checkpoint = safe_torch_load(
        fold_dir / "search_best_model.pt"
    )
    shutil.copy2(
        fold_dir / "search_best_model.pt",
        model_path,
    )
    shutil.copy2(
        fold_dir / "search_best_history.csv",
        fold_dir / "training_history.csv",
    )

    final_config = {
        "analysis": analysis,
        "method": method,
        "workflow": METHODS[method][0],
        "scaling_method": METHODS[method][1],
        "description": METHODS[method][2],
        "input_size": int(X_train.shape[1]),
        "n_channels": len(channels),
        "n_timepoints": N_TIMEPOINTS,
        "n_classes": len(labels),
        "weighted_loss": analysis == "weighted",
        "optimizer": "Adam",
        "loss": (
            "weighted cross-entropy"
            if analysis == "weighted"
            else "cross-entropy"
        ),
        "initialization": "Xavier uniform",
        "reuse_primary_hyperparameters": reuse_primary_hyperparameters,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        **best_config,
    }
    (fold_dir / "best_hyperparameters.json").write_text(
        json.dumps(final_config, indent=2),
        encoding="utf-8",
    )

    model = build_model(
        int(search_checkpoint["input_size"]),
        int(search_checkpoint["hidden_size"]),
        int(search_checkpoint["n_classes"]),
    )
    model.load_state_dict(search_checkpoint["state_dict"])
    model = model.to(device)

    probabilities, predictions = predict(
        model,
        X_test,
        int(best_config["batch_size"]),
        device,
    )

    metrics = {
        "analysis": analysis,
        "method": method,
        "workflow": METHODS[method][0],
        "scaling_method": METHODS[method][1],
        "fold": fold,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "n_channels": len(channels),
        "input_size": X_train.shape[1],
        "hidden_size": best_config["hidden_size"],
        "learning_rate": best_config["learning_rate"],
        "batch_size": best_config["batch_size"],
        "best_epoch": best_config["best_epoch"],
        "best_validation_loss": best_config["best_val_loss"],
        **calculate_metrics(y_test, predictions),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        fold_dir / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(labels))),
    )
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(fold_dir / "confusion_matrix.csv")

    meta_columns = [
        column
        for column in (
            KEY_COLUMNS
            + OPTIONAL_META
            + ["sample_id", "group_id", "outer_fold"]
        )
        if column in selected.columns
    ]
    predictions_df = (
        selected.iloc[test_idx][meta_columns]
        .copy()
        .reset_index(drop=True)
    )
    predictions_df["analysis"] = analysis
    predictions_df["method"] = method
    predictions_df["fold"] = fold
    predictions_df["true_label"] = [
        id_to_label[int(value)] for value in y_test
    ]
    predictions_df["predicted_label"] = [
        id_to_label[int(value)] for value in predictions
    ]
    predictions_df["correct"] = (
        predictions_df["true_label"]
        == predictions_df["predicted_label"]
    )
    for class_id, label in enumerate(labels):
        predictions_df[f"prob_{label}"] = probabilities[:, class_id]
    predictions_df.to_csv(
        fold_dir / "test_predictions.csv",
        index=False,
    )

    print(
        f"  Fold {fold}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )

    del model, X_train, X_val, X_test
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def compile_analysis(
    analysis_dir: Path,
) -> None:
    metric_files = sorted(
        analysis_dir.glob("*/fold_*/metrics.csv")
    )
    if not metric_files:
        return

    metrics = pd.concat(
        [pd.read_csv(path) for path in metric_files],
        ignore_index=True,
    ).sort_values(["method", "fold"])
    metrics.to_csv(
        analysis_dir / "all_fold_metrics.csv",
        index=False,
    )

    summary_rows = []
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    for method, part in metrics.groupby("method"):
        row = {
            "analysis": part["analysis"].iloc[0],
            "method": method,
            "workflow": part["workflow"].iloc[0],
            "scaling_method": part["scaling_method"].iloc[0],
            "completed_folds": len(part),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(
                part[metric].std(ddof=1)
                if len(part) > 1
                else 0.0
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        analysis_dir / "summary_mean_std.csv",
        index=False,
    )

    prediction_files = sorted(
        analysis_dir.glob("*/fold_*/test_predictions.csv")
    )
    if prediction_files:
        pd.concat(
            [pd.read_csv(path) for path in prediction_files],
            ignore_index=True,
        ).to_csv(
            analysis_dir / "all_test_predictions.csv",
            index=False,
        )

    lookup = {}
    for _, row in summary.iterrows():
        lookup[row["method"]] = (
            f"{100*row['accuracy_mean']:.2f} ± "
            f"{100*row['accuracy_std']:.2f}"
        )

    table = pd.DataFrame(
        [
            {
                "Pooling workflow": "WF1 (pool -> scale)",
                "Feature-wise": lookup.get(
                    "wf1_featurewise",
                    "-",
                ),
                "Time series-wise": lookup.get(
                    "wf1_timeserieswise",
                    "-",
                ),
                "Trial-wise": "-",
            },
            {
                "Pooling workflow": "WF2 (scale -> pool)",
                "Feature-wise": lookup.get(
                    "wf2_featurewise",
                    "-",
                ),
                "Time series-wise": lookup.get(
                    "wf2_timeserieswise",
                    "-",
                ),
                "Trial-wise": "-",
            },
            {
                "Pooling workflow": "WF3",
                "Feature-wise": "-",
                "Time series-wise": "-",
                "Trial-wise": lookup.get(
                    "wf3_trialwise",
                    "-",
                ),
            },
        ]
    )
    table.to_csv(
        analysis_dir / "table_style_results.csv",
        index=False,
    )


def save_feature_map(
    channels: List[str],
    analysis_dir: Path,
) -> None:
    rows = []
    for channel_index, channel in enumerate(channels):
        for time_index in range(N_TIMEPOINTS):
            rows.append(
                {
                    "flat_index": (
                        channel_index * N_TIMEPOINTS
                        + time_index
                    ),
                    "channel_index": channel_index,
                    "feature": channel,
                    "time_index": time_index,
                    "gait_cycle_percent": (
                        100.0
                        * time_index
                        / (N_TIMEPOINTS - 1)
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        analysis_dir / "feature_index_map.csv",
        index=False,
    )


def train_analysis(
    args,
    analysis: str,
    methods: List[str],
    folds: List[int],
) -> None:
    data_dir = Path(args.data_dir)
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_root / analysis
    analysis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv(data_dir)
    metadata = read_metadata(csv_path)
    plan = get_or_create_plan(
        metadata,
        output_root,
        analysis,
        args.seed,
    )
    folds_df = get_or_create_outer_folds(
        plan,
        output_root,
        analysis,
        args.n_splits,
        args.seed,
    )

    X, y, selected, channels, label_to_id = load_tensor(
        csv_path,
        folds_df,
    )
    save_feature_map(channels, analysis_dir)

    configuration = {
        "input_csv": str(csv_path.resolve()),
        "analysis": analysis,
        "grid": args.grid,
        "reuse_primary_hyperparameters": (
            args.reuse_primary_hyperparameters
        ),
        "expected_grid_combinations": (
            1
            if args.reuse_primary_hyperparameters
            else len(grid_combinations(args.grid))
        ),
        "methods": methods,
        "folds": folds,
        "n_samples": len(selected),
        "class_counts": (
            selected["DB"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "n_channels": len(channels),
        "n_timepoints": N_TIMEPOINTS,
        "input_size": len(channels) * N_TIMEPOINTS,
        "n_splits": args.n_splits,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "device": str(args.device_obj),
    }
    (analysis_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    combinations = grid_combinations(args.grid)
    groups = selected["group_id"].astype(str).to_numpy()
    outer_folds = selected["outer_fold"].to_numpy(dtype=int)

    print("\nCase 3 full MLP training", flush=True)
    print("========================", flush=True)
    print(f"Analysis: {analysis}", flush=True)
    print(f"Samples: {len(selected)}", flush=True)
    print(
        selected["DB"].value_counts().sort_index(),
        flush=True,
    )
    print(f"Channels: {len(channels)}", flush=True)
    print(
        f"Input size: {len(channels)*N_TIMEPOINTS}",
        flush=True,
    )
    print(f"Methods: {methods}", flush=True)
    print(f"Folds: {folds}", flush=True)
    print(
        "Fits per workflow/fold: "
        + (
            "1 reused primary configuration"
            if args.reuse_primary_hyperparameters
            else f"{len(combinations)} grid combinations"
        ),
        flush=True,
    )
    print(f"Device: {args.device_obj}", flush=True)

    for method in methods:
        print(
            f"\n{method}: {METHODS[method][2]}",
            flush=True,
        )
        method_dir = analysis_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        for fold in folds:
            test_idx = np.where(outer_folds == fold)[0]
            trainval_idx = np.where(outer_folds != fold)[0]
            train_idx, val_idx = inner_split(
                trainval_idx,
                y,
                groups,
                args.seed + fold,
            )

            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            test_groups = set(groups[test_idx])

            if train_groups & val_groups:
                raise RuntimeError("Train/validation leakage.")
            if train_groups & test_groups:
                raise RuntimeError("Train/test leakage.")
            if val_groups & test_groups:
                raise RuntimeError("Validation/test leakage.")

            train_one_fold(
                analysis=analysis,
                method=method,
                fold=fold,
                fold_dir=method_dir / f"fold_{fold}",
                X=X,
                y=y,
                selected=selected,
                channels=channels,
                label_to_id=label_to_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                combinations=combinations,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                seed=args.seed,
                device=args.device_obj,
                force_fold=args.force_fold,
                reuse_primary_hyperparameters=(
                    args.reuse_primary_hyperparameters
                ),
                output_root=output_root,
            )
            compile_analysis(analysis_dir)

    compile_analysis(analysis_dir)
    print(
        f"\nCompleted analysis: {analysis}",
        flush=True,
    )
    print(
        f"Summary: {analysis_dir/'summary_mean_std.csv'}",
        flush=True,
    )


def indices_from_split(
    selected: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str,
) -> np.ndarray:
    ids = set(
        pd.to_numeric(
            split_df.loc[
                split_df["split"].astype(str) == split_name,
                "sample_id",
            ],
            errors="raise",
        ).astype(int)
    )
    indices = selected.index[
        selected["sample_id"].astype(int).isin(ids)
    ].to_numpy(dtype=int)
    if len(indices) == 0:
        raise RuntimeError(f"No {split_name} rows found.")
    return np.sort(indices)


def run_zennit_batch(
    model,
    X_batch: np.ndarray,
    predicted_classes: np.ndarray,
    epsilon: float,
    device,
):
    torch, nn, _, _ = require_torch()
    Gradient, LayerMapComposite, Epsilon, Pass = (
        require_zennit()
    )

    inputs = torch.from_numpy(
        X_batch.astype(np.float32)
    ).to(device)
    inputs.requires_grad_(True)

    targets = torch.zeros(
        (len(X_batch), model[-1].out_features),
        dtype=torch.float32,
        device=device,
    )
    targets[
        torch.arange(len(X_batch), device=device),
        torch.from_numpy(
            predicted_classes.astype(np.int64)
        ).to(device),
    ] = 1.0

    composite = LayerMapComposite(
        [
            (nn.Linear, Epsilon(epsilon=epsilon)),
            (nn.ReLU, Pass()),
        ]
    )

    with Gradient(model=model, composite=composite) as attributor:
        output, relevance = attributor(inputs, targets)

    return (
        output.detach().cpu().numpy(),
        relevance.detach().cpu().numpy(),
    )


def modality_name(channel: str) -> str:
    """Return the Case 3 modality label for LRP summaries."""
    return "ID"


def save_lrp_heatmap(
    average_map: np.ndarray,
    channels: List[str],
    title: str,
    path: Path,
) -> None:
    if plt is None:
        return
    figure_height = max(6.0, len(channels) * 0.18 + 2.0)
    plt.figure(figsize=(12, figure_height))
    plt.imshow(
        average_map,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    plt.xlabel("Gait cycle (%)")
    plt.ylabel("Feature channel")
    plt.title(title)
    positions = np.linspace(0, N_TIMEPOINTS - 1, 11)
    plt.xticks(
        positions,
        [str(int(value)) for value in np.linspace(0, 100, 11)],
    )
    plt.yticks(
        np.arange(len(channels)),
        channels,
        fontsize=7,
    )
    plt.colorbar(label="Normalized positive relevance")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def run_lrp_for_analysis(
    args,
    analysis: str,
    methods: List[str],
    folds: List[int],
) -> None:
    torch, _, _, _ = require_torch()
    require_zennit()

    output_root = Path(args.out_dir)
    analysis_dir = output_root / analysis
    fold_plan_path = analysis_dir / "outer_fold_assignments.csv"

    if not fold_plan_path.exists():
        raise FileNotFoundError(
            f"Training results not found for {analysis}: "
            f"{fold_plan_path}"
        )

    plan = pd.read_csv(fold_plan_path)
    for column in KEY_COLUMNS + ["group_id"]:
        plan[column] = plan[column].astype(str)
    plan = plan.sort_values("sample_id").reset_index(drop=True)

    csv_path = resolve_csv(Path(args.data_dir))
    X, y, selected, channels, label_to_id = load_tensor(
        csv_path,
        plan,
    )
    labels = [
        label
        for label, _ in sorted(
            label_to_id.items(),
            key=lambda item: item[1],
        )
    ]
    id_to_label = {
        value: key for key, value in label_to_id.items()
    }

    print(
        f"\nStrict Zennit LRP-epsilon: {analysis}",
        flush=True,
    )
    print(f"Methods: {methods}", flush=True)
    print(f"Folds: {folds}", flush=True)
    print(f"Epsilon: {args.epsilon}", flush=True)

    for method in methods:
        method_dir = analysis_dir / method
        lrp_dir = method_dir / "zennit_lrp_epsilon"
        lrp_dir.mkdir(parents=True, exist_ok=True)

        relevance_parts = []
        metadata_parts = []

        for fold in folds:
            fold_dir = method_dir / f"fold_{fold}"
            model_path = fold_dir / "best_model.pt"
            split_path = fold_dir / "sample_split.csv"
            config_path = fold_dir / "best_hyperparameters.json"

            if not (
                model_path.exists()
                and split_path.exists()
                and config_path.exists()
            ):
                raise FileNotFoundError(
                    f"Incomplete trained fold: {fold_dir}"
                )

            split_df = pd.read_csv(split_path)
            train_idx = indices_from_split(
                selected,
                split_df,
                "train",
            )
            val_idx = indices_from_split(
                selected,
                split_df,
                "validation",
            )
            test_idx = indices_from_split(
                selected,
                split_df,
                "test",
            )

            _, _, X_test = scale_splits(
                method,
                X,
                y,
                train_idx,
                val_idx,
                test_idx,
                len(labels),
            )
            y_test = y[test_idx]

            checkpoint = safe_torch_load(model_path)
            config = json.loads(
                config_path.read_text(encoding="utf-8")
            )

            model = build_model(
                int(checkpoint["input_size"]),
                int(checkpoint["hidden_size"]),
                int(checkpoint["n_classes"]),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(args.device_obj)
            model.eval()

            probabilities, predictions = predict(
                model,
                X_test,
                int(config["batch_size"]),
                args.device_obj,
            )

            eligible = predictions == y_test
            if args.include_incorrect:
                eligible[:] = True
            local_indices = np.where(eligible)[0]

            if args.lrp_max_per_class_per_fold > 0:
                rng = np.random.default_rng(
                    args.seed + fold
                )
                limited = []
                for class_id in range(len(labels)):
                    class_indices = local_indices[
                        y_test[local_indices] == class_id
                    ]
                    if (
                        len(class_indices)
                        > args.lrp_max_per_class_per_fold
                    ):
                        class_indices = rng.choice(
                            class_indices,
                            size=args.lrp_max_per_class_per_fold,
                            replace=False,
                        )
                    limited.extend(
                        int(index) for index in class_indices
                    )
                local_indices = np.array(
                    sorted(limited),
                    dtype=int,
                )

            fold_relevance = []
            fold_metadata = []
            test_meta = (
                selected.iloc[test_idx]
                .copy()
                .reset_index(drop=True)
            )

            for start in range(
                0,
                len(local_indices),
                args.lrp_batch_size,
            ):
                batch_indices = local_indices[
                    start : start + args.lrp_batch_size
                ]

                _, raw_relevance = run_zennit_batch(
                    model,
                    X_test[batch_indices],
                    predictions[batch_indices],
                    args.epsilon,
                    args.device_obj,
                )

                if args.keep_negative:
                    processed = raw_relevance
                    denominator = np.max(
                        np.abs(processed),
                        axis=1,
                        keepdims=True,
                    )
                else:
                    processed = np.maximum(
                        raw_relevance,
                        0.0,
                    )
                    denominator = np.max(
                        processed,
                        axis=1,
                        keepdims=True,
                    )

                denominator = np.where(
                    denominator > 0.0,
                    denominator,
                    1.0,
                )
                normalized = (
                    processed / denominator
                ).astype(np.float32)
                normalized = normalized.reshape(
                    len(batch_indices),
                    len(channels),
                    N_TIMEPOINTS,
                )
                fold_relevance.append(normalized)

                for local_index in batch_indices:
                    row = test_meta.iloc[int(local_index)]
                    fold_metadata.append(
                        {
                            "sample_id": int(row["sample_id"]),
                            "DB": str(row["DB"]),
                            "Participant": str(
                                row["Participant"]
                            ),
                            "Trial": str(row["Trial"]),
                            "Leg_Used": str(row["Leg_Used"]),
                            "analysis": analysis,
                            "method": method,
                            "fold": fold,
                            "true_label": id_to_label[
                                int(y_test[local_index])
                            ],
                            "predicted_label": id_to_label[
                                int(predictions[local_index])
                            ],
                            "correct": bool(
                                y_test[local_index]
                                == predictions[local_index]
                            ),
                            "predicted_probability": float(
                                probabilities[
                                    local_index,
                                    predictions[local_index],
                                ]
                            ),
                        }
                    )

            if fold_relevance:
                relevance_parts.append(
                    np.concatenate(fold_relevance, axis=0)
                )
                metadata_parts.append(
                    pd.DataFrame(fold_metadata)
                )

            print(
                f"  {method} fold {fold}: "
                f"explained {len(fold_metadata)} samples",
                flush=True,
            )

            del model, X_test
            gc.collect()

        if not relevance_parts:
            raise RuntimeError(
                f"No LRP maps generated for {method}."
            )

        relevance = np.concatenate(relevance_parts, axis=0)
        metadata = pd.concat(
            metadata_parts,
            ignore_index=True,
        )

        metadata.to_csv(
            lrp_dir / "lrp_metadata.csv",
            index=False,
        )
        np.savez_compressed(
            lrp_dir / "lrp_relevance_maps.npz",
            relevance=relevance,
            feature_names=np.asarray(channels, dtype=str),
            sample_ids=metadata["sample_id"].to_numpy(),
            true_labels=metadata["true_label"].to_numpy(),
            predicted_labels=metadata[
                "predicted_label"
            ].to_numpy(),
            folds=metadata["fold"].to_numpy(),
        )

        channel_rows = []
        time_rows = []
        modality_rows = []
        summary_rows = []

        for label in labels:
            mask = (
                metadata["true_label"].astype(str).to_numpy()
                == label
            )
            count = int(mask.sum())
            if count == 0:
                continue

            average_map = relevance[mask].mean(axis=0)

            heatmap_csv = pd.DataFrame(
                average_map.T,
                columns=channels,
            )
            heatmap_csv.insert(
                0,
                "gait_cycle_percent",
                np.linspace(0.0, 100.0, N_TIMEPOINTS),
            )
            heatmap_csv.to_csv(
                lrp_dir / f"class_heatmap_{label}.csv",
                index=False,
            )
            save_lrp_heatmap(
                average_map,
                channels,
                (
                    f"{analysis} | {METHOD_LABELS[method]} "
                    f"| {label} | Zennit LRP-epsilon"
                ),
                lrp_dir / f"class_heatmap_{label}.png",
            )

            channel_values = average_map.sum(axis=1)
            channel_total = float(channel_values.sum())
            channel_relative = (
                channel_values / channel_total
                if channel_total > 0
                else np.zeros_like(channel_values)
            )

            for index, channel in enumerate(channels):
                channel_rows.append(
                    {
                        "DB": label,
                        "n_samples": count,
                        "feature": channel,
                        "modality": modality_name(channel),
                        "relative_contribution": float(
                            channel_relative[index]
                        ),
                        "relative_contribution_percent": float(
                            100.0 * channel_relative[index]
                        ),
                    }
                )

            time_values = average_map.sum(axis=0)
            time_total = float(time_values.sum())
            time_relative = (
                time_values / time_total
                if time_total > 0
                else np.zeros_like(time_values)
            )

            for time_index, value in enumerate(time_relative):
                time_rows.append(
                    {
                        "DB": label,
                        "n_samples": count,
                        "time_index": time_index,
                        "gait_cycle_percent": (
                            100.0
                            * time_index
                            / (N_TIMEPOINTS - 1)
                        ),
                        "relative_contribution": float(value),
                        "relative_contribution_percent": float(
                            100.0 * value
                        ),
                    }
                )

            modality_totals: Dict[str, float] = {}
            for index, channel in enumerate(channels):
                modality = modality_name(channel)
                modality_totals[modality] = (
                    modality_totals.get(modality, 0.0)
                    + float(channel_relative[index])
                )

            for modality, value in sorted(
                modality_totals.items()
            ):
                modality_rows.append(
                    {
                        "DB": label,
                        "n_samples": count,
                        "modality": modality,
                        "relative_contribution": value,
                        "relative_contribution_percent": (
                            100.0 * value
                        ),
                    }
                )

            ranked = sorted(
                zip(channels, channel_relative),
                key=lambda item: item[1],
                reverse=True,
            )

            summary_rows.append(
                {
                    "analysis": analysis,
                    "method": method,
                    "DB": label,
                    "n_explained_samples": count,
                    "peak_gait_cycle_percent": (
                        100.0
                        * int(np.argmax(time_relative))
                        / (N_TIMEPOINTS - 1)
                    ),
                    "top_feature_1": ranked[0][0],
                    "top_feature_1_percent": (
                        100.0 * ranked[0][1]
                    ),
                    "top_feature_2": ranked[1][0],
                    "top_feature_2_percent": (
                        100.0 * ranked[1][1]
                    ),
                    "top_feature_3": ranked[2][0],
                    "top_feature_3_percent": (
                        100.0 * ranked[2][1]
                    ),
                }
            )

        pd.DataFrame(channel_rows).to_csv(
            lrp_dir / "lrp_class_channel_importance.csv",
            index=False,
        )
        pd.DataFrame(time_rows).to_csv(
            lrp_dir / "lrp_class_time_importance.csv",
            index=False,
        )
        pd.DataFrame(modality_rows).to_csv(
            lrp_dir / "lrp_class_modality_importance.csv",
            index=False,
        )
        pd.DataFrame(summary_rows).to_csv(
            lrp_dir / "lrp_summary.csv",
            index=False,
        )

        print(
            f"  Saved LRP output: {lrp_dir}",
            flush=True,
        )


def save_figure(path: Path) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def visualize_analysis(
    output_root: Path,
    analysis: str,
) -> Optional[pd.DataFrame]:
    if plt is None:
        raise ImportError("matplotlib is required.")

    analysis_dir = output_root / analysis
    summary_path = analysis_dir / "summary_mean_std.csv"
    metrics_path = analysis_dir / "all_fold_metrics.csv"
    predictions_path = (
        analysis_dir / "all_test_predictions.csv"
    )
    plan_path = analysis_dir / "sample_plan.csv"

    if not (
        summary_path.exists()
        and metrics_path.exists()
        and predictions_path.exists()
    ):
        print(
            f"Skipping incomplete analysis: {analysis}",
            flush=True,
        )
        return None

    summary = pd.read_csv(summary_path)
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    plan = (
        pd.read_csv(plan_path)
        if plan_path.exists()
        else None
    )

    visual_dir = output_root / "visual_summary" / analysis
    visual_dir.mkdir(parents=True, exist_ok=True)

    if plan is not None:
        counts = (
            plan["DB"]
            .astype(str)
            .value_counts()
            .reindex(DB_LABELS, fill_value=0)
        )
        plt.figure(figsize=(8, 5))
        plt.bar(counts.index, counts.values)
        plt.xlabel("Dataset source")
        plt.ylabel("Number of trials")
        plt.title(f"{analysis.capitalize()} class distribution")
        for index, value in enumerate(counts.values):
            plt.text(
                index,
                value + max(counts.max(), 1) * 0.02,
                str(int(value)),
                ha="center",
            )
        save_figure(
            visual_dir / "01_class_distribution.png"
        )

    ordered = (
        summary.set_index("method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    x = np.arange(len(METHOD_ORDER))
    width = 0.24

    plt.figure(figsize=(12, 6))
    for index, (mean_col, std_col, label) in enumerate(
        [
            ("accuracy_mean", "accuracy_std", "Accuracy"),
            (
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "Balanced accuracy",
            ),
            ("macro_f1_mean", "macro_f1_std", "Macro F1"),
        ]
    ):
        plt.bar(
            x + (index - 1) * width,
            ordered[mean_col].to_numpy() * 100,
            width,
            yerr=ordered[std_col].fillna(0).to_numpy()
            * 100,
            capsize=4,
            label=label,
        )
    plt.xticks(
        x,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=20,
        ha="right",
    )
    plt.ylabel("Performance (%)")
    plt.ylim(0, 101)
    plt.title(
        f"{analysis.capitalize()}: five-fold mean ± SD"
    )
    plt.legend()
    save_figure(visual_dir / "02_mean_sd_metrics.png")

    for metric, filename in [
        ("accuracy", "03_fold_accuracy.png"),
        (
            "balanced_accuracy",
            "04_fold_balanced_accuracy.png",
        ),
        ("macro_f1", "05_fold_macro_f1.png"),
    ]:
        plt.figure(figsize=(10, 6))
        for method in METHOD_ORDER:
            part = metrics[
                metrics["method"] == method
            ].sort_values("fold")
            if len(part):
                plt.plot(
                    part["fold"],
                    part[metric] * 100,
                    marker="o",
                    label=METHOD_LABELS[method],
                )
        plt.xlabel("Outer fold")
        plt.ylabel(
            metric.replace("_", " ").title() + " (%)"
        )
        plt.xticks(range(1, 6))
        plt.ylim(0, 101)
        plt.title(
            f"{analysis.capitalize()}: "
            f"{metric.replace('_', ' ').title()}"
        )
        plt.legend()
        save_figure(visual_dir / filename)

    per_db_rows = []
    confusion_by_method = {}

    for method in METHOD_ORDER:
        part = predictions[
            predictions["method"] == method
        ]
        if part.empty:
            continue
        report = classification_report(
            part["true_label"],
            part["predicted_label"],
            labels=DB_LABELS,
            output_dict=True,
            zero_division=0,
        )
        for db in DB_LABELS:
            per_db_rows.append(
                {
                    "method": method,
                    "DB": db,
                    "precision": report[db]["precision"],
                    "recall": report[db]["recall"],
                    "f1": report[db]["f1-score"],
                    "support": report[db]["support"],
                }
            )
        confusion_by_method[method] = confusion_matrix(
            part["true_label"],
            part["predicted_label"],
            labels=DB_LABELS,
        )

    per_db = pd.DataFrame(per_db_rows)
    per_db.to_csv(
        visual_dir / "per_db_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))
    x = np.arange(len(DB_LABELS))
    width = 0.16
    for index, method in enumerate(METHOD_ORDER):
        part = (
            per_db[per_db["method"] == method]
            .set_index("DB")
            .reindex(DB_LABELS)
        )
        if len(part):
            plt.bar(
                x + (index - 2) * width,
                part["recall"].to_numpy() * 100,
                width,
                label=METHOD_LABELS[method],
            )
    plt.xticks(x, DB_LABELS)
    plt.ylabel("Recall (%)")
    plt.ylim(0, 101)
    plt.title(
        f"{analysis.capitalize()}: per-DB recall"
    )
    plt.legend()
    save_figure(visual_dir / "06_per_db_recall.png")

    for index, method in enumerate(
        METHOD_ORDER,
        start=7,
    ):
        if method not in confusion_by_method:
            continue
        matrix = confusion_by_method[method].astype(float)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix),
            where=row_sum > 0,
        )
        plt.figure(figsize=(7, 6))
        plt.imshow(normalized, vmin=0, vmax=1)
        plt.colorbar(label="Row-normalized proportion")
        plt.xticks(range(len(DB_LABELS)), DB_LABELS)
        plt.yticks(range(len(DB_LABELS)), DB_LABELS)
        plt.xlabel("Predicted dataset")
        plt.ylabel("True dataset")
        plt.title(
            f"{analysis.capitalize()}: "
            f"{METHOD_LABELS[method]}"
        )
        for row in range(len(DB_LABELS)):
            for column in range(len(DB_LABELS)):
                plt.text(
                    column,
                    row,
                    (
                        f"{int(matrix[row, column])}\n"
                        f"({100*normalized[row, column]:.1f}%)"
                    ),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        save_figure(
            visual_dir
            / f"{index:02d}_confusion_{method}.png"
        )

    lrp_visual_dir = visual_dir / "lrp"
    lrp_visual_dir.mkdir(parents=True, exist_ok=True)

    for method in METHOD_ORDER:
        lrp_dir = (
            analysis_dir
            / method
            / "zennit_lrp_epsilon"
        )
        modality_path = (
            lrp_dir / "lrp_class_modality_importance.csv"
        )
        channel_path = (
            lrp_dir / "lrp_class_channel_importance.csv"
        )
        time_path = (
            lrp_dir / "lrp_class_time_importance.csv"
        )

        if not (
            modality_path.exists()
            and channel_path.exists()
            and time_path.exists()
        ):
            continue

        modality = pd.read_csv(modality_path)
        channel = pd.read_csv(channel_path)
        time_df = pd.read_csv(time_path)

        pivot = modality.pivot(
            index="DB",
            columns="modality",
            values="relative_contribution_percent",
        ).reindex(DB_LABELS).fillna(0)

        plt.figure(figsize=(9, 6))
        x = np.arange(len(DB_LABELS))
        columns = list(pivot.columns)
        width = (
            0.36
            if len(columns) <= 2
            else max(0.16, 0.75 / len(columns))
        )
        for index, column in enumerate(columns):
            plt.bar(
                x
                + (
                    index
                    - (len(columns) - 1) / 2
                )
                * width,
                pivot[column].to_numpy(),
                width,
                label=column,
            )
        plt.xticks(x, DB_LABELS)
        plt.ylabel("Relative relevance (%)")
        plt.title(
            f"{analysis.capitalize()} | "
            f"{METHOD_LABELS[method]} | modality LRP"
        )
        plt.legend()
        save_figure(
            lrp_visual_dir
            / f"lrp_modality_{method}.png"
        )

        top_channels = (
            channel.groupby(
                "feature",
                as_index=False,
            )["relative_contribution_percent"]
            .mean()
            .sort_values(
                "relative_contribution_percent",
                ascending=False,
            )
            .head(15)
            .sort_values(
                "relative_contribution_percent"
            )
        )

        plt.figure(figsize=(11, 7))
        plt.barh(
            top_channels["feature"],
            top_channels[
                "relative_contribution_percent"
            ],
        )
        plt.xlabel(
            "Equal-DB mean relevance contribution (%)"
        )
        plt.ylabel("Feature channel")
        plt.title(
            f"{analysis.capitalize()} | "
            f"{METHOD_LABELS[method]} | top LRP channels"
        )
        save_figure(
            lrp_visual_dir
            / f"lrp_top_channels_{method}.png"
        )

        plt.figure(figsize=(10, 6))
        for db in DB_LABELS:
            part = time_df[
                time_df["DB"] == db
            ].sort_values("gait_cycle_percent")
            if len(part):
                plt.plot(
                    part["gait_cycle_percent"],
                    part[
                        "relative_contribution_percent"
                    ],
                    label=db,
                )
        plt.xlabel("Gait cycle (%)")
        plt.ylabel("Relative relevance (%)")
        plt.title(
            f"{analysis.capitalize()} | "
            f"{METHOD_LABELS[method]} | temporal LRP"
        )
        plt.legend()
        save_figure(
            lrp_visual_dir
            / f"lrp_time_{method}.png"
        )

        heatmap_output = (
            lrp_visual_dir / "class_heatmaps"
        )
        heatmap_output.mkdir(
            parents=True,
            exist_ok=True,
        )
        for db in DB_LABELS:
            source = lrp_dir / f"class_heatmap_{db}.png"
            if source.exists():
                shutil.copy2(
                    source,
                    heatmap_output
                    / f"{analysis}_{method}_{db}.png",
                )

    print(
        f"Saved PNG files: {visual_dir}",
        flush=True,
    )
    return summary


def visualize_all(
    output_root: Path,
    analyses: List[str],
) -> None:
    summaries = {}
    for analysis in analyses:
        result = visualize_analysis(
            output_root,
            analysis,
        )
        if result is not None:
            summaries[analysis] = result

    if plt is None or len(summaries) < 2:
        return

    cross_dir = output_root / "visual_summary"
    for metric, filename in [
        (
            "accuracy_mean",
            "90_cross_analysis_accuracy.png",
        ),
        (
            "balanced_accuracy_mean",
            "91_cross_analysis_balanced_accuracy.png",
        ),
        (
            "macro_f1_mean",
            "92_cross_analysis_macro_f1.png",
        ),
    ]:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(METHOD_ORDER))
        width = 0.22
        available = [
            analysis
            for analysis in ANALYSES
            if analysis in summaries
        ]
        for index, analysis in enumerate(available):
            part = (
                summaries[analysis]
                .set_index("method")
                .reindex(METHOD_ORDER)
            )
            plt.bar(
                x
                + (
                    index
                    - (len(available) - 1) / 2
                )
                * width,
                part[metric].to_numpy() * 100,
                width,
                label=analysis.capitalize(),
            )
        plt.xticks(
            x,
            [
                METHOD_LABELS[method]
                for method in METHOD_ORDER
            ],
            rotation=20,
            ha="right",
        )
        plt.ylabel(
            metric.replace("_mean", "")
            .replace("_", " ")
            .title()
            + " (%)"
        )
        plt.ylim(0, 101)
        plt.title(
            metric.replace("_mean", "")
            .replace("_", " ")
            .title()
            + ": analysis comparison"
        )
        plt.legend()
        save_figure(cross_dir / filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["train", "lrp", "visualize", "all"],
        default="train",
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument(
        "--out-dir",
        default="case3_results_new",
    )
    parser.add_argument(
        "--analysis",
        choices=["primary", "weighted", "balanced", "all"],
        default="primary",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
    )
    parser.add_argument(
        "--grid",
        choices=["quick", "paper"],
        default="paper",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["all"],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=5e-3,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--force-fold",
        action="store_true",
        help="Reevaluate a completed fold.",
    )
    parser.add_argument(
        "--reuse-primary-hyperparameters",
        action="store_true",
        help=(
            "For weighted or balanced analysis, skip grid search and "
            "reuse the best hidden size, learning rate, and batch size "
            "from the matching primary workflow/fold. This produces "
            "one fit per workflow/fold (25 fits for all workflows/folds)."
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--lrp-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lrp-max-per-class-per-fold",
        type=int,
        default=0,
        help="0 explains all eligible test samples.",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
    )
    parser.add_argument(
        "--keep-negative",
        action="store_true",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.device_obj = choose_device(args.device)
    methods = parse_methods(args.methods)
    folds = parse_folds(args.folds, args.n_splits)
    analyses = parse_analyses(args.analysis)

    if args.reuse_primary_hyperparameters:
        if args.stage not in {"train", "all"}:
            raise ValueError(
                "--reuse-primary-hyperparameters applies only to training."
            )
        if "primary" in analyses:
            raise ValueError(
                "Do not use --reuse-primary-hyperparameters with "
                "--analysis primary or --analysis all. Run it separately "
                "with --analysis balanced or --analysis weighted."
            )

    if args.stage in {"train", "all"}:
        for analysis in analyses:
            train_analysis(
                args,
                analysis,
                methods,
                folds,
            )

    if args.stage in {"lrp", "all"}:
        for analysis in analyses:
            run_lrp_for_analysis(
                args,
                analysis,
                methods,
                folds,
            )

    if args.stage in {"visualize", "all"}:
        visualize_all(
            Path(args.out_dir),
            analyses,
        )


if __name__ == "__main__":
    main()
