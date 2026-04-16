import random
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_parse_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        ts = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = ts.dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = ts.dt.month.fillna(0).astype(int)
        df[f"{col}_day"] = ts.dt.day.fillna(0).astype(int)
        df[f"{col}_hour"] = ts.dt.hour.fillna(0).astype(int)
        df[f"{col}_minute"] = ts.dt.minute.fillna(0).astype(int)
        df[f"{col}_second"] = ts.dt.second.fillna(0).astype(int)
        df = df.drop(columns=[col])
    return df


def load_and_preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    df = df.drop_duplicates()

    if "FraudLabel" not in df.columns:
        raise ValueError(f"'FraudLabel' column not found in {csv_path}")

    if "Timestamp" in df.columns:
        df = _safe_parse_timestamp(df, "Timestamp")

    drop_cols = [c for c in ["TransactionID", "CustomerID"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df["FraudLabel"].astype(int).values
    X = df.drop(columns=["FraudLabel"]).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if not X[c].dropna().empty else 0.0)

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")

    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    meta = {
        "feature_count": X_train.shape[1],
        "rows_total": len(df),
        "rows_train": len(y_train),
        "rows_test": len(y_test),
    }

    return X_train, X_test, y_train, y_test, X_train.shape[1], meta


def make_torch_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError("Mismatch between model parameters and provided weights.")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.array(y_true).reshape(-1)
    y_prob = np.array(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)
    })
    return metrics