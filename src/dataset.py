from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_col: str


def _normalize(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def detect_target_column(columns: List[str]) -> Optional[str]:
    # Common in the Kaggle no-show dataset
    for c in columns:
        if _normalize(c) in {"no show", "no show status", "noshow", "no-show"}:
            return c
    # Fallback guesses
    for c in columns:
        n = _normalize(c)
        if "target" in n or "label" in n or "outcome" in n:
            return c
    return None


def to_binary_target(s: pd.Series) -> np.ndarray:
    # Kaggle dataset uses 'Yes' = no-show (missed), 'No' = showed
    if pd.api.types.is_numeric_dtype(s):
        vals = s.fillna(0).astype(int).to_numpy()
        if set(np.unique(vals)).issubset({0, 1}):
            return vals
        med = np.median(vals)
        return (vals > med).astype(int)

    s2 = s.astype(str).str.strip().str.lower()
    positives = {"yes", "y", "true", "1", "no-show", "noshow", "no show", "missed"}
    negatives = {"no", "n", "false", "0", "show", "attended"}

    y = np.zeros(len(s2), dtype=np.int64)
    y[s2.isin(positives)] = 1
    y[s2.isin(negatives)] = 0

    # If mapping failed, factorize safely
    if len(np.unique(y)) == 1 and len(pd.unique(s2)) > 1:
        codes, _ = pd.factorize(s2)
        y = (codes == codes.max()).astype(np.int64)

    return y


def load_and_prepare(
    csv_path: str = "data/appointments.csv",
    target_col: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DataBundle:
    df = pd.read_csv(csv_path)

    if target_col is None:
        target_col = detect_target_column(df.columns.tolist())

    if target_col is None:
        raise ValueError(
            "Could not detect the label column.\n"
            "Run: python -c \"import pandas as pd; df=pd.read_csv('data/appointments.csv'); print(df.columns.tolist())\"\n"
            "Then set target_col explicitly in load_and_prepare(...)."
        )

    y = to_binary_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    # Drop obvious ID columns if present
    drop_names = {"patientid", "appointmentid", "id"}
    for c in list(X.columns):
        if _normalize(c).replace(" ", "") in drop_names:
            X.drop(columns=[c], inplace=True)

    # Convert datetimes to useful numeric features if present
    for c in list(X.columns):
        n = _normalize(c)
        if "day" in n or "date" in n or "time" in n:
            try:
                dt = pd.to_datetime(X[c], errors="coerce")
                if dt.notna().sum() > 0:
                    X[f"{c}_dow"] = dt.dt.dayofweek
                    X[f"{c}_hour"] = dt.dt.hour
                    X.drop(columns=[c], inplace=True)
            except Exception:
                pass

    X = X.replace([np.inf, -np.inf], np.nan)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")

    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    feature_names = X_enc.columns.tolist()
    X_np = X_enc.to_numpy(dtype=np.float32)

    stratify = y if len(np.unique(y)) > 1 else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_np, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    val_frac = val_size / (1.0 - test_size)
    strat_tv = y_trainval if len(np.unique(y_trainval)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=random_state, stratify=strat_tv
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return DataBundle(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train.astype(np.int64), y_val=y_val.astype(np.int64), y_test=y_test.astype(np.int64),
        feature_names=feature_names, target_col=target_col
    )
