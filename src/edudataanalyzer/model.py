"""Risk prediction model built on top of scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["grade", "attendance_rate", "assignments_completed", "absences"]
TARGET_COLUMN = "risk_label"


@dataclass
class TrainedModel:
    model: Pipeline
    metrics: Dict[str, float]


def _feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for training: {sorted(missing)}")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def train_risk_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Train a logistic regression model to predict student risk."""

    X, y = _feature_target_split(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    except ValueError:
        # roc_auc requires both classes; if absent, fall back silently
        pass

    return TrainedModel(model=model, metrics=metrics)


def predict_risk(model: Pipeline, records: Iterable[Dict[str, float]]) -> np.ndarray:
    """Predict dropout probability for one or multiple records."""

    X = pd.DataFrame(records, columns=FEATURE_COLUMNS).astype(float)
    return model.predict_proba(X)[:, 1]
