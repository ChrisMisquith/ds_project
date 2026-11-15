"""Data validation, preprocessing, and evaluation shared by the notebook and tests.

All learned preprocessing belongs inside a fitted sklearn Pipeline. The positive
class is always HeartDisease=1. Functions never choose a model from test scores.
"""

from __future__ import annotations

import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

SEED = 42
TARGET = "HeartDisease"
FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
CATEGORIES = {
    "Sex": {"M", "F"},
    "ChestPainType": {"TA", "ATA", "NAP", "ASY"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"Y", "N"},
    "ST_Slope": {"Up", "Flat", "Down"},
}


def load_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    """Verify the preserved source hash and schema before analysis."""
    path = root / "data/raw/heart.csv"
    provenance = json.loads(
        (path.parent / "provenance.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["csv_sha256"]:
        raise ValueError("Raw source hash does not match provenance.json.")
    frame = pd.read_csv(path)
    validate_raw(frame)
    return frame, provenance


def validate_raw(frame: pd.DataFrame) -> None:
    if list(frame.columns) != FEATURES + [TARGET]:
        raise ValueError("Unexpected columns or column order.")
    if len(frame) != 918:
        raise ValueError("This analysis expects the verified 918-record source.")
    if frame.isna().any().any():
        raise ValueError("Unexpected empty source cells; update the data audit first.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("HeartDisease must contain labels 0 and 1.")
    if not frame["FastingBS"].isin([0, 1]).all():
        raise ValueError("FastingBS must be binary.")
    for col, allowed in CATEGORIES.items():
        if not set(frame[col].unique()).issubset(allowed):
            raise ValueError(f"Unexpected categories in {col}.")
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric values found.")
    # Splitting duplicate feature vectors could leak the same observation.
    if frame[FEATURES].duplicated().any():
        raise ValueError(
            "Repeated feature vectors found; implement grouped splitting before analysis."
        )


def audit_data(frame: pd.DataFrame) -> pd.DataFrame:
    """A quality audit, not an automatic deletion of legitimate zero values."""
    rows = []
    for col in frame.columns:
        rows.append(
            {
                "feature": col,
                "dtype": str(frame[col].dtype),
                "empty_cells": int(frame[col].isna().sum()),
                "unique_values": int(frame[col].nunique()),
                "invalid_nonpositive": (
                    int((frame[col] <= 0).sum())
                    if col in ["RestingBP", "Cholesterol"]
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


class InvalidMeasurementCleaner(TransformerMixin, BaseEstimator):
    """Replace nonpositive BP/cholesterol with NaN; keep all rows.

    This rule uses no fitted statistics. Oldpeak=0, negative oldpeak, and binary
    zeros are retained. Median imputation is learned later in the pipeline.
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass a pandas DataFrame with named columns.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Input columns must match the fitted pipeline.")
        clean = X.copy()
        for col in ["RestingBP", "Cholesterol"]:
            if col in clean:
                clean[col] = clean[col].astype(float).mask(clean[col] <= 0, np.nan)
        for col in CATEGORICAL:
            if col in clean:
                clean[col] = clean[col].astype(object)
        return clean

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


def make_model_pipeline(estimator, *, scale_numeric=False, features=None) -> Pipeline:
    features = FEATURES if features is None else features
    numeric = [x for x in NUMERIC if x in features]
    categorical = [x for x in CATEGORICAL if x in features]
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("clean", InvalidMeasurementCleaner()),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a binomial proportion."""
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def calculate_metrics(y_true, y_pred, probability) -> dict:
    """Metrics with rows=actual, columns=predicted, class order [0, 1]."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc_ci = wilson_interval(int(tp + tn), int(tp + tn + fp + fn))
    recall_ci = wilson_interval(int(tp), int(tp + fn))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy_ci_low": acc_ci[0],
        "accuracy_ci_high": acc_ci[1],
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
    }


def evaluate_pipeline(pipeline, X, y) -> tuple[dict, np.ndarray, np.ndarray]:
    classes = list(pipeline.classes_)
    probability = pipeline.predict_proba(X)[:, classes.index(1)]
    predicted = pipeline.predict(X)
    return calculate_metrics(y, predicted, probability), predicted, probability


def subgroup_metrics(y, predictions, features: pd.DataFrame) -> pd.DataFrame:
    """Descriptive test-set audit; small samples do not establish fairness."""
    groups = {
        "Sex: female": features["Sex"].eq("F").to_numpy(),
        "Sex: male": features["Sex"].eq("M").to_numpy(),
        "Age: below 55": features["Age"].lt(55).to_numpy(),
        "Age: 55 or above": features["Age"].ge(55).to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        actual, pred = np.asarray(y)[mask], np.asarray(predictions)[mask]
        tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
        lo, hi = wilson_interval(int(tp), int(tp + fn))
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "positive_cases": int(tp + fn),
                "accuracy": (
                    float(accuracy_score(actual, pred)) if len(actual) else np.nan
                ),
                "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
                "recall_ci_low": lo,
                "recall_ci_high": hi,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "false_negatives": int(fn),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def extract_leaf_rules(pipeline) -> pd.DataFrame:
    """Full paths to fitted CART leaves, with training support and class fraction."""
    tree = pipeline.named_steps["model"].tree_
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    classes = list(pipeline.classes_)
    rows = []

    def condition(feature, threshold, left):
        op = "<=" if left else ">"
        if feature.startswith("missingindicator_"):
            return f"{feature.removeprefix('missingindicator_')} is {'observed' if left else 'missing'}"
        for col in CATEGORICAL:
            if feature.startswith(col + "_") and abs(threshold - 0.5) < 1e-8:
                value = feature[len(col) + 1 :]
                return f"{col} {'is not' if left else 'is'} {value}"
        return f"{feature} {op} {threshold:.2f}"

    def walk(node, path):
        if tree.children_left[node] == tree.children_right[node]:
            values = tree.value[node][0]
            rows.append(
                {
                    "leaf": int(node),
                    "rule": " AND ".join(path) or "All records",
                    "training_records": int(tree.n_node_samples[node]),
                    "predicted_class": int(classes[int(np.argmax(values))]),
                    "fitted_disease_fraction": float(
                        values[classes.index(1)] / values.sum()
                    ),
                }
            )
            return
        feature = str(names[tree.feature[node]])
        walk(
            tree.children_left[node],
            path + [condition(feature, tree.threshold[node], True)],
        )
        walk(
            tree.children_right[node],
            path + [condition(feature, tree.threshold[node], False)],
        )

    walk(0, [])
    return (
        pd.DataFrame(rows)
        .sort_values("training_records", ascending=False)
        .reset_index(drop=True)
    )
