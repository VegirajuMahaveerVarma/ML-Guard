"""Dataset-level performance metrics for classification and regression."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from mlguard.exceptions import InputValidationError
from mlguard.results import EvaluationReport

Task = Literal["classification", "regression"]
Average = Literal["macro", "weighted", "micro"]


def _one_dimensional(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} cannot be converted to an array.") from exc
    if array.ndim != 1:
        raise InputValidationError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise InputValidationError(f"{name} must contain at least one sample.")
    return array


def _label_scalar(value: Any, *, name: str) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise InputValidationError(
        f"{name} must contain finite numbers, strings, or boolean class labels."
    )


def _same_label(left: Any, right: Any) -> bool:
    return bool(left == right)


def _ordered_unique(values: Sequence[Any]) -> list[str | int | float | bool]:
    unique: list[str | int | float | bool] = []
    for value in values:
        if not any(_same_label(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Average,
    labels: Sequence[Any] | None,
) -> dict[str, Any]:
    true_values = [_label_scalar(value, name="y_true") for value in y_true]
    pred_values = [_label_scalar(value, name="y_pred") for value in y_pred]
    observed = true_values + pred_values

    if labels is None:
        class_labels = _ordered_unique(observed)
    else:
        class_labels = [
            _label_scalar(value, name="labels") for value in labels
        ]
        if not class_labels:
            raise InputValidationError("labels must contain at least one class.")
        if len(_ordered_unique(class_labels)) != len(class_labels):
            raise InputValidationError("labels must not contain duplicate classes.")
        absent = [
            value
            for value in observed
            if not any(_same_label(value, known) for known in class_labels)
        ]
        if absent:
            raise InputValidationError(
                "labels must include every class present in y_true and y_pred."
            )

    confusion = [
        [0 for _ in class_labels]
        for _ in class_labels
    ]
    per_class: list[dict[str, Any]] = []
    supports: list[int] = []
    true_positives: list[int] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for true_value, pred_value in zip(true_values, pred_values, strict=True):
        true_index = next(
            index
            for index, label in enumerate(class_labels)
            if _same_label(true_value, label)
        )
        pred_index = next(
            index
            for index, label in enumerate(class_labels)
            if _same_label(pred_value, label)
        )
        confusion[true_index][pred_index] += 1

    for index, label in enumerate(class_labels):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        fp = predicted_count - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        supports.append(support)
        true_positives.append(tp)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    sample_count = len(true_values)
    correct = sum(true_positives)
    accuracy = correct / sample_count
    if average == "macro":
        precision_score = float(np.mean(precisions))
        recall_score = float(np.mean(recalls))
        f1_score = float(np.mean(f1_scores))
    elif average == "weighted":
        precision_score = float(np.average(precisions, weights=supports))
        recall_score = float(np.average(recalls, weights=supports))
        f1_score = float(np.average(f1_scores, weights=supports))
    else:
        # For single-label multiclass classification, micro P/R/F1 equal accuracy.
        precision_score = accuracy
        recall_score = accuracy
        f1_score = accuracy

    return {
        "accuracy": float(accuracy),
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "average": average,
        "labels": class_labels,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    try:
        actual = y_true.astype(float)
        predicted = y_pred.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "Regression evaluation requires numeric y_true and predictions."
        ) from exc

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise InputValidationError(
            "Regression evaluation requires finite numeric targets and predictions."
        )

    errors = predicted - actual
    absolute = np.abs(errors)
    squared = np.square(errors)
    residual_sum = float(squared.sum())
    centered_sum = float(np.square(actual - actual.mean()).sum())

    r2_score: float | None
    if len(actual) < 2:
        r2_score = None
    elif centered_sum == 0:
        r2_score = 1.0 if residual_sum == 0 else 0.0
    else:
        r2_score = float(1.0 - residual_sum / centered_sum)

    return {
        "mae": float(absolute.mean()),
        "mse": float(squared.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "r2": r2_score,
        "max_absolute_error": float(absolute.max(initial=0.0)),
    }


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    task: Task,
    average: Average = "macro",
    labels: Sequence[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Compute evaluation metrics from ground-truth targets and model outputs.

    Classification is single-label and supports binary and multiclass targets.
    Regression currently supports one numeric target per sample. Classification
    averaging may be macro, weighted, or micro. This function computes metrics
    locally using NumPy and does not require scikit-learn.
    """
    if task not in {"classification", "regression"}:
        raise InputValidationError("task must be 'classification' or 'regression'.")
    if average not in {"macro", "weighted", "micro"}:
        raise InputValidationError(
            "average must be 'macro', 'weighted', or 'micro'."
        )

    actual = _one_dimensional(y_true, name="y_true")
    predicted = _one_dimensional(y_pred, name="y_pred")
    if len(actual) != len(predicted):
        raise InputValidationError(
            "y_true and y_pred must contain the same number of samples."
        )

    if task == "classification":
        metrics = _classification_metrics(
            actual,
            predicted,
            average=average,
            labels=labels,
        )
    else:
        if labels is not None:
            raise InputValidationError(
                "labels is only supported for classification evaluation."
            )
        metrics = _regression_metrics(actual, predicted)

    return EvaluationReport(
        task=task,
        sample_count=int(len(actual)),
        metrics=metrics,
        metadata=dict(metadata or {}),
    )
