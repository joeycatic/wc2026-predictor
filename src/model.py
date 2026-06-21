"""Model training, evaluation, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.features import FEATURE_COLUMNS

RANDOM_SEED = 42
CLASS_LABELS = {"A": "Win A", "D": "Draw", "B": "Win B"}


def build_ensemble() -> VotingClassifier:
    """Create the soft-voting ensemble specified by the project prompt.

    Returns:
        Configured sklearn VotingClassifier.
    """
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_SEED,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=RANDOM_SEED,
    )
    return VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("mlp", mlp)],
        voting="soft",
    )


def chronological_split(
    features: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature rows by time order without shuffling.

    Args:
        features: Feature dataframe sorted or sortable by date.
        train_fraction: Fraction of rows assigned to training.

    Returns:
        Tuple of train and test dataframes.
    """
    ordered = features.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    split_index = min(max(split_index, 1), len(ordered) - 1)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def multiclass_brier_score(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    """Compute the multiclass Brier score.

    Args:
        y_true: Encoded true labels.
        probabilities: Predicted class probabilities.
        classes: Model class labels matching probability columns.

    Returns:
        Mean squared probability error.
    """
    class_to_index = {label: index for index, label in enumerate(classes)}
    observed = np.zeros_like(probabilities, dtype=float)
    for row_index, label in enumerate(y_true):
        observed[row_index, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    classes: np.ndarray,
    output_path: Path,
) -> None:
    """Save a row-normalized confusion matrix heatmap.

    Args:
        y_true: Encoded true labels.
        y_pred: Encoded predicted labels.
        label_encoder: Fitted label encoder.
        classes: Class order used by the model.
        output_path: PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = label_encoder.inverse_transform(classes)
    display_labels = [CLASS_LABELS.get(label, label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            annotations[row, column] = (
                f"{matrix[row, column]}\n{normalized[row, column] * 100:.1f}%"
            )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Greens",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Row-normalized share"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix - FIFA WC 2026 Predictor", color="#FFFFFF")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    features: pd.DataFrame,
    outputs_dir: Path,
    visualizations_dir: Path,
) -> tuple[VotingClassifier, LabelEncoder, dict[str, Any]]:
    """Train the ensemble and write all requested evaluation artifacts.

    Args:
        features: Leakage-safe feature dataframe.
        outputs_dir: Directory for metrics and model artifacts.
        visualizations_dir: Directory for plots.

    Returns:
        Fitted model, fitted label encoder, and metrics dictionary.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    train, test = chronological_split(features)
    label_encoder = LabelEncoder()
    label_encoder.fit(features["result"])

    x_train = train[FEATURE_COLUMNS]
    y_train = label_encoder.transform(train["result"])
    x_test = test[FEATURE_COLUMNS]
    y_test = label_encoder.transform(test["result"])

    model = build_ensemble()
    cv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        model,
        features[FEATURE_COLUMNS],
        label_encoder.transform(features["result"]),
        cv=cv,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classes = model.classes_
    labels = label_encoder.inverse_transform(classes)
    target_names = [CLASS_LABELS.get(label, label) for label in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(accuracy_score(y_test, y_pred))
    brier = multiclass_brier_score(y_test, probabilities, classes)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "classification_report": report,
        "brier_score": brier,
        "confusion_matrix": matrix.tolist(),
        "classes": labels.tolist(),
        "time_series_cv_accuracy_mean": float(np.mean(cv_scores)),
        "time_series_cv_accuracy_std": float(np.std(cv_scores)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_confusion_matrix(
        y_test,
        y_pred,
        label_encoder,
        classes,
        visualizations_dir / "confusion_matrix.png",
    )
    joblib.dump(model, outputs_dir / "ensemble_model.pkl")
    joblib.dump(label_encoder, outputs_dir / "label_encoder.pkl")

    print(
        f"TimeSeriesSplit accuracy: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}"
    )
    print(f"Chronological test accuracy: {accuracy:.3f}")
    print(f"Multiclass Brier score: {brier:.3f}")
    return model, label_encoder, metrics
