"""Model training, evaluation, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.features import get_feature_columns
from src.player_features import PLAYER_FEATURE_COLUMNS

RANDOM_SEED = 42
CLASS_LABELS = {"A": "Win A", "D": "Draw", "B": "Win B"}
LOGGER = logging.getLogger(__name__)


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


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    """Compute top-label expected calibration error."""
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidences = np.max(probabilities, axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
        )
    return ece


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Return compact numeric metrics for a fitted classifier."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier_score": multiclass_brier_score(y_true, probabilities, classes),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, classes
        ),
    }


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


def save_feature_importance(
    model: VotingClassifier,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Save mean tree-based feature importances from ensemble members."""
    importances: list[np.ndarray] = []
    for estimator in model.estimators_:
        values = getattr(estimator, "feature_importances_", None)
        if values is not None:
            total = float(np.sum(values))
            importances.append(values / total if total else values)

    if importances:
        mean_importance = np.mean(importances, axis=0)
    else:
        mean_importance = np.zeros(len(feature_columns))
    output = pd.DataFrame(
        {"feature": feature_columns, "importance": mean_importance}
    ).sort_values("importance", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def train_baseline_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_encoder: LabelEncoder,
    outputs_dir: Path,
) -> dict[str, Any]:
    """Train compact ELO-only and ELO+player baselines."""
    baseline_specs = {
        "elo_only": ["elo_diff", "elo_a", "elo_b"],
    }
    if set(PLAYER_FEATURE_COLUMNS).issubset(train.columns):
        baseline_specs["elo_player"] = [
            "elo_diff",
            "elo_a",
            "elo_b",
            *PLAYER_FEATURE_COLUMNS,
        ]

    y_train = label_encoder.transform(train["result"])
    y_test = label_encoder.transform(test["result"])
    metrics: dict[str, Any] = {}
    for name, columns in baseline_specs.items():
        baseline = RandomForestClassifier(
            n_estimators=150, max_depth=4, random_state=RANDOM_SEED
        )
        baseline.fit(train[columns], y_train)
        probabilities = baseline.predict_proba(test[columns])
        y_pred = baseline.predict(test[columns])
        metrics[name] = evaluate_predictions(
            y_test, y_pred, probabilities, baseline.classes_
        )
        metrics[name]["features"] = columns

    with (outputs_dir / "baseline_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def train_and_evaluate(
    features: pd.DataFrame,
    outputs_dir: Path,
    visualizations_dir: Path,
    use_player_features: bool = False,
) -> tuple[VotingClassifier, LabelEncoder, dict[str, Any]]:
    """Train the ensemble and write all requested evaluation artifacts.

    Args:
        features: Leakage-safe feature dataframe.
        outputs_dir: Directory for metrics and model artifacts.
        visualizations_dir: Directory for plots.
        use_player_features: Whether to train on player-enhanced columns.

    Returns:
        Fitted model, fitted label encoder, and metrics dictionary.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    train, test = chronological_split(features)
    feature_columns = get_feature_columns(use_player_features)
    label_encoder = LabelEncoder()
    label_encoder.fit(features["result"])

    x_train = train[feature_columns]
    y_train = label_encoder.transform(train["result"])
    x_test = test[feature_columns]
    y_test = label_encoder.transform(test["result"])

    model = build_ensemble()
    encoded_all = label_encoder.transform(features["result"])
    if len(features) > 6:
        cv = TimeSeriesSplit(n_splits=min(5, len(features) - 1))
        try:
            cv_scores = cross_val_score(
                model,
                features[feature_columns],
                encoded_all,
                cv=cv,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping time-series CV: %s", exc)
            cv_scores = np.array([np.nan])
    else:
        LOGGER.warning("Skipping time-series CV: not enough rows")
        cv_scores = np.array([np.nan])
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
    compact_metrics = evaluate_predictions(y_test, y_pred, probabilities, classes)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    baseline_metrics = train_baseline_metrics(train, test, label_encoder, outputs_dir)

    metrics: dict[str, Any] = {
        **compact_metrics,
        "feature_set": "player_enhanced" if use_player_features else "base",
        "feature_columns": feature_columns,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "classes": labels.tolist(),
        "time_series_cv_accuracy_mean": float(np.nanmean(cv_scores)),
        "time_series_cv_accuracy_std": float(np.nanstd(cv_scores)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline_metrics_file": "baseline_metrics.json",
    }
    if use_player_features:
        base_brier = baseline_metrics.get("elo_only", {}).get("brier_score")
        player_brier = baseline_metrics.get("elo_player", {}).get("brier_score")
        base_accuracy = baseline_metrics.get("elo_only", {}).get("accuracy")
        player_accuracy = baseline_metrics.get("elo_player", {}).get("accuracy")
        if (
            base_brier is not None
            and player_brier is not None
            and base_accuracy is not None
            and player_accuracy is not None
        ):
            metrics["player_feature_note"] = (
                "player features help accuracy but hurt calibration"
                if player_accuracy > base_accuracy and player_brier > base_brier
                else "player baseline comparison recorded"
            )

    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_confusion_matrix(
        y_test,
        y_pred,
        label_encoder,
        classes,
        visualizations_dir / "confusion_matrix.png",
    )
    save_feature_importance(
        model, feature_columns, outputs_dir / "feature_importance.csv"
    )
    joblib.dump(model, outputs_dir / "ensemble_model.pkl")
    joblib.dump(label_encoder, outputs_dir / "label_encoder.pkl")

    print(
        "TimeSeriesSplit accuracy: "
        f"{np.nanmean(cv_scores):.3f} +/- {np.nanstd(cv_scores):.3f}"
    )
    print(f"Chronological test accuracy: {compact_metrics['accuracy']:.3f}")
    print(f"Multiclass Brier score: {compact_metrics['brier_score']:.3f}")
    print(f"Log loss: {compact_metrics['log_loss']:.3f}")
    return model, label_encoder, metrics
