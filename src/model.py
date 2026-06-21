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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility with older scikit-learn
    FrozenEstimator = None  # type: ignore[assignment]

RANDOM_SEED = 42
CLASS_LABELS = {"A": "Win A", "D": "Draw", "B": "Win B"}
LOGGER = logging.getLogger(__name__)
MODEL_FILE = "ensemble_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"


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


def chronological_train_calibration_test_split(
    features: pd.DataFrame,
    train_fraction: float = 0.7,
    calibration_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by time into train, calibration, and test partitions."""
    ordered = features.sort_values("date").reset_index(drop=True)
    if len(ordered) < 6:
        train, test = chronological_split(ordered, train_fraction=0.8)
        return train, test.copy(), test

    train_end = int(len(ordered) * train_fraction)
    calibration_end = int(len(ordered) * (train_fraction + calibration_fraction))
    train_end = min(max(train_end, 1), len(ordered) - 2)
    calibration_end = min(max(calibration_end, train_end + 1), len(ordered) - 1)
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:calibration_end].copy(),
        ordered.iloc[calibration_end:].copy(),
    )


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
    model: Any,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Save mean tree-based feature importances from ensemble members."""
    if isinstance(model, CalibratedClassifierCV):
        first_calibrated = model.calibrated_classifiers_[0]
        model = getattr(first_calibrated, "estimator", model)

    importances: list[np.ndarray] = []
    for estimator in getattr(model, "estimators_", []):
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


def save_calibration_curves(
    y_true: np.ndarray,
    before_probabilities: np.ndarray,
    after_probabilities: np.ndarray,
    classes: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    """Save one-vs-rest calibration curves before and after calibration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
    labels = label_encoder.inverse_transform(classes)
    for ax, class_id, label in zip(axes, classes, labels, strict=True):
        observed = (y_true == class_id).astype(int)
        before_true, before_pred = calibration_curve(
            observed,
            before_probabilities[:, list(classes).index(class_id)],
            n_bins=8,
            strategy="uniform",
        )
        after_true, after_pred = calibration_curve(
            observed,
            after_probabilities[:, list(classes).index(class_id)],
            n_bins=8,
            strategy="uniform",
        )
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        ax.plot(before_pred, before_true, marker="o", label="Before")
        ax.plot(after_pred, after_true, marker="o", label="After")
        ax.set_title(CLASS_LABELS.get(str(label), str(label)))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Calibration Curves - Chronological Test Set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_analysis(
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> pd.DataFrame:
    """Save per-team and per-confederation test-set error diagnostics."""
    from src.simulate import CONFEDERATIONS

    label_to_column = {label: index for index, label in enumerate(classes)}
    rows: list[dict[str, Any]] = []
    test = test.reset_index(drop=True)
    pred_labels = label_encoder.inverse_transform(y_pred)
    true_labels = label_encoder.inverse_transform(y_true)

    expanded: list[dict[str, Any]] = []
    for index, match in test.iterrows():
        actual = str(true_labels[index])
        predicted = str(pred_labels[index])
        true_column = label_to_column[y_true[index]]
        brier_vector = np.zeros(len(classes))
        brier_vector[true_column] = 1.0
        for team_column in ("team_a", "team_b"):
            team = str(match[team_column])
            expanded.append(
                {
                    "team": team,
                    "confederation": CONFEDERATIONS.get(team, "Other"),
                    "actual": actual,
                    "predicted": predicted,
                    "correct": actual == predicted,
                    "actual_probability": probabilities[index, true_column],
                    "brier": float(np.sum((probabilities[index] - brier_vector) ** 2)),
                    "confidence": float(np.max(probabilities[index])),
                }
            )
    expanded_frame = pd.DataFrame(expanded)
    if expanded_frame.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        expanded_frame.to_csv(output_path, index=False)
        return expanded_frame

    for scope, group_key in (("team", "team"), ("confederation", "confederation")):
        for name, group in expanded_frame.groupby(group_key):
            failures = group[group["correct"] == 0]
            if failures.empty:
                failure_mode = "none"
            else:
                failure_mode = (
                    failures.assign(
                        mode=failures["actual"] + "->" + failures["predicted"]
                    )["mode"]
                    .value_counts()
                    .idxmax()
                )
            rows.append(
                {
                    "scope": scope,
                    "name": name,
                    "sample_count": int(len(group)),
                    "accuracy": float(group["correct"].mean()),
                    "log_loss": float(-np.log(np.clip(group["actual_probability"], 1e-15, 1)).mean()),
                    "brier_score": float(group["brier"].mean()),
                    "calibration_error": float(
                        abs(group["correct"].mean() - group["confidence"].mean())
                    ),
                    "most_common_failure_mode": failure_mode,
                }
            )
    output = pd.DataFrame(rows).sort_values(["scope", "accuracy", "sample_count"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def plot_error_analysis(error_analysis: pd.DataFrame, output_path: Path) -> None:
    """Plot teams with the highest test-set log loss among meaningful samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if error_analysis.empty:
        return
    team_rows = error_analysis[
        (error_analysis["scope"] == "team") & (error_analysis["sample_count"] >= 2)
    ].copy()
    if team_rows.empty:
        team_rows = error_analysis[error_analysis["scope"] == "team"].copy()
    top = team_rows.sort_values("log_loss", ascending=False).head(16)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["name"], top["log_loss"], color="#EF4444")
    ax.set_title("Highest Team Error - Chronological Test Set")
    ax.set_xlabel("Log loss")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    player_feature_status: dict[str, Any] | None = None,
    optional_source_status: dict[str, Any] | None = None,
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

    train, calibration, test = chronological_train_calibration_test_split(features)
    feature_columns = get_feature_columns(use_player_features)
    label_encoder = LabelEncoder()
    label_encoder.fit(features["result"])

    x_train = train[feature_columns]
    y_train = label_encoder.transform(train["result"])
    x_calibration = calibration[feature_columns]
    y_calibration = label_encoder.transform(calibration["result"])
    x_test = test[feature_columns]
    y_test = label_encoder.transform(test["result"])

    base_model = build_ensemble()
    encoded_all = label_encoder.transform(features["result"])
    if len(features) > 6:
        cv = TimeSeriesSplit(n_splits=min(5, len(features) - 1))
        try:
            cv_scores = cross_val_score(
                base_model,
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
    base_model.fit(x_train, y_train)

    before_probabilities = base_model.predict_proba(x_test)
    before_pred = base_model.predict(x_test)
    before_metrics = evaluate_predictions(
        y_test, before_pred, before_probabilities, base_model.classes_
    )

    try:
        if FrozenEstimator is None:
            model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
        else:
            model = CalibratedClassifierCV(
                FrozenEstimator(base_model),
                method="sigmoid",
            )
        model.fit(x_calibration, y_calibration)
    except ValueError as exc:
        LOGGER.warning("Skipping calibration and using uncalibrated model: %s", exc)
        model = base_model

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
    baseline_train = pd.concat([train, calibration], ignore_index=True)
    baseline_metrics = train_baseline_metrics(
        baseline_train, test, label_encoder, outputs_dir
    )

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
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "calibration": {
            "enabled": isinstance(model, CalibratedClassifierCV),
            "method": "sigmoid",
            "before": before_metrics,
            "after": compact_metrics,
        },
        "baseline_metrics_file": "baseline_metrics.json",
    }
    if player_feature_status is not None:
        metrics["player_feature_coverage"] = player_feature_status
    if optional_source_status is not None:
        metrics["optional_sources"] = optional_source_status
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
    save_calibration_curves(
        y_test,
        before_probabilities,
        probabilities,
        classes,
        label_encoder,
        visualizations_dir / "calibration_curves.png",
    )
    error_analysis = save_error_analysis(
        test,
        y_test,
        y_pred,
        probabilities,
        classes,
        label_encoder,
        outputs_dir / "error_analysis.csv",
    )
    plot_error_analysis(error_analysis, visualizations_dir / "error_analysis.png")
    save_feature_importance(
        base_model, feature_columns, outputs_dir / "feature_importance.csv"
    )
    joblib.dump(model, outputs_dir / MODEL_FILE)
    joblib.dump(label_encoder, outputs_dir / LABEL_ENCODER_FILE)

    print(
        "TimeSeriesSplit accuracy: "
        f"{np.nanmean(cv_scores):.3f} +/- {np.nanstd(cv_scores):.3f}"
    )
    print(f"Chronological test accuracy: {compact_metrics['accuracy']:.3f}")
    print(f"Multiclass Brier score: {compact_metrics['brier_score']:.3f}")
    print(f"Log loss: {compact_metrics['log_loss']:.3f}")
    return model, label_encoder, metrics


def load_cached_model(outputs_dir: Path) -> tuple[Any, LabelEncoder]:
    """Load a previously trained model and label encoder."""
    model_path = outputs_dir / MODEL_FILE
    label_path = outputs_dir / LABEL_ENCODER_FILE
    missing = [str(path) for path in (model_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "--no-train requires cached model artifacts; missing: "
            + ", ".join(missing)
        )
    return joblib.load(model_path), joblib.load(label_path)
