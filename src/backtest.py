"""Historical FIFA World Cup backtesting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from src.model import multiclass_brier_score


def _actual_champion(year_matches: pd.DataFrame) -> str:
    ordered = year_matches.sort_values("date")
    for match in reversed(list(ordered.itertuples(index=False))):
        if match.result == "A":
            return str(match.team_a)
        if match.result == "B":
            return str(match.team_b)
    return "Unknown"


def run_world_cup_backtest(
    model: Any,
    label_encoder: LabelEncoder,
    features: pd.DataFrame,
    feature_columns: list[str],
    outputs_dir: Path,
    visualizations_dir: Path,
) -> pd.DataFrame:
    """Evaluate model predictions on historical FIFA World Cup tournaments."""
    world_cup = features[
        features["tournament"].astype(str).eq("FIFA World Cup")
    ].copy()
    world_cup = world_cup[world_cup["year"] < 2026]
    rows: list[dict[str, Any]] = []
    if world_cup.empty:
        output = pd.DataFrame(
            columns=[
                "tournament",
                "year",
                "predicted_champion",
                "actual_champion",
                "predicted_finalist_1",
                "predicted_finalist_2",
                "match_accuracy",
                "log_loss",
                "brier_score",
                "champion_hit",
                "matches",
            ]
        )
    else:
        for year, group in world_cup.groupby("year"):
            y_true = label_encoder.transform(group["result"])
            probabilities = model.predict_proba(group[feature_columns])
            y_pred = model.predict(group[feature_columns])
            labels = label_encoder.inverse_transform(model.classes_)
            classes = model.classes_
            team_scores: dict[str, float] = {}
            for match, row_probabilities in zip(
                group.itertuples(index=False), probabilities, strict=True
            ):
                probability_map = {
                    str(label): float(probability)
                    for label, probability in zip(labels, row_probabilities, strict=True)
                }
                team_scores[str(match.team_a)] = team_scores.get(str(match.team_a), 0.0) + probability_map.get("A", 0.0) + 0.5 * probability_map.get("D", 0.0)
                team_scores[str(match.team_b)] = team_scores.get(str(match.team_b), 0.0) + probability_map.get("B", 0.0) + 0.5 * probability_map.get("D", 0.0)
            ranked = sorted(team_scores, key=team_scores.get, reverse=True)
            predicted_champion = ranked[0] if ranked else "Unknown"
            predicted_finalists = (ranked + ["Unknown", "Unknown"])[:2]
            actual_champion = _actual_champion(group)
            rows.append(
                {
                    "tournament": "FIFA World Cup",
                    "year": int(year),
                    "predicted_champion": predicted_champion,
                    "actual_champion": actual_champion,
                    "predicted_finalist_1": predicted_finalists[0],
                    "predicted_finalist_2": predicted_finalists[1],
                    "match_accuracy": float(accuracy_score(y_true, y_pred)),
                    "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
                    "brier_score": multiclass_brier_score(y_true, probabilities, classes),
                    "champion_hit": predicted_champion == actual_champion,
                    "matches": int(len(group)),
                }
            )
        output = pd.DataFrame(rows).sort_values("year")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(outputs_dir / "backtest_results.csv", index=False)
    plot_backtest_summary(output, visualizations_dir / "backtest_summary.png")
    return output


def plot_backtest_summary(backtest: pd.DataFrame, output_path: Path) -> None:
    """Save a summary plot for historical World Cup backtests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    if backtest.empty:
        ax.text(0.5, 0.5, "No FIFA World Cup backtest rows", ha="center", va="center")
        ax.axis("off")
    else:
        ax.plot(backtest["year"], backtest["match_accuracy"], marker="o", label="Accuracy")
        ax.plot(
            backtest["year"],
            1 - np.clip(backtest["brier_score"], 0, 1),
            marker="o",
            label="1 - Brier",
        )
        hits = backtest[backtest["champion_hit"]]
        ax.scatter(
            hits["year"],
            hits["match_accuracy"],
            s=80,
            color="#00FF87",
            label="Champion hit",
            zorder=3,
        )
        ax.set_ylim(0, 1)
        ax.set_title("FIFA World Cup Historical Backtest")
        ax.set_xlabel("Tournament year")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
