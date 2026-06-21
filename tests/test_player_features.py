from __future__ import annotations

import pandas as pd

from src.features import build_feature_dataset
from src.player_features import (
    PLAYER_BASE_COLUMNS,
    aggregate_player_features,
    player_feature_defaults,
)


def _player_feature_frame(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in PLAYER_BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 70.0
    return frame


def test_player_country_aggregation_normalizes_country_names(tmp_path) -> None:
    csv_path = tmp_path / "players.csv"
    pd.DataFrame(
        [
            {
                "nationality_name": "USA",
                "fifa_version": 23,
                "overall": 80,
                "player_positions": "ST",
                "age": 24,
                "international_reputation": 2,
                "value_eur": 1000000,
                "weak_foot": 4,
                "skill_moves": 3,
            },
            {
                "nationality_name": "USA",
                "fifa_version": 23,
                "overall": 70,
                "player_positions": "CB",
                "age": 26,
                "international_reputation": 1,
                "value_eur": 500000,
                "weak_foot": 3,
                "skill_moves": 2,
            },
        ]
    ).to_csv(csv_path, index=False)

    features = aggregate_player_features(csv_path)

    row = features.iloc[0]
    assert row["team"] == "United States"
    assert row["season"] == 2023
    assert row["avg_overall"] == 75.0


def test_feature_join_uses_latest_snapshot_before_match_year() -> None:
    results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-01"),
                "home_team": "Germany",
                "away_team": "France",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "stage": "Group",
            }
        ]
    )
    elo = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2023-01-01"),
                "team": "Germany",
                "elo_rating": 1800.0,
            },
            {
                "date": pd.Timestamp("2023-01-01"),
                "team": "France",
                "elo_rating": 1810.0,
            },
        ]
    )
    player_features = _player_feature_frame(
        [
            {"team": "Germany", "season": 2022, "avg_overall": 70.0},
            {"team": "Germany", "season": 2024, "avg_overall": 90.0},
            {"team": "France", "season": 2022, "avg_overall": 80.0},
        ]
    )

    features = build_feature_dataset(results, elo, player_features)

    assert features.loc[0, "player_avg_overall_a"] == 70.0
    assert features.loc[0, "player_avg_overall_b"] == 80.0
    assert features.loc[0, "player_avg_overall_diff"] == -10.0


def test_missing_player_data_uses_neutral_defaults() -> None:
    defaults = player_feature_defaults(None)

    assert defaults["avg_overall"] == 70.0
    assert defaults["avg_age"] == 26.0
