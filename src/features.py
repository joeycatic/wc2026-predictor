"""Leakage-safe feature engineering for football match prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.player_features import (
    PLAYER_FEATURE_COLUMNS,
    build_player_lookup,
    lookup_player_snapshot,
    player_feature_defaults,
    player_pair_features,
)
from src.preprocessing import normalize_team_name

BASE_FEATURE_COLUMNS = [
    "elo_diff",
    "elo_a",
    "elo_b",
    "form_a",
    "form_b",
    "goals_scored_avg_a",
    "goals_scored_avg_b",
    "goals_conceded_avg_a",
    "goals_conceded_avg_b",
    "h2h_wins_a",
    "h2h_wins_b",
    "stage_weight",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS

STAGE_WEIGHTS = {
    "group": 1,
    "first stage": 1,
    "round of 32": 2,
    "r32": 2,
    "round of 16": 2,
    "r16": 2,
    "quarter-final": 3,
    "quarterfinal": 3,
    "quarter-finals": 3,
    "quarterfinals": 3,
    "semi-final": 4,
    "semifinal": 4,
    "semi-finals": 4,
    "semifinals": 4,
    "third-place": 4,
    "third place": 4,
    "final": 5,
}


@dataclass(frozen=True)
class TeamSnapshot:
    """Historical team-state features computed before a match."""

    form: float
    goals_scored_avg: float
    goals_conceded_avg: float
    consistency: float


def stage_to_weight(stage: object) -> int:
    """Encode a tournament stage as an ordinal importance weight.

    Args:
        stage: Raw stage label.

    Returns:
        Stage weight from 1 to 5.
    """
    value = str(stage or "group").strip().lower()
    for key, weight in STAGE_WEIGHTS.items():
        if key in value:
            return weight
    return 1


def result_label(score_a: int, score_b: int) -> str:
    """Map a scoreline to the target class.

    Args:
        score_a: Team A goals.
        score_b: Team B goals.

    Returns:
        A, B, or D.
    """
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "D"


def get_elo_before(
    elo: pd.DataFrame, team: str, match_date: pd.Timestamp, fallback: float
) -> float:
    """Return the latest available ELO for a team before a match date.

    Args:
        elo: Cleaned ELO dataframe.
        team: Team name.
        match_date: Match date.
        fallback: Rating used when no prior team rating exists.

    Returns:
        Latest pre-match ELO rating.
    """
    normalized = normalize_team_name(team)
    team_rows = elo[(elo["team"] == normalized) & (elo["date"] < match_date)]
    if team_rows.empty:
        return fallback
    return float(team_rows.iloc[-1]["elo_rating"])


def build_elo_lookup(elo: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build numpy arrays for fast pre-match ELO lookup.

    Args:
        elo: Cleaned ELO dataframe.

    Returns:
        Mapping from team name to sorted date and rating arrays.
    """
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for team, group in elo.sort_values("date").groupby("team"):
        lookup[str(team)] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["elo_rating"].to_numpy(dtype=float),
        )
    return lookup


def lookup_elo_before(
    lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    team: str,
    match_date: pd.Timestamp,
    fallback: float,
) -> float:
    """Return the latest available ELO before a date from a lookup table.

    Args:
        lookup: Team ELO lookup from build_elo_lookup.
        team: Team name.
        match_date: Match date.
        fallback: Rating used when no prior rating exists.

    Returns:
        Latest pre-match ELO.
    """
    dates_and_ratings = lookup.get(normalize_team_name(team))
    if dates_and_ratings is None:
        return fallback
    dates, ratings = dates_and_ratings
    index = int(np.searchsorted(dates, np.datetime64(match_date), side="left") - 1)
    if index < 0:
        return fallback
    return float(ratings[index])


def team_snapshot(history: list[tuple[int, int]]) -> TeamSnapshot:
    """Compute last-five form and goal averages from previous matches.

    Args:
        history: Previous matches as (goals_for, goals_against).

    Returns:
        TeamSnapshot with neutral defaults when history is absent.
    """
    recent = history[-5:]
    if not recent:
        return TeamSnapshot(0.5, 1.0, 1.0, 0.5)

    wins = sum(1 for goals_for, goals_against in recent if goals_for > goals_against)
    goals_for_values = [goals_for for goals_for, _ in recent]
    goals_against_values = [goals_against for _, goals_against in recent]
    results = [
        1.0 if goals_for > goals_against else 0.5 if goals_for == goals_against else 0.0
        for goals_for, goals_against in recent
    ]
    consistency = 1.0 - min(float(np.std(results)), 0.5) / 0.5
    return TeamSnapshot(
        form=wins / len(recent),
        goals_scored_avg=float(np.mean(goals_for_values)),
        goals_conceded_avg=float(np.mean(goals_against_values)),
        consistency=consistency,
    )


def h2h_wins(
    pair_history: list[tuple[str, int, int]], team_a: str, team_b: str
) -> tuple[int, int]:
    """Count head-to-head wins over the last five previous meetings.

    Args:
        pair_history: Previous pair meetings as (home_team, home_score, away_score).
        team_a: Team A name.
        team_b: Team B name.

    Returns:
        Tuple of team A wins and team B wins.
    """
    wins_a = 0
    wins_b = 0
    for home_team, home_score, away_score in pair_history[-5:]:
        away_team = team_b if home_team == team_a else team_a
        winner = result_label(home_score, away_score)
        if winner == "D":
            continue
        winning_team = home_team if winner == "A" else away_team
        if winning_team == team_a:
            wins_a += 1
        elif winning_team == team_b:
            wins_b += 1
    return wins_a, wins_b


def filter_training_matches(results: pd.DataFrame) -> pd.DataFrame:
    """Prefer World Cup matches but fall back to all results if needed.

    Args:
        results: Cleaned match results.

    Returns:
        Filtered training match dataframe.
    """
    world_cup_mask = (
        results["tournament"]
        .astype(str)
        .str.contains("FIFA World Cup", case=False, na=False)
    )
    world_cup = results[world_cup_mask].copy()
    if len(world_cup) >= 100:
        return world_cup
    return results.copy()


def get_feature_columns(use_player_features: bool) -> list[str]:
    """Return model feature columns for the requested feature set."""
    if use_player_features:
        return FEATURE_COLUMNS
    return BASE_FEATURE_COLUMNS.copy()


def build_feature_dataset(
    results: pd.DataFrame,
    elo: pd.DataFrame,
    player_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a leakage-safe feature matrix from historical matches.

    Args:
        results: Cleaned match results.
        elo: Cleaned ELO ratings.
        player_features: Optional team-season player aggregates.

    Returns:
        Feature dataframe containing target labels and metadata.
    """
    matches = (
        filter_training_matches(results).sort_values("date").reset_index(drop=True)
    )
    fallback_elo = float(elo["elo_rating"].median()) if not elo.empty else 1500.0
    elo_lookup = build_elo_lookup(elo)
    player_lookup = build_player_lookup(player_features)
    player_defaults = player_feature_defaults(player_features)
    team_histories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pair_histories: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(
        list
    )
    rows: list[dict[str, Any]] = []

    for _, match in matches.iterrows():
        team_a = normalize_team_name(match["home_team"])
        team_b = normalize_team_name(match["away_team"])
        match_date = pd.Timestamp(match["date"])
        score_a = int(match["home_score"])
        score_b = int(match["away_score"])
        pair_key = tuple(sorted((team_a, team_b)))

        snapshot_a = team_snapshot(team_histories[team_a])
        snapshot_b = team_snapshot(team_histories[team_b])
        h2h_a, h2h_b = h2h_wins(pair_histories[pair_key], team_a, team_b)
        elo_a = lookup_elo_before(elo_lookup, team_a, match_date, fallback_elo)
        elo_b = lookup_elo_before(elo_lookup, team_b, match_date, fallback_elo)
        player_a = lookup_player_snapshot(
            player_lookup, team_a, match_date.year, player_defaults, before_year=True
        )
        player_b = lookup_player_snapshot(
            player_lookup, team_b, match_date.year, player_defaults, before_year=True
        )

        row = {
            "date": match_date,
            "year": int(match_date.year),
            "tournament": str(match.get("tournament", "Unknown")),
            "stage": str(match.get("stage", "Group")),
            "team_a": team_a,
            "team_b": team_b,
            "elo_diff": elo_a - elo_b,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "form_a": snapshot_a.form,
            "form_b": snapshot_b.form,
            "goals_scored_avg_a": snapshot_a.goals_scored_avg,
            "goals_scored_avg_b": snapshot_b.goals_scored_avg,
            "goals_conceded_avg_a": snapshot_a.goals_conceded_avg,
            "goals_conceded_avg_b": snapshot_b.goals_conceded_avg,
            "h2h_wins_a": h2h_a,
            "h2h_wins_b": h2h_b,
            "stage_weight": stage_to_weight(match.get("stage", "Group")),
            "result": result_label(score_a, score_b),
        }
        row.update(player_pair_features(player_a, player_b))
        rows.append(row)

        team_histories[team_a].append((score_a, score_b))
        team_histories[team_b].append((score_b, score_a))
        pair_histories[pair_key].append((team_a, score_a, score_b))

    features = pd.DataFrame(rows)
    return features.dropna(subset=FEATURE_COLUMNS + ["result"]).reset_index(drop=True)


def build_match_features(
    team_a: str,
    team_b: str,
    match_date: pd.Timestamp,
    results: pd.DataFrame,
    elo: pd.DataFrame,
    stage: str = "Group",
    player_features: pd.DataFrame | None = None,
    use_player_features: bool = False,
) -> pd.DataFrame:
    """Build a one-row feature frame for a future match.

    Args:
        team_a: First team.
        team_b: Second team.
        match_date: Prediction date.
        results: Cleaned historical match results.
        elo: Cleaned ELO ratings.
        stage: Match stage.
        player_features: Optional team-season player aggregates.
        use_player_features: Whether to include player feature columns.

    Returns:
        One-row dataframe with model feature columns.
    """
    normalized_a = normalize_team_name(team_a)
    normalized_b = normalize_team_name(team_b)
    prior = results[results["date"] < match_date].sort_values("date")
    team_histories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pair_history: list[tuple[str, int, int]] = []

    for _, match in prior.iterrows():
        home = normalize_team_name(match["home_team"])
        away = normalize_team_name(match["away_team"])
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        team_histories[home].append((home_score, away_score))
        team_histories[away].append((away_score, home_score))
        if set((home, away)) == set((normalized_a, normalized_b)):
            pair_history.append((home, home_score, away_score))

    snapshot_a = team_snapshot(team_histories[normalized_a])
    snapshot_b = team_snapshot(team_histories[normalized_b])
    fallback_elo = float(elo["elo_rating"].median()) if not elo.empty else 1500.0
    elo_a = get_elo_before(elo, normalized_a, match_date, fallback_elo)
    elo_b = get_elo_before(elo, normalized_b, match_date, fallback_elo)
    h2h_a, h2h_b = h2h_wins(pair_history, normalized_a, normalized_b)
    row = {
        "elo_diff": elo_a - elo_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "form_a": snapshot_a.form,
        "form_b": snapshot_b.form,
        "goals_scored_avg_a": snapshot_a.goals_scored_avg,
        "goals_scored_avg_b": snapshot_b.goals_scored_avg,
        "goals_conceded_avg_a": snapshot_a.goals_conceded_avg,
        "goals_conceded_avg_b": snapshot_b.goals_conceded_avg,
        "h2h_wins_a": h2h_a,
        "h2h_wins_b": h2h_b,
        "stage_weight": stage_to_weight(stage),
    }
    if use_player_features:
        player_lookup = build_player_lookup(player_features)
        player_defaults = player_feature_defaults(player_features)
        player_a = lookup_player_snapshot(
            player_lookup, normalized_a, match_date.year, player_defaults
        )
        player_b = lookup_player_snapshot(
            player_lookup, normalized_b, match_date.year, player_defaults
        )
        row.update(player_pair_features(player_a, player_b))

    return pd.DataFrame(
        [row],
        columns=get_feature_columns(use_player_features),
    )


def latest_team_elo(
    elo: pd.DataFrame, team: str, fallback: float | None = None
) -> float:
    """Return a team's latest available ELO rating.

    Args:
        elo: Cleaned ELO ratings.
        team: Team name.
        fallback: Optional fallback if no rating exists.

    Returns:
        Latest ELO rating.
    """
    if fallback is None:
        fallback = float(elo["elo_rating"].median()) if not elo.empty else 1500.0
    normalized = normalize_team_name(team)
    rows = elo[elo["team"] == normalized]
    if rows.empty:
        return fallback
    return float(rows.sort_values("date").iloc[-1]["elo_rating"])
