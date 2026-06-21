"""World Cup 2026 tournament simulation and visualization utilities."""

from __future__ import annotations

import json
import logging
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from src.features import (
    get_feature_columns,
    h2h_wins,
    latest_team_elo,
    stage_to_weight,
    team_snapshot,
)
from src.player_features import (
    build_player_lookup,
    lookup_player_snapshot,
    player_feature_defaults,
    player_pair_features,
)
from src.preprocessing import normalize_team_name
from src.tournament import (
    TournamentConfig,
    all_config_teams,
    build_knockout_field,
    load_tournament_config,
)

RANDOM_SEED = 42
ACCENT = "#00FF87"
WHITE = "#FFFFFF"
LOGGER = logging.getLogger(__name__)
ROUND_STAGES = ("R32", "R16", "QF", "SF", "Final", "Champion")

WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

CONFEDERATIONS = {
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Australia": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "South Korea": "AFC",
    "Uzbekistan": "AFC",
    "Algeria": "CAF",
    "Cape Verde": "CAF",
    "DR Congo": "CAF",
    "Egypt": "CAF",
    "Ghana": "CAF",
    "Ivory Coast": "CAF",
    "Morocco": "CAF",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "Tunisia": "CAF",
    "Canada": "CONCACAF",
    "Curacao": "CONCACAF",
    "Haiti": "CONCACAF",
    "Mexico": "CONCACAF",
    "Panama": "CONCACAF",
    "United States": "CONCACAF",
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Croatia": "UEFA",
    "Czech Republic": "UEFA",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Netherlands": "UEFA",
    "Norway": "UEFA",
    "Portugal": "UEFA",
    "Scotland": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkey": "UEFA",
    "Jordan": "AFC",
    "New Zealand": "OFC",
}

CONFEDERATION_COLORS = {
    "UEFA": "#3B82F6",
    "CONMEBOL": "#FACC15",
    "CONCACAF": "#22C55E",
    "CAF": "#EF4444",
    "AFC": "#A855F7",
    "OFC": "#14B8A6",
}

KIT_COLORS = {
    "Argentina": "#75AADB",
    "Brazil": "#FEDD00",
    "England": "#FFFFFF",
    "France": "#1D4ED8",
    "Germany": "#FFFFFF",
    "Netherlands": "#FF7F00",
    "Portugal": "#C8102E",
    "Spain": "#AA151B",
    "United States": "#3C3B6E",
}


@dataclass
class PredictionContext:
    """Cached historical state used for future match prediction."""

    team_features: dict[str, dict[str, float]]
    h2h: dict[tuple[str, str], list[tuple[str, int, int]]]
    fallback_elo: float
    prediction_date: pd.Timestamp
    tournament_config: TournamentConfig
    use_player_features: bool
    feature_columns: list[str]


def all_teams(config: TournamentConfig | None = None) -> list[str]:
    """Return the 48 World Cup 2026 teams in group order.

    Returns:
        List of team names.
    """
    tournament_config = config or load_tournament_config()
    return all_config_teams(tournament_config)


def build_prediction_context(
    results: pd.DataFrame,
    elo: pd.DataFrame,
    prediction_date: pd.Timestamp | None = None,
    player_features: pd.DataFrame | None = None,
    use_player_features: bool = False,
    tournament_config: TournamentConfig | None = None,
) -> PredictionContext:
    """Precompute team and pair histories for fast future predictions.

    Args:
        results: Cleaned historical match results.
        elo: Cleaned ELO ratings.
        prediction_date: Optional date for future prediction context.
        player_features: Optional processed player-strength features.
        use_player_features: Whether prediction rows include player features.
        tournament_config: Optional loaded tournament configuration.

    Returns:
        PredictionContext instance.
    """
    if prediction_date is None:
        latest_result = results["date"].max()
        latest_rating = elo["date"].max()
        prediction_date = max(latest_result, latest_rating) + pd.Timedelta(days=1)
    config = tournament_config or load_tournament_config()

    prior = results[results["date"] < prediction_date].sort_values("date")
    team_histories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pair_histories: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(
        list
    )
    for _, match in prior.iterrows():
        home = normalize_team_name(match["home_team"])
        away = normalize_team_name(match["away_team"])
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        team_histories[home].append((home_score, away_score))
        team_histories[away].append((away_score, home_score))
        pair_histories[tuple(sorted((home, away)))].append(
            (home, home_score, away_score)
        )

    fallback_elo = float(elo["elo_rating"].median()) if not elo.empty else 1500.0
    player_lookup = build_player_lookup(player_features)
    player_defaults = player_feature_defaults(player_features)
    team_features: dict[str, dict[str, float]] = {}
    missing_elo: list[str] = []
    missing_player: list[str] = []
    for team in all_teams(config):
        normalized = normalize_team_name(team)
        snapshot = team_snapshot(team_histories[normalized])
        if elo[elo["team"] == normalized].empty:
            missing_elo.append(team)
        player_values = lookup_player_snapshot(
            player_lookup,
            normalized,
            int(prediction_date.year),
            player_defaults,
            before_year=False,
        )
        if normalized not in player_lookup:
            missing_player.append(team)
        team_features[team] = {
            "elo": latest_team_elo(elo, normalized, fallback_elo),
            "form": snapshot.form,
            "goals_scored_avg": snapshot.goals_scored_avg,
            "goals_conceded_avg": snapshot.goals_conceded_avg,
            "consistency": snapshot.consistency,
            **{f"player_{key}": value for key, value in player_values.items()},
        }
    if missing_elo:
        LOGGER.warning("Missing ELO mappings for teams: %s", ", ".join(missing_elo))
    if use_player_features and missing_player:
        LOGGER.warning(
            "Missing player mappings for teams: %s; neutral values used",
            ", ".join(missing_player),
        )
    return PredictionContext(
        team_features,
        dict(pair_histories),
        fallback_elo,
        prediction_date,
        config,
        use_player_features,
        get_feature_columns(use_player_features),
    )


def feature_frame_from_context(
    team_a: str,
    team_b: str,
    context: PredictionContext,
    stage: str,
) -> pd.DataFrame:
    """Create one feature row from a precomputed prediction context.

    Args:
        team_a: Team A.
        team_b: Team B.
        context: PredictionContext.
        stage: Match stage.

    Returns:
        One-row feature dataframe.
    """
    normalized_a = normalize_team_name(team_a)
    normalized_b = normalize_team_name(team_b)
    stats_a = context.team_features.get(team_a) or context.team_features.get(
        normalized_a
    )
    stats_b = context.team_features.get(team_b) or context.team_features.get(
        normalized_b
    )
    if stats_a is None:
        stats_a = {
            "elo": context.fallback_elo,
            "form": 0.5,
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "consistency": 0.5,
        }
    if stats_b is None:
        stats_b = {
            "elo": context.fallback_elo,
            "form": 0.5,
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "consistency": 0.5,
        }
    pair_key = tuple(sorted((normalized_a, normalized_b)))
    h2h_a, h2h_b = h2h_wins(context.h2h.get(pair_key, []), normalized_a, normalized_b)
    row = {
        "elo_diff": stats_a["elo"] - stats_b["elo"],
        "elo_a": stats_a["elo"],
        "elo_b": stats_b["elo"],
        "form_a": stats_a["form"],
        "form_b": stats_b["form"],
        "goals_scored_avg_a": stats_a["goals_scored_avg"],
        "goals_scored_avg_b": stats_b["goals_scored_avg"],
        "goals_conceded_avg_a": stats_a["goals_conceded_avg"],
        "goals_conceded_avg_b": stats_b["goals_conceded_avg"],
        "h2h_wins_a": h2h_a,
        "h2h_wins_b": h2h_b,
        "stage_weight": stage_to_weight(stage),
    }
    if context.use_player_features:
        player_a = {
            key.removeprefix("player_"): value
            for key, value in stats_a.items()
            if key.startswith("player_")
        }
        player_b = {
            key.removeprefix("player_"): value
            for key, value in stats_b.items()
            if key.startswith("player_")
        }
        row.update(player_pair_features(player_a, player_b))
    return pd.DataFrame([row], columns=context.feature_columns)


def predict_probabilities(
    model: Any,
    label_encoder: LabelEncoder,
    team_a: str,
    team_b: str,
    context: PredictionContext,
    stage: str,
    cache: dict[tuple[str, str, str], dict[str, float]],
) -> dict[str, float]:
    """Predict A/D/B probabilities for a match with caching.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        team_a: Team A.
        team_b: Team B.
        context: PredictionContext.
        stage: Match stage.
        cache: Mutable prediction cache.

    Returns:
        Probability dictionary with keys A, D, and B.
    """
    key = (team_a, team_b, stage)
    if key in cache:
        return cache[key]

    feature_frame = feature_frame_from_context(team_a, team_b, context, stage)
    probabilities = model.predict_proba(feature_frame)[0]
    labels = label_encoder.inverse_transform(model.classes_)
    result = {"A": 0.0, "D": 0.0, "B": 0.0}
    for label, probability in zip(labels, probabilities, strict=True):
        result[str(label)] = float(probability)
    total = sum(result.values())
    if total:
        result = {label: value / total for label, value in result.items()}
    cache[key] = result
    return result


def precompute_probability_cache(
    model: Any,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    stages: list[str],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Batch-predict probabilities for every ordered team pair and stage.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        context: PredictionContext.
        stages: Stage labels to precompute.

    Returns:
        Populated probability cache.
    """
    teams = all_teams(context.tournament_config)
    keys: list[tuple[str, str, str]] = []
    rows: list[dict[str, float]] = []
    for stage in stages:
        for team_a in teams:
            for team_b in teams:
                if team_a == team_b:
                    continue
                keys.append((team_a, team_b, stage))
                rows.append(
                    feature_frame_from_context(team_a, team_b, context, stage)
                    .iloc[0]
                    .to_dict()
                )

    feature_frame = pd.DataFrame(rows, columns=context.feature_columns)
    probabilities = model.predict_proba(feature_frame)
    labels = label_encoder.inverse_transform(model.classes_)
    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, row_probabilities in zip(keys, probabilities, strict=True):
        result = {"A": 0.0, "D": 0.0, "B": 0.0}
        for label, probability in zip(labels, row_probabilities, strict=True):
            result[str(label)] = float(probability)
        total = sum(result.values())
        if total:
            result = {label: value / total for label, value in result.items()}
        cache[key] = result
    return cache


def sample_legacy_scoreline(
    outcome: str,
    team_a: str,
    team_b: str,
    context: PredictionContext,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a plausible scoreline consistent with an outcome class.

    Args:
        outcome: A, B, or D.
        team_a: Team A.
        team_b: Team B.
        context: PredictionContext.
        rng: Random generator.

    Returns:
        Goals for team A and team B.
    """
    stats_a = context.team_features[team_a]
    stats_b = context.team_features[team_b]
    expected_a = max(
        0.25, (stats_a["goals_scored_avg"] + stats_b["goals_conceded_avg"]) / 2
    )
    expected_b = max(
        0.25, (stats_b["goals_scored_avg"] + stats_a["goals_conceded_avg"]) / 2
    )
    goals_a = int(min(rng.poisson(expected_a), 6))
    goals_b = int(min(rng.poisson(expected_b), 6))

    if outcome == "D":
        level = int(round((goals_a + goals_b) / 2))
        return min(level, 4), min(level, 4)
    if outcome == "A" and goals_a <= goals_b:
        goals_a = goals_b + 1
    if outcome == "B" and goals_b <= goals_a:
        goals_b = goals_a + 1
    return min(goals_a, 7), min(goals_b, 7)


def sample_poisson_scoreline(
    outcome: str,
    team_a: str,
    team_b: str,
    context: PredictionContext,
    rng: np.random.Generator,
    draw_probability: float = 0.25,
) -> tuple[int, int]:
    """Sample a Poisson scoreline constrained to the selected outcome."""
    stats_a = context.team_features[team_a]
    stats_b = context.team_features[team_b]
    elo_delta = (stats_a["elo"] - stats_b["elo"]) / 400
    attack_a = (stats_a["goals_scored_avg"] + stats_b["goals_conceded_avg"]) / 2
    attack_b = (stats_b["goals_scored_avg"] + stats_a["goals_conceded_avg"]) / 2
    expected_a = float(np.clip(attack_a * np.exp(0.18 * elo_delta), 0.20, 3.80))
    expected_b = float(np.clip(attack_b * np.exp(-0.18 * elo_delta), 0.20, 3.80))

    if outcome == "D":
        draw_lambda = float(np.clip((expected_a + expected_b) / 2, 0.2, 2.8))
        if draw_probability > 0.34:
            draw_lambda *= 0.85
        goals = int(min(rng.poisson(draw_lambda), 5))
        return goals, goals

    for _ in range(40):
        goals_a = int(min(rng.poisson(expected_a), 7))
        goals_b = int(min(rng.poisson(expected_b), 7))
        if outcome == "A" and goals_a > goals_b:
            return goals_a, goals_b
        if outcome == "B" and goals_b > goals_a:
            return goals_a, goals_b

    goals_a = int(min(rng.poisson(expected_a), 6))
    goals_b = int(min(rng.poisson(expected_b), 6))
    if outcome == "A":
        return max(goals_a, goals_b + 1), goals_b
    return goals_a, max(goals_b, goals_a + 1)


def sample_scoreline(
    outcome: str,
    team_a: str,
    team_b: str,
    context: PredictionContext,
    rng: np.random.Generator,
    *,
    scoreline_model: str = "poisson",
    draw_probability: float = 0.25,
) -> tuple[int, int]:
    """Sample a scoreline using the requested scoreline model."""
    if scoreline_model == "legacy":
        return sample_legacy_scoreline(outcome, team_a, team_b, context, rng)
    if scoreline_model != "poisson":
        raise ValueError("scoreline_model must be 'poisson' or 'legacy'")
    return sample_poisson_scoreline(
        outcome,
        team_a,
        team_b,
        context,
        rng,
        draw_probability=draw_probability,
    )


def simulate_group_stage(
    model: Any,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    rng: np.random.Generator,
    cache: dict[tuple[str, str, str], dict[str, float]],
    scoreline_model: str = "poisson",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Simulate all group-stage matches and select 32 qualifiers.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        context: PredictionContext.
        rng: Random generator.
        cache: Prediction cache.

    Returns:
        Knockout field and detailed group table rows.
    """
    third_place: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    group_rankings: dict[str, list[str]] = {}
    ranked_groups: dict[str, list[dict[str, Any]]] = {}

    for group, teams in context.tournament_config.groups.items():
        table = {
            team: {
                "team": team,
                "group": group,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            }
            for team in teams
        }
        for team_a, team_b in combinations(teams, 2):
            probabilities = predict_probabilities(
                model, label_encoder, team_a, team_b, context, "Group", cache
            )
            outcome = str(
                rng.choice(
                    ["A", "D", "B"],
                    p=[probabilities["A"], probabilities["D"], probabilities["B"]],
                )
            )
            goals_a, goals_b = sample_scoreline(
                outcome,
                team_a,
                team_b,
                context,
                rng,
                scoreline_model=scoreline_model,
                draw_probability=probabilities["D"],
            )
            table[team_a]["gf"] += goals_a
            table[team_a]["ga"] += goals_b
            table[team_b]["gf"] += goals_b
            table[team_b]["ga"] += goals_a
            if outcome == "A":
                table[team_a]["points"] += 3
                table[team_a]["wins"] += 1
                table[team_b]["losses"] += 1
            elif outcome == "B":
                table[team_b]["points"] += 3
                table[team_b]["wins"] += 1
                table[team_a]["losses"] += 1
            else:
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1
                table[team_a]["draws"] += 1
                table[team_b]["draws"] += 1

        ranked = sorted(
            table.values(),
            key=lambda row: (
                row["points"],
                row["gf"] - row["ga"],
                row["gf"],
                context.team_features[row["team"]]["elo"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            row["gd"] = row["gf"] - row["ga"]
        group_rankings[group] = [row["team"] for row in ranked]
        ranked_groups[group] = ranked
        third_place.append(ranked[2])

    best_third = sorted(
        third_place,
        key=lambda row: (
            row["points"],
            row["gf"] - row["ga"],
            row["gf"],
            context.team_features[row["team"]]["elo"],
        ),
        reverse=True,
    )[:8]
    best_third_teams = [row["team"] for row in best_third]
    best_third_set = set(best_third_teams)
    for group, ranked in ranked_groups.items():
        for row in ranked:
            qualified = row["rank"] <= 2 or row["team"] in best_third_set
            table_rows.append(
                {
                    **row,
                    "group": group,
                    "qualified": qualified,
                    "best_third_qualified": row["rank"] == 3
                    and row["team"] in best_third_set,
                }
            )
    knockout_field = build_knockout_field(
        context.tournament_config, group_rankings, best_third_teams
    )
    return knockout_field, table_rows


def summarize_group_stage_results(
    table_rows: list[dict[str, Any]],
    simulations: int,
    config: TournamentConfig,
) -> pd.DataFrame:
    """Convert per-simulation group tables into team-level probabilities.

    Args:
        table_rows: Group table rows from every simulation.
        simulations: Number of tournament runs.
        config: Tournament configuration.

    Returns:
        One row per team with expected table stats and rank/qualification chances.
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    frame = pd.DataFrame(table_rows)
    rows: list[dict[str, Any]] = []
    for group, teams in config.groups.items():
        for team in teams:
            team_rows = frame[frame["team"] == team] if not frame.empty else frame
            rank_probabilities = {
                f"rank_{rank}_probability": (
                    float((team_rows["rank"] == rank).sum() / simulations)
                    if not team_rows.empty
                    else 0.0
                )
                for rank in range(1, 5)
            }
            expected_finish = sum(
                rank * rank_probabilities[f"rank_{rank}_probability"]
                for rank in range(1, 5)
            )
            rows.append(
                {
                    "group": group,
                    "team": team,
                    "expected_points": (
                        float(team_rows["points"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                    "expected_goal_difference": (
                        float(team_rows["gd"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                    "expected_goals_for": (
                        float(team_rows["gf"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                    "expected_goals_against": (
                        float(team_rows["ga"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                    "expected_wins": (
                        float(team_rows["wins"].sum() / simulations)
                        if not team_rows.empty and "wins" in team_rows
                        else 0.0
                    ),
                    "expected_draws": (
                        float(team_rows["draws"].sum() / simulations)
                        if not team_rows.empty and "draws" in team_rows
                        else 0.0
                    ),
                    "expected_losses": (
                        float(team_rows["losses"].sum() / simulations)
                        if not team_rows.empty and "losses" in team_rows
                        else 0.0
                    ),
                    "expected_finish": expected_finish,
                    **rank_probabilities,
                    "qualification_probability": (
                        float(team_rows["qualified"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                    "best_third_probability": (
                        float(team_rows["best_third_qualified"].sum() / simulations)
                        if not team_rows.empty
                        else 0.0
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["group", "expected_finish", "expected_points", "expected_goal_difference"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )


def validate_group_stage_summary(summary: pd.DataFrame) -> None:
    """Validate rank and qualification probability contracts."""
    rank_columns = [
        "rank_1_probability",
        "rank_2_probability",
        "rank_3_probability",
        "rank_4_probability",
    ]
    rank_totals = summary[rank_columns].sum(axis=1)
    if not np.allclose(rank_totals, 1.0, atol=1e-9):
        raise ValueError("Group rank probabilities must sum to one for every team")
    qualification = (
        summary["rank_1_probability"]
        + summary["rank_2_probability"]
        + summary["best_third_probability"]
    )
    if not np.allclose(summary["qualification_probability"], qualification, atol=1e-9):
        raise ValueError("Qualification probability does not match rank components")


def build_group_most_likely_tables(group_summary: pd.DataFrame) -> pd.DataFrame:
    """Create projected group tables from group-stage simulation means."""
    frame = group_summary.copy()
    frame["rank"] = (
        frame[
            [
                "rank_1_probability",
                "rank_2_probability",
                "rank_3_probability",
                "rank_4_probability",
            ]
        ]
        .to_numpy()
        .argmax(axis=1)
        + 1
    )
    output = frame.rename(
        columns={
            "expected_wins": "projected_wins",
            "expected_draws": "projected_draws",
            "expected_losses": "projected_losses",
            "expected_points": "projected_points",
            "expected_goals_for": "projected_goals_for",
            "expected_goals_against": "projected_goals_against",
            "expected_goal_difference": "projected_goal_difference",
        }
    )
    columns = [
        "group",
        "rank",
        "team",
        "projected_wins",
        "projected_draws",
        "projected_losses",
        "projected_points",
        "projected_goals_for",
        "projected_goals_against",
        "projected_goal_difference",
        "qualification_probability",
        "best_third_probability",
    ]
    return output[columns].sort_values(
        ["group", "rank", "projected_points", "projected_goal_difference"],
        ascending=[True, True, False, False],
    )


def knockout_advancement_probability(
    probabilities: dict[str, float], elo_a: float, elo_b: float
) -> float:
    """Convert A/D/B match probabilities into team A knockout advancement odds."""
    elo_tiebreak = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    non_draw = probabilities["A"] + probabilities["B"]
    regulation_share = probabilities["A"] / non_draw if non_draw else 0.5
    return float(
        np.clip(
            probabilities["A"] + probabilities["D"] * (0.65 * elo_tiebreak + 0.35 * regulation_share),
            0.02,
            0.98,
        )
    )


def decide_knockout_winner(
    team_a: str,
    team_b: str,
    stage: str,
    model: Any,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    rng: np.random.Generator,
    cache: dict[tuple[str, str, str], dict[str, float]],
) -> str:
    """Simulate a knockout match with no draw outcome.

    Args:
        team_a: Team A.
        team_b: Team B.
        stage: Knockout stage.
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        context: PredictionContext.
        rng: Random generator.
        cache: Prediction cache.

    Returns:
        Winning team name.
    """
    probabilities = predict_probabilities(
        model, label_encoder, team_a, team_b, context, stage, cache
    )
    elo_a = context.team_features[team_a]["elo"]
    elo_b = context.team_features[team_b]["elo"]
    chance_a = knockout_advancement_probability(probabilities, elo_a, elo_b)
    return team_a if rng.random() < chance_a else team_b


def simulate_knockouts(
    qualifiers: list[str],
    model: Any,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    rng: np.random.Generator,
    cache: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[str, dict[str, list[str]]]:
    """Simulate the Round of 32 through the final.

    Args:
        qualifiers: Group-stage qualifiers.
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        context: PredictionContext.
        rng: Random generator.
        cache: Prediction cache.

    Returns:
        Champion and round participants/winners.
    """
    rounds: dict[str, list[str]] = {"R32": qualifiers}
    current = qualifiers
    for stage, next_name in [
        ("Round of 32", "R16"),
        ("Round of 16", "QF"),
        ("Quarterfinal", "SF"),
        ("Semifinal", "Final"),
        ("Final", "Champion"),
    ]:
        winners: list[str] = []
        for index in range(0, len(current), 2):
            team_a = current[index]
            team_b = current[index + 1]
            winners.append(
                decide_knockout_winner(
                    team_a, team_b, stage, model, label_encoder, context, rng, cache
                )
            )
        rounds[next_name] = winners
        current = winners
    return current[0], rounds


def run_monte_carlo(
    model: Any,
    label_encoder: LabelEncoder,
    results: pd.DataFrame,
    elo: pd.DataFrame,
    simulations: int = 10_000,
    player_features: pd.DataFrame | None = None,
    use_player_features: bool = False,
    tournament_config: TournamentConfig | None = None,
    seed: int = RANDOM_SEED,
    scoreline_model: str = "poisson",
) -> tuple[
    pd.DataFrame,
    dict[str, Counter[str]],
    PredictionContext,
    pd.DataFrame,
    Counter[tuple[tuple[str, ...], ...]],
]:
    """Run full-tournament Monte Carlo simulations.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        results: Cleaned historical results.
        elo: Cleaned ELO ratings.
        simulations: Number of tournament runs.
        player_features: Optional processed player-strength features.
        use_player_features: Whether to include player features in prediction rows.
        tournament_config: Optional tournament configuration.
        seed: Random seed for reproducible simulations.
        scoreline_model: Score sampler, either poisson or legacy.

    Returns:
        Aggregated results, round counters, prediction context, group-stage summary,
        and path counters.
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if scoreline_model not in {"poisson", "legacy"}:
        raise ValueError("scoreline_model must be 'poisson' or 'legacy'")
    rng = np.random.default_rng(seed)
    context = build_prediction_context(
        results,
        elo,
        player_features=player_features,
        use_player_features=use_player_features,
        tournament_config=tournament_config,
    )
    cache = precompute_probability_cache(
        model,
        label_encoder,
        context,
        ["Group", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
    )
    round_counts: dict[str, Counter[str]] = {
        "SF": Counter(),
        "Final": Counter(),
        "Champion": Counter(),
        "R32": Counter(),
        "R16": Counter(),
        "QF": Counter(),
    }
    path_counts: Counter[tuple[tuple[str, ...], ...]] = Counter()
    group_stage_rows: list[dict[str, Any]] = []

    for simulation in range(simulations):
        qualifiers, table_rows = simulate_group_stage(
            model,
            label_encoder,
            context,
            rng,
            cache,
            scoreline_model=scoreline_model,
        )
        group_stage_rows.extend(table_rows)
        champion, rounds = simulate_knockouts(
            qualifiers, model, label_encoder, context, rng, cache
        )
        for stage in ("R32", "R16", "QF", "SF", "Final"):
            round_counts[stage].update(rounds[stage])
        round_counts["Champion"].update([champion])
        path_counts.update([tuple(tuple(rounds[stage]) for stage in ROUND_STAGES)])
        if (simulation + 1) % max(simulations // 5, 1) == 0:
            print(f"Monte Carlo progress: {simulation + 1}/{simulations}")

    rows = []
    for team in all_teams(context.tournament_config):
        rows.append(
            {
                "team": team,
                "confederation": CONFEDERATIONS.get(team, "Other"),
                "r32_probability": round_counts["R32"][team] / simulations,
                "r16_probability": round_counts["R16"][team] / simulations,
                "qf_probability": round_counts["QF"][team] / simulations,
                "sf_probability": round_counts["SF"][team] / simulations,
                "final_probability": round_counts["Final"][team] / simulations,
                "win_probability": round_counts["Champion"][team] / simulations,
            }
        )
    summary = pd.DataFrame(rows).sort_values("win_probability", ascending=False)
    group_stage_summary = summarize_group_stage_results(
        group_stage_rows,
        simulations,
        context.tournament_config,
    )
    validate_group_stage_summary(group_stage_summary)
    validate_round_counts(round_counts, simulations)
    return (
        summary.reset_index(drop=True),
        round_counts,
        context,
        group_stage_summary,
        path_counts,
    )


def save_simulation_results(summary: pd.DataFrame, output_path: Path) -> None:
    """Save Monte Carlo summary to CSV.

    Args:
        summary: Simulation summary dataframe.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def validate_round_counts(round_counts: dict[str, Counter[str]], simulations: int) -> None:
    """Validate expected total entrants per knockout round."""
    expected = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}
    for stage, entrants in expected.items():
        total = sum(round_counts[stage].values())
        if total != entrants * simulations:
            raise ValueError(
                f"{stage} has {total} counted entrants, expected {entrants * simulations}"
            )


def round_counts_to_jsonable(round_counts: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    """Convert round counters into JSON-serializable dictionaries."""
    return {
        stage: {team: int(count) for team, count in counter.items()}
        for stage, counter in round_counts.items()
    }


def round_counts_from_jsonable(payload: dict[str, dict[str, int]]) -> dict[str, Counter[str]]:
    """Convert saved round-count JSON back into counters."""
    return {stage: Counter(values) for stage, values in payload.items()}


def path_counts_to_jsonable(
    path_counts: Counter[tuple[tuple[str, ...], ...]]
) -> list[dict[str, Any]]:
    """Convert path counters into JSON-serializable rows."""
    rows = []
    for path, count in path_counts.items():
        rows.append(
            {
                "count": int(count),
                "path": {
                    stage: list(teams)
                    for stage, teams in zip(ROUND_STAGES, path, strict=True)
                },
            }
        )
    return rows


def path_counts_from_jsonable(
    payload: list[dict[str, Any]]
) -> Counter[tuple[tuple[str, ...], ...]]:
    """Convert saved path-count rows back into a counter."""
    counter: Counter[tuple[tuple[str, ...], ...]] = Counter()
    for row in payload:
        stage_map = row["path"]
        path = tuple(tuple(stage_map[stage]) for stage in ROUND_STAGES)
        counter[path] = int(row["count"])
    return counter


def save_round_counts(round_counts: dict[str, Counter[str]], output_path: Path) -> None:
    """Save round counters to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(round_counts_to_jsonable(round_counts), handle, indent=2)


def load_round_counts(input_path: Path) -> dict[str, Counter[str]]:
    """Load round counters from JSON."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        return round_counts_from_jsonable(json.load(handle))


def save_path_counts(
    path_counts: Counter[tuple[tuple[str, ...], ...]], output_path: Path
) -> None:
    """Save bracket path counters to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(path_counts_to_jsonable(path_counts), handle, indent=2)


def load_path_counts(input_path: Path) -> Counter[tuple[tuple[str, ...], ...]]:
    """Load bracket path counters from JSON."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        return path_counts_from_jsonable(json.load(handle))


def plot_heatmap_win_probabilities(
    model: VotingClassifier,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    output_path: Path,
) -> None:
    """Generate the 48-team head-to-head win probability heatmap.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        context: PredictionContext.
        output_path: Output PNG path.
    """
    cache = precompute_probability_cache(model, label_encoder, context, ["Group"])
    teams = all_teams(context.tournament_config)
    matrix = np.full((len(teams), len(teams)), np.nan)
    annotations = np.full((len(teams), len(teams)), "", dtype=object)
    for row, team_a in enumerate(teams):
        for column, team_b in enumerate(teams):
            if team_a == team_b:
                continue
            probabilities = predict_probabilities(
                model, label_encoder, team_a, team_b, context, "Group", cache
            )
            matrix[row, column] = probabilities["A"] * 100
            annotations[row, column] = f"{matrix[row, column]:.0f}%"

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(30, 26))
    sns.heatmap(
        matrix,
        cmap="Greens",
        xticklabels=teams,
        yticklabels=teams,
        annot=annotations,
        fmt="",
        linewidths=0.25,
        linecolor="#222222",
        cbar_kws={"label": "P(Team A wins)"},
        ax=ax,
    )
    ax.set_title("Head-to-Head Win Probability Matrix - FIFA WC 2026", fontsize=24)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_radar_team_strengths(context: PredictionContext, output_path: Path) -> None:
    """Generate radar charts for the top 16 teams by ELO.

    Args:
        context: PredictionContext.
        output_path: Output PNG path.
    """
    teams = sorted(
        all_teams(context.tournament_config),
        key=lambda team: context.team_features[team]["elo"],
        reverse=True,
    )[:16]
    max_elo = max(context.team_features[team]["elo"] for team in teams)
    min_elo = min(context.team_features[team]["elo"] for team in teams)
    labels = ["Attack", "Defense", "Form", "H2H", "ELO", "Consistency"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 4, subplot_kw={"polar": True}, figsize=(16, 16))
    for ax, team in zip(axes.flat, teams, strict=True):
        stats = context.team_features[team]
        elo_scaled = (stats["elo"] - min_elo) / max(max_elo - min_elo, 1)
        values = [
            min(stats["goals_scored_avg"] / 3, 1),
            min(1 / max(stats["goals_conceded_avg"], 0.25) / 2, 1),
            stats["form"],
            0.5,
            elo_scaled,
            stats["consistency"],
        ]
        values += values[:1]
        color = KIT_COLORS.get(team, ACCENT)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels([])
        ax.set_title(team, fontsize=10, color=WHITE, pad=12)
    fig.suptitle("Top 16 Team Strength Profiles - FIFA WC 2026", fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_winners(
    summary: pd.DataFrame, output_path: Path, simulations: int | None = None
) -> None:
    """Plot top 16 tournament win probabilities.

    Args:
        summary: Simulation summary dataframe.
        output_path: Output PNG path.
    """
    top = summary.head(16).sort_values("win_probability")
    colors = [
        CONFEDERATION_COLORS.get(confederation, ACCENT)
        for confederation in top["confederation"]
    ]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["team"], top["win_probability"] * 100, color=colors)
    note = ""
    if simulations:
        max_se = np.sqrt(
            top["win_probability"] * (1 - top["win_probability"]) / simulations
        ).max()
        note = f"\nMonte Carlo SE up to +/- {max_se * 100:.2f} pp"
    ax.set_title(f"Monte Carlo Tournament Winners - FIFA WC 2026{note}", fontsize=18)
    ax.set_xlabel("Win probability (%)")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, top["win_probability"] * 100, strict=True):
        ax.text(
            value + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            color=WHITE,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_stage_predictions(
    group_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot group-stage rank and qualification probabilities.

    Args:
        group_summary: Team-level group-stage probability summary.
        output_path: Output PNG path.
    """
    rank_columns = [
        "rank_1_probability",
        "rank_2_probability",
        "rank_3_probability",
        "rank_4_probability",
    ]
    rank_colors = ["#00FF87", "#2DD4BF", "#FACC15", "#EF4444"]
    groups = sorted(group_summary["group"].unique())

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 3, figsize=(22, 24), sharex=True)
    fig.patch.set_facecolor("#050505")
    for ax, group in zip(axes.flat, groups, strict=True):
        group_data = group_summary[group_summary["group"] == group].sort_values(
            ["expected_finish", "expected_points", "expected_goal_difference"],
            ascending=[True, False, False],
        )
        ax.set_facecolor("#050505")
        ax.set_xlim(-0.58, 1.34)
        ax.set_ylim(-0.65, len(group_data) - 0.15)
        ax.axis("off")
        ax.set_title(f"Group {group}", color=ACCENT, fontsize=15, weight="bold", pad=10)
        ax.text(-0.56, len(group_data) - 0.35, "Team", fontsize=8, color="#BBBBBB")
        ax.text(
            0.00,
            len(group_data) - 0.35,
            "Finish probabilities",
            fontsize=8,
            color="#BBBBBB",
        )
        ax.text(1.03, len(group_data) - 0.35, "Qual", fontsize=8, color="#BBBBBB")
        ax.text(1.18, len(group_data) - 0.35, "Pts", fontsize=8, color="#BBBBBB")
        ax.text(1.28, len(group_data) - 0.35, "GD", fontsize=8, color="#BBBBBB")

        for row_index, row in enumerate(group_data.itertuples(index=False)):
            y_pos = len(group_data) - row_index - 1
            ax.text(
                -0.56,
                y_pos,
                row.team,
                ha="left",
                va="center",
                fontsize=9,
                color=WHITE,
                weight="bold" if row_index < 2 else "normal",
            )
            left = 0.0
            for rank_index, column in enumerate(rank_columns, start=1):
                probability = float(getattr(row, column))
                ax.barh(
                    y_pos,
                    probability,
                    left=left,
                    height=0.40,
                    color=rank_colors[rank_index - 1],
                    alpha=0.92,
                )
                if probability >= 0.13:
                    ax.text(
                        left + probability / 2,
                        y_pos,
                        f"{probability * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#050505",
                        weight="bold",
                    )
                left += probability
            ax.barh(
                y_pos - 0.30,
                row.qualification_probability,
                left=0,
                height=0.08,
                color=ACCENT,
                alpha=0.95,
            )
            ax.text(
                1.03,
                y_pos,
                f"{row.qualification_probability * 100:.0f}%",
                ha="left",
                va="center",
                fontsize=8,
                color=ACCENT,
                weight="bold",
            )
            ax.text(
                1.18,
                y_pos,
                f"{row.expected_points:.1f}",
                ha="left",
                va="center",
                fontsize=8,
                color=WHITE,
            )
            ax.text(
                1.28,
                y_pos,
                f"{row.expected_goal_difference:+.1f}",
                ha="left",
                va="center",
                fontsize=8,
                color=WHITE,
            )

        legend_x = [0.00, 0.18, 0.36, 0.54]
        for label, color, x_pos in zip(
            ["1st", "2nd", "3rd", "4th"], rank_colors, legend_x, strict=True
        ):
            ax.scatter(x_pos, -0.45, s=28, color=color)
            ax.text(
                x_pos + 0.025, -0.45, label, va="center", fontsize=7, color="#CCCCCC"
            )

    fig.suptitle(
        "World Cup 2026 Group-Stage Prediction Probabilities",
        fontsize=22,
        color=WHITE,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_bracket_probability_slots(
    path_counts: Counter[tuple[tuple[str, ...], ...]],
    round_counts: dict[str, Counter[str]],
    simulations: int,
    config: TournamentConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate most likely teams by bracket position for each knockout round."""
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    stages = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    stage_sizes = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}
    position_counts = {
        stage: [Counter() for _ in range(stage_sizes[stage])] for stage in stages
    }

    for path, count in path_counts.items():
        for stage, teams in zip(stages, path, strict=False):
            for slot_index, team in enumerate(teams[: stage_sizes[stage]]):
                position_counts[stage][slot_index][team] += count

    if not path_counts:
        for stage in stages:
            for slot_index, (team, count) in enumerate(
                round_counts[stage].most_common(stage_sizes[stage])
            ):
                position_counts[stage][slot_index][team] += count

    bracket_slots: dict[str, list[dict[str, Any]]] = {}
    for stage in stages:
        rows = []
        for slot_index, counter in enumerate(position_counts[stage]):
            if counter:
                team, count = counter.most_common(1)[0]
            else:
                team, count = "TBD", 0
            rows.append(
                {
                    "slot_index": slot_index,
                    "slot_label": (
                        config.bracket_order[slot_index]
                        if stage == "R32"
                        else f"{stage} {slot_index + 1}"
                    ),
                    "team": team,
                    "probability": count / simulations,
                    "count": count,
                }
            )
        bracket_slots[stage] = rows

    return bracket_slots


def bracket_probability_slots_frame(
    path_counts: Counter[tuple[tuple[str, ...], ...]],
    round_counts: dict[str, Counter[str]],
    simulations: int,
    config: TournamentConfig,
) -> pd.DataFrame:
    """Return bracket slot probabilities as a flat dataframe."""
    slots = build_bracket_probability_slots(path_counts, round_counts, simulations, config)
    rows = []
    for stage, stage_slots in slots.items():
        for slot in stage_slots:
            rows.append({"round": stage, **slot})
    return pd.DataFrame(rows)


def save_bracket_slot_probabilities(
    path_counts: Counter[tuple[tuple[str, ...], ...]],
    round_counts: dict[str, Counter[str]],
    simulations: int,
    config: TournamentConfig,
    output_path: Path,
) -> pd.DataFrame:
    """Save bracket slot probabilities to CSV."""
    frame = bracket_probability_slots_frame(
        path_counts,
        round_counts,
        simulations,
        config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def plot_team_odds_table(
    summary: pd.DataFrame,
    output_path: Path,
    simulations: int | None = None,
) -> None:
    """Render each team's round probabilities as a table-style heatmap."""
    columns = [
        "r32_probability",
        "r16_probability",
        "qf_probability",
        "sf_probability",
        "final_probability",
        "win_probability",
    ]
    labels = ["R32", "R16", "QF", "SF", "Final", "Win"]
    data = summary.sort_values("win_probability", ascending=False).reset_index(drop=True)
    values = data[columns].to_numpy() * 100
    annotations = np.vectorize(lambda value: f"{value:.1f}%")(values)
    wrapped_teams = ["\n".join(textwrap.wrap(team, width=18)) for team in data["team"]]
    height = max(12, len(data) * 0.28)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, height))
    sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=wrapped_teams,
        cbar_kws={"label": "Probability (%)"},
        ax=ax,
    )
    note = ""
    if simulations:
        max_se = np.sqrt(data["win_probability"] * (1 - data["win_probability"]) / simulations).max()
        note = f"\nWin-probability Monte Carlo SE up to +/- {max_se * 100:.2f} pp"
    ax.set_title(f"World Cup 2026 Team Round Odds{note}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_most_likely_tables(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render projected group tables."""
    groups = sorted(table["group"].unique())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    fig.patch.set_facecolor("#050505")
    for ax, group in zip(axes.flat, groups, strict=True):
        group_data = table[table["group"] == group].sort_values("rank")
        ax.axis("off")
        ax.set_title(f"Group {group}", color=ACCENT, fontsize=14, weight="bold")
        display = group_data[
            [
                "rank",
                "team",
                "projected_wins",
                "projected_draws",
                "projected_losses",
                "projected_points",
                "projected_goal_difference",
                "qualification_probability",
            ]
        ].copy()
        display["team"] = display["team"].map(lambda value: "\n".join(textwrap.wrap(value, 16)))
        for column in [
            "projected_wins",
            "projected_draws",
            "projected_losses",
            "projected_points",
            "projected_goal_difference",
        ]:
            display[column] = display[column].map(lambda value: f"{value:.1f}")
        display["qualification_probability"] = display["qualification_probability"].map(
            lambda value: f"{value * 100:.0f}%"
        )
        display.columns = ["Rk", "Team", "W", "D", "L", "Pts", "GD", "Qual"]
        table_artist = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            loc="center",
            cellLoc="center",
        )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(8)
        table_artist.scale(1, 1.4)
        for cell in table_artist.get_celld().values():
            cell.set_edgecolor("#333333")
            cell.set_facecolor("#111111")
            cell.get_text().set_color(WHITE)
    fig.suptitle("Most Likely Group Tables - FIFA WC 2026", color=WHITE, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bracket_simulation(
    summary: pd.DataFrame,
    round_counts: dict[str, Counter[str]],
    path_counts: Counter[tuple[tuple[str, ...], ...]],
    simulations: int,
    output_path: Path,
    config: TournamentConfig,
) -> None:
    """Draw a probability bracket from most likely teams at each slot.

    Args:
        summary: Simulation summary dataframe.
        round_counts: Monte Carlo round counters.
        path_counts: Complete simulated round paths.
        simulations: Number of tournament runs.
        output_path: Output PNG path.
        config: Tournament configuration with official-style bracket slots.
    """
    stages = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    stage_sizes = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}
    stage_labels = {
        "R32": "Round of 32",
        "R16": "Round of 16",
        "QF": "Quarterfinals",
        "SF": "Semifinals",
        "Final": "Final",
        "Champion": "Champion",
    }
    bracket_slots = build_bracket_probability_slots(
        path_counts,
        round_counts,
        simulations,
        config,
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(24, 18))
    fig.patch.set_facecolor("#050505")
    ax.set_facecolor("#050505")
    ax.axis("off")
    ax.set_ylim(-0.02, 1.06)
    x_positions = np.array([0.06, 0.26, 0.44, 0.61, 0.77, 0.92])
    for stage_index, stage in enumerate(stages):
        slots = bracket_slots[stage]
        y_positions = (
            np.linspace(0.02, 0.93, len(slots))[::-1] if len(slots) > 1 else [0.48]
        )
        for slot, y_pos in zip(slots, y_positions, strict=True):
            probability = slot["probability"] * 100
            label_prefix = f"{slot['slot_label']}  " if stage == "R32" else ""
            ax.text(
                x_positions[stage_index],
                y_pos,
                f"{label_prefix}{slot['team']}\n{probability:.1f}% reach",
                ha="center",
                va="center",
                fontsize=6.5 if stage in {"R32", "R16"} else 9,
                color=WHITE,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#111111",
                    "edgecolor": ACCENT if stage == "Champion" else "#666666",
                    "linewidth": 0.8,
                },
            )
        ax.text(
            x_positions[stage_index],
            0.985,
            stage_labels[stage],
            ha="center",
            va="bottom",
            fontsize=12,
            color=ACCENT,
            weight="bold",
        )

    for left_index in range(len(stages) - 1):
        left_ys = np.linspace(0.02, 0.93, stage_sizes[stages[left_index]])[::-1]
        right_ys = np.linspace(0.02, 0.93, stage_sizes[stages[left_index + 1]])[::-1]
        for pair_index, right_y in enumerate(right_ys):
            source = left_ys[pair_index * 2 : pair_index * 2 + 2]
            for left_y in source:
                path = MplPath(
                    [
                        (x_positions[left_index] + 0.045, left_y),
                        (
                            (x_positions[left_index] + x_positions[left_index + 1]) / 2,
                            left_y,
                        ),
                        (
                            (x_positions[left_index] + x_positions[left_index + 1]) / 2,
                            right_y,
                        ),
                        (x_positions[left_index + 1] - 0.045, right_y),
                    ],
                    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
                )
                ax.add_patch(
                    PathPatch(path, edgecolor="#444444", facecolor="none", lw=0.65)
                )

    champion = summary.iloc[0]
    fig.suptitle(
        f"World Cup 2026 Probability Bracket - Most Likely Champion: {champion['team']}"
        f" ({champion['win_probability'] * 100:.2f}%)",
        fontsize=20,
        color=WHITE,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_visualizations(
    model: Any,
    label_encoder: LabelEncoder,
    summary: pd.DataFrame,
    round_counts: dict[str, Counter[str]],
    context: PredictionContext,
    group_stage_summary: pd.DataFrame,
    visualizations_dir: Path,
    simulations: int,
    path_counts: Counter[tuple[tuple[str, ...], ...]] | None = None,
) -> None:
    """Generate all simulation visualizations.

    Args:
        model: Fitted ensemble.
        label_encoder: Fitted label encoder.
        summary: Simulation summary dataframe.
        round_counts: Monte Carlo round counters.
        context: PredictionContext.
        group_stage_summary: Group-stage probability summary dataframe.
        visualizations_dir: Output directory.
        simulations: Number of tournament runs.
        path_counts: Complete simulated path counters.
    """
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap_win_probabilities(
        model,
        label_encoder,
        context,
        visualizations_dir / "heatmap_win_probabilities.png",
    )
    plot_radar_team_strengths(context, visualizations_dir / "radar_team_strengths.png")
    plot_bracket_simulation(
        summary,
        round_counts,
        path_counts or Counter(),
        simulations,
        visualizations_dir / "bracket_simulation.png",
        context.tournament_config,
    )
    plot_group_stage_predictions(
        group_stage_summary,
        visualizations_dir / "group_stage_predictions.png",
    )
    plot_monte_carlo_winners(
        summary,
        visualizations_dir / "monte_carlo_winners.png",
        simulations=simulations,
    )
    plot_team_odds_table(
        summary,
        visualizations_dir / "team_odds_table.png",
        simulations=simulations,
    )
    plot_group_most_likely_tables(
        build_group_most_likely_tables(group_stage_summary),
        visualizations_dir / "group_most_likely_tables.png",
    )
