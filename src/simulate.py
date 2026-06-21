"""World Cup 2026 tournament simulation and visualization utilities."""

from __future__ import annotations

import logging
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
    model: VotingClassifier,
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
    model: VotingClassifier,
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


def sample_scoreline(
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


def simulate_group_stage(
    model: VotingClassifier,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    rng: np.random.Generator,
    cache: dict[tuple[str, str, str], dict[str, float]],
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

    for group, teams in context.tournament_config.groups.items():
        table = {
            team: {"team": team, "group": group, "points": 0, "gf": 0, "ga": 0}
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
            goals_a, goals_b = sample_scoreline(outcome, team_a, team_b, context, rng)
            table[team_a]["gf"] += goals_a
            table[team_a]["ga"] += goals_b
            table[team_b]["gf"] += goals_b
            table[team_b]["ga"] += goals_a
            if outcome == "A":
                table[team_a]["points"] += 3
            elif outcome == "B":
                table[team_b]["points"] += 3
            else:
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1

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
            table_rows.append(row.copy())
        group_rankings[group] = [row["team"] for row in ranked]
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
    knockout_field = build_knockout_field(
        context.tournament_config, group_rankings, best_third_teams
    )
    return knockout_field, table_rows


def decide_knockout_winner(
    team_a: str,
    team_b: str,
    stage: str,
    model: VotingClassifier,
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
    if probabilities["D"] > 0.3:
        chance_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    else:
        non_draw = probabilities["A"] + probabilities["B"]
        chance_a = probabilities["A"] / non_draw if non_draw else 0.5
    return team_a if rng.random() < chance_a else team_b


def simulate_knockouts(
    qualifiers: list[str],
    model: VotingClassifier,
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
    model: VotingClassifier,
    label_encoder: LabelEncoder,
    results: pd.DataFrame,
    elo: pd.DataFrame,
    simulations: int = 10_000,
    player_features: pd.DataFrame | None = None,
    use_player_features: bool = False,
    tournament_config: TournamentConfig | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, Counter[str]],
    PredictionContext,
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

    Returns:
        Aggregated results, round counters, prediction context, and path counters.
    """
    rng = np.random.default_rng(RANDOM_SEED)
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
    path_stages = ("R32", "R16", "QF", "SF", "Final", "Champion")

    for simulation in range(simulations):
        qualifiers, _ = simulate_group_stage(model, label_encoder, context, rng, cache)
        champion, rounds = simulate_knockouts(
            qualifiers, model, label_encoder, context, rng, cache
        )
        for stage in ("R32", "R16", "QF", "SF", "Final"):
            round_counts[stage].update(rounds[stage])
        round_counts["Champion"].update([champion])
        path_counts.update([tuple(tuple(rounds[stage]) for stage in path_stages)])
        if (simulation + 1) % max(simulations // 5, 1) == 0:
            print(f"Monte Carlo progress: {simulation + 1}/{simulations}")

    rows = []
    for team in all_teams(context.tournament_config):
        rows.append(
            {
                "team": team,
                "confederation": CONFEDERATIONS.get(team, "Other"),
                "win_probability": round_counts["Champion"][team] / simulations,
                "final_probability": round_counts["Final"][team] / simulations,
                "sf_probability": round_counts["SF"][team] / simulations,
            }
        )
    summary = pd.DataFrame(rows).sort_values("win_probability", ascending=False)
    return summary.reset_index(drop=True), round_counts, context, path_counts


def save_simulation_results(summary: pd.DataFrame, output_path: Path) -> None:
    """Save Monte Carlo summary to CSV.

    Args:
        summary: Simulation summary dataframe.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


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


def plot_monte_carlo_winners(summary: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_title("Monte Carlo Tournament Winners - FIFA WC 2026", fontsize=18)
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


def plot_bracket_simulation(
    summary: pd.DataFrame,
    round_counts: dict[str, Counter[str]],
    path_counts: Counter[tuple[tuple[str, ...], ...]],
    simulations: int,
    output_path: Path,
) -> None:
    """Draw a compact bracket-style summary from most frequent advancers.

    Args:
        summary: Simulation summary dataframe.
        round_counts: Monte Carlo round counters.
        path_counts: Complete simulated round paths.
        simulations: Number of tournament runs.
        output_path: Output PNG path.
    """
    stages = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    stage_sizes = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}
    if path_counts:
        path, _ = path_counts.most_common(1)[0]
        stage_teams = {
            stage: list(teams) for stage, teams in zip(stages, path, strict=True)
        }
    else:
        stage_teams = {
            stage: [
                team for team, _ in round_counts[stage].most_common(stage_sizes[stage])
            ]
            for stage in stages
        }

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    x_positions = np.linspace(0.05, 0.95, len(stages))
    for stage_index, stage in enumerate(stages):
        teams = stage_teams[stage]
        y_positions = np.linspace(0.05, 0.95, len(teams)) if len(teams) > 1 else [0.5]
        for team, y_pos in zip(teams, y_positions, strict=True):
            probability = round_counts[stage][team] / simulations * 100
            ax.text(
                x_positions[stage_index],
                y_pos,
                f"{team}\n{probability:.1f}%",
                ha="center",
                va="center",
                fontsize=8 if stage != "Champion" else 13,
                color=WHITE,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#111111",
                    "edgecolor": ACCENT,
                    "linewidth": 0.8,
                },
            )
        ax.text(
            x_positions[stage_index],
            1.02,
            stage,
            ha="center",
            va="bottom",
            fontsize=12,
            color=ACCENT,
            weight="bold",
        )

    for left_index in range(len(stages) - 1):
        left_ys = np.linspace(0.05, 0.95, stage_sizes[stages[left_index]])
        right_ys = np.linspace(0.05, 0.95, stage_sizes[stages[left_index + 1]])
        for pair_index, right_y in enumerate(right_ys):
            source = left_ys[pair_index * 2 : pair_index * 2 + 2]
            for left_y in source:
                path = MplPath(
                    [
                        (x_positions[left_index] + 0.035, left_y),
                        (
                            (x_positions[left_index] + x_positions[left_index + 1]) / 2,
                            left_y,
                        ),
                        (
                            (x_positions[left_index] + x_positions[left_index + 1]) / 2,
                            right_y,
                        ),
                        (x_positions[left_index + 1] - 0.035, right_y),
                    ],
                    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
                )
                ax.add_patch(
                    PathPatch(path, edgecolor="#555555", facecolor="none", lw=0.7)
                )

    champion = summary.iloc[0]
    ax.set_title(
        f"Most Likely Tournament Path - Champion: {champion['team']}"
        f" ({champion['win_probability'] * 100:.2f}%)",
        fontsize=18,
        color=WHITE,
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_visualizations(
    model: VotingClassifier,
    label_encoder: LabelEncoder,
    summary: pd.DataFrame,
    round_counts: dict[str, Counter[str]],
    context: PredictionContext,
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
    )
    plot_monte_carlo_winners(summary, visualizations_dir / "monte_carlo_winners.png")
