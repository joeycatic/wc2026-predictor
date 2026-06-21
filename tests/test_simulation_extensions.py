from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.live_data import (
    apply_what_if,
    build_match_predictions,
    normalize_match_frame,
    parse_what_if,
)
from src.model import load_cached_model
from src.optional_data import (
    load_betting_odds,
    load_fifa_rankings,
    load_wc2026_fixtures,
    optional_source_status,
)
from src.simulate import (
    PredictionContext,
    build_bracket_probability_slots,
    load_round_counts,
    run_monte_carlo,
    sample_scoreline,
    save_bracket_slot_probabilities,
    simulate_group_stage,
    simulation_integrity_report,
    validate_round_counts,
)
from src.tournament import load_tournament_config


class ConstantModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, frame):
        return np.tile(np.array([0.46, 0.29, 0.25]), (len(frame), 1))

    def predict(self, frame):
        return np.repeat(0, len(frame))


def _context() -> PredictionContext:
    config = load_tournament_config()
    teams = [team for group in config.groups.values() for team in group]
    features = {
        team: {
            "elo": 1600.0 + index,
            "form": 0.5,
            "goals_scored_avg": 1.3,
            "goals_conceded_avg": 1.1,
            "consistency": 0.5,
        }
        for index, team in enumerate(teams)
    }
    return PredictionContext(
        team_features=features,
        h2h={},
        fallback_elo=1500.0,
        prediction_date=pd.Timestamp("2026-06-01"),
        tournament_config=config,
        use_player_features=False,
        feature_columns=[
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
        ],
    )


def test_poisson_scoreline_respects_selected_outcome() -> None:
    context = _context()
    rng = np.random.default_rng(7)
    teams = list(context.team_features)[:2]

    a_goals, b_goals = sample_scoreline("A", *teams, context, rng)
    assert a_goals > b_goals
    a_goals, b_goals = sample_scoreline("B", *teams, context, rng)
    assert b_goals > a_goals
    a_goals, b_goals = sample_scoreline("D", *teams, context, rng)
    assert a_goals == b_goals


def test_round_count_validation_and_monte_carlo_summary_contract() -> None:
    config = load_tournament_config()
    label_encoder = LabelEncoder().fit(["A", "B", "D"])
    teams = [team for group in config.groups.values() for team in group]
    results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "home_team": teams[0],
                "away_team": teams[1],
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "stage": "Group",
            }
        ]
    )
    elo = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "team": team,
                "elo_rating": 1500.0 + index,
            }
            for index, team in enumerate(teams)
        ]
    )

    summary, round_counts, _, group_stage_summary, _ = run_monte_carlo(
        ConstantModel(),
        label_encoder,
        results,
        elo,
        simulations=2,
        tournament_config=config,
        seed=123,
    )

    validate_round_counts(round_counts, 2)
    integrity = simulation_integrity_report(
        summary, group_stage_summary, round_counts, 2
    )
    assert integrity["valid"] is True
    assert summary["r32_probability"].sum() == pytest.approx(32)
    assert summary["r16_probability"].sum() == pytest.approx(16)
    assert summary["qf_probability"].sum() == pytest.approx(8)
    assert summary["sf_probability"].sum() == pytest.approx(4)
    assert summary["final_probability"].sum() == pytest.approx(2)
    assert summary["win_probability"].sum() == pytest.approx(1)


def test_saved_bracket_slot_probabilities_include_round(tmp_path) -> None:
    config = load_tournament_config()
    r32 = [team for group in config.groups.values() for team in group[:2]]
    r32.extend([group[2] for group in list(config.groups.values())[:8]])
    r32 = r32[:32]
    path = tuple(
        tuple(stage) for stage in [r32, r32[:16], r32[:8], r32[:4], r32[:2], r32[:1]]
    )
    round_counts = {
        "R32": Counter(r32),
        "R16": Counter(r32[:16]),
        "QF": Counter(r32[:8]),
        "SF": Counter(r32[:4]),
        "Final": Counter(r32[:2]),
        "Champion": Counter(r32[:1]),
    }

    frame = save_bracket_slot_probabilities(
        Counter({path: 1}),
        round_counts,
        1,
        config,
        tmp_path / "slots.csv",
    )

    assert (tmp_path / "slots.csv").exists()
    assert set(
        ["round", "slot_index", "slot_label", "team", "count", "probability"]
    ).issubset(frame.columns)
    assert (
        len(
            build_bracket_probability_slots(
                Counter({path: 1}), round_counts, 1, config
            )["R32"]
        )
        == 32
    )


def test_live_match_normalization_and_what_if_override() -> None:
    frame = normalize_match_frame(
        pd.DataFrame(
            [
                {
                    "date": "2026-06-11T19:00:00Z",
                    "home_team": "USA",
                    "away_team": "Mexico",
                    "status": "scheduled",
                }
            ]
        )
    )

    override = parse_what_if("United States 2-1 Mexico")
    updated = apply_what_if(frame, "United States 2-1 Mexico")

    assert override["home_team"] == "United States"
    assert updated.iloc[0]["status"] == "FINISHED"
    assert updated.iloc[0]["home_score"] == 2
    assert updated.iloc[0]["away_score"] == 1


def test_completed_group_match_is_locked_into_simulated_table() -> None:
    context = _context()
    label_encoder = LabelEncoder().fit(["A", "B", "D"])
    group = next(iter(context.tournament_config.groups))
    team_a, team_b = context.tournament_config.groups[group][:2]
    live_matches = normalize_match_frame(
        pd.DataFrame(
            [
                {
                    "stage": "Group",
                    "group": group,
                    "home_team": team_a,
                    "away_team": team_b,
                    "status": "FINISHED",
                    "home_score": 0,
                    "away_score": 4,
                }
            ]
        )
    )

    _, table_rows = simulate_group_stage(
        ConstantModel(),
        label_encoder,
        context,
        np.random.default_rng(3),
        {},
        live_matches=live_matches,
    )
    row_a = next(row for row in table_rows if row["team"] == team_a)
    row_b = next(row for row in table_rows if row["team"] == team_b)

    assert row_a["ga"] >= 4
    assert row_b["gf"] >= 4


def test_match_predictions_score_completed_matches() -> None:
    context = _context()
    label_encoder = LabelEncoder().fit(["A", "B", "D"])
    teams = list(context.team_features)[:2]
    matches = normalize_match_frame(
        pd.DataFrame(
            [
                {
                    "stage": "Group",
                    "home_team": teams[0],
                    "away_team": teams[1],
                    "status": "FINISHED",
                    "home_score": 2,
                    "away_score": 0,
                }
            ]
        )
    )

    predictions = build_match_predictions(
        ConstantModel(),
        label_encoder,
        context,
        matches,
        seed=5,
    )

    assert predictions.iloc[0]["actual_outcome"] == "A"
    assert predictions.iloc[0]["predicted_outcome"] == "A"
    assert bool(predictions.iloc[0]["prediction_correct"]) is True


def test_optional_csv_loaders_and_absent_status(tmp_path) -> None:
    config = load_tournament_config()
    assert load_fifa_rankings(tmp_path) is None
    assert load_betting_odds(tmp_path) is None
    assert load_wc2026_fixtures(tmp_path) is None

    pd.DataFrame(
        [{"date": "2026-01-01", "team": "USA", "rank": 10, "points": 1700}]
    ).to_csv(tmp_path / "fifa_rankings.csv", index=False)
    pd.DataFrame(
        [{"date": "2026-01-01", "team": "USA", "implied_win_probability": 0.08}]
    ).to_csv(tmp_path / "betting_odds.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "stage": "Group",
                "home_team": "USA",
                "away_team": "Mexico",
                "venue": "Example Stadium",
            }
        ]
    ).to_csv(tmp_path / "wc2026_fixtures.csv", index=False)

    status = optional_source_status(tmp_path, config)

    assert status["fifa_rankings"]["present"] is True
    assert status["fifa_rankings"]["team_coverage"] == 1
    assert status["betting_odds"]["team_coverage"] == 1
    assert status["wc2026_fixtures"]["team_coverage"] == 2


def test_missing_cached_model_and_visual_artifact_fail_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--no-train requires cached model"):
        load_cached_model(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        load_round_counts(tmp_path / "round_counts.json")
