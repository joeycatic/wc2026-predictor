from __future__ import annotations

from collections import Counter

import pytest

from src.simulate import build_bracket_probability_slots, summarize_group_stage_results
from src.tournament import (
    TournamentConfig,
    build_knockout_field,
    load_tournament_config,
    validate_tournament_config,
)


def test_default_tournament_config_is_valid() -> None:
    config = load_tournament_config()

    teams = [team for group in config.groups.values() for team in group]
    assert len(config.groups) == 12
    assert len(teams) == 48
    assert len(set(teams)) == 48
    assert len(config.bracket_order) == 32


def test_tournament_config_rejects_duplicate_teams() -> None:
    groups = {
        chr(65 + index): [f"Team {index}-{slot}" for slot in range(4)]
        for index in range(12)
    }
    groups["A"][0] = groups["B"][0]
    config = TournamentConfig(groups=groups, bracket_order=["1A"] * 32)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_tournament_config(config)


def test_bracket_mapping_produces_32_unique_knockout_teams() -> None:
    config = load_tournament_config()
    group_rankings = {group: teams.copy() for group, teams in config.groups.items()}
    best_thirds = [teams[2] for teams in list(config.groups.values())[:8]]

    field = build_knockout_field(config, group_rankings, best_thirds)

    assert len(field) == 32
    assert len(set(field)) == 32


def test_group_stage_summary_has_expected_probability_contract() -> None:
    config = load_tournament_config()
    table_rows = []
    for group_index, (group, teams) in enumerate(config.groups.items()):
        for rank, team in enumerate(teams, start=1):
            qualified_as_best_third = rank == 3 and group_index < 8
            table_rows.append(
                {
                    "group": group,
                    "team": team,
                    "points": 10 - rank,
                    "gf": 6 - rank,
                    "ga": rank,
                    "gd": 6 - (2 * rank),
                    "rank": rank,
                    "qualified": rank <= 2 or qualified_as_best_third,
                    "best_third_qualified": qualified_as_best_third,
                }
            )

    summary = summarize_group_stage_results(table_rows, 1, config)

    assert len(summary) == 48
    assert summary["group"].nunique() == 12
    rank_columns = [
        "rank_1_probability",
        "rank_2_probability",
        "rank_3_probability",
        "rank_4_probability",
    ]
    rank_totals = summary[rank_columns].sum(axis=1)
    assert rank_totals.to_list() == pytest.approx([1.0] * len(summary))
    qualification_from_ranks = (
        summary["rank_1_probability"]
        + summary["rank_2_probability"]
        + summary["best_third_probability"]
    )
    assert summary["qualification_probability"].to_list() == pytest.approx(
        qualification_from_ranks.to_list()
    )
    best_third_rows = summary[summary["best_third_probability"] > 0]
    assert all(best_third_rows["rank_3_probability"] > 0)


def test_bracket_probability_slots_preserve_round_of_32_order() -> None:
    config = load_tournament_config()
    group_rankings = {group: teams.copy() for group, teams in config.groups.items()}
    best_thirds = [teams[2] for teams in list(config.groups.values())[:8]]
    r32 = build_knockout_field(config, group_rankings, best_thirds)
    r16 = r32[::2]
    qf = r16[::2]
    sf = qf[::2]
    final = sf[::2]
    champion = final[:1]
    path = tuple(tuple(stage) for stage in [r32, r16, qf, sf, final, champion])
    round_counts = {
        "R32": Counter(r32),
        "R16": Counter(r16),
        "QF": Counter(qf),
        "SF": Counter(sf),
        "Final": Counter(final),
        "Champion": Counter(champion),
    }

    slots = build_bracket_probability_slots(Counter({path: 1}), round_counts, 1, config)

    assert len(slots["R32"]) == 32
    assert [slot["slot_label"] for slot in slots["R32"]] == config.bracket_order
    assert [slot["team"] for slot in slots["R32"]] == r32
    assert all(slot["probability"] == pytest.approx(1.0) for slot in slots["R32"])
