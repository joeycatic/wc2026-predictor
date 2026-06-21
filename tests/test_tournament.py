from __future__ import annotations

import pytest

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
