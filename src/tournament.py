"""Tournament configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.preprocessing import normalize_team_name


@dataclass(frozen=True)
class TournamentConfig:
    """World Cup group and bracket configuration."""

    groups: dict[str, list[str]]
    bracket_order: list[str]


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "config" / "wc2026_groups.yml"
)


def load_tournament_config(path: Path | None = None) -> TournamentConfig:
    """Load and validate tournament groups and knockout slot order."""
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    groups = raw.get("groups", {})
    bracket_order = raw.get("bracket_order", [])
    config = TournamentConfig(
        groups={
            str(group): [normalize_team_name(team) for team in teams]
            for group, teams in groups.items()
        },
        bracket_order=[str(slot) for slot in bracket_order],
    )
    validate_tournament_config(config)
    return config


def validate_tournament_config(config: TournamentConfig) -> None:
    """Validate the supported 48-team, 12-group tournament shape."""
    if len(config.groups) != 12:
        raise ValueError(f"Expected 12 groups, found {len(config.groups)}")

    teams = [team for group_teams in config.groups.values() for team in group_teams]
    if len(teams) != 48:
        raise ValueError(f"Expected 48 group slots, found {len(teams)}")
    duplicates = sorted({team for team in teams if teams.count(team) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tournament teams: {duplicates}")

    for group, group_teams in config.groups.items():
        if len(group_teams) != 4:
            raise ValueError(f"Group {group} must contain 4 teams")

    if len(config.bracket_order) != 32:
        raise ValueError(
            f"Expected 32 knockout bracket slots, found {len(config.bracket_order)}"
        )

    group_ids = set(config.groups)
    for slot in config.bracket_order:
        if slot.startswith("BT"):
            number = int(slot.removeprefix("BT"))
            if number < 1 or number > 8:
                raise ValueError(f"Invalid best-third slot {slot}")
            continue
        if len(slot) < 2 or slot[0] not in {"1", "2"} or slot[1:] not in group_ids:
            raise ValueError(f"Invalid bracket slot {slot}")


def all_config_teams(config: TournamentConfig) -> list[str]:
    """Return teams in group order."""
    return [team for teams in config.groups.values() for team in teams]


def build_knockout_field(
    config: TournamentConfig,
    group_rankings: dict[str, list[str]],
    best_thirds: list[str],
) -> list[str]:
    """Build a 32-team knockout field from configured bracket slots."""
    field: list[str] = []
    for slot in config.bracket_order:
        if slot.startswith("BT"):
            field.append(best_thirds[int(slot.removeprefix("BT")) - 1])
            continue
        rank = int(slot[0]) - 1
        group = slot[1:]
        field.append(group_rankings[group][rank])

    if len(field) != 32 or len(set(field)) != 32:
        raise ValueError("Knockout field must contain 32 unique teams")
    return field
