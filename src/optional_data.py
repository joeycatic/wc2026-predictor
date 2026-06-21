"""Optional local data inputs and data-status reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing import normalize_team_name, parse_mixed_dates
from src.tournament import TournamentConfig, all_config_teams

FIFA_RANKINGS_FILE = "fifa_rankings.csv"
BETTING_ODDS_FILE = "betting_odds.csv"
WC2026_FIXTURES_FILE = "wc2026_fixtures.csv"


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_fifa_rankings(raw_dir: Path) -> pd.DataFrame | None:
    """Load optional FIFA ranking priors when present."""
    frame = _load_optional_csv(raw_dir / FIFA_RANKINGS_FILE)
    if frame is None:
        return None
    for column in ("date", "team", "rank"):
        if column not in frame.columns:
            raise ValueError(f"{FIFA_RANKINGS_FILE} is missing column {column}")
    frame = frame.copy()
    frame["date"] = parse_mixed_dates(frame["date"])
    frame["team"] = frame["team"].map(normalize_team_name)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    if "points" in frame.columns:
        frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
    return frame.dropna(subset=["date", "team", "rank"]).reset_index(drop=True)


def load_betting_odds(raw_dir: Path) -> pd.DataFrame | None:
    """Load optional betting-market priors when present."""
    frame = _load_optional_csv(raw_dir / BETTING_ODDS_FILE)
    if frame is None:
        return None
    required = {"date", "team"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{BETTING_ODDS_FILE} is missing columns {missing}")
    if not {"win_odds", "implied_win_probability"} & set(frame.columns):
        raise ValueError(
            f"{BETTING_ODDS_FILE} needs win_odds or implied_win_probability"
        )
    frame = frame.copy()
    frame["date"] = parse_mixed_dates(frame["date"])
    frame["team"] = frame["team"].map(normalize_team_name)
    if "win_odds" in frame.columns:
        frame["win_odds"] = pd.to_numeric(frame["win_odds"], errors="coerce")
    if "implied_win_probability" in frame.columns:
        frame["implied_win_probability"] = pd.to_numeric(
            frame["implied_win_probability"], errors="coerce"
        )
    return frame.dropna(subset=["date", "team"]).reset_index(drop=True)


def load_wc2026_fixtures(raw_dir: Path) -> pd.DataFrame | None:
    """Load optional World Cup 2026 fixture metadata when present."""
    frame = _load_optional_csv(raw_dir / WC2026_FIXTURES_FILE)
    if frame is None:
        return None
    required = {"date", "stage", "home_team", "away_team"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{WC2026_FIXTURES_FILE} is missing columns {missing}")
    frame = frame.copy()
    frame["date"] = parse_mixed_dates(frame["date"])
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    return frame.dropna(subset=["date", "home_team", "away_team"]).reset_index(
        drop=True
    )


def load_optional_inputs(raw_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Load all supported optional local CSVs."""
    return {
        "fifa_rankings": load_fifa_rankings(raw_dir),
        "betting_odds": load_betting_odds(raw_dir),
        "wc2026_fixtures": load_wc2026_fixtures(raw_dir),
    }


def _team_coverage(frame: pd.DataFrame | None, team_column: str, teams: list[str]) -> int:
    if frame is None or frame.empty:
        return 0
    return len(set(frame[team_column].dropna()) & set(teams))


def optional_source_status(
    raw_dir: Path,
    config: TournamentConfig,
    optional_inputs: dict[str, pd.DataFrame | None] | None = None,
) -> dict[str, Any]:
    """Return compact availability and coverage metadata for optional CSV inputs."""
    inputs = optional_inputs if optional_inputs is not None else load_optional_inputs(raw_dir)
    teams = all_config_teams(config)
    rankings = inputs.get("fifa_rankings")
    betting = inputs.get("betting_odds")
    fixtures = inputs.get("wc2026_fixtures")
    fixture_teams: set[str] = set()
    if fixtures is not None and not fixtures.empty:
        fixture_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])

    return {
        "fifa_rankings": {
            "present": rankings is not None,
            "rows": 0 if rankings is None else int(len(rankings)),
            "team_coverage": _team_coverage(rankings, "team", teams),
            "missing_teams": (
                teams
                if rankings is None
                else sorted(set(teams) - set(rankings["team"].dropna()))
            ),
        },
        "betting_odds": {
            "present": betting is not None,
            "rows": 0 if betting is None else int(len(betting)),
            "team_coverage": _team_coverage(betting, "team", teams),
            "missing_teams": (
                teams if betting is None else sorted(set(teams) - set(betting["team"]))
            ),
        },
        "wc2026_fixtures": {
            "present": fixtures is not None,
            "rows": 0 if fixtures is None else int(len(fixtures)),
            "team_coverage": len(fixture_teams & set(teams)),
            "missing_teams": teams if fixtures is None else sorted(set(teams) - fixture_teams),
            "has_venue_fields": bool(
                fixtures is not None
                and {"venue", "city", "country"} & set(fixtures.columns)
            ),
        },
    }


def build_data_status(
    results: pd.DataFrame,
    elo: pd.DataFrame,
    config: TournamentConfig,
    raw_dir: Path,
    player_features: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build the CLI data-status report payload."""
    tournament_teams = all_config_teams(config)
    player_teams = (
        set() if player_features is None or player_features.empty else set(player_features["team"])
    )
    optional_status = optional_source_status(raw_dir, config)
    result_teams = set(results["home_team"]) | set(results["away_team"])
    elo_teams = set(elo["team"])
    return {
        "latest_result_date": str(pd.Timestamp(results["date"].max()).date()),
        "latest_elo_date": str(pd.Timestamp(elo["date"].max()).date()),
        "tournament_teams": len(tournament_teams),
        "missing_result_history_teams": sorted(set(tournament_teams) - result_teams),
        "missing_elo_teams": sorted(set(tournament_teams) - elo_teams),
        "player_feature_coverage": {
            "status": (
                "neutral_fallback"
                if not player_teams
                else "real"
                if len(player_teams & set(tournament_teams)) == len(tournament_teams)
                else "partial"
            ),
            "covered_teams": len(player_teams & set(tournament_teams)),
            "missing_teams": sorted(set(tournament_teams) - player_teams),
        },
        "optional_sources": optional_status,
    }
