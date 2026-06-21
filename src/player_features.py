"""Kaggle player-rating preprocessing and leakage-safe team joins."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import normalize_team_name, parse_mixed_dates

LOGGER = logging.getLogger(__name__)

PLAYER_BASE_COLUMNS = [
    "avg_overall",
    "top15_overall",
    "top5_attack",
    "top5_midfield",
    "top5_defense",
    "top5_gk",
    "avg_age",
    "avg_international_reputation",
    "avg_value_eur",
    "avg_weak_foot",
    "avg_skill_moves",
    "depth_score",
]
PLAYER_FEATURE_COLUMNS = [
    *(f"player_{column}_a" for column in PLAYER_BASE_COLUMNS),
    *(f"player_{column}_b" for column in PLAYER_BASE_COLUMNS),
    *(f"player_{column}_diff" for column in PLAYER_BASE_COLUMNS),
]


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _parse_money(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("EUR", "").strip()
    match = re.fullmatch(r"[€$]?([0-9]*\.?[0-9]+)([KkMm]?)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return amount


def _infer_year(frame: pd.DataFrame) -> pd.Series:
    year_column = _first_existing(
        frame.columns,
        ("season", "year", "fifa_version", "sofifa_version", "fifa_update"),
    )
    if year_column is not None:
        years = pd.to_numeric(frame[year_column], errors="coerce")
        years = years.where(years >= 1900, years + 2000)
        if years.notna().any():
            return years.astype("Int64")

    date_column = _first_existing(frame.columns, ("fifa_update_date", "date"))
    if date_column is not None:
        return parse_mixed_dates(frame[date_column]).dt.year.astype("Int64")

    return pd.Series([2023] * len(frame), index=frame.index, dtype="Int64")


def _position_family(positions: object) -> str:
    text = str(positions or "").upper()
    tokens = {token.strip() for token in re.split(r"[,/ ]+", text) if token.strip()}
    if tokens & {"GK"}:
        return "gk"
    if tokens & {"ST", "CF", "LW", "RW", "LF", "RF"}:
        return "attack"
    if tokens & {"CAM", "CM", "CDM", "LM", "RM"}:
        return "midfield"
    if tokens & {"CB", "LB", "RB", "LWB", "RWB"}:
        return "defense"
    return "midfield"


def discover_fifa_player_csv(raw_dir: Path) -> Path | None:
    """Find a plausible Sofifa/FIFA player CSV under the raw data directory."""
    candidates = [
        raw_dir / "fifa-23-complete-player-dataset" / "male_players.csv",
        raw_dir / "fifa-23-complete-player-dataset" / "players_23.csv",
        raw_dir / "male_players.csv",
        raw_dir / "players_23.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in raw_dir.rglob("*.csv"):
        lower = path.name.lower()
        if "player" in lower and "world_cup_2022" not in str(path).lower():
            return path
    return None


def aggregate_player_features(player_csv: Path) -> pd.DataFrame:
    """Aggregate player snapshots to team-season strength features."""
    frame = pd.read_csv(player_csv, low_memory=False)
    nationality_column = _first_existing(
        frame.columns, ("nationality_name", "nationality", "nation", "country")
    )
    overall_column = _first_existing(frame.columns, ("overall", "rating", "ova"))
    if nationality_column is None or overall_column is None:
        raise ValueError(
            "Player CSV must contain nationality and overall/rating columns"
        )

    position_column = _first_existing(
        frame.columns, ("player_positions", "positions", "position", "club_position")
    )
    age_column = _first_existing(frame.columns, ("age",))
    reputation_column = _first_existing(
        frame.columns, ("international_reputation", "international reputation")
    )
    value_column = _first_existing(frame.columns, ("value_eur", "value"))
    weak_foot_column = _first_existing(frame.columns, ("weak_foot", "weak foot"))
    skill_column = _first_existing(frame.columns, ("skill_moves", "skill moves"))

    clean = pd.DataFrame(
        {
            "team": frame[nationality_column].map(normalize_team_name),
            "season": _infer_year(frame),
            "overall": pd.to_numeric(frame[overall_column], errors="coerce"),
            "position_family": (
                frame[position_column].map(_position_family)
                if position_column
                else "midfield"
            ),
            "age": (
                pd.to_numeric(frame[age_column], errors="coerce")
                if age_column
                else np.nan
            ),
            "international_reputation": (
                pd.to_numeric(frame[reputation_column], errors="coerce")
                if reputation_column
                else np.nan
            ),
            "value_eur": (
                frame[value_column].map(_parse_money) if value_column else np.nan
            ),
            "weak_foot": (
                pd.to_numeric(frame[weak_foot_column], errors="coerce")
                if weak_foot_column
                else np.nan
            ),
            "skill_moves": (
                pd.to_numeric(frame[skill_column], errors="coerce")
                if skill_column
                else np.nan
            ),
        }
    )
    clean = clean.dropna(subset=["team", "season", "overall"])
    clean["season"] = clean["season"].astype(int)

    rows: list[dict[str, float | int | str]] = []
    for (team, season), group in clean.groupby(["team", "season"]):
        row: dict[str, float | int | str] = {"team": team, "season": int(season)}
        ordered = group.sort_values("overall", ascending=False)
        row["avg_overall"] = float(group["overall"].mean())
        row["top15_overall"] = float(ordered.head(15)["overall"].mean())
        for family in ("attack", "midfield", "defense", "gk"):
            family_values = group[group["position_family"] == family].sort_values(
                "overall", ascending=False
            )
            row[f"top5_{family}"] = float(family_values.head(5)["overall"].mean())
        for column in (
            "age",
            "international_reputation",
            "value_eur",
            "weak_foot",
            "skill_moves",
        ):
            row[f"avg_{column}"] = float(group[column].mean())
        row["depth_score"] = float((group["overall"] >= 70).sum())
        rows.append(row)

    features = pd.DataFrame(rows)
    neutral_defaults = player_feature_defaults(None)
    for column in PLAYER_BASE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
        fallback = features[column].median()
        if pd.isna(fallback):
            fallback = neutral_defaults[column]
        features[column] = features[column].fillna(fallback)
    return features.sort_values(["team", "season"]).reset_index(drop=True)


def save_team_player_features(raw_dir: Path, processed_dir: Path) -> Path | None:
    """Build and save processed player features if the Kaggle CSV exists."""
    player_csv = discover_fifa_player_csv(raw_dir)
    if player_csv is None:
        LOGGER.warning("No Sofifa/FIFA player CSV found under %s", raw_dir)
        return None
    processed_dir.mkdir(parents=True, exist_ok=True)
    features = aggregate_player_features(player_csv)
    output_path = processed_dir / "team_player_features.csv"
    features.to_csv(output_path, index=False)
    LOGGER.info("Saved player features to %s", output_path)
    return output_path


def load_team_player_features(path: Path) -> pd.DataFrame | None:
    """Load processed team-player features, returning None when absent."""
    if not path.exists():
        LOGGER.warning(
            "Player feature file %s is missing; using neutral player values", path
        )
        return None
    frame = pd.read_csv(path)
    frame["team"] = frame["team"].map(normalize_team_name)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    for column in PLAYER_BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["team", "season"]).reset_index(drop=True)


def player_feature_defaults(player_features: pd.DataFrame | None) -> dict[str, float]:
    """Return neutral player-feature defaults from dataset medians."""
    if player_features is None or player_features.empty:
        return {
            "avg_overall": 70.0,
            "top15_overall": 72.0,
            "top5_attack": 72.0,
            "top5_midfield": 72.0,
            "top5_defense": 72.0,
            "top5_gk": 70.0,
            "avg_age": 26.0,
            "avg_international_reputation": 1.0,
            "avg_value_eur": 1_000_000.0,
            "avg_weak_foot": 3.0,
            "avg_skill_moves": 2.5,
            "depth_score": 18.0,
        }
    return {
        column: float(player_features[column].median())
        for column in PLAYER_BASE_COLUMNS
    }


def build_player_lookup(
    player_features: pd.DataFrame | None,
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Build team -> sorted years/frame lookup for player snapshots."""
    lookup: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    if player_features is None or player_features.empty:
        return lookup
    for team, group in player_features.sort_values("season").groupby("team"):
        lookup[str(team)] = (group["season"].to_numpy(dtype=int), group)
    return lookup


def lookup_player_snapshot(
    lookup: dict[str, tuple[np.ndarray, pd.DataFrame]],
    team: str,
    year: int,
    defaults: dict[str, float],
    *,
    before_year: bool = True,
) -> dict[str, float]:
    """Return a team's latest snapshot before a year, or latest if requested."""
    item = lookup.get(normalize_team_name(team))
    if item is None:
        return defaults.copy()
    years, group = item
    side = "left" if before_year else "right"
    index = int(np.searchsorted(years, year, side=side) - 1)
    if index < 0:
        return defaults.copy()
    row = group.iloc[index]
    return {column: float(row[column]) for column in PLAYER_BASE_COLUMNS}


def player_columns_for_pair(prefix: str, values: dict[str, float]) -> dict[str, float]:
    """Prefix a single team's player features for model input."""
    return {f"player_{column}_{prefix}": value for column, value in values.items()}


def player_pair_features(
    values_a: dict[str, float], values_b: dict[str, float]
) -> dict[str, float]:
    """Create A, B, and difference player feature columns."""
    row = {}
    row.update(player_columns_for_pair("a", values_a))
    row.update(player_columns_for_pair("b", values_b))
    for column in PLAYER_BASE_COLUMNS:
        row[f"player_{column}_diff"] = values_a[column] - values_b[column]
    return row
