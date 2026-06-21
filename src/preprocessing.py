"""Data loading and preprocessing for the World Cup 2026 predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPECTED_RESULTS_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
}
EXPECTED_ELO_COLUMNS = {"date", "team", "elo_rating"}


def configure_logging() -> None:
    """Configure compact console logging for command-line runs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse dates that may mix ISO and slash-separated formats.

    Args:
        values: Raw date values.

    Returns:
        Parsed pandas datetime series.
    """
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def resolve_data_file(raw_dir: Path, candidates: Iterable[str]) -> Path:
    """Return the first existing data file from a list of candidate names.

    Args:
        raw_dir: Directory containing raw input files.
        candidates: Candidate relative paths, checked in order.

    Returns:
        Path to the first matching CSV.

    Raises:
        FileNotFoundError: If none of the candidates exist.
    """
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path
    joined = ", ".join(str(raw_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any expected raw data file: {joined}")


def validate_columns(
    frame: pd.DataFrame, expected: set[str], dataset_name: str
) -> None:
    """Log a warning if an input dataset is missing expected columns.

    Args:
        frame: Loaded dataset.
        expected: Expected column names.
        dataset_name: Human-readable dataset label for logging.
    """
    missing = sorted(expected - set(frame.columns))
    if missing:
        LOGGER.warning("%s is missing expected columns: %s", dataset_name, missing)


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw results and ELO ratings from supported repository layouts.

    Args:
        raw_dir: Raw data directory.

    Returns:
        Tuple of raw match results and raw ELO ratings.
    """
    results_path = resolve_data_file(
        raw_dir,
        (
            "wc_results.csv",
            "footballresults/results.csv",
            "results.csv",
        ),
    )
    elo_path = resolve_data_file(raw_dir, ("elo_ratings.csv", "eloratings.csv"))

    LOGGER.info("Loading match results from %s", results_path)
    LOGGER.info("Loading ELO ratings from %s", elo_path)
    results = pd.read_csv(results_path)
    elo = pd.read_csv(elo_path)

    validate_columns(results, EXPECTED_RESULTS_COLUMNS, "Match results")
    validate_columns(
        elo.rename(columns={"rating": "elo_rating"}),
        EXPECTED_ELO_COLUMNS,
        "ELO ratings",
    )
    return results, elo


def normalize_team_name(name: object) -> str:
    """Normalize a team name to the convention used by the historical data.

    Args:
        name: Raw team name.

    Returns:
        Normalized team name.
    """
    if pd.isna(name):
        return "Unknown"

    aliases = {
        "Czechia": "Czech Republic",
        "Korea Republic": "South Korea",
        "Republic of Korea": "South Korea",
        "USA": "United States",
        "United States of America": "United States",
        "Türkiye": "Turkey",
        "Turkey": "Turkey",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "IR Iran": "Iran",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde",
        "DR Congo": "DR Congo",
        "Congo DR": "DR Congo",
        "Democratic Republic of the Congo": "DR Congo",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    }
    cleaned = str(name).strip()
    return aliases.get(cleaned, cleaned)


def infer_stage(row: pd.Series) -> str:
    """Infer a coarse tournament stage when the source dataset has no stage.

    Args:
        row: Match row.

    Returns:
        Stage name.
    """
    stage = str(row.get("stage", "") or "").strip()
    if stage:
        return stage
    return "Group"


def preprocess_results(results: pd.DataFrame) -> pd.DataFrame:
    """Clean match results while preserving all usable historical rows.

    Args:
        results: Raw match results.

    Returns:
        Cleaned match results with standard columns.
    """
    frame = results.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    for column in required:
        if column not in frame.columns:
            LOGGER.warning(
                "Missing required results column %s; filling with NA", column
            )
            frame[column] = pd.NA

    if "tournament" not in frame.columns:
        LOGGER.warning("Missing tournament column; using 'Unknown'")
        frame["tournament"] = "Unknown"

    frame["date"] = parse_mixed_dates(frame["date"])
    for column in ("home_score", "away_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["stage"] = frame.apply(infer_stage, axis=1)

    before = len(frame)
    frame = frame.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    dropped = before - len(frame)
    if dropped:
        LOGGER.warning("Dropped %s result rows with unusable core fields", dropped)

    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def preprocess_elo(elo: pd.DataFrame) -> pd.DataFrame:
    """Clean ELO ratings and normalize the rating column name.

    Args:
        elo: Raw ELO ratings.

    Returns:
        Cleaned ELO ratings with date, team, and elo_rating columns.
    """
    frame = elo.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if "elo_rating" not in frame.columns and "rating" in frame.columns:
        frame = frame.rename(columns={"rating": "elo_rating"})

    for column in ("date", "team", "elo_rating"):
        if column not in frame.columns:
            LOGGER.warning("Missing ELO column %s; filling with NA", column)
            frame[column] = pd.NA

    frame["date"] = parse_mixed_dates(frame["date"])
    frame["team"] = frame["team"].map(normalize_team_name)
    frame["elo_rating"] = pd.to_numeric(frame["elo_rating"], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=["date", "team", "elo_rating"])
    dropped = before - len(frame)
    if dropped:
        LOGGER.warning("Dropped %s ELO rows with unusable core fields", dropped)

    frame["elo_rating"] = frame["elo_rating"].astype(float)
    return (
        frame[["date", "team", "elo_rating"]]
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )


def load_and_preprocess(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess all raw input datasets.

    Args:
        raw_dir: Raw data directory.

    Returns:
        Tuple of cleaned match results and cleaned ELO ratings.
    """
    results, elo = load_raw_data(raw_dir)
    return preprocess_results(results), preprocess_elo(elo)
