"""Live fixture/result ingestion and match-prediction helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import normalize_team_name, parse_mixed_dates
from src.simulate import (
    PredictionContext,
    predict_probabilities,
    sample_scoreline,
)
from src.tournament import TournamentConfig

LIVE_MATCH_COLUMNS = [
    "match_id",
    "utc_date",
    "stage",
    "group",
    "home_team",
    "away_team",
    "status",
    "minute",
    "home_score",
    "away_score",
    "venue",
    "source",
    "last_updated",
]

COMPLETED_STATUSES = {"FINISHED", "AWARDED"}
LIVE_STATUSES = {"IN_PLAY", "PAUSED", "LIVE"}
UPCOMING_STATUSES = {"SCHEDULED", "TIMED", "POSTPONED"}


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional until dependencies are installed
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class LiveDataResult:
    """Normalized fixture/result payload plus source metadata."""

    matches: pd.DataFrame
    source: str
    fetched_at: pd.Timestamp


def normalize_status(value: object) -> str:
    """Normalize provider-specific match statuses."""
    status = str(value or "SCHEDULED").strip().upper()
    return status if status else "SCHEDULED"


def normalize_match_frame(frame: pd.DataFrame, source: str = "csv") -> pd.DataFrame:
    """Return a normalized live-match dataframe."""
    output = frame.copy()
    rename_map = {
        "date": "utc_date",
        "home": "home_team",
        "away": "away_team",
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "home_goals": "home_score",
        "away_goals": "away_score",
    }
    output = output.rename(
        columns={key: value for key, value in rename_map.items() if key in output}
    )
    for column in LIVE_MATCH_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    output["utc_date"] = parse_mixed_dates(output["utc_date"])
    output["home_team"] = output["home_team"].map(normalize_team_name)
    output["away_team"] = output["away_team"].map(normalize_team_name)
    output["stage"] = output["stage"].fillna("Group").astype(str)
    output["group"] = output["group"].fillna("").astype(str)
    output["status"] = output["status"].map(normalize_status)
    output["minute"] = pd.to_numeric(output["minute"], errors="coerce")
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output["source"] = output["source"].fillna(source).replace("", source)
    output["last_updated"] = output["last_updated"].fillna(
        pd.Timestamp.now(tz="UTC").isoformat()
    )
    if output["match_id"].isna().any():
        generated = (
            output["utc_date"].dt.strftime("%Y%m%d%H%M").fillna("undated")
            + "-"
            + output["home_team"].astype(str).str.replace(r"\W+", "-", regex=True)
            + "-"
            + output["away_team"].astype(str).str.replace(r"\W+", "-", regex=True)
        )
        output["match_id"] = output["match_id"].fillna(generated)

    output = output.dropna(subset=["home_team", "away_team"]).copy()
    return output[LIVE_MATCH_COLUMNS].reset_index(drop=True)


class CsvFixtureProvider:
    """Read fixtures/results from a local CSV fallback."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self) -> LiveDataResult:
        if not self.path.exists():
            return LiveDataResult(
                pd.DataFrame(columns=LIVE_MATCH_COLUMNS),
                "csv",
                pd.Timestamp.now(tz="UTC"),
            )
        frame = normalize_match_frame(pd.read_csv(self.path), source="csv")
        return LiveDataResult(frame, "csv", pd.Timestamp.now(tz="UTC"))


class FootballDataProvider:
    """Fetch FIFA World Cup matches from football-data.org."""

    API_URL = "https://api.football-data.org/v4/competitions/WC/matches"

    def __init__(self, token: str | None = None, season: int = 2026) -> None:
        self.token = token or os.environ.get("FOOTBALL_DATA_API_TOKEN")
        self.season = season

    def fetch(self) -> LiveDataResult:
        if not self.token:
            raise RuntimeError("FOOTBALL_DATA_API_TOKEN is not set")
        response = requests.get(
            self.API_URL,
            params={"season": self.season},
            headers={"X-Auth-Token": self.token},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        rows = [
            self._normalize_provider_match(match)
            for match in payload.get("matches", [])
        ]
        frame = normalize_match_frame(pd.DataFrame(rows), source="football-data")
        return LiveDataResult(frame, "football-data", pd.Timestamp.now(tz="UTC"))

    @staticmethod
    def _normalize_provider_match(match: dict[str, Any]) -> dict[str, Any]:
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        group = match.get("group") or ""
        return {
            "match_id": match.get("id"),
            "utc_date": match.get("utcDate"),
            "stage": match.get("stage") or "Group",
            "group": group,
            "home_team": home.get("name") or home.get("shortName"),
            "away_team": away.get("name") or away.get("shortName"),
            "status": match.get("status"),
            "minute": match.get("minute"),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
            "venue": match.get("venue"),
            "source": "football-data",
        }


def fetch_live_matches(
    raw_dir: Path,
    source: str = "auto",
    token: str | None = None,
) -> LiveDataResult:
    """Fetch live matches from the requested provider with CSV fallback."""
    if source not in {"auto", "football-data", "csv"}:
        raise ValueError("source must be auto, football-data, or csv")
    csv_provider = CsvFixtureProvider(raw_dir / "wc2026_fixtures.csv")
    if source == "csv":
        return csv_provider.fetch()
    try:
        return FootballDataProvider(token=token).fetch()
    except Exception:
        if source == "football-data":
            raise
        return csv_provider.fetch()


def save_live_matches(matches: pd.DataFrame, output_path: Path) -> None:
    """Persist normalized fixtures/results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_match_frame(matches).to_csv(output_path, index=False)


def load_processed_live_matches(path: Path) -> pd.DataFrame:
    """Load normalized processed live fixtures/results if present."""
    if not path.exists():
        return pd.DataFrame(columns=LIVE_MATCH_COLUMNS)
    return normalize_match_frame(pd.read_csv(path), source="processed")


def actual_outcome(row: pd.Series) -> str | None:
    """Return A/D/B actual outcome for completed match rows."""
    if normalize_status(row.get("status")) not in COMPLETED_STATUSES:
        return None
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return None
    home_score = int(row["home_score"])
    away_score = int(row["away_score"])
    if home_score > away_score:
        return "A"
    if away_score > home_score:
        return "B"
    return "D"


def parse_what_if(value: str) -> dict[str, Any]:
    """Parse strings like 'Brazil 2-1 Morocco' into a scenario override."""
    pattern = re.compile(
        r"^\s*(?P<home>.+?)\s+(?P<hs>\d+)\s*-\s*(?P<as>\d+)\s+(?P<away>.+?)\s*$"
    )
    match = pattern.match(value)
    if not match:
        raise ValueError("what-if must look like 'Home Team 2-1 Away Team'")
    return {
        "home_team": normalize_team_name(match.group("home")),
        "away_team": normalize_team_name(match.group("away")),
        "home_score": int(match.group("hs")),
        "away_score": int(match.group("as")),
        "status": "FINISHED",
        "source": "what-if",
    }


def apply_what_if(matches: pd.DataFrame, value: str | None) -> pd.DataFrame:
    """Apply a single completed-result scenario override."""
    if not value:
        return matches
    override = parse_what_if(value)
    frame = (
        normalize_match_frame(matches)
        if not matches.empty
        else pd.DataFrame(columns=LIVE_MATCH_COLUMNS)
    )
    mask = (
        (frame["home_team"] == override["home_team"])
        & (frame["away_team"] == override["away_team"])
    ) | (
        (frame["home_team"] == override["away_team"])
        & (frame["away_team"] == override["home_team"])
    )
    if mask.any():
        index = frame[mask].index[0]
        home_matches_override = frame.at[index, "home_team"] == override["home_team"]
        frame.at[index, "status"] = "FINISHED"
        frame.at[index, "source"] = "what-if"
        if home_matches_override:
            frame.at[index, "home_score"] = override["home_score"]
            frame.at[index, "away_score"] = override["away_score"]
        else:
            frame.at[index, "home_score"] = override["away_score"]
            frame.at[index, "away_score"] = override["home_score"]
        return frame
    row = {column: pd.NA for column in LIVE_MATCH_COLUMNS}
    row.update(override)
    row["match_id"] = f"what-if-{override['home_team']}-{override['away_team']}"
    row["stage"] = "Group"
    row["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    return normalize_match_frame(
        pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    )


def fixture_frame_from_config(config: TournamentConfig) -> pd.DataFrame:
    """Build a fixture-like frame from configured group combinations."""
    rows: list[dict[str, Any]] = []
    for group, teams in config.groups.items():
        for index, home in enumerate(teams):
            for away in teams[index + 1 :]:
                rows.append(
                    {
                        "match_id": f"group-{group}-{home}-{away}",
                        "stage": "Group",
                        "group": group,
                        "home_team": home,
                        "away_team": away,
                        "status": "SCHEDULED",
                        "source": "config",
                    }
                )
    return normalize_match_frame(pd.DataFrame(rows), source="config")


def build_match_predictions(
    model: Any,
    label_encoder: LabelEncoder,
    context: PredictionContext,
    matches: pd.DataFrame | None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create per-fixture model predictions and completed-match scoring."""
    frame = (
        normalize_match_frame(matches)
        if matches is not None and not matches.empty
        else fixture_frame_from_config(context.tournament_config)
    )
    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, match in frame.iterrows():
        stage = str(match.get("stage") or "Group")
        home = str(match["home_team"])
        away = str(match["away_team"])
        probabilities = predict_probabilities(
            model, label_encoder, home, away, context, stage, cache
        )
        predicted_outcome = max(probabilities, key=probabilities.get)
        predicted_home_score, predicted_away_score = sample_scoreline(
            predicted_outcome,
            home,
            away,
            context,
            rng,
            draw_probability=probabilities["D"],
        )
        actual = actual_outcome(match)
        rows.append(
            {
                **{column: match.get(column) for column in LIVE_MATCH_COLUMNS},
                "prob_home_win": probabilities["A"],
                "prob_draw": probabilities["D"],
                "prob_away_win": probabilities["B"],
                "predicted_outcome": predicted_outcome,
                "predicted_label": {
                    "A": f"{home} win",
                    "D": "Draw",
                    "B": f"{away} win",
                }[predicted_outcome],
                "predicted_home_score": predicted_home_score,
                "predicted_away_score": predicted_away_score,
                "actual_outcome": actual,
                "prediction_correct": (
                    bool(actual == predicted_outcome) if actual is not None else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def team_paths_frame(
    team: str,
    summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    match_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact one-team path view for dashboard/CSV export."""
    normalized = normalize_team_name(team)
    rows: list[dict[str, Any]] = []
    team_summary = summary[summary["team"] == normalized]
    if not team_summary.empty:
        for column, label in [
            ("r32_probability", "Reach Round of 32"),
            ("r16_probability", "Reach Round of 16"),
            ("qf_probability", "Reach Quarterfinal"),
            ("sf_probability", "Reach Semifinal"),
            ("final_probability", "Reach Final"),
            ("win_probability", "Win Tournament"),
        ]:
            rows.append(
                {
                    "team": normalized,
                    "step": label,
                    "probability": float(team_summary.iloc[0][column]),
                }
            )
    group_row = group_summary[group_summary["team"] == normalized]
    if not group_row.empty:
        rows.append(
            {
                "team": normalized,
                "step": "Qualify from group",
                "probability": float(group_row.iloc[0]["qualification_probability"]),
            }
        )
    fixtures = match_predictions[
        (match_predictions["home_team"] == normalized)
        | (match_predictions["away_team"] == normalized)
    ]
    for _, row in fixtures.iterrows():
        opponent = (
            row["away_team"] if row["home_team"] == normalized else row["home_team"]
        )
        rows.append(
            {
                "team": normalized,
                "step": f"{row['stage']} vs {opponent}",
                "probability": (
                    float(row["prob_home_win"])
                    if row["home_team"] == normalized
                    else float(row["prob_away_win"])
                ),
            }
        )
    return pd.DataFrame(rows)
