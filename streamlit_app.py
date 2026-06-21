"""Streamlit dashboard for WC 2026 prediction artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.live_data import (
    COMPLETED_STATUSES,
    LIVE_STATUSES,
    UPCOMING_STATUSES,
    fetch_live_matches,
    load_processed_live_matches,
    save_live_matches,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LIVE_MATCHES_PATH = PROCESSED_DIR / "wc2026_live_matches.csv"
ROUND_COLUMNS = [
    "r32_probability",
    "r16_probability",
    "qf_probability",
    "sf_probability",
    "final_probability",
    "win_probability",
]
ROUND_LABELS = {
    "r32_probability": "R32",
    "r16_probability": "R16",
    "qf_probability": "QF",
    "sf_probability": "SF",
    "final_probability": "Final",
    "win_probability": "Win",
}


st.set_page_config(
    page_title="WC 2026 Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --pitch: #07130f;
      --panel: #0f1c18;
      --panel-2: #14251f;
      --line: rgba(211, 255, 225, 0.16);
      --text: #eef9f1;
      --muted: #9fb5a8;
      --accent: #4dff91;
      --gold: #f2c94c;
      --danger: #ff5c5c;
    }
    .stApp {
      background:
        linear-gradient(135deg, rgba(77,255,145,.07), transparent 28%),
        repeating-linear-gradient(90deg, rgba(255,255,255,.025), rgba(255,255,255,.025) 1px, transparent 1px, transparent 48px),
        radial-gradient(circle at 78% 0%, rgba(242,201,76,.12), transparent 36%),
        var(--pitch);
      color: var(--text);
    }
    .block-container {
      padding-top: 1.25rem;
      padding-bottom: 2.5rem;
      max-width: 1480px;
    }
    h1, h2, h3 {
      letter-spacing: 0;
    }
    .app-header {
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(15,28,24,.94), rgba(20,37,31,.82));
      padding: 20px 22px;
      border-radius: 8px;
      margin-bottom: 14px;
      box-shadow: 0 16px 40px rgba(0,0,0,.22);
    }
    .kicker {
      color: var(--accent);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .headline {
      font-size: clamp(2rem, 4.2vw, 4rem);
      font-weight: 900;
      line-height: 1;
      margin: 0;
    }
    .subline {
      color: var(--muted);
      margin: 8px 0 0;
      font-size: 0.98rem;
    }
    [data-testid="stMetric"] {
      background: linear-gradient(180deg, rgba(20,37,31,.98), rgba(11,24,19,.98));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 104px;
    }
    [data-testid="stMetricValue"] {
      color: var(--text);
      font-weight: 900;
    }
    div[data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .status-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0 12px;
    }
    .status-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: rgba(15, 28, 24, .88);
    }
    .status-label {
      color: var(--muted);
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .1em;
      margin-bottom: 5px;
    }
    .status-value {
      color: var(--text);
      font-size: 1.15rem;
      font-weight: 850;
    }
    .ok { color: var(--accent); }
    .warn { color: var(--gold); }
    .bad { color: var(--danger); }
    .section-rule {
      border-top: 1px solid var(--line);
      margin: 8px 0 18px;
    }
    .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
      background: rgba(15,28,24,.7);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
    }
    @media (max-width: 900px) {
      .status-strip { grid-template-columns: 1fr 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce") * 100
            ).round(2)
    return output


def format_match_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output = frame[[column for column in columns if column in frame.columns]].copy()
    for column in ["prob_home_win", "prob_draw", "prob_away_win"]:
        if column in output.columns:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce") * 100
            ).round(1)
    return output


def merge_live_with_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    live = load_processed_live_matches(LIVE_MATCHES_PATH)
    if live.empty:
        return predictions
    if predictions.empty:
        return live

    prediction_columns = [
        "stage",
        "home_team",
        "away_team",
        "predicted_label",
        "predicted_outcome",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "predicted_home_score",
        "predicted_away_score",
    ]
    available = [
        column for column in prediction_columns if column in predictions.columns
    ]
    merged = live.merge(
        predictions[available].drop_duplicates(
            subset=["stage", "home_team", "away_team"]
        ),
        on=["stage", "home_team", "away_team"],
        how="left",
    )
    return merged


def artifact_status() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]
):
    summary = load_csv(OUTPUTS_DIR / "simulation_results.csv")
    groups = load_csv(OUTPUTS_DIR / "group_stage_predictions.csv")
    matches = merge_live_with_predictions(
        load_csv(OUTPUTS_DIR / "match_predictions.csv")
    )
    metadata = load_json(OUTPUTS_DIR / "run_metadata.json")
    return summary, groups, matches, metadata


def match_tables(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matches.empty or "status" not in matches.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    statuses = matches["status"].astype(str).str.upper()
    completed = matches[statuses.isin(COMPLETED_STATUSES)].copy()
    live = matches[statuses.isin(LIVE_STATUSES)].copy()
    upcoming = matches[statuses.isin(UPCOMING_STATUSES)].copy()
    for frame in [completed, live, upcoming]:
        if "utc_date" in frame.columns:
            frame.sort_values("utc_date", inplace=True, na_position="last")
    return completed, live, upcoming


def top_contenders_chart(summary: pd.DataFrame) -> alt.Chart:
    data = summary.sort_values("win_probability", ascending=False).head(14).copy()
    data["win_percent"] = data["win_probability"] * 100
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            x=alt.X("win_percent:Q", title="Win probability (%)"),
            y=alt.Y("team:N", title="", sort="-x"),
            color=alt.Color("confederation:N", legend=alt.Legend(title="Confed")),
            tooltip=[
                alt.Tooltip("team:N"),
                alt.Tooltip("win_percent:Q", title="Win %", format=".2f"),
                alt.Tooltip("final_probability:Q", title="Final", format=".1%"),
            ],
        )
        .properties(height=360)
    )


def advancement_chart(summary: pd.DataFrame, team: str) -> alt.Chart:
    row = summary[summary["team"] == team]
    if row.empty:
        return alt.Chart(pd.DataFrame({"round": [], "probability": []})).mark_bar()
    data = pd.DataFrame(
        {
            "round": [ROUND_LABELS[column] for column in ROUND_COLUMNS],
            "probability": [
                float(row.iloc[0][column]) * 100 for column in ROUND_COLUMNS
            ],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "probability:Q",
                title="Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("round:N", title="", sort=list(reversed(data["round"].tolist()))),
            color=alt.value("#4dff91"),
            tooltip=[
                alt.Tooltip("round:N"),
                alt.Tooltip("probability:Q", format=".2f"),
            ],
        )
        .properties(height=220)
    )


def group_rank_chart(group_data: pd.DataFrame) -> alt.Chart:
    rank_columns = [
        "rank_1_probability",
        "rank_2_probability",
        "rank_3_probability",
        "rank_4_probability",
    ]
    chart_data = group_data.melt(
        id_vars=["team"],
        value_vars=rank_columns,
        var_name="finish",
        value_name="probability",
    )
    chart_data["probability"] *= 100
    chart_data["finish"] = chart_data["finish"].str.extract(r"rank_(\d)_")[0] + "st"
    chart_data["finish"] = chart_data["finish"].replace(
        {"1st": "1st", "2st": "2nd", "3st": "3rd", "4st": "4th"}
    )
    team_order = group_data.sort_values(
        ["expected_finish", "qualification_probability"],
        ascending=[True, False],
    )["team"].tolist()
    return (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", title="Finish probability (%)", stack="normalize"),
            y=alt.Y("team:N", title="", sort=team_order),
            color=alt.Color(
                "finish:N",
                scale=alt.Scale(range=["#4dff91", "#8be0ff", "#f2c94c", "#ff5c5c"]),
                legend=alt.Legend(title="Finish"),
            ),
            tooltip=[
                alt.Tooltip("team:N"),
                alt.Tooltip("finish:N"),
                alt.Tooltip("probability:Q", format=".1f"),
            ],
        )
        .properties(height=220)
    )


def integrity_status(metadata: dict[str, Any]) -> tuple[str, str]:
    integrity = metadata.get("simulation_integrity", {})
    if not integrity:
        return "Not recorded", "warn"
    return ("Valid", "ok") if integrity.get("valid") else ("Invalid", "bad")


def render_status_strip(metadata: dict[str, Any], matches: pd.DataFrame) -> None:
    completed, live, upcoming = match_tables(matches)
    integrity_label, integrity_class = integrity_status(metadata)
    token_label = (
        "Configured" if os.environ.get("FOOTBALL_DATA_API_TOKEN") else "Missing"
    )
    token_class = "ok" if token_label == "Configured" else "warn"
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item">
            <div class="status-label">Simulation integrity</div>
            <div class="status-value {integrity_class}">{integrity_label}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Live token</div>
            <div class="status-value {token_class}">{token_label}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Match state</div>
            <div class="status-value">{len(completed)} done / {len(live)} live / {len(upcoming)} scheduled</div>
          </div>
          <div class="status-item">
            <div class="status-label">Data source</div>
            <div class="status-value">{metadata.get("live_source", "none")}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


summary, groups, matches, metadata = artifact_status()
completed_matches, live_matches, upcoming_matches = match_tables(matches)

st.markdown(
    """
    <div class="app-header">
      <div class="kicker">World Cup 2026 Monte Carlo Control Room</div>
      <h1 class="headline">Prediction board</h1>
      <p class="subline">Live fixtures, locked results, group odds, bracket paths, and run integrity in one view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

action_left, action_right = st.columns([0.72, 0.28], vertical_alignment="center")
with action_left:
    render_status_strip(metadata, matches)
with action_right:
    if st.button("Refresh live data", width="stretch"):
        try:
            result = fetch_live_matches(RAW_DIR, source="auto")
            save_live_matches(result.matches, LIVE_MATCHES_PATH)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Live refresh failed: {exc}")

metric_columns = st.columns(5)
if not summary.empty:
    leader = summary.sort_values("win_probability", ascending=False).iloc[0]
    metric_columns[0].metric("Favorite", leader["team"])
    metric_columns[1].metric("Win", f"{leader['win_probability'] * 100:.2f}%")
    metric_columns[2].metric("Final", f"{leader['final_probability'] * 100:.2f}%")
else:
    metric_columns[0].metric("Favorite", "No run")
    metric_columns[1].metric("Win", "0.00%")
    metric_columns[2].metric("Final", "0.00%")
metric_columns[3].metric("Simulations", f"{metadata.get('simulations', 0):,}")
metric_columns[4].metric("Locked results", f"{metadata.get('locked_match_count', 0):,}")

overview_tab, matches_tab, groups_tab, bracket_tab, team_tab, data_tab = st.tabs(
    ["Overview", "Matches", "Groups", "Bracket", "Team", "Data"]
)

with overview_tab:
    if summary.empty:
        st.info("No simulation artifact found.")
    else:
        chart_col, table_col = st.columns([0.58, 0.42])
        with chart_col:
            st.subheader("Title race")
            st.altair_chart(top_contenders_chart(summary), width="stretch")
        with table_col:
            st.subheader("Round probabilities")
            display = percent_frame(summary.head(16), ROUND_COLUMNS)
            st.dataframe(
                display[["team", "confederation", *ROUND_COLUMNS]],
                width="stretch",
                hide_index=True,
            )

with matches_tab:
    completed_columns = [
        "utc_date",
        "stage",
        "group",
        "home_team",
        "home_score",
        "away_score",
        "away_team",
        "predicted_label",
        "prediction_correct",
    ]
    live_columns = [
        "utc_date",
        "stage",
        "group",
        "home_team",
        "home_score",
        "away_score",
        "away_team",
        "minute",
        "predicted_label",
    ]
    upcoming_columns = [
        "utc_date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "venue",
        "predicted_label",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    ]
    c1, c2, c3 = st.columns(3)
    c1.metric("Completed", len(completed_matches))
    c2.metric("Live", len(live_matches))
    c3.metric("Scheduled", len(upcoming_matches))
    st.subheader("Completed matches")
    st.dataframe(
        format_match_table(completed_matches, completed_columns),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Live matches")
    st.dataframe(
        format_match_table(live_matches, live_columns),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Schedule")
    st.dataframe(
        format_match_table(upcoming_matches, upcoming_columns),
        width="stretch",
        hide_index=True,
    )

with groups_tab:
    if groups.empty:
        st.info("No group-stage artifact found.")
    else:
        group_filter = st.selectbox("Group", sorted(groups["group"].unique()), index=0)
        group_data = groups[groups["group"] == group_filter].copy()
        chart_col, table_col = st.columns([0.55, 0.45])
        with chart_col:
            st.subheader(f"Group {group_filter} finish distribution")
            st.altair_chart(group_rank_chart(group_data), width="stretch")
        with table_col:
            st.subheader("Projected table")
            st.dataframe(
                percent_frame(
                    group_data,
                    [
                        "rank_1_probability",
                        "rank_2_probability",
                        "rank_3_probability",
                        "rank_4_probability",
                        "qualification_probability",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )

with bracket_tab:
    slots = load_csv(OUTPUTS_DIR / "bracket_slot_probabilities.csv")
    if slots.empty:
        st.info("No bracket artifact found.")
    else:
        round_filter = st.selectbox(
            "Round",
            ["R32", "R16", "QF", "SF", "Final", "Champion"],
            index=0,
        )
        round_slots = slots[slots["round"] == round_filter].copy()
        round_slots["probability_percent"] = round_slots["probability"] * 100
        bracket_chart = (
            alt.Chart(round_slots)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("probability_percent:Q", title="Slot probability (%)"),
                y=alt.Y("slot_label:N", title="Slot", sort=None),
                color=alt.value("#8be0ff"),
                tooltip=[
                    alt.Tooltip("slot_label:N"),
                    alt.Tooltip("team:N"),
                    alt.Tooltip("probability_percent:Q", format=".1f"),
                ],
            )
            .properties(height=max(220, min(680, len(round_slots) * 24)))
        )
        st.altair_chart(bracket_chart, width="stretch")
        st.dataframe(
            percent_frame(round_slots, ["probability"]),
            width="stretch",
            hide_index=True,
        )

with team_tab:
    teams = sorted(summary["team"].unique()) if not summary.empty else []
    if not teams:
        st.info("No team artifact found.")
    else:
        selected_team = st.selectbox("Team", teams)
        team_summary = summary[summary["team"] == selected_team]
        team_groups = groups[groups["team"] == selected_team]
        if {"home_team", "away_team"}.issubset(matches.columns):
            team_matches = matches[
                (matches["home_team"] == selected_team)
                | (matches["away_team"] == selected_team)
            ]
        else:
            team_matches = pd.DataFrame()
        chart_col, table_col = st.columns([0.45, 0.55])
        with chart_col:
            st.subheader(f"{selected_team} path")
            st.altair_chart(
                advancement_chart(summary, selected_team),
                width="stretch",
            )
        with table_col:
            st.subheader("Team probabilities")
            st.dataframe(
                percent_frame(team_summary, ROUND_COLUMNS),
                width="stretch",
                hide_index=True,
            )
        st.subheader("Group profile")
        st.dataframe(team_groups, width="stretch", hide_index=True)
        st.subheader("Matches")
        st.dataframe(team_matches, width="stretch", hide_index=True)

with data_tab:
    integrity = metadata.get("simulation_integrity", {})
    if integrity:
        st.subheader("Simulation integrity")
        st.json(integrity)
        st.subheader("Round count checks")
        round_checks = pd.DataFrame.from_dict(
            integrity.get("round_totals", {}),
            orient="index",
        ).reset_index(names="round")
        st.dataframe(round_checks, width="stretch", hide_index=True)
        st.subheader("Probability total checks")
        probability_checks = pd.DataFrame.from_dict(
            integrity.get("probability_totals", {}),
            orient="index",
        ).reset_index(names="probability")
        st.dataframe(probability_checks, width="stretch", hide_index=True)
    st.subheader("Run metadata")
    st.json(metadata)
