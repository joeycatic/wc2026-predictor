from __future__ import annotations

import pandas as pd

from src.preprocessing import normalize_team_name, parse_mixed_dates


def test_parse_mixed_dates_handles_iso_and_slash_formats() -> None:
    parsed = parse_mixed_dates(pd.Series(["2024-01-02", "03/04/2024"]))

    assert parsed.notna().all()
    assert parsed.iloc[0].year == 2024
    assert parsed.iloc[1].year == 2024


def test_normalize_team_name_uses_aliases() -> None:
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("Korea Republic") == "South Korea"
    assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"
    assert normalize_team_name("United\xa0States") == "United States"
    assert normalize_team_name("Democratic Republic of Congo") == "DR Congo"
