"""
Unit tests για το src/data_cleaning.py
Τρέξιμο: pytest -v (από τον root φάκελο του project)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import clean


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "white_rating": [1500, 1600, None, 1400],
        "black_rating": [1400, 1550, 1600, 1450],
        "opening_name": ["Sicilian Defense", "Italian Game", "London System", "Slav Defense"],
        "opening_eco": ["B20", "C50", "D02", "D10"],
        "time_control": ["Blitz", "Rapid", "Bullet", "Classical"],
        "num_moves": [35, 40, 50, 60],
        "result": ["white", "black", "draw", "invalid_value"],
    })


def test_drops_rows_with_missing_ratings(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["white_rating"].isna().sum() == 0


def test_removes_invalid_results(raw_df):
    cleaned = clean(raw_df)
    assert set(cleaned["result"].unique()).issubset({"white", "black", "draw"})


def test_rating_diff_calculation(raw_df):
    cleaned = clean(raw_df)
    row = cleaned[cleaned["opening_name"] == "Sicilian Defense"].iloc[0]
    assert row["rating_diff"] == 1500 - 1400


def test_higher_rated_won_flag(raw_df):
    cleaned = clean(raw_df)
    row = cleaned[cleaned["opening_name"] == "Sicilian Defense"].iloc[0]
    # white_rating(1500) > black_rating(1400) και νίκησε ο λευκός -> True
    assert row["higher_rated_won"] == True  # noqa: E712
