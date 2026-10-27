"""Runs every student's feature function in dependency order on one shared dataframe.

Order: A and B first (no dependencies among the engineered features), then C (needs
A's DailyReturn), then D (no dependencies). If a student later adds a feature that
depends on another's, update this order accordingly.
"""

import pandas as pd

from crypto_features import student_a, student_b, student_c, student_d

REQUIRED_COLUMNS = ["token", "date", "open", "high", "low", "close", "volume"]


def build_all_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute every shared feature for all coins, sorted by date within each token."""
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")

    df = raw.sort_values(["token", "date"]).reset_index(drop=True)
    df = student_a.add_features(df)
    df = student_b.add_features(df)
    df = student_c.add_features(df)
    df = student_d.add_features(df)
    return df
