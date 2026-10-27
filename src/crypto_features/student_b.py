"""Student B's features: MA5, MA20, TrendStrength.

Reference implementation from the group's agreed formulas (feature-ownership table).
Student B owns this file — replace with their own implementation; the function
signature and output column names must stay the same so the rest of the pipeline
doesn't need to change.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5, MA20, TrendStrength. Requires columns: token, close.

    Computed per `token`, on data already sorted by date, so rolling windows never
    mix coins.
    """
    close = df.groupby("token")["close"]
    df["MA5"] = close.transform(lambda s: s.rolling(5).mean())
    df["MA20"] = close.transform(lambda s: s.rolling(20).mean())
    df["TrendStrength"] = df["MA5"] / df["MA20"] - 1
    return df
