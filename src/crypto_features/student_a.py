"""Student A's features: DailyReturn, Return5, Return20, Momentum.

Reference implementation from the group's agreed formulas (feature-ownership table).
Student A owns this file — replace with their own implementation; the function
signature and output column names must stay the same so the rest of the pipeline
doesn't need to change.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add DailyReturn, Return5, Return20, Momentum. Requires columns: token, close.

    Computed per `token`, on data already sorted by date, so rolling windows never
    mix coins.
    """
    close = df.groupby("token")["close"]
    df["DailyReturn"] = close.pct_change(1)
    df["Return5"] = close.pct_change(5)
    df["Return20"] = close.pct_change(20)
    df["Momentum"] = df["close"] - close.shift(20)
    return df
