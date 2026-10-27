"""Student C's features: Volatility20, MaxPrice20, MaxDrawdown, GapRisk.

Owned by Harshini Prasad (26025369). Depends on Student A's DailyReturn, so
student_a.add_features must run first (see crypto_features.build_all_features).
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volatility20, MaxPrice20, MaxDrawdown, GapRisk.

    Requires columns: token, open, high, close, DailyReturn (from Student A).
    Computed per `token`, on data already sorted by date, so rolling windows never
    mix coins.
    """
    if "DailyReturn" not in df.columns:
        raise ValueError("student_c.add_features requires DailyReturn (Student A) first")

    grouped = df.groupby("token")

    # Volatility20: 20-day rolling std of DailyReturn, sample std (ddof=1).
    df["Volatility20"] = grouped["DailyReturn"].transform(lambda s: s.rolling(20).std(ddof=1))

    # MaxPrice20: 20-day rolling max of High (not Close).
    df["MaxPrice20"] = grouped["high"].transform(lambda s: s.rolling(20).max())

    # MaxDrawdown: distance of today's close below the 20-day high, as a fraction.
    df["MaxDrawdown"] = (df["MaxPrice20"] - df["close"]) / df["MaxPrice20"]

    # GapRisk: 20-day rolling mean of the absolute overnight gap, relative to
    # yesterday's close (Anthony, 6 Oct: 20-day rolling mean, not a single-day gap).
    prev_close = grouped["close"].shift(1)
    daily_gap = (df["open"] - prev_close).abs() / prev_close
    df["GapRisk"] = daily_gap.groupby(df["token"]).transform(lambda s: s.rolling(20).mean())

    return df
