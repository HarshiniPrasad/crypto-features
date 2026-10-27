"""Student D's features: SMA20(Volume), RelativeVolume.

Reference implementation from the group's agreed formulas (feature-ownership table).
Student D owns this file — replace with their own implementation; the function
signature and output column names must stay the same so the rest of the pipeline
doesn't need to change.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA20(Volume), RelativeVolume. Requires columns: token, volume.

    Computed per `token`, on data already sorted by date, so rolling windows never
    mix coins.
    """
    volume = df.groupby("token")["volume"]
    df["SMA20(Volume)"] = volume.transform(lambda s: s.rolling(20).mean())
    df["RelativeVolume"] = df["volume"] / df["SMA20(Volume)"]
    return df
