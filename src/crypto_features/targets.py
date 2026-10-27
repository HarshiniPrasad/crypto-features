"""Target construction for all 4 students, from the group's agreed formulas.

Every target is computed per coin from the pooled feature table (see pipeline.py),
then shifted forward FORECAST_HORIZON days so it represents the value that many
days after the feature date — this is what turns each target into a genuine
forecast instead of same-day curve fitting.
"""

import pandas as pd

FORECAST_HORIZON = 7

TREND_LABELS = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
TOR_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def _shift(df: pd.DataFrame, column: str, horizon: int = FORECAST_HORIZON) -> pd.Series:
    return df.groupby("token")[column].shift(-horizon)


def add_student_a_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Asset Performance Index (API), 0-100, regression."""
    rs = ((df["Return20"] + 0.20) / 0.40).clip(0, 1)
    ts = ((df["TrendStrength"] + 0.10) / 0.20).clip(0, 1)
    rv = (df["RelativeVolume"].clip(upper=2)) / 2
    ps = 1 - df["Volatility20"]
    df["API"] = 100 * (0.40 * rs + 0.30 * ts + 0.20 * rv + 0.10 * ps)
    df[f"API_t+{horizon}"] = _shift(df, "API", horizon)
    return df


def add_student_b_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Market Risk Index (MRI), 0-100, regression."""
    dr = (df["MaxPrice20"] - df["close"]) / df["MaxPrice20"]
    vrn = (df["Volatility20"] / 0.10).clip(upper=1)
    drn = (dr / 0.40).clip(upper=1)
    rrn = (df["Return20"].clip(upper=0).abs() / 0.30).clip(upper=1)
    grn = (df["GapRisk"] / 0.10).clip(upper=1)
    df["MRI"] = 100 * (0.40 * vrn + 0.30 * drn + 0.20 * rrn + 0.10 * grn)
    df[f"MRI_t+{horizon}"] = _shift(df, "MRI", horizon)
    return df


def add_student_c_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Trend Category (TC), 5 classes, multiclass classification. Owned by Student C."""
    df["TrendScore"] = 0.7 * (df["MA5"] / df["MA20"] - 1) + 0.3 * df["Return20"]
    bins = [-float("inf"), -0.10, -0.02, 0.02, 0.10, float("inf")]
    df["TrendCategory"] = pd.cut(df["TrendScore"], bins=bins, labels=TREND_LABELS, right=False)
    df[f"TrendCategory_t+{horizon}"] = _shift(df, "TrendCategory", horizon)
    return df


def add_student_d_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Trading Opportunity Rating (TOR), 5 classes, multiclass classification."""
    m = (df["Return20"] + 0.20) / 0.40
    t = df["MA5"] / df["MA20"]
    v = (df["RelativeVolume"].clip(upper=2)) / 2
    r = (df["Volatility20"] / 0.10).clip(upper=1)
    score = 0.40 * m + 0.30 * t + 0.20 * v - 0.10 * r
    bins = [-float("inf"), 0.20, 0.40, 0.60, 0.80, float("inf")]
    df["TOR"] = pd.cut(score, bins=bins, labels=TOR_LABELS, right=False)
    df[f"TOR_t+{horizon}"] = _shift(df, "TOR", horizon)
    return df


def drop_unlabeled_tail(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop the last `horizon` rows per coin, which have no future value yet."""
    return df[df[target_column].notna()].reset_index(drop=True)
