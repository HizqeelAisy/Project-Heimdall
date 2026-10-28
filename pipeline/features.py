"""Step 2: features for a forecast made H hours ahead.

Rule that prevents leakage: every load feature for hour t uses only readings up to t-H.
Weather at hour t is used as-is, which assumes a perfect weather forecast (stated caveat).
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def make_features(clean: pd.DataFrame, horizon: int, scale: pd.Series) -> pd.DataFrame:
    df = clean.sort_values(["building", "timestamp"]).copy()
    df["y"] = df["kw"] / df["building"].map(scale)          # normalised load, so one model fits all buildings
    g = df.groupby("building")["y"]

    lags = sorted({horizon, horizon + 1, horizon + 2, 24 if horizon <= 24 else horizon, 48, 168, 336})
    lags = [l for l in lags if l >= horizon]
    for l in lags:
        df[f"lag_{l}"] = g.shift(l)
    known = g.shift(horizon)                                # last reading available when forecasting
    df["roll24_mean"] = known.groupby(df["building"]).transform(lambda s: s.rolling(24, min_periods=12).mean())
    df["roll24_max"] = known.groupby(df["building"]).transform(lambda s: s.rolling(24, min_periods=12).max())
    df["roll168_mean"] = known.groupby(df["building"]).transform(lambda s: s.rolling(168, min_periods=48).mean())

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = ts.dt.month
    hol = USFederalHolidayCalendar().holidays(ts.min(), ts.max())   # Malaysia deployment: swap for MY holidays
    df["holiday"] = ts.dt.normalize().isin(hol).astype(int)

    tg = df.groupby("building")["temp"]
    df["temp_roll24"] = tg.transform(lambda s: s.rolling(24, min_periods=6).mean())   # thermal mass of the building
    df["temp_max_today"] = df.groupby(["building", ts.dt.date])["temp"].transform("max")
    df["cdd"] = (df["temp"] - 18).clip(lower=0)                    # cooling degree hours

    df["building_cat"] = df["building"].astype("category")
    df["site_cat"] = df["site"].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = ["hour", "dow", "weekend", "month", "holiday", "temp", "dew", "temp_roll24",
            "temp_max_today", "cdd", "roll24_mean", "roll24_max", "roll168_mean", "building_cat", "site_cat"]
    return base + [c for c in df.columns if c.startswith("lag_")]
