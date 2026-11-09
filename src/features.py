"""Date and entity features for the tabular models (panel: one row per date x item)."""
from __future__ import annotations

import pandas as pd

HOLDOUT_START = pd.Timestamp("2021-07-01")  # notebook: last six months are the test set
DATE_FEATURES = ["weekday", "quarter", "year", "month", "day_year", "day_month", "week_num"]
PANEL_FEATURES = DATE_FEATURES + ["store_id", "item_id", "price", "kcal"]


def add_date_features(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    d = df[col]
    out = df.copy()
    out["weekday"] = d.dt.dayofweek
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day_year"] = d.dt.dayofyear
    out["day_month"] = d.dt.day
    out["week_num"] = d.dt.isocalendar().week.astype(int)
    return out


def panel(df: pd.DataFrame) -> pd.DataFrame:
    """Merged frame -> model rows with PANEL_FEATURES and target item_count."""
    p = add_date_features(df[["date", "store_id", "item_id", "price", "kcal", "item_count"]])
    return p[["date", *PANEL_FEATURES, "item_count"]]


def daily_total(df: pd.DataFrame, value: str = "item_count") -> pd.DataFrame:
    """One row per date with the summed value and date features (the notebook's time_series)."""
    ts = df.groupby("date", as_index=False)[value].sum()
    return add_date_features(ts)


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame[frame.date < HOLDOUT_START], frame[frame.date >= HOLDOUT_START]


def future_rows(items: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Cartesian product of future dates x items with PANEL_FEATURES (prices are the catalogue cost)."""
    dates = pd.DataFrame({"date": pd.date_range(start, periods=days, freq="D")})
    it = items.rename(columns={"id": "item_id", "cost": "price"})[["item_id", "store_id", "price", "kcal"]]
    rows = dates.merge(it, how="cross")
    return add_date_features(rows)[["date", *PANEL_FEATURES]]
