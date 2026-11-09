"""Load and merge the three Capstone 3 CSVs into one frame (read-only source, DATA_ROOT env)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path.home() / "Caltech CTME" / "PGC AIML - Capstone" / "Datasets"


def data_dir() -> Path:
    root = Path(os.getenv("DATA_ROOT", DEFAULT_ROOT))
    return root / "Capstone 3" if (root / "Capstone 3").exists() else root


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data_dir()
    sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
    items = pd.read_csv(d / "items.csv")
    restaurants = pd.read_csv(d / "resturants.csv")
    return sales, items, restaurants


@lru_cache(maxsize=1)
def load_merged() -> pd.DataFrame:
    """date, item_id, price, item_count, store_id, item_name, kcal, restaurant_name, sales_amnt (+ date parts)."""
    sales, items, restaurants = load_raw()
    df = sales.merge(items, left_on="item_id", right_on="id", how="left").drop(columns=["id"])
    df = df.merge(restaurants.rename(columns={"id": "store_id", "name": "restaurant_name"}), on="store_id", how="left")
    df = df.rename(columns={"name": "item_name"}).drop(columns=["cost"])
    df["sales_amnt"] = df.item_count * df.price
    df["year"] = df.date.dt.year
    df["quarter"] = df.date.dt.quarter
    df["month"] = df.date.dt.month
    df["weekday"] = df.date.dt.dayofweek
    return df.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)


def stores() -> dict[int, str]:
    _, _, r = load_raw()
    return dict(zip(r.id.astype(int), r.name))


def items_table() -> pd.DataFrame:
    _, items, _ = load_raw()
    return items
