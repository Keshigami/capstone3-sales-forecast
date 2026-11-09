"""Serve-time forecasting from the artifacts in models/ (no dataset needed once trained)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import PROJECT
from .features import PANEL_FEATURES, future_rows

MODELS = PROJECT / "models"


class NotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def artifacts() -> dict:
    if not (MODELS / "best_model.joblib").exists() or not (MODELS / "metrics.json").exists():
        raise NotTrained("models/ is empty: run `just train` first")
    import joblib

    metrics = json.loads((MODELS / "metrics.json").read_text())
    cat = json.loads((MODELS / "catalogue.json").read_text())
    recent = pd.read_csv(MODELS / "recent.csv", parse_dates=["date"])
    return {"model": joblib.load(MODELS / "best_model.joblib"), "metrics": metrics, "stores": {int(k): v for k, v in cat["stores"].items()},
            "items": pd.DataFrame(cat["items"]), "recent": recent, "last_date": pd.Timestamp(metrics["baseline"]["last_date"])}


def loaded() -> bool:
    try:
        artifacts()
        return True
    except NotTrained:
        return False


def resolve_store(store) -> int | None:
    """'all' | id | name (case-insensitive, prefix ok) -> store_id or None for all."""
    if store in (None, "", "all", "All", 0, "0"):
        return None
    st = artifacts()["stores"]
    if str(store).isdigit() and int(store) in st:
        return int(store)
    s = str(store).strip().lower()
    for k, v in st.items():
        if v.lower() == s or v.lower().startswith(s) or s in v.lower():
            return k
    raise ValueError(f"unknown store {store!r}; stores: {', '.join(f'{k}={v}' for k, v in st.items())}")


def resolve_item(item, store_id: int | None) -> int | None:
    if item in (None, "", "all", "All", 0, "0"):
        return None
    items = artifacts()["items"]
    if store_id is not None:
        items = items[items.store_id == store_id]
    if str(item).isdigit() and int(item) in set(items.item_id):
        return int(item)
    s = str(item).strip().lower()
    hit = items[items.name.str.lower().str.contains(s, regex=False)]
    if hit.empty:
        raise ValueError(f"unknown item {item!r}")
    return int(hit.iloc[0].item_id)


def forecast(store="all", item="all", days: int = 30, model: str = "best") -> dict:
    """Daily forecast for the horizon after the last date in the data. Values are units (best tabular) or $ (lstm)."""
    a = artifacts()
    days = max(1, min(int(days), 365))
    start = a["last_date"] + pd.Timedelta(days=1)
    if model == "lstm":
        return lstm_forecast(days, start, a)
    sid = resolve_store(store)
    iid = resolve_item(item, sid)
    items = a["items"]
    if sid is not None:
        items = items[items.store_id == sid]
    if iid is not None:
        items = items[items.item_id == iid]
    rows = future_rows(items, start, days)
    pred = np.clip(a["model"].predict(rows[PANEL_FEATURES]), 0, None)
    daily = pd.Series(pred).groupby(rows.date.values).sum()
    hist = a["recent"]
    if sid is not None:
        hist = hist[hist.store_id == sid]
    if iid is not None:
        hist = hist[hist.item_id == iid]
    hist = hist.groupby("date").item_count.sum().tail(60)
    return {
        "model": a["metrics"]["best_tabular"], "unit": "units",
        "store": a["stores"][sid] if sid else "all stores", "store_id": sid,
        "item": str(items.iloc[0]["name"]) if iid else "all items", "item_id": iid, "days": days,
        "dates": [str(d.date()) for d in daily.index], "values": [round(float(v), 1) for v in daily.values],
        "total": round(float(daily.sum()), 1), "mean_per_day": round(float(daily.mean()), 2),
        "history": {"dates": [str(d.date()) for d in hist.index], "values": [float(v) for v in hist.values]},
    }


def lstm_forecast(days: int, start: pd.Timestamp, a: dict) -> dict:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    meta = json.loads((MODELS / "lstm_meta.json").read_text())
    model = tf.keras.models.load_model(MODELS / "lstm.keras")
    batch = np.array(meta["seed"], dtype="float32").reshape(1, -1, 1)
    out = []
    for _ in range(days):
        p = float(np.clip(model.predict(batch, verbose=0)[0, 0], 0.0, 1.5))
        out.append(p)
        batch = np.append(batch[:, 1:, :], [[[p]]], axis=1)
    lo, hi = meta["scaler_min"], meta["scaler_max"]
    vals = np.clip(np.array(out) * (hi - lo) + lo, 0, None)
    dates = pd.date_range(start, periods=days)
    hist = a["recent"].groupby("date").sales_amnt.sum().tail(60)
    return {"model": "lstm", "unit": "revenue $", "store": "all stores", "store_id": None, "item": "all items", "item_id": None,
            "days": days, "dates": [str(d.date()) for d in dates], "values": [round(float(v), 2) for v in vals],
            "total": round(float(vals.sum()), 2), "mean_per_day": round(float(vals.mean()), 2),
            "history": {"dates": [str(d.date()) for d in hist.index], "values": [round(float(v), 2) for v in hist.values]}}
