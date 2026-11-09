"""Train LR / RF / XGB on the (date x item) panel and an LSTM on daily revenue; write models/ and metrics.json.

    uv run python -m src.train --quick   # a couple of minutes
    uv run python -m src.train --full    # the notebook's setup (RF 100x20, XGB 1000 trees, LSTM batch 1 + early stop)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .data import PROJECT, items_table, load_merged, stores
from .features import DATE_FEATURES, HOLDOUT_START, PANEL_FEATURES, daily_total, panel, split

MODELS = PROJECT / "models"
LSTM_LENGTH = 12  # notebook: 12-day window for the evaluated model, 90 for the final refit (tanh cell instead of relu: relu diverged when fed its own output)
NOTEBOOK_REPORTED = {"linear_regression": {"rmse": 244.21, "mae": 196.04, "r2": 0.0765},
                     "random_forest": {"rmse": 59.91, "mae": 47.48, "r2": 0.9444},
                     "xgboost": {"rmse": 72.30, "mae": 59.45, "r2": 0.9190},
                     "lstm": {"mape": 1.3407}}


def scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}
    if (y > 0).all():
        out["mape"] = float(mean_absolute_percentage_error(y, p))
    out["r2"] = float(r2_score(y, p))
    return {k: round(v, 4) for k, v in out.items()}


def make_models(full: bool) -> dict:
    lr = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["store_id", "item_id", "weekday", "month"])],
                                               remainder="passthrough")), ("lr", LinearRegression())])
    rf = RandomForestRegressor(n_estimators=100 if full else 40, max_depth=20 if full else 14, n_jobs=-1, random_state=42)
    xgb = XGBRegressor(n_estimators=1000 if full else 300, learning_rate=0.02 if full else 0.08, max_depth=6, subsample=0.9,
                       colsample_bytree=0.9, early_stopping_rounds=50, random_state=42, n_jobs=0)
    return {"linear_regression": lr, "random_forest": rf, "xgboost": xgb}


def fit(name, model, X, y, Xv, yv):
    t = time.time()
    if name == "xgboost":
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        model.fit(X, y)
    return round(time.time() - t, 1)


def train_tabular(df: pd.DataFrame, full: bool) -> tuple[dict, str, object]:
    rows = panel(df)
    train, test = split(rows)
    # early-stopping slice for XGB: last 90 days of train (the holdout stays untouched)
    cut = HOLDOUT_START - pd.Timedelta(days=90)
    fit_part, val_part = train[train.date < cut], train[train.date >= cut]
    X, y = train[PANEL_FEATURES], train.item_count
    Xt, yt = test[PANEL_FEATURES], test.item_count
    results, fitted = {}, {}
    for name, model in make_models(full).items():
        if name == "xgboost":
            secs = fit(name, model, fit_part[PANEL_FEATURES], fit_part.item_count, val_part[PANEL_FEATURES], val_part.item_count)
        else:
            secs = fit(name, model, X, y, None, None)
        pred = np.clip(model.predict(Xt), 0, None)
        daily = pd.DataFrame({"date": test.date.values, "y": yt.values, "p": pred}).groupby("date").sum()
        by_store = pd.DataFrame({"store": test.store_id.values, "date": test.date.values, "y": yt.values, "p": pred}).groupby(["store", "date"]).sum()
        results[name] = {"fit_seconds": secs, "row_level": scores(yt, pred), "daily_total": scores(daily.y, daily.p),
                         "store_daily": {str(s): scores(g.y, g.p) for s, g in by_store.groupby(level=0)}}
        fitted[name] = model
        print(f"  {name:18s} fit {secs:6.1f}s  rows rmse {results[name]['row_level']['rmse']:8.2f}  daily-total rmse "
              f"{results[name]['daily_total']['rmse']:8.2f}  mape {results[name]['daily_total'].get('mape', float('nan')):.3f}")
    best = min(results, key=lambda k: results[k]["daily_total"]["rmse"])
    return results, best, fitted[best]


def notebook_replica(df: pd.DataFrame, full: bool) -> dict:
    """The notebook's exact framing: one daily-total series, date features only, last six months held out."""
    ts = daily_total(df)
    train, test = split(ts)
    X, y, Xt, yt = train[DATE_FEATURES], train.item_count, test[DATE_FEATURES], test.item_count
    out = {}
    lr = LinearRegression().fit(X, y)
    out["linear_regression"] = scores(yt, lr.predict(Xt))
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1).fit(X, y)
    out["random_forest"] = scores(yt, rf.predict(Xt))
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, early_stopping_rounds=50, random_state=42)
    xgb.fit(X, y, eval_set=[(Xt, yt)], verbose=False)  # the notebook early-stops on the test set; kept for comparability
    out["xgboost"] = scores(yt, xgb.predict(Xt))
    return out


def train_lstm(df: pd.DataFrame, full: bool) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    ts = df.groupby("date").sales_amnt.sum()
    train, test = ts[ts.index < HOLDOUT_START], ts[ts.index >= HOLDOUT_START]

    def windows(arr, length):
        xs = np.stack([arr[i:i + length] for i in range(len(arr) - length)])
        return xs.reshape(-1, length, 1), arr[length:]

    def build(length):
        m = tf.keras.Sequential([tf.keras.Input((length, 1)), tf.keras.layers.LSTM(100), tf.keras.layers.Dense(1)])
        m.compile(optimizer="adam", loss="mse")
        return m

    def recursive(model, seed, steps):
        batch = seed.reshape(1, -1, 1)
        out = []
        for _ in range(steps):
            p = float(np.clip(model.predict(batch, verbose=0)[0, 0], 0.0, 1.5))  # bounded in scaled space
            out.append(p)
            batch = np.append(batch[:, 1:, :], [[[p]]], axis=1)
        return np.array(out)

    t = time.time()
    scaler = MinMaxScaler().fit(train.values.reshape(-1, 1))
    s_train = scaler.transform(train.values.reshape(-1, 1)).ravel()
    s_test = scaler.transform(test.values.reshape(-1, 1)).ravel()
    Xw, yw = windows(s_train, LSTM_LENGTH)
    Xv, yv = windows(s_test, LSTM_LENGTH)
    model = build(LSTM_LENGTH)
    epochs, batch = (20, 1) if full else (15, 16)
    hist = model.fit(Xw, yw, validation_data=(Xv, yv), epochs=epochs, batch_size=batch, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    rec = scaler.inverse_transform(recursive(model, s_train[-LSTM_LENGTH:], len(test)).reshape(-1, 1)).ravel()
    one_step = scaler.inverse_transform(model.predict(Xv, verbose=0)).ravel()
    eval_out = {"epochs_run": len(hist.history["loss"]), "final_val_loss": round(float(hist.history["val_loss"][-1]), 5),
                "recursive_184_days": scores(test.values, rec), "one_step_ahead": scores(test.values[LSTM_LENGTH:], one_step)}

    # Final model on the full series, as the notebook does before forecasting next year.
    length = 90 if full else LSTM_LENGTH
    full_scaler = MinMaxScaler().fit(ts.values.reshape(-1, 1))
    s_full = full_scaler.transform(ts.values.reshape(-1, 1)).ravel()
    Xf, yf = windows(s_full, length)
    final = build(length)
    final.fit(Xf, yf, epochs=8 if full else 15, batch_size=1 if full else 16, verbose=0)
    MODELS.mkdir(exist_ok=True)
    final.save(MODELS / "lstm.keras")
    meta = {"target": "daily revenue, all stores", "length": length, "scaler_min": float(full_scaler.data_min_[0]),
            "scaler_max": float(full_scaler.data_max_[0]), "last_date": str(ts.index.max().date()),
            "seed": [float(v) for v in s_full[-length:]], "fit_seconds": round(time.time() - t, 1)}
    (MODELS / "lstm_meta.json").write_text(json.dumps(meta))
    print(f"  lstm               fit {meta['fit_seconds']:6.1f}s  recursive mape {eval_out['recursive_184_days']['mape']:.3f}  "
          f"one-step mape {eval_out['one_step_ahead']['mape']:.3f}")
    return {**eval_out, "fit_seconds": meta["fit_seconds"], "length": length}


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM (tests)")
    a = ap.parse_args(argv)
    full = bool(a.full)
    mode = "full" if full else "quick"
    t0 = time.time()
    df = load_merged()
    MODELS.mkdir(exist_ok=True)
    print(f"[{mode}] panel rows {len(df):,}, holdout from {HOLDOUT_START.date()}")
    tab, best, model = train_tabular(df, full)
    joblib.dump(model, MODELS / "best_model.joblib")
    rep = notebook_replica(df, full)
    lstm = None if a.no_lstm else train_lstm(df, full)

    recent = df[df.date > df.date.max() - pd.Timedelta(days=120)][["date", "store_id", "item_id", "item_count", "sales_amnt"]]
    recent.to_csv(MODELS / "recent.csv", index=False)
    items = items_table()
    cat = {"stores": {str(k): v for k, v in stores().items()},
           "items": items.rename(columns={"id": "item_id", "cost": "price"}).to_dict(orient="records")}
    (MODELS / "catalogue.json").write_text(json.dumps(cat))

    last28 = df[df.date > df.date.max() - pd.Timedelta(days=28)]
    metrics = {
        "mode": mode, "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout": {"start": str(HOLDOUT_START.date()), "end": str(df.date.max().date()),
                    "days": int(df[df.date >= HOLDOUT_START].date.nunique()), "rows": int((df.date >= HOLDOUT_START).sum())},
        "target": "item_count per (date, store, item); daily_total = predictions summed per day over all 100 items",
        "features": PANEL_FEATURES,
        "models": {**tab, **({"lstm": lstm} if lstm else {})},
        "best_tabular": best, "winner_metric": "daily_total.rmse on the 184-day holdout",
        "notebook_replica": {"note": "daily-total series, date features only, as in the notebook", "scores": rep,
                             "notebook_reported": NOTEBOOK_REPORTED},
        "baseline": {"last_date": str(df.date.max().date()),
                     "last_28_days_mean_units_per_day": {"all": round(float(last28.groupby("date").item_count.sum().mean()), 2),
                                                         **{str(s): round(float(g.groupby("date").item_count.sum().mean()), 2)
                                                            for s, g in last28.groupby("store_id")}},
                     "last_28_days_mean_revenue_per_day": round(float(last28.groupby("date").sales_amnt.sum().mean()), 2)},
        "train_seconds": round(time.time() - t0, 1),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"best tabular model: {best}; wrote models/metrics.json in {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
