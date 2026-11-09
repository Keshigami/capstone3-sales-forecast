# How it was built

The starting point was the capstone notebook: 164 cells that merge three CSVs, print the EDA tables and charts, and fit linear regression, random forest, XGBoost and an LSTM on one daily total. This repo turns that into a service with the same numbers checked along the way.

## Steps
1. **Read the notebook and the problem statement.** Listed every table and chart the notebook produces and the three scores it reports (LR 244.21, RF 59.91, XGB 72.30 RMSE; LSTM MAPE 1.34).
2. **Data** (`src/data.py`): merges sales, items and restaurants exactly as the notebook does and adds revenue. 109,600 rows: 1,096 days by 100 items. The source folder is never written to.
3. **EDA** (`src/eda.py`): recomputes every notebook table into `static/eda/findings.json` with sentences generated from the numbers, and draws eight charts.
4. **A better framing.** The notebook forecasts one daily total, which cannot answer "how many for store 4 next week". The models here train on the date by item panel with eleven features: weekday, quarter, year, month, day of year, day of month, ISO week, store, item, price, kcal. They are scored per row and on the daily total, so the notebook's numbers remain comparable. A replica of the notebook's own framing is trained too and matches it to the decimal for linear regression.
5. **Honest evaluation.** The notebook's holdout is kept: train to 2021-06-30, hold out 2021-07-01 to 2021-12-31 (184 days). Metrics: RMSE, MAE, R², plus MAPE. XGBoost early-stops on the last 90 days of the training window, never on the holdout.
6. **LSTM** as in the notebook (12-day window, 100 units, MinMax on revenue, recursive 184-day evaluation), with two changes: a tanh cell instead of relu, which diverged when fed its own output, and a fixed seed with deterministic ops.
7. **Service** (`app.py`): FastAPI with `POST /forecast`, `GET /metrics`, `GET /health`, and one Jinja2 page with an inline SVG chart; no build step. The page runs a forecast on load.
8. **Tests, justfile, README.** Seven offline tests (health, metrics, forecast shape and bounds, dashboard), one `justfile` with setup, train, train-full, run, up, down, test and demo.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.11 via uv | TensorFlow and XGBoost wheels, reproducible lockfile |
| Models | scikit-learn LR and RF, XGBoost, Keras LSTM; pandas features | the notebook's four models; XGBoost won on the holdout |
| Interface | FastAPI, one Jinja2 page, inline SVG | loads in a second, works offline |
| Data | `sales.csv` 109,600 rows, `items.csv` 100, `resturants.csv` 6 | the capstone dataset, read-only |
| Tests | pytest with TestClient, 7 tests | health, metrics, forecast shape and bounds, dashboard |

## Libraries
pandas and numpy (merge, features, aggregation), scikit-learn (LinearRegression, RandomForestRegressor, metrics, scaler, one-hot), xgboost (XGBRegressor with early stopping), tensorflow (Keras LSTM), matplotlib (eight EDA charts), joblib (model file), fastapi, uvicorn and jinja2 (API and page), pytest and httpx (tests), ruff (lint).

## Layout
```
app.py              the service and the page
src/data.py         merge and features
src/eda.py          findings.json and the eight charts
src/train.py        the four models, two framings, metrics.json
src/forecast.py     the forecast the API serves
models/             best_model.joblib, lstm.keras, metrics.json, catalogue.json
static/eda/         charts and findings
tests/              seven offline tests
data/Capstone 3/    the three CSVs
demo/               film.mp4, film-60s.mp4, demo.mp4, poster.png
```
