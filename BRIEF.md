---
title: Restaurant sales forecast
group: capstone
role: Capstone
port: 8104
status: working
one_liner: Three years of restaurant item sales explored and forecast per store and item with XGBoost, random forest, linear regression and an LSTM, served on a dashboard.
---
# Restaurant sales forecast

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (79 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8104/.

Two-minute walkthrough:
1. Capstone 3: a 30-day forecast for Bob's Diner runs on load
2. Change the store: Beachfront Bar
3. Horizon 60 days, Forecast
4. Switch to the LSTM: daily revenue over all stores
5. Model scores on the held-out six months: linear regression, random forest, XGBoost, LSTM
6. EDA: the overall sales pattern
7. Friday and Saturday sell most; Sunday is the quietest day
8. Each restaurant over time: Bob's Diner is 99 percent of all volume
9. Most popular items, and the most expensive item per store
10. Most popular item at each store
<!-- /demo:auto -->

## Brief
Fresh Analytics wants to understand and predict demand for items across six restaurants so that purchasing, staffing and
planning can run on numbers instead of guesses. The data is 1,096 days of sales for 100 items. Solved means: the
patterns (day of week, month, quarter, store, item, price versus volume) are stated with numbers, three regression models
plus a deep model are compared on the last six months, and the best one gives a forecast for any store and item over any
horizon up to a year.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | TensorFlow and XGBoost wheels, same as the rest of the repo |
| Model or core logic | scikit-learn LR and RF, XGBoost, Keras LSTM; pandas feature engineering | the notebook's four models; XGBoost won on the holdout |
| Interface | FastAPI + one Jinja2 HTML page, inline SVG chart, no build step | loads in a second, scriptable ids, works offline |
| Data | `sales.csv` 109,600 rows, `items.csv` 100, `resturants.csv` 6, read from `DATA_ROOT` | the capstone dataset, read-only |
| Tests | pytest + TestClient, 7 tests, offline | health, metrics, forecast shape and bounds, dashboard |

## Libraries used
`pandas` / `numpy` (merge, features, aggregation), `scikit-learn` (LinearRegression, RandomForestRegressor, metrics, scaler, one-hot), `xgboost` (XGBRegressor with early stopping), `tensorflow` (Keras LSTM), `matplotlib` (eight EDA charts on the dark surface), `joblib` (model file), `fastapi` + `uvicorn` + `jinja2` (API and page), `pytest` + `httpx` (tests), `ruff` (lint).

## How it was built
1. Extracted the 164 notebook cells and the problem statement PDF; listed every table and chart the notebook produces and the three model scores it reports (LR 244.21, RF 59.91, XGB 72.30 RMSE; LSTM MAPE 1.34).
2. `src/data.py` merges the three CSVs exactly as the notebook does (sales → items → restaurants) and adds revenue.
3. `src/eda.py` recomputes every notebook table into `findings.json` with sentences generated from the numbers, and draws eight charts with the shared dark palette.
4. Changed the model framing: the notebook forecasts one daily total, which cannot answer "how many for store 4 next week". The models here train on the date x item panel with store, item, price and kcal as features, and are scored both per row and on the daily total so the notebook's numbers remain comparable. A replica of the notebook's own framing is trained too and matches it to the decimal for LR.
5. Kept the notebook's holdout (last six months) and metrics (RMSE, MAE, R², plus MAPE). XGBoost early-stops on the last 90 days of the training window, not on the holdout.
6. LSTM as in the notebook (12-day window, 100 units, MinMax on revenue, recursive 184-day evaluation), with two changes: a tanh cell instead of relu, which diverged when fed its own output, and a fixed seed with deterministic ops.
7. `app.py` serves the forecast (`POST /forecast`), metrics, health and the dashboard; the page runs a forecast on load.
8. Tests, justfile (`setup`, `train`, `train-full`, `run`, `up`, `down`, `test`, `demo`), README.

## Results
Quick mode (`just train`, 49 s), holdout 2021-07-01 to 2021-12-31, scores on the daily total over all items:

| Model | RMSE (units/day) | MAE | MAPE | R² |
|---|---|---|---|---|
| Linear regression | 136.4 | 98.3 | 25.1% | 0.71 |
| Random forest | 86.1 | 69.5 | 15.4% | 0.89 |
| XGBoost (winner) | 59.9 | 47.5 | 9.9% | 0.94 |
| LSTM recursive, $ revenue | 6,088 | 5,086 | 116.8% | -5.3 |
| LSTM one step ahead, $ revenue | 979 | 779 | 15.6% | 0.83 |

Full mode (`just train-full`, 3 min 40 s): random forest 84.9 / 68.4 / 15.2%, XGBoost 59.9 / 47.6 / 9.6% (still the winner), LSTM recursive 2,166 / 1,825 / 29.1% and one step ahead 825 / 646 / 12.7%. The notebook reported LR 244.21, RF 59.91, XGB 72.30 on its aggregate series and LSTM MAPE 1.34; the replica here gives 244.21, 60.34, 73.44 and 0.29 (full) / 1.17 (quick).

Daily totals run 175 to 1,201 units, so RMSE 60 is roughly a 10 percent error on a typical day; the notebook's best was 59.91 on the aggregate series. A good production number for daily restaurant demand is a MAPE under 10 percent, which XGBoost reaches here.

## Demo details
- Run: `just setup && just train && just up`, open `http://localhost:8104/`.
- Two-minute walkthrough: the forecast for Bob's Diner over 30 days is already on screen; change the store to Fou Cher and the horizon to 90 (small numbers, as the EDA says); switch the model to LSTM for revenue; scroll to the score tiles and the eight captioned charts.
- Video clip: `demo/clips/capstone3-sales-forecast.webm` (recorded by `just demo-video`).
- Sample inputs: the store and item dropdowns; `just demo` posts a 7-day forecast for Bob's Diner and a 5-day LSTM revenue forecast.

## Limits and next steps
- Tree models carry the 2021 level forward with the weekly and seasonal shape but no growth trend; a trend feature or a hybrid (trend model plus tree residuals) is the next step.
- Bob's Diner is 99 percent of the volume; the other five stores are one or two units a day and their relative errors are large.
- The recursive LSTM is unstable over 184 steps in the notebook and here; a direct multi-horizon model or a seq2seq would fix that.
- Quick mode uses 40 trees / 300 rounds / 15 epochs; `just train-full` uses the notebook's settings and takes a few minutes.
- No holidays, weather or promotions in the data; price never changes, so elasticity cannot be learned.
