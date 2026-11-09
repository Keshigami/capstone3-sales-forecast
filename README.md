# capstone3-sales-forecast (port 8104)

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

Restaurant sales forecasting for "Fresh Analytics": three years of daily item sales across six restaurants, the
notebook's EDA as charts with the findings written out, then Linear Regression, Random Forest, XGBoost and an LSTM
compared on the last six months, with the best model serving a per-store, per-item forecast on a dashboard.

## Data

`DATA_ROOT` (default `~/Caltech CTME/PGC AIML - Capstone/Datasets`) `/Capstone 3/`:

| File | Rows | Columns |
|---|---|---|
| `sales.csv` | 109,600 | date, item_id, price, item_count (2019-01-01 to 2021-12-31, 1,096 days x 100 items) |
| `items.csv` | 100 | id, store_id, name, kcal, cost |
| `resturants.csv` | 6 | id, name |

Merged in `src/data.py` into one frame (date, item, price, count, store, item name, kcal, store name, revenue). No
missing values; the source folder is never written to.

## Findings (`static/eda/findings.json`, charts in `static/eda/`)

- 694,787 units and $6.40M revenue over 1,096 days; 634 units a day on average (min 175, max 1,201 on 2021-07-09). The series climbs year on year with a strong weekly rhythm and a dip at each year end.
- Friday and Saturday sell most (136,300 and 135,644 units); Sunday is the quietest (68,032), about half of the peak.
- July is the busiest month (77,125 units), December the slowest (41,224). Q2 sells most (211,200), Q4 least (138,297); every quarter of 2021 beats the same quarter of 2019.
- Bob's Diner is 99.0 percent of all volume (687,527 units, 627 a day). Among the other five, Surfs Up sells the most units (1,803) and Fou Cher the fewest (1,106).
- Volume and money disagree once Bob's Diner is set aside: Fou Cher earns the most per day ($25) on one unit a day, Sweet Shack the least ($2). Unit ranking: Bob's > Surfs Up > Sweet Shack > Corner Cafe > Beachfront Bar > Fou Cher. Revenue ranking: Bob's > Fou Cher > Corner Cafe > Surfs Up > Beachfront Bar > Sweet Shack.
- Best seller: Strawberry Smoothy at Bob's Diner, 236,337 units (145 kcal, $2.89); all five top items are Bob's Diner items, three cost under $5.
- Most expensive item anywhere: Blue Ribbon Fruity Vegi Lunch at Fou Cher, $53.98 and 881 kcal; every store's priciest item is at least 538 kcal.

## Models (`src/train.py`)

Two framings, both with the notebook's split (train to 2021-06-30, hold out 2021-07-01 to 2021-12-31, 184 days):

1. **Panel models** (what the dashboard serves): one row per date x item (109,600 rows), target `item_count`, features
   weekday, quarter, year, month, day of year, day of month, ISO week, store_id, item_id, price, kcal. LR gets
   one-hot store/item/weekday/month; RF and XGB take the integers. Scores are reported per row and on the daily total
   (predictions summed over all 100 items per day), which is the number the notebook reports. The winner is the lowest
   daily-total RMSE and is saved to `models/best_model.joblib`.
2. **Notebook replica**: one daily-total series with date features only, same three models, to check the numbers.
3. **LSTM** (`models/lstm.keras`): daily revenue for all stores, MinMax-scaled, 12-day window, LSTM(100) + Dense(1),
   early stopping on the holdout, then evaluated the notebook's way (fed its own output for 184 days, MAPE) and one
   step ahead. A second LSTM is refit on the full series for serving.

### Scores, quick mode (`just train`, 49 s on an M4 Max)

| Model | Daily-total RMSE | MAE | MAPE | R² | Row RMSE |
|---|---|---|---|---|---|
| Linear regression | 136.4 | 98.3 | 25.1% | 0.71 | 18.4 |
| Random forest (40 trees, depth 14) | 86.1 | 69.5 | 15.4% | 0.89 | 6.9 |
| **XGBoost (300 trees, lr 0.08, depth 6)** | **59.9** | **47.5** | **9.9%** | **0.94** | **5.9** |
| LSTM, recursive 184 days ($ revenue) | 6,088 | 5,086 | 116.8% | -5.3 | |
| LSTM, one step ahead ($ revenue) | 979 | 779 | 15.6% | 0.83 | |

Notebook replica on the daily-total series, here versus the notebook: LR 244.21 vs 244.21, RF 60.34 vs 59.91,
XGB 72.21 vs 72.30 RMSE; LSTM recursive MAPE 1.17 here vs 1.34 in the notebook. The panel XGBoost (59.9) matches the
notebook's best aggregate model (RF, 59.9) while also forecasting each store and item.

### Scores, full mode (`just train-full`, 3 min 40 s on an M4 Max; `models/` currently holds these)

| Model | Daily-total RMSE | MAE | MAPE | R² | Row RMSE |
|---|---|---|---|---|---|
| Linear regression | 136.4 | 98.3 | 25.1% | 0.71 | 18.4 |
| Random forest (100 trees, depth 20) | 84.9 | 68.4 | 15.2% | 0.89 | 6.9 |
| **XGBoost (1000 rounds, lr 0.02, early stop at 90-day validation)** | **59.9** | **47.6** | **9.6%** | **0.94** | **6.1** |
| LSTM, recursive 184 days ($ revenue; batch 1, 20 epochs, early stop at 8) | 2,166 | 1,825 | 29.1% | 0.21 | |
| LSTM, one step ahead ($ revenue) | 825 | 646 | 12.7% | 0.88 | |

Full mode trains the LSTM the notebook's way (batch size 1) and refits the serving LSTM with the notebook's 90-day
window; the recursive MAPE drops from 1.17 (quick) to 0.29, against 1.34 in the notebook.

What the numbers mean: the daily total runs 175 to 1,201 units, so an RMSE of 60 is about 10 percent of a typical day.
The recursive LSTM is poor in both the notebook and here: feeding a one-day model its own output for 184 days drifts;
the one-step score (15.6%) is the fairer view of what it learned.

## Run

```bash
just setup        # uv sync (Python 3.11), copy common/ui.css to static/
just train        # EDA charts + findings, LR/RF/XGB/LSTM quick, models/ + metrics.json (about a minute)
just train-full   # notebook settings: RF 100x20, XGB 1000 trees lr 0.02, LSTM batch 1 with early stopping, 90-day final window
just run          # http://127.0.0.1:8104  (just up / just down for background with a pid file)
just test         # ruff + 7 pytest tests, offline (they skip the model tests until just train has run)
just demo         # health, metrics, a 7-day forecast for Bob's Diner, a 5-day LSTM revenue forecast
```

API: `GET /health` → `{"ok": true, "service": "capstone3-sales-forecast", "model_loaded": true}`; `GET /metrics` → `models/metrics.json`;
`POST /forecast {"store": "Bob's Diner" | 1 | "all", "item": "Strawberry" | 19 | "all", "days": 1..365, "model": "best" | "lstm"}` →
dates, values (units, or $ for lstm), total, mean per day, and the last 60 days of actuals for context. The horizon starts 2022-01-01, the day after the data ends.

## Two-minute walkthrough

1. Open http://127.0.0.1:8104/. The forecast panel is already filled (Bob's Diner, 30 days) and the chart shows the last 60 days of actuals in grey and the XGBoost forecast in teal, with the weekly rhythm carried forward.
2. Change the store to Fou Cher: the forecast drops to about one unit a day, matching the EDA table at the bottom. Pick one item (the list filters by store) and a 90-day horizon.
3. Switch the model to LSTM: the same panel forecasts total daily revenue.
4. Scroll: the four score tiles (the winner is highlighted) and the notebook comparison under the details line, then the eight charts, each with its finding in the caption.
5. `curl -X POST :8104/forecast -d '{"store":"Surfs Up","days":14}'` for the JSON the agent on :8105 consumes.

## Limits

- The tree models cannot extrapolate the yearly growth: `year=2022` is treated like 2021, so the forecast carries the 2021 level forward with the right weekly and seasonal shape but no trend.
- Five of the six stores sell one or two units a day; their forecasts are small numbers with large relative error. The scores above are dominated by Bob's Diner.
- Quick mode uses fewer trees and epochs; `just train-full` reproduces the notebook's settings.
- The recursive LSTM is fragile across seeds (a seed is fixed, and `enable_op_determinism` is on, so a run repeats on the same machine). Revenue, not units, is what it forecasts, as in the notebook.
- No exogenous inputs (holidays, weather, promotions); price is the catalogue cost, which never changes in this data.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Dataset: included under `data/`; run with `DATA_ROOT=data` (the justfile sets it when the folder exists).
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
