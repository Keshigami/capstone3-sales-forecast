# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) (Python manager) and [just](https://github.com/casey/just) (command runner). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv; TensorFlow and XGBoost wheels exist for Apple Silicon and x86.
- No API key. Nothing in this project calls an external service.

## Three commands
```bash
git clone https://github.com/Keshigami/capstone3-sales-forecast.git && cd capstone3-sales-forecast
just setup        # uv sync, copies the shared stylesheet into static/
just up           # serves http://127.0.0.1:8104/ in the background; just down stops it
```
`just run` serves in the foreground instead. The dataset is in `data/Capstone 3/` and the trained models in `models/`, so the dashboard runs without training.

## What you see
1. The forecast panel is already filled: Bob's Diner, all items, 30 days, best tabular model (XGBoost). The chart shows the last 60 days of actuals in grey and the forecast in teal, with the weekly rhythm carried forward.
2. Change the store to Beachfront Bar or Fou Cher: the forecast drops to a unit or two a day, matching the EDA table at the bottom.
3. Set the horizon to 60 and press Forecast.
4. Switch the model to LSTM: the same panel forecasts total daily revenue over all stores.
5. Scroll: the four score tiles with the winner highlighted, the notebook comparison, then eight charts, each with its finding written in the caption.

## Retrain
```bash
just train        # EDA charts and findings, LR / RF / XGBoost / LSTM in quick mode, about a minute
just train-full   # the notebook's settings: RF 100 trees, XGBoost 1000 rounds with early stopping, LSTM batch 1; a few minutes
just test         # ruff and seven offline pytest tests
just demo         # health, metrics, a 7-day forecast for Bob's Diner, a 5-day LSTM revenue forecast
```

## The API
```bash
curl -s localhost:8104/health
curl -s localhost:8104/metrics
curl -s localhost:8104/forecast -H 'content-type: application/json' \
  -d '{"store":"Bob\u0027s Diner","item":"all","days":7}'
```
`POST /forecast` takes `store` (name, id or `"all"`), `item` (name, id or `"all"`), `days` (1 to 365) and `model` (`"best"` or `"lstm"`). It returns the dates, the values (units, or dollars for the LSTM), the total, the mean per day and the last 60 days of actuals for context. The horizon starts on 2022-01-01, the day after the data ends.

The 7-day response for Bob's Diner is the receipt in the film: 753.7, 292.3, 278.7, 320.1, 369.2, 548.6, 678.1 units, total 3,240.8.
