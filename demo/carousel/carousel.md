# Restaurant sales forecast · carousel

Spec: film/specs/capstone3-sales-forecast.json · 8 slides, 1080x1350.

## 1/8 Hook

- This receipt is from next week.

## 2/8 The question

- Bob's Diner is 99% of volume.
- Can one model fit all six?
- Bob's Diner: 687,527 units · 99% of volume

## 3/8 What it does

- Restaurant sales forecast
- Capstone 3 · Fresh Analytics
- Four models, one dashboard, under 10% error
- max@baluyot ~ $ curl -s localhost:8104/forecast -d '{"store":"Bob's Diner","days":7}'

## 4/8 How it is built

- 1. Sales, items, restaurants — pandas
- 2. Explore — matplotlib
- 3. Date features — pandas
- 4. LR · RF · XGBoost · LSTM — scikit-learn · xgboost · tensorflow
- 5. POST /forecast — FastAPI
- 6. Dashboard — HTML + ui.css
- Early stop on the last 90 training days

## 5/8 The result

- Six months held out. XGBoost wins.
- Linear regression 25.1%
- Random forest 15.2%
- LSTM 29.1%
- XGBoost 9.6% (winner)

## 6/8 The proof

- Beachfront Bar, 60 days, one click
- screen: capstone3-sales-forecast/demo/poster.png

## 7/8 The answer and the limit

- 9.6%
- MAPE · XGBoost · held-out six months
- Carries the 2021 level forward. No trend.
- Notebook 72.30 → 59.95 here. And it ships.

## 8/8 Try it

- Try it: Beachfront Bar, 60 days, Forecast.
- max@baluyot ~ $ git clone https://github.com/Keshigami/capstone3-sales-forecast.git
- max@baluyot ~ $ just setup
- max@baluyot ~ $ just up
- github.com/Keshigami/capstone3-sales-forecast
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
