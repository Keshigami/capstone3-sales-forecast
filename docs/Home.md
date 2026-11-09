# Restaurant sales forecast

Three years of daily item sales across six restaurants, explored and forecast per store and item. Four models compared on the same held-out six months; the winner, XGBoost, serves a 1 to 365 day forecast for any store and item on a dashboard. Caltech CTME PGC AIML, Capstone 3, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/capstone3-sales-forecast/blob/main/demo/film.mp4). It opens on a receipt for next week, printed by the API.

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the notebook to a service, step by step.
- [Results](Results.md): every model, every score, and what the numbers mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show and how the film was made.

## The one-line answer
XGBoost on a date by item panel: 9.6% MAPE on the daily total over the last six months (RMSE 59.9 units a day, R² 0.94). The notebook's best XGBoost RMSE was 72.30 on its aggregate series; this one lands at 59.95 and also forecasts each store and item.

## Honest limits
Bob's Diner is 99% of the volume. The tree models carry the 2021 level forward with the right weekly and seasonal shape but no growth trend. Details on the [limits page](Limits-and-Next-Steps.md).
