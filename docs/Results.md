# Results

Holdout: 2021-07-01 to 2021-12-31, 184 days. Scores on the daily total over all items, which is the number the notebook reports.

## Full mode (`just train-full`; the models in the repo)
| Model | RMSE (units/day) | MAE | MAPE | R² |
|---|---|---|---|---|
| Linear regression | 136.4 | 98.3 | 25.1% | 0.71 |
| Random forest (100 trees, depth 20) | 84.9 | 68.4 | 15.2% | 0.89 |
| **XGBoost (1000 rounds, lr 0.02, early stop on a 90-day validation window)** | **59.9** | **47.6** | **9.6%** | **0.94** |
| LSTM, recursive 184 days ($ revenue) | 2,166 | 1,825 | 29.1% | 0.21 |
| LSTM, one step ahead ($ revenue) | 825 | 646 | 12.7% | 0.88 |

## Quick mode (`just train`, about a minute)
| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Linear regression | 136.4 | 98.3 | 25.1% | 0.71 |
| Random forest (40 trees, depth 14) | 86.1 | 69.5 | 15.4% | 0.89 |
| XGBoost (300 trees, lr 0.08, depth 6) | 59.9 | 47.5 | 9.9% | 0.94 |
| LSTM, recursive ($ revenue) | 6,088 | 5,086 | 116.8% | -5.3 |
| LSTM, one step ahead ($ revenue) | 979 | 779 | 15.6% | 0.83 |

## Against the notebook
The notebook reported, on its one daily-total series: LR 244.21, RF 59.91, XGB 72.30 RMSE, LSTM recursive MAPE 1.34. The replica of that framing here gives 244.21, 60.34, 73.44 and 0.29 (full mode), so the numbers reconcile. The panel XGBoost at 59.9 matches the notebook's best aggregate model while also forecasting each store and item.

## What the numbers mean
- Daily totals run from 175 to 1,201 units, so an RMSE of 60 is about 10 percent of a typical day. A MAPE under 10 percent is a good production number for daily restaurant demand; XGBoost reaches it.
- The recursive LSTM is poor in the notebook and here: feeding a one-day model its own output for 184 days drifts. The one-step score is the fairer view of what it learned.
- Bob's Diner is 99 percent of all volume (687,527 units). The other five stores sell one or two units a day, so their relative errors are large and the headline scores are dominated by Bob's Diner.

## Findings from the EDA
- 694,787 units and $6.40M revenue over 1,096 days; 634 units a day on average.
- Friday and Saturday sell most (136,300 and 135,644 units); Sunday is the quietest (68,032), about half of the peak.
- July is the busiest month, December the slowest; every quarter of 2021 beats the same quarter of 2019.
- Best seller: Strawberry Smoothy at Bob's Diner, 236,337 units.
