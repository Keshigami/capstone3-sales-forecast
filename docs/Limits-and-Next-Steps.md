# Limits and next steps

- **No trend.** The tree models treat 2022 like 2021: the forecast carries the 2021 level forward with the right weekly and seasonal shape but no growth. Next: a trend feature, or a hybrid (trend model plus tree residuals).
- **One store dominates.** Bob's Diner is 99 percent of the volume; the other five stores are one or two units a day and their relative errors are large.
- **Recursive LSTM is unstable** over 184 steps, in the notebook and here. A direct multi-horizon model or a seq2seq would fix that.
- **Quick mode is a sketch.** 40 trees, 300 rounds, 15 epochs; `just train-full` uses the notebook's settings and takes a few minutes.
- **No external signals.** No holidays, weather or promotions in the data; price never changes, so elasticity cannot be learned.
