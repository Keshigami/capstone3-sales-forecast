# The film

`demo/film.mp4` is 90 seconds in four chapters, made for engineers and project managers who have ninety seconds.

1. **What it does** (0 to 17 s): a terminal prompt runs the real `curl` to `/forecast`, and the response prints as a receipt for next week, line by line. Then the six restaurants as dots, Bob's Diner at 99 percent of volume, and the question: can one model fit all six?
2. **How it is built** (17 to 40 s): a paper model of the pipeline, data to dashboard, with the four-model race on the held-out six months (XGBoost lands last and lowest), the early-stopping line in `src/train.py`, the API call, and the tests.
3. **The proof** (40 to 68 s): the real dashboard, recorded: Beachfront Bar, 60 days; the LSTM revenue view; the scores page.
4. **The answer** (68 to 90 s): 9.6 percent. The notebook's 72.30 counting down to 59.95. One limit, in red. Then the prompt types `just setup && just up`.

`demo/film-60s.mp4` is the same without the proof chapter. `demo/demo.mp4` is the plain captioned walkthrough of the app.

Every number in the film comes from `models/metrics.json`, `static/eda/findings.json` or a live API response; nothing is drawn by hand. The film pipeline (Pillow and Three.js scenes rendered frame by frame with Playwright, composed with ffmpeg, narrated with an offline Kokoro voice) lives in the author's workspace, not in this repo.
