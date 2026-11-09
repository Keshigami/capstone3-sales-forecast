export DATA_ROOT := if path_exists("data") == "true" { "data" } else { env_var_or_default("DATA_ROOT", "") }
# capstone3-sales-forecast: EDA dashboard + per-store forecast on port 8104
set shell := ["bash", "-cu"]
port := "8104"

setup:
    uv sync
    mkdir -p static models
    cp common/ui.css static/ui.css

# EDA charts + LR/RF/XGB + LSTM, quick settings (about a minute)
train:
    uv run python -m src.eda
    uv run python -m src.train --quick

# the notebook's settings (RF 100x20, XGB 1000 trees, LSTM batch 1 with early stopping; a few minutes)
train-full:
    uv run python -m src.eda
    uv run python -m src.train --full

run:
    uv run uvicorn app:app --host 127.0.0.1 --port {{port}}

up:
    @if [ -f .pid ] && kill -0 $(cat .pid) 2>/dev/null; then echo "already up (pid $(cat .pid)) on {{port}}"; exit 0; fi
    nohup uv run uvicorn app:app --host 127.0.0.1 --port {{port}} > .server.log 2>&1 & echo $! > .pid
    for i in $(seq 1 40); do curl -sf http://127.0.0.1:{{port}}/health >/dev/null && break; sleep 0.5; done
    echo "up on http://127.0.0.1:{{port}} (pid $(cat .pid))"

down:
    @if [ -f .pid ]; then kill $(cat .pid) 2>/dev/null || true; rm -f .pid; echo "down"; else echo "not running"; fi

test:
    uv run ruff check .
    uv run pytest -q

# scripted end-to-end call against the running server (just up first)
demo:
    curl -s http://127.0.0.1:{{port}}/health; echo
    curl -s http://127.0.0.1:{{port}}/metrics | uv run python -c "import json,sys; m=json.load(sys.stdin); print('best:', m['best_tabular']); [print(f\"  {k:18s} holdout RMSE {v.get('daily_total', v.get('recursive_184_days'))['rmse']:9.1f}  {'$ revenue/day, recursive' if k=='lstm' else 'units/day'}\") for k,v in m['models'].items()]"
    curl -s -X POST http://127.0.0.1:{{port}}/forecast -H 'content-type: application/json' -d '{"store":"Bob'"'"'s Diner","days":7}' | uv run python -c "import json,sys; r=json.load(sys.stdin); print(r['store'], r['model'], 'next 7 days:', r['values'], 'total', r['total'])"
    curl -s -X POST http://127.0.0.1:{{port}}/forecast -H 'content-type: application/json' -d '{"store":"all","days":5,"model":"lstm"}' | uv run python -c "import json,sys; r=json.load(sys.stdin); print('LSTM revenue next 5 days:', r['values'])"
