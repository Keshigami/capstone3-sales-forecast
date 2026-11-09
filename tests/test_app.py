"""Offline smoke tests against the FastAPI app with TestClient (needs models/ from `just train`)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app

client = TestClient(app, raise_server_exceptions=False)
TRAINED = (ROOT / "models" / "metrics.json").exists()
needs_model = pytest.mark.skipif(not TRAINED, reason="run `just train` first")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["service"] == "capstone3-sales-forecast" and isinstance(body["model_loaded"], bool)


@needs_model
def test_metrics_lists_all_models_with_scores():
    m = client.get("/metrics").json()
    for name in ("linear_regression", "random_forest", "xgboost", "lstm"):
        assert name in m["models"], name
    for name in ("linear_regression", "random_forest", "xgboost"):
        s = m["models"][name]["daily_total"]
        assert s["rmse"] > 0 and s["mae"] > 0 and 0 < s["mape"] < 5
    assert m["models"]["lstm"]["recursive_184_days"]["mape"] > 0
    assert m["best_tabular"] in ("linear_regression", "random_forest", "xgboost")


@needs_model
def test_forecast_for_a_store_returns_requested_days_non_negative():
    r = client.post("/forecast", json={"store": "Bob's Diner", "days": 14})
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["dates"]) == 14 and len(b["values"]) == 14
    assert all(v >= 0 for v in b["values"]) and b["total"] >= 0
    assert b["store"] == "Bob's Diner" and b["store_id"] == 1
    assert b["dates"][0] > "2021-12-31"


@needs_model
def test_forecast_by_id_item_and_all():
    r = client.post("/forecast", json={"store": 4, "item": "all", "days": 3}).json()
    assert r["store"] == "Fou Cher" and len(r["values"]) == 3
    r = client.post("/forecast", json={"store": "all", "item": "all", "days": 2}).json()
    assert r["store"] == "all stores" and len(r["values"]) == 2
    r = client.post("/forecast", json={"store": 1, "item": "Strawberry", "days": 2}).json()
    assert "Strawberry" in r["item"] and len(r["values"]) == 2


@needs_model
def test_forecast_rejects_unknown_store_and_bad_days():
    assert client.post("/forecast", json={"store": "Nowhere", "days": 3}).status_code == 422
    assert client.post("/forecast", json={"store": 1, "days": 0}).status_code == 422


@needs_model
def test_lstm_forecast_revenue():
    r = client.post("/forecast", json={"days": 4, "model": "lstm"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model"] == "lstm" and len(b["values"]) == 4 and all(v >= 0 for v in b["values"])


def test_dashboard_renders():
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert 'id="result"' in html and "Restaurant sales forecast" in html
    if TRAINED:
        assert 'id="input"' in html and 'id="run"' in html
        f = json.loads((ROOT / "static" / "eda" / "findings.json").read_text())
        assert all(f"/static/{p}" in html for p in f["charts"].values())
