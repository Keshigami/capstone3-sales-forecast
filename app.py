"""Capstone 3 sales forecast: EDA dashboard + forecast API on port 8104."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src import eda
from src import forecast as fc

ROOT = Path(__file__).resolve().parent
PORT = 8104
SERVICE = "capstone3-sales-forecast"
app = FastAPI(title="Restaurant sales forecast", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

CHART_ORDER = ["daily_total", "weekday", "month", "quarter", "stores", "revenue_vs_volume", "popular_items", "expensive_items"]
CHART_TITLES = {"daily_total": "Overall sales pattern", "weekday": "Day of the week", "month": "Month of the year",
                "quarter": "Quarters, by year", "stores": "Each restaurant over time", "revenue_vs_volume": "Volume versus money",
                "popular_items": "Most popular items", "expensive_items": "Most expensive item per store"}
MODEL_LABELS = {"linear_regression": "Linear regression", "random_forest": "Random forest", "xgboost": "XGBoost", "lstm": "LSTM"}


class ForecastIn(BaseModel):
    store: str | int = "all"
    item: str | int = "all"
    days: int = Field(30, ge=1, le=365)
    model: str = Field("best", pattern="^(best|lstm)$")


def metrics() -> dict | None:
    p = ROOT / "models" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE, "model_loaded": fc.loaded(), "port": PORT}


@app.get("/metrics")
def get_metrics():
    m = metrics()
    if m is None:
        raise HTTPException(503, "no models/metrics.json yet: run `just train`")
    return m


@app.post("/forecast")
def post_forecast(q: ForecastIn):
    try:
        return fc.forecast(q.store, q.item, q.days, q.model)
    except fc.NotTrained as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def index(request: Request):
    f = eda.load_findings() or {}
    m = metrics()
    charts = [{"key": k, "title": CHART_TITLES[k], "src": "/static/" + f["charts"][k], "caption": f["findings"][k]}
              for k in CHART_ORDER if f and k in f.get("charts", {})]
    stores, items = {}, []
    if fc.loaded():
        a = fc.artifacts()
        stores = a["stores"]
        items = a["items"].sort_values(["store_id", "name"]).to_dict(orient="records")
    tiles = []
    if m:
        for k, v in m["models"].items():
            s = v.get("daily_total") or v.get("recursive_184_days") or {}
            tiles.append({"key": k, "label": MODEL_LABELS.get(k, k), "rmse": s.get("rmse"), "mae": s.get("mae"),
                          "mape": s.get("mape"), "best": k == m["best_tabular"]})
    return templates.TemplateResponse(request, "index.html", {
        "port": PORT, "findings": f, "metrics": m, "charts": charts, "stores": stores, "items": items, "tiles": tiles,
        "model_loaded": fc.loaded(), "best_label": MODEL_LABELS.get(m["best_tabular"], "") if m else "",
    })


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)
