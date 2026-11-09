"""EDA from the capstone notebook: findings as JSON tables plus PNG charts under static/eda/.

Run:  uv run python -m src.eda      (writes static/eda/findings.json and 8 charts)
"""
from __future__ import annotations

import calendar
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .data import PROJECT, items_table, load_merged, stores

OUT = PROJECT / "static" / "eda"
# Dark surface from common/ui.css and the validated dark categorical palette (dataviz skill).
BG, PANEL, INK, INK2, GRID, ACCENT = "#0c1018", "#182033", "#e8ecf6", "#aab3c8", "#2c2c2a", "#4fd1c5"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
DAYS = list(calendar.day_name)
MONTHS = list(calendar.month_name)[1:]


def _style():
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "grid.color": GRID, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    return f"eda/{name}"


def _bar(ax, labels, values, color=ACCENT, fmt="{:,.0f}"):
    bars = ax.bar(labels, values, color=color, width=0.72)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    fontsize=9, color=INK2, xytext=(0, 2), textcoords="offset points")
    ax.margins(y=0.15)


def compute(df: pd.DataFrame) -> dict:
    """All notebook tables as plain dicts (numbers only, no charts)."""
    names = stores()
    items = items_table()
    df = df.assign(weekday_name=df.date.dt.day_name(), month_name=df.date.dt.month_name(),
                   restaurant_name=df.store_id.map(names))
    daily = df.groupby("date").item_count.sum()
    weekday = df.groupby("weekday_name").item_count.sum().reindex(DAYS)
    month = df.groupby("month_name").item_count.sum().reindex(MONTHS)
    quarter = df.groupby("quarter").item_count.sum()
    qy = df.groupby(["year", "quarter"]).item_count.sum()
    store_units = df.groupby("restaurant_name").item_count.sum().sort_values(ascending=False)
    store_rev = df.groupby("restaurant_name").sales_amnt.sum().sort_values(ascending=False)
    store_year_units = df.pivot_table(index="restaurant_name", columns="year", values="item_count", aggfunc="sum")
    store_year_rev = df.pivot_table(index="restaurant_name", columns="year", values="sales_amnt", aggfunc="sum")
    n_days = df.date.nunique()
    per_day = pd.DataFrame({"units_per_day": store_units / n_days, "revenue_per_day": store_rev / n_days})
    per_day = per_day.sort_values("revenue_per_day", ascending=False)

    top_items = (df.groupby(["item_id", "item_name", "restaurant_name", "kcal", "price"]).item_count.sum()
                 .sort_values(ascending=False).head(5).reset_index())
    pop_store = (df.groupby(["restaurant_name", "item_name", "kcal", "price"]).item_count.sum().reset_index()
                 .sort_values("item_count", ascending=False).groupby("restaurant_name").head(1)
                 .sort_values("item_count", ascending=False))
    expensive = (items.assign(restaurant_name=items.store_id.map(names)).sort_values("cost", ascending=False)
                 .groupby("restaurant_name").head(1)[["restaurant_name", "name", "cost", "kcal"]]
                 .rename(columns={"name": "item_name"}).sort_values("cost", ascending=False))
    cost_stats = items.assign(restaurant_name=items.store_id.map(names)).groupby("restaurant_name").cost.agg(
        ["max", "min", "mean", "count"]).round(2)

    out = {
        "rows": len(df), "n_items": int(df.item_id.nunique()), "n_stores": int(df.store_id.nunique()),
        "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()), "n_days": int(n_days),
        "total_units": float(df.item_count.sum()), "total_revenue": round(float(df.sales_amnt.sum()), 2),
        "daily_units": {"mean": round(float(daily.mean()), 1), "min": float(daily.min()), "max": float(daily.max()),
                        "peak_date": str(daily.idxmax().date())},
        "weekday_units": {k: float(v) for k, v in weekday.items()},
        "month_units": {k: float(v) for k, v in month.items()},
        "quarter_units": {f"Q{k}": float(v) for k, v in quarter.items()},
        "quarter_year_units": {f"Q{q}-{y}": float(v) for (y, q), v in qy.items()},
        "store_units": {k: float(v) for k, v in store_units.items()},
        "store_revenue": {k: round(float(v), 2) for k, v in store_rev.items()},
        "store_year_units": {k: {str(c): float(v) for c, v in r.items()} for k, r in store_year_units.iterrows()},
        "store_year_revenue": {k: {str(c): round(float(v), 2) for c, v in r.items()} for k, r in store_year_rev.iterrows()},
        "store_per_day": {k: {c: round(float(v), 2) for c, v in r.items()} for k, r in per_day.iterrows()},
        "top_items": top_items.to_dict(orient="records"),
        "popular_item_per_store": pop_store.to_dict(orient="records"),
        "most_expensive_per_store": expensive.to_dict(orient="records"),
        "item_cost_per_store": {k: {c: float(v) for c, v in r.items()} for k, r in cost_stats.iterrows()},
    }
    out["findings"] = sentences(out)
    return out


def sentences(f: dict) -> dict[str, str]:
    """The findings in plain words, each keyed by the chart it belongs to."""
    wd, mo, q = f["weekday_units"], f["month_units"], f["quarter_units"]
    su, sr = f["store_units"], f["store_revenue"]
    top_store = next(iter(su))
    share = su[top_store] / f["total_units"] * 100
    rank_units = list(su)
    rank_rev = list(sr)
    others_units = {k: v for k, v in su.items() if k != top_store}
    others_rev = {k: v for k, v in sr.items() if k != top_store}
    mx_u, mn_u = max(others_units, key=others_units.get), min(others_units, key=others_units.get)
    mx_r, mn_r = max(others_rev, key=others_rev.get), min(others_rev, key=others_rev.get)
    top = f["top_items"][0]
    exp = f["most_expensive_per_store"][0]
    pd_ = f["store_per_day"]
    top2 = sorted(wd, key=wd.get, reverse=True)[:2]
    cheap = sum(1 for t in f["top_items"] if t["price"] < 5)
    min_kcal = min(e["kcal"] for e in f["most_expensive_per_store"])
    return {
        "daily_total": (f"{f['n_days']} days from {f['date_min']} to {f['date_max']}: {f['total_units']:,.0f} units sold, "
                        f"{f['daily_units']['mean']:,.0f} a day on average (min {f['daily_units']['min']:,.0f}, "
                        f"max {f['daily_units']['max']:,.0f} on {f['daily_units']['peak_date']}). The series climbs year on year "
                        "with a strong weekly rhythm and a dip around the end of each year."),
        "weekday": (f"{top2[0]} and {top2[1]} sell most ({wd[top2[0]]:,.0f} and {wd[top2[1]]:,.0f} units); "
                    f"{min(wd, key=wd.get)} is the quietest day ({wd[min(wd, key=wd.get)]:,.0f}), about half of the peak. "
                    "Volume builds through the week and collapses on Sunday."),
        "month": (f"{max(mo, key=mo.get)} is the busiest month ({mo[max(mo, key=mo.get)]:,.0f} units) and "
                  f"{min(mo, key=mo.get)} the slowest ({mo[min(mo, key=mo.get)]:,.0f}); summer months lead."),
        "quarter": (f"Summed over the three years, {max(q, key=q.get)} sells most ({q[max(q, key=q.get)]:,.0f}) and "
                    f"{min(q, key=q.get)} least ({q[min(q, key=q.get)]:,.0f}). Every quarter of 2021 beats the same quarter of 2019."),
        "stores": (f"{top_store} is in another league: {su[top_store]:,.0f} units, {share:.1f} percent of all volume. "
                   f"Among the other five, {mx_u} sells the most units and {mn_u} the fewest."),
        "revenue_vs_volume": (f"Volume and money disagree once {top_store} is set aside: {mx_r} earns the most per day "
                              f"(${pd_[mx_r]['revenue_per_day']:,.0f}) on only {pd_[mx_r]['units_per_day']:.1f} units, while "
                              f"{mn_r} earns the least (${pd_[mn_r]['revenue_per_day']:,.0f} a day). Unit ranking: "
                              f"{' > '.join(rank_units)}. Revenue ranking: {' > '.join(rank_rev)}."),
        "popular_items": (f"The best seller is {top['item_name']} at {top['restaurant_name']} with {top['item_count']:,.0f} units "
                          f"({top['kcal']} kcal, ${top['price']:.2f}); {cheap} of the top five cost under $5 and all five are sold at {top['restaurant_name']}."),
        "expensive_items": (f"The priciest item on any menu is {exp['item_name']} at {exp['restaurant_name']} "
                            f"(${exp['cost']:.2f}, {exp['kcal']} kcal). Price and calories go together: every store's priciest item is "
                            f"at least {min_kcal} kcal, and the two costliest are also the two most caloric."),
    }


def charts(df: pd.DataFrame, f: dict) -> dict[str, str]:
    _style()
    names = stores()
    out = {}
    daily = df.groupby("date").item_count.sum()

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(daily.index, daily.values, color=ACCENT, lw=1.2)
    ax.set_title("Units sold per day, all stores")
    ax.set_ylabel("units")
    out["daily_total"] = _save(fig, "daily_total.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    _bar(ax, DAYS, [f["weekday_units"][d] for d in DAYS])
    ax.set_title("Units by day of week")
    out["weekday"] = _save(fig, "weekday.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    _bar(ax, [m[:3] for m in MONTHS], [f["month_units"][m] for m in MONTHS])
    ax.set_title("Units by month, summed over 2019 to 2021")
    out["month"] = _save(fig, "month.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    keys = [f"Q{q}-{y}" for y in (2019, 2020, 2021) for q in (1, 2, 3, 4)]
    vals = [f["quarter_year_units"].get(k, 0) for k in keys]
    cols = [SERIES[i // 4] for i in range(12)]
    _bar(ax, keys, vals, color=cols)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Units by quarter and year")
    out["quarter"] = _save(fig, "quarter.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sn = list(f["store_units"])
    for ax, key, title in ((axes[0], "store_units", "Total units (log scale)"),
                           (axes[1], "store_revenue", "Total revenue, $ (log scale)")):
        vals = [f[key][s] for s in sn]
        ax.barh(sn[::-1], vals[::-1], color=[SERIES[sn.index(s) % 8] for s in sn[::-1]], height=0.7)
        ax.set_xscale("log")
        ax.grid(axis="x")
        ax.set_title(title)
        for i, v in enumerate(vals[::-1]):
            ax.annotate(f"{v:,.0f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.margins(x=0.25)
    out["revenue_vs_volume"] = _save(fig, "revenue_vs_volume.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
    piv = df.pivot_table(index="date", columns="store_id", values="item_count", aggfunc="sum")
    for i, (sid, ax) in enumerate(zip(sorted(piv.columns), axes.ravel())):
        ax.plot(piv.index, piv[sid].values, color=SERIES[i % 8], lw=0.9)
        ax.set_title(f"{names[sid]} (store {sid})")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Units per day by store", x=0.01, ha="left", fontweight="bold", color=INK)
    out["stores"] = _save(fig, "store_daily.png")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ti = f["top_items"]
    labels = [f"{t['item_name']}\n{t['restaurant_name']}" for t in ti][::-1]
    ax.barh(labels, [t["item_count"] for t in ti][::-1], color=ACCENT, height=0.65)
    ax.grid(axis="x")
    for i, t in enumerate(ti[::-1]):
        ax.annotate(f"{t['item_count']:,.0f}  ({t['kcal']} kcal, ${t['price']:.2f})", (t["item_count"], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.margins(x=0.4)
    ax.set_title("Five most popular items overall (log scale)")
    out["popular_items"] = _save(fig, "popular_items.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ex = f["most_expensive_per_store"]
    labels = [f"{e['item_name'][:34]}\n{e['restaurant_name']}" for e in ex][::-1]
    ax.barh(labels, [e["cost"] for e in ex][::-1], color=SERIES[3], height=0.65)
    ax.grid(axis="x")
    for i, e in enumerate(ex[::-1]):
        ax.annotate(f"${e['cost']:.2f}, {e['kcal']} kcal", (e["cost"], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.margins(x=0.3)
    ax.set_title("Most expensive item per store, with calories")
    out["expensive_items"] = _save(fig, "expensive_items.png")
    return out


def run() -> dict:
    df = load_merged()
    f = compute(df)
    f["charts"] = charts(df, f)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "findings.json").write_text(json.dumps(f, indent=1, default=float))
    return f


def load_findings() -> dict | None:
    p = OUT / "findings.json"
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == "__main__":
    f = run()
    for k, v in f["findings"].items():
        print(f"- {k}: {v}")
    print("charts:", ", ".join(Path(v).name for v in f["charts"].values()))
