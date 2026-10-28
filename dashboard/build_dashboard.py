"""Rebuild index.html (the dashboard) from the pipeline outputs.

Run after `bash pipeline/run_all.sh`:   python dashboard/build_dashboard.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline" / "outputs"
NAMES = {
    "Fox_office_Alice": ("Office tower", "Tempe, Arizona", "Office"),
    "Fox_education_Wendell": ("Lecture hall block", "Tempe, Arizona", "Education"),
    "Fox_assembly_Johnnie": ("Event venue", "Tempe, Arizona", "Assembly"),
    "Bull_office_Lilla": ("Office tower", "Austin, Texas", "Office"),
    "Bull_education_Arthur": ("Campus building", "Austin, Texas", "Education"),
    "Bull_public_Hyun": ("Public services building", "Austin, Texas", "Public services"),
}


def arr(s):
    return [None if pd.isna(v) else int(round(v)) for v in s]


IMP_NAMES = {"lag_168": "Same hour last week", "lag_24": "Same hour yesterday", "building_cat": "Which building",
             "lag_336": "Same hour 2 weeks ago", "dow": "Day of week", "lag_25": "Yesterday, 1 hour earlier",
             "temp_roll24": "Last 24h temperature", "temp_max_today": "Today\u2019s max temperature"}


def add_model_stages(buildings, order):
    """Forecasts from the first 1, 10, 50 and 200 trees, for the 'Inside the model' section."""
    import sys
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT / "pipeline"))
    from features import make_features, feature_columns
    scale = pd.Series(json.load(open(ROOT / "models" / "scale.json")))
    data = pd.read_csv(ROOT / "pipeline" / "sample" / "demo_buildings.csv.gz", parse_dates=["timestamp"])
    feat = make_features(data, 24, scale)
    cols = feature_columns(feat)
    model = lgb.Booster(model_file=str(ROOT / "models" / "model_h24_p50.txt"))
    for b, rec in zip(order, buildings):
        g = feat[(feat.building == b) & (feat.timestamp >= "2017-04-01")].sort_values("timestamp")
        rec["stg"] = [[int(round(v)) for v in model.predict(g[cols], num_iteration=n) * scale[b]] for n in (1, 10, 50, 200)]
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(8).round(1)
    return dict(trees=model.num_trees(), inputs=len(model.feature_name()), stages=[1, 10, 50, 200, model.num_trees()],
                imp=[[IMP_NAMES.get(k, k), float(v)] for k, v in imp.items()])


def main():
    h = pd.read_csv(OUT / "dashboard_hourly.csv", parse_dates=["timestamp"])
    m = pd.read_csv(OUT / "dashboard_monthly.csv")
    summ = json.load(open(OUT / "summary.json"))
    order = list(NAMES)
    buildings = []
    for b in order:
        g = h[h.building == b].sort_values("timestamp")
        fl = (g.routine_hour.astype(int) + g.counts_toward_peak.astype(int) * 2
              + g.waste_flag.fillna(False).astype(int) * 4 + g.boost_hour.astype(int) * 8).tolist()
        n, s, t = NAMES[b]
        buildings.append(dict(id=b, name=n, site=s, type=t, a=arr(g.actual_kw), p50=arr(g.forecast_kw),
                              p95=arr(g.forecast_high_kw), plan=arr(g.kw_if_plan_followed),
                              tgt=arr(g.target_kw), fl=fl))
    months = [dict(b=order.index(r.building), month=r.month, before=round(r.md_baseline_kw, 1),
                   after=round(r.md_with_plan_kw, 1), rm=int(round(r.net_rm)), boostDays=int(r.boost_days),
                   wasteH=int(r.waste_hours)) for r in m.itertuples()]
    summary = dict(mape_day=summ["day_ahead_mape_pct"], mape_hour=summ["hour_ahead_mape_pct"],
                   mape_naive=summ["shortcut_mape_pct"], strategies=summ["strategies"], buildings=summ["buildings"])
    model = add_model_stages(buildings, order)
    data = json.dumps(dict(start="2017-04-01", buildings=buildings, months=months, summary=summary, model=model),
                      separators=(",", ":"))
    page = (Path(__file__).parent / "template.html").read_text().replace("/*DATA*/", data)
    head, body = page.split('<div class="wrap">', 1)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' + head +
            '</head>\n<body>\n<div class="wrap">' + body + '\n</body>\n</html>\n')
    (ROOT / "index.html").write_text(full)
    print(f"Wrote {ROOT / 'index.html'} ({len(full) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
