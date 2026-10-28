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


def main():
    h = pd.read_csv(OUT / "dashboard_hourly.csv", parse_dates=["timestamp"])
    m = pd.read_csv(OUT / "dashboard_monthly.csv")
    acc = pd.read_csv(OUT / "forecast_accuracy.csv").set_index(["horizon_h", "group"])
    exp = pd.read_csv(OUT / "experiments_all.csv")
    summ = json.load(open(OUT / "summary.json"))
    order = list(NAMES)
    buildings = []
    for b in order:
        g = h[h.building == b].sort_values("timestamp")
        fl = (g.alert.astype(int) + g.counts_toward_peak.astype(int) * 2
              + g.waste_flag.fillna(False).astype(int) * 4).tolist()
        n, s, t = NAMES[b]
        buildings.append(dict(id=b, name=n, site=s, type=t, a=arr(g.actual_kw), p50=arr(g.forecast_kw),
                              p95=arr(g.forecast_high_kw), plan=arr(g.kw_if_alerts_followed),
                              tgt=arr(g.target_kw), fl=fl))
    months = [dict(b=order.index(r.building), month=r.month, before=round(r.md_baseline_kw, 1),
                   after=round(r.md_with_alerts_kw, 1), rm=int(r.rm_saved), alertDays=int(r.alert_days),
                   wasteH=int(r.waste_hours)) for r in m.itertuples()]
    pick = lambda tariff: exp[(exp.tariff == tariff) & (exp["quantile"] == "p95") &
                              (exp.flex == 0.10) & (exp.target_cut == 0.10)].iloc[0]
    tou, gen = pick("tou"), pick("general")
    summary = dict(
        mape_day=float(acc.loc[(24, "all buildings"), "model_mape"]),
        mape_hour=float(acc.loc[(1, "all buildings"), "model_mape"]),
        mape_naive=float(acc.loc[(24, "all buildings"), "naive_last_week_mape"]),
        all_cut=float(tou.avg_peak_reduction_pct), all_perfect=float(tou.perfect_foresight_pct),
        all_rm=int(tou.rm_saved_per_building_month), all_kw=float(tou.avg_peak_reduction_kw),
        precision=float(tou.precision), recall=float(tou.recall), alert_days=float(tou.alert_days_per_month),
        gen_cut=float(gen.avg_peak_reduction_pct), gen_perfect=float(gen.perfect_foresight_pct),
        demo_cut=summ["avg_peak_reduction_pct"], rate=97.06, buildings=99)
    data = json.dumps(dict(start="2017-04-01", buildings=buildings, months=months, summary=summary),
                      separators=(",", ":"))
    page = (Path(__file__).parent / "template.html").read_text().replace("/*DATA*/", data)
    head, body = page.split('<div class="wrap">', 1)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' + head +
            '</head>\n<body>\n<div class="wrap">' + body + '\n</body>\n</html>\n')
    (ROOT / "index.html").write_text(full)
    print(f"Wrote {ROOT / 'index.html'} ({len(full) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
