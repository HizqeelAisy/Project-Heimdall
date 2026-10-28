"""Step 6: is a forecast worth it? Compare peak strategies under realistic limits, and export the
Heimdall policy for the dashboard.

Question from the team: "If everyone knows 2pm-10pm is peak time, why not just pre-cool every weekday?"
Answer, tested on 99 buildings over Apr-Dec 2017: a daily routine captures most of the saving, and the model
earns its place by telling the building which days to push harder.

Realistic limits applied to every strategy:
  - in an eased hour, chiller load drops by FLEX of that hour's load; the energy is moved into the 6 hours before
  - pre-cooling wastes LOSS extra energy, charged at the off-peak energy rate
  - target = 10% below last month's peak; a broken target rises to the new peak (it is already on the bill)

Strategies:
  routine          every weekday, pre-cool and ease 2pm-5pm (no model needed)
  heimdall         routine + on days the model's cautious (95%) forecast breaks the target, ease up to
                   EXTRA_HOURS more of the riskiest hours
  heimdall_shortcut same as heimdall, but "same hour last week" decides the risky days
  routine_plus     routine + EXTRA_HOURS more every weekday, blindly
  light            only on high-risk days, up to ROUTINE_HOURS hours chosen by the model (no daily routine)

Outputs: outputs/strategy_comparison.csv, outputs/baseline_accuracy.csv,
         outputs/dashboard_hourly.csv, outputs/dashboard_monthly.csv, outputs/summary.json
"""
import json
import numpy as np
import pandas as pd
import config as C
import backtest as B

FLEX, ROUTINE_HOURS, EXTRA_HOURS, LOSS, WINDOW = 0.10, 3, 2, 0.15, 6
RATE_MD, RATE_ENERGY_OFFPEAK = C.MD_RATE["tou"], 0.2723   # RM/kW/month, RM/kWh (MV Time-of-Use off-peak)
FORECAST = {"routine": "p50", "routine_plus": "p50", "heimdall": "p95", "heimdall_shortcut": "naive", "light": "p95"}


def choose_hours(strategy, ok, hour, fc, cap):
    in_routine = (hour >= 14) & (hour < 14 + ROUTINE_HOURS)
    routine = np.where(ok & in_routine)[0]
    if strategy == "routine":
        return routine, np.array([], int)
    if strategy == "routine_plus":
        return routine, np.where(ok & (hour >= 14 + ROUTINE_HOURS) & (hour < 14 + ROUTINE_HOURS + EXTRA_HOURS))[0]
    risky = np.where(ok & (fc > cap))[0]
    if strategy == "light":
        return np.array([], int), risky[np.argsort(-fc[risky])][:ROUTINE_HOURS]
    boost = np.where(ok & ~in_routine & (fc > cap))[0]
    return routine, boost[np.argsort(-fc[boost])][:EXTRA_HOURS]


def simulate(g: pd.DataFrame, strategy: str, hourly=False):
    g = g.sort_values("timestamp").reset_index(drop=True)
    ts = g.timestamp
    hour = ts.dt.hour.values
    counted = ((ts.dt.dayofweek < 5) & (ts.dt.hour >= 14) & (ts.dt.hour < 22)).values
    kw = g.kw.values.astype(float)
    fc = g[FORECAST[strategy]].values.astype(float)
    new = kw.copy()
    mark = np.zeros(len(g), dtype=int)          # 1 = routine hour, 2 = boost hour
    target = np.full(len(g), np.nan)
    loss_kwh = np.zeros(len(g))
    months, dates = ts.dt.to_period("M").values, ts.dt.normalize().values
    rows, prev_peak = [], None
    for m in dict.fromkeys(months):
        mi = np.where(months == m)[0]
        cap = (1 - C.TARGET_CUT) * prev_peak if prev_peak is not None else np.nan
        days = 0
        for d in dict.fromkeys(dates[mi]):
            di = mi[dates[mi] == d]
            target[di] = cap
            if not np.isfinite(cap):
                continue
            ok = counted[di] & ~np.isnan(kw[di]) & ~np.isnan(fc[di])
            routine, boost = choose_hours(strategy, ok, hour[di], fc[di], cap)
            acted = False
            for jl, kind in [(j, 1) for j in routine] + [(j, 2) for j in boost]:
                j = di[jl]
                before = [k for k in range(jl - WINDOW, jl)
                          if k >= 0 and not counted[di[k]] and not np.isnan(kw[di[k]]) and mark[di[k]] == 0]
                if not before:
                    continue
                shed = FLEX * kw[j]
                new[j] -= shed
                for k in before:
                    new[di[k]] += shed * (1 + LOSS) / len(before)
                loss_kwh[j] = shed * LOSS
                mark[j] = kind
                acted = True
            days += acted
            day_counted = new[di][counted[di]]
            day_counted = day_counted[~np.isnan(day_counted)]
            if day_counted.size:
                cap = max(cap, day_counted.max())
        peak_before = np.nanmax(kw[mi][counted[mi]])
        peak_after = np.nanmax(new[mi][counted[mi]])
        rows.append(dict(month=str(m), md_baseline_kw=peak_before, md_with_plan_kw=peak_after,
                         reduction_kw=peak_before - peak_after,
                         loss_rm=loss_kwh[mi].sum() * RATE_ENERGY_OFFPEAK,
                         routine_hours=int((mark[mi] == 1).sum()), boost_hours=int((mark[mi] == 2).sum()),
                         boost_days=int(pd.Series(dates[mi][mark[mi] == 2]).nunique()), action_days=days))
        prev_peak = peak_before
    monthly = pd.DataFrame(rows[1:])            # first month only sets the starting target
    monthly["peak_rm"] = monthly.reduction_kw * RATE_MD
    monthly["net_rm"] = monthly.peak_rm - monthly.loss_rm
    if hourly:
        h = g[["building", "timestamp", "kw", "p50", "p95"]].copy()
        h["target_kw"], h["kw_with_plan"], h["mark"], h["counted"] = target, new, mark, counted
        return monthly, h
    return monthly


def summarise(monthly: pd.DataFrame) -> dict:
    return dict(peak_cut_pct=float((monthly.reduction_kw / monthly.md_baseline_kw).mean() * 100),
                peak_rm=float(monthly.peak_rm.mean()), loss_rm=float(monthly.loss_rm.mean()),
                net_rm=float(monthly.net_rm.mean()),
                eased_hours=float((monthly.routine_hours + monthly.boost_hours).mean()),
                boost_days=float(monthly.boost_days.mean()))


def baseline_accuracy(p: pd.DataFrame) -> pd.DataFrame:
    """Hourly error inside the 2pm-10pm window, and error estimating the month's peak, per forecast."""
    t = p[(p.timestamp > C.VALID_END) & (p.timestamp.dt.dayofweek < 5)
          & (p.timestamp.dt.hour >= 14) & (p.timestamp.dt.hour < 22)].copy()
    t["month"] = t.timestamp.dt.to_period("M")
    g = t.groupby(["building", "month"]).agg(actual=("kw", "max"), model=("p50", "max"), shortcut=("naive", "max"))
    rows = []
    for f, col in [("model", "p50"), ("shortcut", "naive")]:
        rows.append(dict(forecast=f, hourly_error_pct_peak_window=round(float((abs(t[col] - t.kw) / t.kw).mean() * 100), 2),
                         monthly_peak_error_pct=round(float((abs(g[f] - g.actual) / g.actual).mean() * 100), 2)))
    return pd.DataFrame(rows)


def export_dashboard(p: pd.DataFrame):
    """Hourly and monthly files the dashboard reads, for the demo buildings, under the Heimdall policy."""
    hourly, monthly = [], []
    for b in C.DEMO_BUILDINGS:
        g = p[p.building == b]
        m, h = simulate(g, "heimdall", hourly=True)
        h = h[h.timestamp > C.VALID_END]
        m["building"] = b
        hourly.append(h)
        monthly.append(m)
    h = pd.concat(hourly, ignore_index=True).sort_values(["building", "timestamp"])
    h["waste_flag"] = B.waste_flags(h).values
    out = h.rename(columns={"kw": "actual_kw", "p50": "forecast_kw", "p95": "forecast_high_kw",
                            "kw_with_plan": "kw_if_plan_followed", "counted": "counts_toward_peak"})
    out["routine_hour"], out["boost_hour"] = out["mark"] == 1, out["mark"] == 2
    out = out[["building", "timestamp", "actual_kw", "forecast_kw", "forecast_high_kw", "target_kw",
               "routine_hour", "boost_hour", "kw_if_plan_followed", "counts_toward_peak", "waste_flag"]]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(1)
    out.to_csv(C.OUT / "dashboard_hourly.csv", index=False)
    mon = pd.concat(monthly, ignore_index=True)
    mon["waste_hours"] = [int(out[(out.building == r.building) &
                                  (out.timestamp.dt.to_period("M").astype(str) == r.month)].waste_flag.sum())
                          for r in mon.itertuples()]
    mon.round(1).to_csv(C.OUT / "dashboard_monthly.csv", index=False)
    return mon


def main():
    p = pd.read_parquet(C.WORK / "pred_h24.parquet")
    p["naive"] = p.groupby("building").kw.shift(168)
    sim = p[p.timestamp > pd.Timestamp(C.VALID_END) - pd.DateOffset(months=1)]
    rows = []
    for strategy in ["routine", "heimdall", "heimdall_shortcut", "routine_plus", "light"]:
        res = pd.DataFrame([summarise(simulate(g, strategy)) for _, g in sim.groupby("building")]).mean()
        rows.append(dict(strategy=strategy, **res.round(2).to_dict()))
    comp = pd.DataFrame(rows)
    comp.to_csv(C.OUT / "strategy_comparison.csv", index=False)
    print(comp.to_string(index=False))
    base = baseline_accuracy(p)
    base.to_csv(C.OUT / "baseline_accuracy.csv", index=False)
    print(base.to_string(index=False))
    mon = export_dashboard(sim)
    acc = pd.read_csv(C.OUT / "forecast_accuracy.csv").set_index(["horizon_h", "group"])
    c = comp.set_index("strategy")
    summary = dict(
        policy=dict(routine="pre-cool and ease 2pm-5pm every weekday", boost=f"up to {EXTRA_HOURS} extra hours on days "
                    "the cautious forecast breaks the target", flex=FLEX, loss=LOSS, target_cut=C.TARGET_CUT,
                    md_rate_rm_per_kw=RATE_MD),
        test_period="Apr-Dec 2017", buildings=99,
        day_ahead_mape_pct=float(acc.loc[(24, "all buildings"), "model_mape"]),
        hour_ahead_mape_pct=float(acc.loc[(1, "all buildings"), "model_mape"]),
        shortcut_mape_pct=float(acc.loc[(24, "all buildings"), "naive_last_week_mape"]),
        strategies=c.to_dict(orient="index"),
        demo_buildings_net_rm=float(mon.net_rm.mean()),
        caveats=["US buildings in hot climates stand in for Malaysian ones",
                 "hourly data approximates TNB's 30-minute intervals",
                 "weather treated as perfectly forecast", "assumes the plan is followed every day",
                 "simplified pre-cooling physics: 10% of an hour's load, 15% energy loss, no comfort limits"])
    json.dump(summary, open(C.OUT / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
