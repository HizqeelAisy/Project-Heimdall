"""Step 4: replay Apr-Dec 2017 as if PeakSense had been running.

Day-ahead logic, simulated one day at a time:
  1. At the start of day D the building knows its cap: max(last month's peak, this month's peak so far).
  2. Any counted hour on day D whose pessimistic forecast is above the cap gets an alert.
  3. In alerted hours the building plans (from the forecast) to shed up to FLEX_FRACTION of its load,
     never below the cap. The shed energy is moved into the 6 hours before (pre-cooling) - but only into
     hours the forecast says have room under the cap, so the shift itself doesn't create a new peak.
     If there is no room, the building does nothing that hour.
  4. The month's peak charge is recomputed on the resulting load.

Honest limits: hourly data stands in for 30-minute intervals; weather is treated as perfectly
forecast; we assume managers act on every alert.
"""
import numpy as np
import pandas as pd
import config as C

REBOUND_WINDOW = 6      # pre-cooling can start up to 6 hours before an alerted hour
REBOUND_SHARE = C.REBOUND_SHARE


def counted_hours(ts: pd.Series, tariff: str) -> np.ndarray:
    if tariff == "general":
        return np.ones(len(ts), dtype=bool)
    # Time-of-use: only weekday 2pm-10pm counts towards the peak charge
    return ((ts.dt.dayofweek < 5) & (ts.dt.hour >= 14) & (ts.dt.hour < 22)).values


def simulate_building(df: pd.DataFrame, qcol: str, flex: float, tariff: str, target_cut: float):
    """df: one building, hourly, sorted, columns timestamp, kw, p50, <qcol>. Returns hourly frame + monthly frame."""
    df = df.copy()
    df["counted"] = counted_hours(df.timestamp, tariff)
    df["date"] = df.timestamp.dt.normalize()
    df["ym"] = df.timestamp.dt.to_period("M")
    kw = df.kw.values.astype(float)
    fc = df[qcol].values.astype(float)
    cnt = df.counted.values
    new = kw.copy()
    alert = np.zeros(len(df), dtype=bool)
    cap_arr = np.full(len(df), np.nan)

    months = df.ym.values
    month_list = list(dict.fromkeys(months))
    prev_md_actual = None
    for m in month_list:
        mi = np.where(months == m)[0]
        # The month's target: last month's peak cut by target_cut. If an unprotected hour breaks it,
        # the new peak is already on the bill, so the cap rises to that level for the rest of the month.
        cap = (1 - target_cut) * prev_md_actual if prev_md_actual is not None else np.nan
        dates = df.date.values[mi]
        for d in dict.fromkeys(dates):
            di = mi[dates == d]
            c = cap if not np.isnan(cap) else -np.inf
            cap_arr[di] = c if np.isfinite(c) else np.nan
            a = cnt[di] & (fc[di] > c) & np.isfinite(c) & ~np.isnan(kw[di])
            alert[di] = a
            added = np.zeros(len(di))           # energy already scheduled into each hour today
            for j_local in np.where(a)[0]:
                j = di[j_local]
                # Plan with the FORECAST (the building can't see the future): how much must go?
                want = min(flex * fc[j], max(fc[j] - c, 0.0))
                if want <= 0:
                    continue
                # Shift into earlier hours that have room: uncounted hours have unlimited room,
                # counted hours only up to the cap (judged on the forecast). Rebound share is moved.
                cands = [k for k in range(j_local - REBOUND_WINDOW, j_local)
                         if k >= 0 and not a[k] and not np.isnan(kw[di[k]]) and not np.isnan(fc[di[k]])]
                room = np.array([np.inf if not cnt[di[k]] else max(c - fc[di[k]] - added[k], 0.0) for k in cands])
                need = want * REBOUND_SHARE
                total_room = room.sum() if len(room) else 0.0
                if need > 0 and total_room <= 0:
                    continue                    # nowhere to put it: no action this hour
                scale = min(1.0, total_room / need) if need > 0 else 1.0
                shed = want * scale
                new[j] -= shed
                if need > 0:
                    move = shed * REBOUND_SHARE
                    fin = np.isfinite(room)
                    if (~fin).any():            # spread over uncounted hours first
                        idx = [cands[i] for i in np.where(~fin)[0]]
                        for k in idx:
                            added[k] += move / len(idx); new[di[k]] += move / len(idx)
                    else:
                        share = room / room.sum()
                        for k, s in zip(cands, share):
                            added[k] += move * s; new[di[k]] += move * s
            day_counted = new[di][cnt[di]]
            day_counted = day_counted[~np.isnan(day_counted)]
            if day_counted.size:
                cap = max(c, day_counted.max()) if np.isfinite(c) else day_counted.max()
        ma = kw[mi][cnt[mi]]
        prev_md_actual = np.nanmax(ma) if ma.size and not np.all(np.isnan(ma)) else prev_md_actual

    df["kw_with_alerts"] = new
    df["alert"] = alert
    df["cap_kw"] = cap_arr
    # an hour is "costly" if, without action, it would push the peak above the cap known at the time
    df["costly"] = df.counted & (df.kw > df.cap_kw)

    rows = []
    for m, g in df.groupby("ym"):
        gc = g[g.counted]
        md0, md1 = gc.kw.max(), gc.kw_with_alerts.max()
        tp = int((gc.alert & gc.costly).sum())
        rows.append(dict(month=str(m), md_baseline_kw=md0, md_with_alerts_kw=md1,
                         reduction_kw=md0 - md1, alert_hours=int(gc.alert.sum()),
                         alert_days=int(gc[gc.alert].date.nunique()), costly_hours=int(gc.costly.sum()),
                         true_alerts=tp))
    return df, pd.DataFrame(rows)


def run(pred: pd.DataFrame, buildings, qcol=None, flex=C.FLEX_FRACTION, tariff=C.TARIFF,
        target_cut=C.TARGET_CUT, start=C.VALID_END):
    qcol = qcol or f"p{int(round(C.QUANTILE * 100))}"
    hourly, monthly = [], []
    for b in buildings:
        g = pred[pred.building == b].sort_values("timestamp")
        # start one month early so the first test month has a cap to work with
        g = g[g.timestamp > pd.Timestamp(start) - pd.DateOffset(months=1)]
        h, m = simulate_building(g, qcol, flex, tariff, target_cut)
        h = h[h.timestamp > start]
        m = m[m.month > str(pd.Timestamp(start).to_period("M"))]
        h["building"], m["building"] = b, b
        hourly.append(h)
        monthly.append(m)
    return pd.concat(hourly, ignore_index=True), pd.concat(monthly, ignore_index=True)


def summarise(monthly: pd.DataFrame, tariff: str) -> dict:
    rate = C.MD_RATE[tariff]
    tp, alerts, costly = monthly.true_alerts.sum(), monthly.alert_hours.sum(), monthly.costly_hours.sum()
    return dict(
        building_months=len(monthly),
        avg_peak_reduction_kw=round(monthly.reduction_kw.mean(), 1),
        avg_peak_reduction_pct=round((monthly.reduction_kw / monthly.md_baseline_kw).mean() * 100, 2),
        rm_saved_per_building_month=round(monthly.reduction_kw.mean() * rate, 0),
        alert_hours_per_month=round(monthly.alert_hours.mean(), 1),
        alert_days_per_month=round(monthly.alert_days.mean(), 1),
        precision=round(tp / alerts, 3) if alerts else np.nan,
        recall=round(tp / costly, 3) if costly else np.nan,
    )


def waste_flags(pred24: pd.DataFrame) -> pd.Series:
    """Actual load above the day-ahead median by WASTE_RATIO for WASTE_MIN_HOURS in a row."""
    over = (pred24.kw > C.WASTE_RATIO * pred24.p50).astype(int)
    run_id = (over != over.groupby(pred24.building).shift()).cumsum()
    run_len = over.groupby([pred24.building, run_id]).transform("sum")
    return (over == 1) & (run_len >= C.WASTE_MIN_HOURS)
