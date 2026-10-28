"""Step 5: write the files the dashboard reads. This is the contract with the dashboard designer.

outputs/dashboard_hourly.csv  - one row per building per hour (test period, demo buildings)
outputs/dashboard_monthly.csv - one row per building per month
outputs/summary.json          - headline numbers for the pitch
"""
import json
import pandas as pd
import config as C
import backtest as B


def main():
    pred = pd.read_parquet(C.WORK / "pred_h24.parquet")
    hourly, monthly = B.run(pred, C.DEMO_BUILDINGS)
    pred_t = pred[pred.timestamp > C.VALID_END].copy()
    pred_t["waste_flag"] = B.waste_flags(pred_t.sort_values(["building", "timestamp"]))
    hourly = hourly.merge(pred_t[["building", "timestamp", "waste_flag"]], on=["building", "timestamp"], how="left")

    out = hourly.rename(columns={"kw": "actual_kw", "p50": "forecast_kw", f"p{int(round(C.QUANTILE*100))}": "forecast_high_kw",
                                 "cap_kw": "target_kw", "kw_with_alerts": "kw_if_alerts_followed",
                                 "counted": "counts_toward_peak"})
    out = out[["building", "timestamp", "actual_kw", "forecast_kw", "forecast_high_kw", "target_kw",
               "alert", "kw_if_alerts_followed", "counts_toward_peak", "waste_flag"]]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(1)
    out.to_csv(C.OUT / "dashboard_hourly.csv", index=False)

    rate = C.MD_RATE[C.TARIFF]
    monthly["rm_saved"] = (monthly.reduction_kw * rate).round(0)
    monthly["waste_hours"] = monthly.apply(
        lambda r: int(out[(out.building == r.building) &
                          (out.timestamp.dt.to_period("M").astype(str) == r.month)].waste_flag.sum()), axis=1)
    monthly.round(1).to_csv(C.OUT / "dashboard_monthly.csv", index=False)

    acc = pd.read_csv(C.OUT / "forecast_accuracy.csv")
    demo_acc = acc[(acc.horizon_h == 24) & (acc.group == "demo buildings")].iloc[0]
    s = B.summarise(monthly, C.TARIFF)
    summary = dict(
        settings=dict(tariff=C.TARIFF, quantile=C.QUANTILE, flex_fraction=C.FLEX_FRACTION,
                      target_cut=C.TARGET_CUT, md_rate_rm_per_kw=rate),
        test_period="Apr-Dec 2017", buildings=C.DEMO_BUILDINGS,
        day_ahead_mape_pct=float(demo_acc.model_mape),
        naive_last_week_mape_pct=float(demo_acc.naive_last_week_mape),
        **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in s.items()},
        waste_hours_per_building_month=round(float(monthly.waste_hours.mean()), 1),
        caveats=["US buildings in hot climates stand in for Malaysian ones",
                 "hourly data approximates TNB's 30-minute intervals",
                 "weather treated as perfectly forecast",
                 "assumes managers act on every alert; RM uses vendor-reported TNB MV rates"],
    )
    json.dump(summary, open(C.OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
