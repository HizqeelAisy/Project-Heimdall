"""The experiment loop: try every combination of alert settings, log results to one table.

Run:  python run_experiments.py            (demo buildings, ~1 min)
      python run_experiments.py --all      (all hot-climate buildings, slower, more robust)
Output: outputs/experiments.csv, sorted best-first within each tariff
"""
import itertools
import sys
import pandas as pd
import config as C
import backtest as B

GRID = dict(
    quantile=["p80", "p90", "p95"],
    flex=[0.05, 0.10, 0.15],
    target_cut=[0.05, 0.10, 0.15],
    tariff=["general", "tou"],
)


def main():
    pred = pd.read_parquet(C.WORK / "pred_h24.parquet")
    pred["oracle"] = pred["kw"]
    buildings = sorted(pred.building.unique()) if "--all" in sys.argv else C.DEMO_BUILDINGS
    rows = []
    combos = list(itertools.product(*GRID.values()))
    oracle_cache = {}
    for i, (q, flex, cut, tariff) in enumerate(combos, 1):
        _, m = B.run(pred, buildings, qcol=q, flex=flex, tariff=tariff, target_cut=cut)
        s = B.summarise(m, tariff)
        key = (flex, cut, tariff)
        if key not in oracle_cache:
            _, mo = B.run(pred, buildings, qcol="oracle", flex=flex, tariff=tariff, target_cut=cut)
            oracle_cache[key] = B.summarise(mo, tariff)["avg_peak_reduction_pct"]
        s.update(quantile=q, flex=flex, target_cut=cut, tariff=tariff,
                 perfect_foresight_pct=oracle_cache[key],
                 share_of_perfect=round(s["avg_peak_reduction_pct"] / oracle_cache[key], 2) if oracle_cache[key] else None)
        rows.append(s)
        print(f"[{i}/{len(combos)}] {tariff} q={q} flex={flex} cut={cut} -> "
              f"{s['avg_peak_reduction_pct']}% peak cut, P={s['precision']} R={s['recall']}")
    df = pd.DataFrame(rows)
    front = ["tariff", "quantile", "flex", "target_cut", "avg_peak_reduction_pct", "perfect_foresight_pct",
             "share_of_perfect", "rm_saved_per_building_month", "alert_hours_per_month", "alert_days_per_month",
             "precision", "recall", "avg_peak_reduction_kw", "building_months"]
    df = df[front].sort_values(["tariff", "avg_peak_reduction_pct"], ascending=[True, False])
    df.to_csv(C.OUT / ("experiments_all.csv" if "--all" in sys.argv else "experiments.csv"), index=False)
    print(df.groupby("tariff").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
