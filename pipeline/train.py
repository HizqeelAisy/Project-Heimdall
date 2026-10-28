"""Step 3: train median and quantile forecasts for each horizon.

Output: data/work/pred_h{H}.parquet with columns building, timestamp, kw, p50, p80, p90, p95
        outputs/forecast_accuracy.csv
"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import config as C
from features import make_features, feature_columns

QUANTILES = [0.5, 0.8, 0.9, 0.95]
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)


def mape(actual, pred):
    m = actual > 0
    return float(np.mean(np.abs(actual[m] - pred[m]) / actual[m]) * 100)


def main():
    C.OUT.mkdir(exist_ok=True)
    clean = pd.read_parquet(C.WORK / "clean.parquet")
    train_mask_raw = clean.timestamp <= C.TRAIN_END
    scale = clean[train_mask_raw].groupby("building")["kw"].mean()     # scale from training period only
    json.dump(scale.to_dict(), open(C.WORK / "scale.json", "w"))

    rows = []
    for H in C.HORIZONS:
        df = make_features(clean, H, scale)
        cols = feature_columns(df)
        ok = df["y"].notna() & df[[c for c in cols if c.startswith("lag_")]].notna().all(axis=1)
        tr = ok & (df.timestamp <= C.TRAIN_END)
        va = ok & (df.timestamp > C.TRAIN_END) & (df.timestamp <= C.VALID_END)
        dtr = lgb.Dataset(df.loc[tr, cols], df.loc[tr, "y"])
        dva = lgb.Dataset(df.loc[va, cols], df.loc[va, "y"], reference=dtr)

        out = df[["building", "timestamp", "kw"]].copy()
        sc = df["building"].map(scale).values
        for q in QUANTILES:
            params = dict(PARAMS, objective="quantile", alpha=q, metric="quantile")
            model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            out[f"p{int(q*100)}"] = model.predict(df[cols], num_iteration=model.best_iteration) * sc
            C.MODELS.mkdir(exist_ok=True)
            model.save_model(str(C.MODELS / f"model_h{H}_p{int(q*100)}.txt"), num_iteration=model.best_iteration)
            if q == 0.5:
                imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
                (imp / imp.sum()).round(4).to_csv(C.OUT / f"feature_importance_h{H}.csv", header=["share"])
            print(f"H={H:>2} q={q:.2f} trees={model.best_iteration}")
        out[~ok] = out[~ok].assign(p50=np.nan, p80=np.nan, p90=np.nan, p95=np.nan)
        out.to_parquet(C.WORK / f"pred_h{H}.parquet")

        # Accuracy on the untouched test period (Apr-Dec 2017)
        te = out[(out.timestamp > C.VALID_END) & out.kw.notna() & out.p50.notna()].copy()
        naive = df.loc[te.index, "lag_168"] * df.loc[te.index, "building"].map(scale)   # same hour last week
        for name, sub in [("all buildings", te), ("demo buildings", te[te.building.isin(C.DEMO_BUILDINGS)])]:
            nv = naive.loc[sub.index]
            rows.append(dict(horizon_h=H, group=name, model_mape=round(mape(sub.kw.values, sub.p50.values), 2),
                             naive_last_week_mape=round(mape(sub.kw.values, nv.values), 2),
                             p90_coverage=round(float((sub.kw <= sub.p90).mean()), 3),
                             rows=len(sub)))
        for b in C.DEMO_BUILDINGS:
            sub = te[te.building == b]
            rows.append(dict(horizon_h=H, group=b, model_mape=round(mape(sub.kw.values, sub.p50.values), 2),
                             naive_last_week_mape=round(mape(sub.kw.values, naive.loc[sub.index].values), 2),
                             p90_coverage=round(float((sub.kw <= sub.p90).mean()), 3), rows=len(sub)))

    acc = pd.DataFrame(rows)
    acc.to_csv(C.OUT / "forecast_accuracy.csv", index=False)
    print(acc.to_string(index=False))


if __name__ == "__main__":
    main()
