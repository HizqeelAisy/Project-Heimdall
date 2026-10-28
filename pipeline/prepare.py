"""Step 1: select buildings, clean meter glitches, join weather.

Output: data/work/clean.parquet (long format: building, timestamp, kw, weather columns)
"""
import numpy as np
import pandas as pd
import config as C


def select_buildings(elec, meta):
    m = meta.set_index("building_id")
    cols = [b for b in elec.columns
            if b in m.index and m.loc[b, "site_id"] in C.SITES
            and m.loc[b, "primaryspaceusage"] in C.USES]
    cov = elec[cols].notna().mean()
    mean_kw = elec[cols].mean()
    keep = [b for b in cols if cov[b] >= C.MIN_COVERAGE and mean_kw[b] >= C.MIN_MEAN_KW]
    return keep


def clean_series(x: pd.Series) -> tuple[pd.Series, int]:
    """Remove meter glitches. A single fake spike would set a fake monthly peak.

    Rule: a reading is a glitch if it is far above both the building's normal top level
    (99.5th percentile) and its own week (rolling median + robust spread).
    """
    x = x.copy()
    x[x <= 0] = np.nan
    med = x.rolling(24 * 7, center=True, min_periods=24).median()
    mad = (x - med).abs().rolling(24 * 7, center=True, min_periods=24).median()
    p995 = x.quantile(0.995)
    glitch = (x > 1.2 * p995) & (x > med + 10 * 1.4826 * mad.clip(lower=1))
    # Accumulation artifacts: a meter that misses readings often dumps the missing energy
    # into the next one (spike right after a gap), or splits a reading into dip + double.
    high = x > 1.6 * med
    after_gap = x.shift(1).isna() & x.notna()
    dip_before = x.shift(1) < 0.5 * med
    dip_after = x.shift(-1) < 0.5 * med
    glitch |= high & (after_gap | dip_before | dip_after)
    # the dip half of a dip+double pair is also wrong
    glitch |= (x < 0.5 * med) & ((x.shift(-1) > 1.6 * med) | (x.shift(1) > 1.6 * med))
    n = int(glitch.sum())
    x[glitch] = np.nan
    x = x.interpolate(limit=3, limit_area="inside")   # fill gaps up to 3 hours only
    return x, n


def remove_site_events(data, share=0.4, ratio=1.4):
    """Hours where many buildings on one site jump together are data-logger events, not demand.
    Blank the affected readings (and the dip just before them) for buildings that jumped."""
    total = 0
    for site, g in data.groupby("site"):
        p = g.pivot(index="timestamp", columns="building", values="kw")
        med = p.rolling(168, center=True, min_periods=24).median()
        r = p / med
        event = (r > ratio).mean(axis=1) > share
        bad = r.gt(1.25).mul(event, axis=0)
        bad |= r.lt(0.8).mul(event.shift(-1, fill_value=False), axis=0)   # dip right before
        total += int(bad.values.sum())
        p = p.mask(bad)
        p = p.interpolate(limit=3, limit_area="inside")
        long = p.stack(future_stack=True).rename("kw_fixed").reset_index()
        data = data.merge(long, on=["timestamp", "building"], how="left")
        sel = data.site == site
        data.loc[sel, "kw"] = data.loc[sel, "kw_fixed"]
        data = data.drop(columns="kw_fixed")
    return data, total


def main():
    C.WORK.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.RAW / "metadata.csv")
    elec = pd.read_parquet(C.ROOT / "data" / "electricity.parquet") if (C.ROOT / "data" / "electricity.parquet").exists() \
        else pd.read_csv(C.RAW / "electricity_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    weather = pd.read_csv(C.RAW / "weather.csv", parse_dates=["timestamp"])

    keep = select_buildings(elec, meta)
    missing = [b for b in C.DEMO_BUILDINGS if b not in keep]
    assert not missing, f"Demo buildings fail filters: {missing}"

    parts, report = [], []
    site_of = meta.set_index("building_id")["site_id"]
    for b in keep:
        x, n = clean_series(elec[b])
        report.append((b, n))
        df = x.rename("kw").to_frame()
        df["building"] = b
        df["site"] = site_of[b]
        parts.append(df.reset_index())
    data = pd.concat(parts, ignore_index=True)
    data, n_site = remove_site_events(data)
    report.append(("(site-wide logger events)", n_site))

    w = weather[weather.site_id.isin(C.SITES)][["timestamp", "site_id", "airTemperature", "dewTemperature"]]
    w = w.rename(columns={"site_id": "site", "airTemperature": "temp", "dewTemperature": "dew"})
    w = w.sort_values("timestamp")
    w[["temp", "dew"]] = w.groupby("site")[["temp", "dew"]].transform(lambda s: s.interpolate(limit=6))
    data = data.merge(w, on=["timestamp", "site"], how="left")

    data.to_parquet(C.WORK / "clean.parquet")
    glitches = pd.DataFrame(report, columns=["building", "glitches_removed"])
    glitches.to_csv(C.WORK / "cleaning_report.csv", index=False)
    print(f"buildings kept: {len(keep)}  rows: {len(data):,}  glitches removed: {glitches.glitches_removed.sum()}")
    print(glitches[glitches.building.isin(C.DEMO_BUILDINGS)].to_string(index=False))


if __name__ == "__main__":
    main()
