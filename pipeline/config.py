"""PeakSense pipeline settings. Change things here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"     # trained LightGBM forecast models (one file per horizon and quantile)

# Hot-climate sites closest to Malaysia: Bull = Austin TX (humid subtropical), Fox = Tempe AZ (hot desert)
SITES = ["Bull", "Fox"]
# Commercial-type buildings only (what we sell to)
USES = ["Office", "Education", "Entertainment/public assembly", "Public services", "Healthcare"]
MIN_COVERAGE = 0.97      # share of hours with an electricity reading
MIN_MEAN_KW = 100        # skip tiny buildings

# Buildings shown in the demo/dashboard (must pass the filters above)
DEMO_BUILDINGS = [
    "Fox_office_Alice", "Fox_education_Wendell", "Fox_assembly_Johnnie",
    "Bull_office_Lilla", "Bull_education_Arthur", "Bull_public_Hyun",
]

# Time-based split: never random (leakage). Train on 2016, tune on Q1 2017, test on Apr-Dec 2017.
TRAIN_END = "2016-12-31 23:00"
VALID_END = "2017-03-31 23:00"

# Forecast horizons in hours. 24 = day-ahead planning alerts, 1 = "act now" alerts.
HORIZONS = [24, 1]

# Peak alert backtest defaults (the experiment loop sweeps these)
QUANTILE = 0.95          # pessimistic forecast used for alerts (best in the experiment loop)
FLEX_FRACTION = 0.10     # share of load assumed shiftable in an alerted hour (chillers, pumps)
REBOUND_SHARE = 1.0      # share of shed energy that comes back earlier (1.0 = pure shifting, conservative)
TARGET_CUT = 0.10       # aim to keep this month's peak this far below last month's
TARIFF = "tou"           # "general" = any hour counts; "tou" = only weekday 2pm-10pm counts
MD_RATE = {"general": 89.27, "tou": 97.06}   # RM per kW per month, TNB MV (vendor-reported, verify)

# Waste flag: actual above the median forecast by this ratio for this many consecutive hours
WASTE_RATIO = 1.20
WASTE_MIN_HOURS = 3
