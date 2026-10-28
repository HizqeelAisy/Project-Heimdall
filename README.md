# Project Heimdall

**Sees tomorrow's electricity peak coming. Tells the building when to pre-cool.**

Team Hyperion · Hackpire Lumina 2026 · Predictive Infrastructure & Energy Analytics

**Live prototype:** https://hizqeelaisy.github.io/Project-Heimdall/

**Show me the model:** [`notebooks/show_me_the_model.ipynb`](notebooks/show_me_the_model.ipynb) loads the trained model, prints one of its decision trees, forecasts a day it never saw, and reproduces the accuracy numbers. [Open in Colab](https://colab.research.google.com/github/HizqeelAisy/Project-Heimdall/blob/main/notebooks/show_me_the_model.ipynb) and choose Runtime › Run all.

## The problem

TNB charges Malaysian medium-voltage buildings for their single highest 30-minute demand each month: RM89.27 per kW on MV General and RM97.06 per kW on MV Time-of-Use since July 2025, about three times the old rate. One hot afternoon sets the whole month's bill, and facility managers can't see it coming.

## What Heimdall does

1. **Forecast** tomorrow's load for every hour (a best guess and a cautious 95th-percentile high) from meter history, weather and the calendar.
2. **Plan:** if the cautious forecast breaks this month's target (10% below last month's peak) between 2pm and 10pm on a weekday, move chiller load into the morning. TNB does not charge maximum demand that occurs off-peak.
3. **Message** the facility manager the evening before, for example on WhatsApp.
4. **Waste check:** flag load running 20%+ above the forecast for 3+ hours.

## Results (backtest, 99 commercial buildings, April to December 2017, never seen in training)

| Measure | Result |
| --- | --- |
| Day-ahead forecast error (MAPE) | 6.1% vs 8.6% for "same hour last week" |
| Hour-ahead forecast error | 3.5% |
| Monthly peak, Time-of-Use tariff | 3.1% lower on average, about RM1,140 per building per month |
| Hours that would have raised the peak and got a plan | 93% |
| Same model under the general tariff | 0.5% lower peak (no off-peak window to shift into) |

Total energy is unchanged in the backtest: load is shifted, not removed.

## Repository

| Path | Contents |
| --- | --- |
| `index.html` | The prototype dashboard (self-contained, works offline except fonts) |
| `pipeline/` | Data cleaning, features, LightGBM training, backtest, experiment loop |
| `pipeline/outputs/` | Forecast accuracy, experiment results, dashboard data |
| `models/` | Trained day-ahead and hour-ahead median models (LightGBM text format) |
| `notebooks/` | `show_me_the_model.ipynb`: the model, running live |
| `pipeline/sample/` | Cleaned hourly data for the six demo buildings (used by the notebook) |
| `dashboard/` | Dashboard template and `build_dashboard.py` to rebuild `index.html` |
| `docs/` | The Hackpire Lumina canvas |

## Reproduce

```bash
pip install -r pipeline/requirements.txt
mkdir -p pipeline/data/raw && cd pipeline/data/raw
B=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
curl -LO $B/metadata/metadata.csv -LO $B/weather/weather.csv -LO $B/meters/cleaned/electricity_cleaned.csv
cd ../../..
bash pipeline/run_all.sh          # about 30 minutes, mostly training
python dashboard/build_dashboard.py
```

## Method

- **Data:** Building Data Genome Project 2 (the open dataset behind ASHRAE's Great Energy Predictor III). 99 commercial buildings in Austin, Texas and Tempe, Arizona, the closest hot climates to Malaysia in open data. Two years of hourly electricity plus weather.
- **Cleaning:** removes meter glitches, catch-up spikes after gaps, and site-wide logger events, so no fake peak sets a fake target.
- **Split by time:** train 2016, tune January to March 2017, test April to December 2017. Load features only use readings available at forecast time; a leakage test confirms this.
- **Model:** LightGBM gradient boosting, quantile objectives (50th, 80th, 90th, 95th percentiles), 24-hour and 1-hour horizons, one model across all buildings with load scaled by each building's 2016 average.
- **Backtest:** replayed day by day. Up to 10% of load in a planned hour is shifted into the six hours before, only where the forecast shows room under the target.

## Limits

- US buildings stand in for Malaysian ones; the next step is a pilot on a Malaysian building's 30-minute TNB data.
- Hourly data approximates TNB's 30-minute intervals.
- Weather is treated as known a day ahead.
- The backtest assumes every plan is followed.
- Tariff rates come from vendor guides to TNB's July 2025 schedule; TNB's tariff page confirms off-peak maximum demand is not charged.
