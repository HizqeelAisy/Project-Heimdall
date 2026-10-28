#!/usr/bin/env bash
# Runs the whole PeakSense pipeline. Each step saves its output, so you can rerun any single step.
set -e
cd "$(dirname "$0")"
python3 prepare.py           # 1. select buildings, remove meter glitches, join weather
python3 train.py             # 2-3. features + LightGBM median/quantile forecasts (slowest step, ~20 min)
python3 run_experiments.py   # 4. the experiment loop: every alert setting, one results table
python3 compare_strategies.py # 5. routine vs Heimdall vs simpler strategies; dashboard files + headline numbers
echo "Done. See outputs/"
