# PowerGenAI

AI-Based Power Generation Forecasting and Power Station Performance Monitoring System.

Built from a real CEA-style single-day power generation snapshot covering 152 Indian
power stations. Full development log (Phases 1–13) is documented in the accompanying
HTML reports; this README covers running the final Streamlit application.

## What this is (and isn't)

- The source data is a **single-day snapshot** (one date across ~335K raw records), not
  a time series. All modeling here is **tabular regression**, not forecasting — there
  was no valid basis for LSTM or any sequence model, so none was used.
- **XGBoost and SHAP were unavailable** in the sandbox this project was built in (no
  internet access for `pip install`). Working substitutes are used throughout:
  - `HistGradientBoostingRegressor` (scikit-learn's built-in, algorithmically similar
    histogram-based gradient boosting) stands in for XGBoost.
  - A **tree-path contribution decomposition** (the same core idea behind SHAP's
    TreeExplainer) stands in for SHAP — verified to exactly reconstruct each
    prediction as `base value + sum(feature contributions)`.
  - Swap in the real libraries (commented out in `requirements.txt`) if you have
    connectivity — the code that uses them (`Random Forest` model object, prediction
    pipeline) doesn't need to change, only `Model Performance` / `Explainable AI` pages
    would optionally use the real thing instead.

## Project structure

```
PowerGenAI/
├── data/
│   ├── raw/                  # Original Excel export
│   └── processed/            # Cleaned + feature-engineered CSVs
├── models/                   # Trained model pipelines (Joblib) + comparison metrics
├── src/
│   ├── preprocessing.py      # Phase 2 cleaning logic
│   ├── features.py           # Phase 3 feature engineering
│   ├── prediction.py         # Model loading, inference, explanation
│   ├── analytics.py          # Phase 11/12 station & maintenance analytics
│   ├── alerts.py             # Phase 13 configurable rule-based alerts
│   └── utils.py              # Formatting & input validation helpers
├── dashboard/
│   └── app.py                # Streamlit application (7 pages)
├── reports/                  # Generated reports land here when saved locally
├── requirements.txt
└── README.md
```

## Running the app

```bash
pip install -r requirements.txt
cd PowerGenAI
streamlit run dashboard/app.py
```

The app loads the pre-trained model from `models/` — it does **not** retrain on
startup. If you want to retrain from scratch (e.g. after changing the cleaning or
feature logic), re-run the training steps documented in the Phase 7 report and
overwrite the `.joblib` files in `models/`.

## Pages

1. **Dashboard** — fleet-wide KPIs, filterable by station
2. **Power Station Analysis** — deep dive into a single station
3. **Generation Prediction** — enter operating conditions, get a predicted output,
   risk level, and a chart of what drove the prediction
4. **Maintenance Analysis** — fleet and station-level maintenance breakdown
5. **Model Performance** — comparison table and charts across all 4 trained models
6. **Explainable AI** — global feature importance + individual prediction explanations
7. **Reports** — generate and download a Markdown performance report per station

## Known limitations (carried over from the Phase 1–13 analysis)

- Train/test evaluation is an **interpolation test** (same stations, unseen records),
  not a test of generalization to a brand-new, never-before-seen station.
- 27 stations with fewer than 5 records exist only in training data, never in test —
  there simply isn't enough data to evaluate them separately.
- Alert thresholds in `src/alerts.py` are **software-defined analytical thresholds**
  calibrated to this dataset's own percentiles — not official grid-code or regulatory
  limits. Recalibrate before any real operational use.
