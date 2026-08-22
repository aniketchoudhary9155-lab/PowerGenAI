# PowerGenAI — Project Summary (Phases 1–16)

A complete AI-based power generation forecasting and station performance monitoring
system, built from a real 335K-row CEA-style Indian power sector dataset.

## What the data actually is (and isn't)

- **Single-day snapshot, not a time series.** Only one unique date exists across all
  335K raw rows. LSTM and forecasting were correctly ruled out from Phase 1 onward —
  this is **tabular regression**: predicting Actual generation from operating
  conditions, not predicting the future.
- **334,994 raw rows → 148,815 after cleaning.** Removed: 3,080 unidentified
  placeholder rows, 51,296 state/UT-level aggregate rows (different granularity than
  individual power stations), and 131,803 exact duplicate rows.
- **152 genuine individual power stations** remain, ranging from small IPPs to
  NTPC Ltd. (India's largest generator, ~19.8 GW monitored capacity in this data).

## Phase-by-phase results

| Phase | Outcome |
|---|---|
| 1. Inspection | Confirmed single-day snapshot; found duplicate/placeholder/state-mixing issues |
| 2. Cleaning | 148,815 clean rows, 152 stations, formulas verified 100% consistent |
| 3. Feature Engineering | 10 derived features (Available Capacity, Maintenance Impact Index, etc.) |
| 4. EDA | 15 visualizations; confirmed scheduling accuracy, size vs. efficiency findings |
| 5. Leakage Analysis | 6 target-derived columns excluded from ML input features |
| 6. ML Problem Framing | Confirmed tabular regression; per-station 80/20 split |
| 7. Model Training | 4 models trained (Linear, RF, GB, HistGB/XGBoost-substitute) |
| 8. Model Evaluation | **Random Forest best**: R²=0.994, MAE=1.9 MW |
| 9. Feature Importance | Available_Capacity (81%) dominates; maintenance detail ≈ irrelevant once known |
| 10. Explainable AI | Tree-path contribution decomposition (SHAP substitute), verified exact |
| 11. Performance Monitoring | Station rankings: best/worst performers, highest shortfall/deviation |
| 12. Maintenance Analytics | Forced (unplanned) maintenance = 64% of all fleet maintenance |
| 13. Alert System | 4 configurable rules, recalibrated to ~90th percentile thresholds |
| 14. Streamlit App | 7-page dashboard, verified end-to-end against real data |
| 15. Testing | 19/19 edge-case tests passed (invalid input, unknown station, extreme values, etc.) |
| 16. Packaging | This file — final structure, notebooks, and reports assembled |

## Honest disclosures (carried through every phase)

1. **XGBoost and SHAP could not be installed** in the sandbox this was built in (no
   internet access). Documented, verified substitutes are used instead:
   `HistGradientBoostingRegressor` for XGBoost, and a tree-path contribution
   decomposition for SHAP (mathematically confirmed to exactly reconstruct each
   prediction).
2. **This is an interpolation test, not a generalization test.** The train/test split
   ensures every station appears in training; 27 stations with <5 records exist
   only in training. Real-world performance on a station never seen at all is
   untested here.
3. **Alert thresholds are software-defined**, calibrated to this dataset's own
   percentiles — not official CEA or grid-code limits. Recalibrate before
   operational use.
4. **Deployed Random Forest is a lighter, compressed variant** (100 trees,
   depth-capped) of the one evaluated in Phase 8 — chosen because the
   full-precision version was 424 MB, impractical to ship. Accuracy is nearly
   identical (MAE 1.98 vs. 1.90 MW).
5. **Bug found and fixed post-deployment**: the Dashboard's "Total Monitored
   Capacity" KPI originally summed `Monitored_Capacity` across all 148,815
   rows, overcounting by **590x** (284 million MW vs. the correct ~481,000 MW)
   — because capacity is a static per-unit attribute repeated across many
   observation-rows (different time-blocks), not a per-row quantity like
   Programme/Actual. Fixed via `analytics.total_monitored_capacity()`, which
   sums only distinct (station, capacity) combinations. A regression test
   now guards against this recurring.

## How to explore this project

- **Run the app:** `pip install -r requirements.txt && streamlit run dashboard/app.py`
- **Read the full analysis:** open the HTML reports in `reports/` (Phase 4 EDA,
  Phase 8 evaluation, Phase 10 explainability, Phase 12 maintenance analytics)
- **Reproduce from scratch:** run the notebooks in `notebooks/` in order
- **Extend it:** `src/` modules are decoupled and independently testable —
  see Phase 15's test suite for the pattern

## Repository structure

See `README.md` for the full folder layout and page-by-page app description.
