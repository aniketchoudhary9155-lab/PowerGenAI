"""
PowerGenAI — AI-Based Power Generation Forecasting and
Power Station Performance Monitoring System

Run with:  streamlit run dashboard/app.py   (from the PowerGenAI/ root folder)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import prediction, analytics, alerts, utils
from src.features import MODEL_FEATURE_COLUMNS

# ------------------------------------------------------------------
# PAGE CONFIG & DATA LOADING
# ------------------------------------------------------------------
st.set_page_config(page_title="PowerGenAI", page_icon="⚡", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'powergeneration_features.csv'))
    station_perf = pd.read_csv(os.path.join(DATA_DIR, 'station_performance_summary.csv'))
    return df, station_perf


@st.cache_resource
def load_models():
    """Load each model independently — if one .joblib fails to unpickle
    (e.g. a scikit-learn version mismatch), the others should still load
    rather than the whole app losing all models."""
    models = {}
    load_errors = {}
    for name in ['Random_Forest', 'Linear_Regression', 'Gradient_Boosting', 'HistGB_XGBoost_substitute']:
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        if not os.path.exists(path):
            continue
        try:
            models[name] = prediction.load_model(name)
        except Exception as e:
            load_errors[name] = str(e)
    return models, load_errors


@st.cache_data
def load_model_comparison():
    path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_feature_importance():
    path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


try:
    df, station_perf = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"⚠️ Could not load processed data: {e}\n\nMake sure the pipeline "
             f"(Phases 2-3) has been run and data/processed/ contains the expected CSVs.")

try:
    models, model_load_errors = load_models()
    MODELS_OK = 'Random_Forest' in models
    if model_load_errors:
        error_lines = "\n".join(f"- **{name}**: {err}" for name, err in model_load_errors.items())
        st.warning(
            f"⚠️ Some models failed to load (likely a scikit-learn version mismatch between "
            f"training and this environment — pin `scikit-learn==1.8.0` per requirements.txt "
            f"to match, or retrain locally):\n\n{error_lines}\n\n"
            f"Working models: {', '.join(models.keys()) if models else 'none'}"
        )
    if not MODELS_OK:
        st.error("⚠️ The Random Forest model (used for predictions) could not be loaded — "
                 "the Generation Prediction and Explainable AI pages won't work until this is fixed.")
except Exception as e:
    MODELS_OK = False
    models = {}
    st.error(f"⚠️ Could not load trained models: {e}\n\nMake sure the models/ folder "
             f"contains the .joblib files from Phase 7 training.")

model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("⚡ PowerGenAI")
st.sidebar.caption("AI-Based Power Generation Forecasting & Monitoring")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🏭 Power Station Analysis",
    "🔮 Generation Prediction",
    "🔧 Maintenance Analysis",
    "📈 Model Performance",
    "🧠 Explainable AI",
    "📄 Reports",
])
st.sidebar.divider()
st.sidebar.caption("Note: XGBoost & SHAP were unavailable in the original "
                    "training sandbox (no internet access). HistGB substitutes "
                    "XGBoost; a tree-path contribution method substitutes SHAP. "
                    "Swap in the real libraries if you have connectivity.")

# ------------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 PowerGenAI Dashboard")

    if not DATA_OK:
        st.stop()

    station_filter = st.selectbox("Filter by station (optional)",
                                   ["All Stations"] + sorted(df['Power_Station'].unique().tolist()))
    view_df = df if station_filter == "All Stations" else df[df['Power_Station'] == station_filter]

    if view_df.empty:
        st.warning("No records match this filter.")
        st.stop()

    total_stations = view_df['Power_Station'].nunique()
    total_capacity = analytics.total_monitored_capacity(view_df)
    total_programme = view_df['Programme'].sum()
    total_actual = view_df['Actual'].sum()
    total_shortfall = view_df.loc[view_df['Excess_Shortfall'] < 0, 'Excess_Shortfall'].sum()
    avg_achievement = (total_actual / total_programme * 100) if total_programme > 0 else np.nan
    total_maintenance = view_df['Total_Maintenance'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Power Stations", f"{total_stations}")
    c2.metric("Total Monitored Capacity", utils.fmt_mw(total_capacity))
    c3.metric("Total Programme", utils.fmt_mw(total_programme))
    c4.metric("Total Actual", utils.fmt_mw(total_actual))

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Shortfall", utils.fmt_mw(total_shortfall))
    c6.metric("Avg Programme Achievement", utils.fmt_pct(avg_achievement))
    c7.metric("Total Maintenance", utils.fmt_mw(total_maintenance))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(view_df, x='Actual', nbins=60, title='Actual Generation Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        comp = analytics.maintenance_composition(view_df)
        fig2 = px.pie(names=list(comp.keys()), values=list(comp.values()),
                      title='Maintenance Composition', color_discrete_sequence=['#2563EB', '#DC2626', '#F59E0B'])
        st.plotly_chart(fig2, use_container_width=True)

    if station_filter == "All Stations":
        top15 = station_perf.sort_values('Actual_sum', ascending=False).head(15)
        fig3 = px.bar(top15, x='Actual_sum', y='Power_Station', orientation='h',
                      title='Top 15 Stations by Total Actual Generation')
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# PAGE: POWER STATION ANALYSIS
# ------------------------------------------------------------------
elif page == "🏭 Power Station Analysis":
    st.title("🏭 Power Station Analysis")

    if not DATA_OK:
        st.stop()

    station = st.selectbox("Select a station", sorted(df['Power_Station'].unique().tolist()))
    station_df = df[df['Power_Station'] == station]
    station_row = station_perf[station_perf['Power_Station'] == station]

    if station_df.empty:
        st.warning("No data available for this station.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    station_capacity = analytics.total_monitored_capacity(station_df)
    n_sub_units = station_df['Monitored_Capacity'].nunique()
    capacity_label = "Total Monitored Capacity" if n_sub_units == 1 else \
        f"Total Monitored Capacity ({n_sub_units} sub-units)"
    c1.metric(capacity_label, utils.fmt_mw(station_capacity))
    c2.metric("Records", f"{len(station_df)}")
    c3.metric("Available Capacity (avg)", utils.fmt_mw(station_df['Available_Capacity'].mean()))

    c4, c5, c6 = st.columns(3)
    c4.metric("Programme (avg)", utils.fmt_mw(station_df['Programme'].mean()))
    c5.metric("Actual (avg)", utils.fmt_mw(station_df['Actual'].mean()))
    c6.metric("Shortfall (avg)", utils.fmt_mw(station_df['Excess_Shortfall'].mean()))

    if not station_row.empty:
        r = station_row.iloc[0]
        c7, c8, c9 = st.columns(3)
        c7.metric("Programme Achievement", utils.fmt_pct(r['Programme_Achievement_pct']))
        c8.metric("Capacity Utilization", utils.fmt_pct(r['Capacity_Utilization_pct']))
        c9.metric("Maintenance Impact Index", utils.fmt_pct(r['Maintenance_Impact_Index_pct']))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(station_df.reset_index(), x=station_df.reset_index().index, y='Actual',
                         title=f'{station}: Actual Generation per Record')
        fig.add_hline(y=station_df['Programme'].mean(), line_dash='dash', line_color='red',
                      annotation_text='Avg Programme')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        maint_totals = station_df[['Planned_Maintenance', 'Forced_Maintenance', 'Other_Reasons']].sum()
        fig2 = px.pie(names=maint_totals.index, values=maint_totals.values,
                      title=f'{station}: Maintenance Breakdown')
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# PAGE: GENERATION PREDICTION
# ------------------------------------------------------------------
elif page == "🔮 Generation Prediction":
    st.title("🔮 Generation Prediction")
    st.caption("Predicts Actual Generation (MW) from operating conditions using the trained Random Forest model.")

    if not DATA_OK or not MODELS_OK:
        st.error("Data or model not available — cannot run predictions.")
        st.stop()

    station_list = sorted(df['Power_Station'].unique().tolist())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            station = st.selectbox("Power Station", station_list)
            monitored_capacity = st.number_input("Monitored Capacity (MW)", min_value=0.0, value=500.0, step=10.0)
            programme = st.number_input("Programme Generation (MW)", min_value=0.0, value=50.0, step=1.0)
        with col2:
            planned_maintenance = st.number_input("Planned Maintenance (MW)", min_value=0.0, value=0.0, step=1.0)
            forced_maintenance = st.number_input("Forced Maintenance (MW)", min_value=0.0, value=0.0, step=1.0)
            other_reasons = st.number_input("Other Reasons (MW)", min_value=0.0, value=0.0, step=1.0)

        submitted = st.form_submit_button("⚡ PREDICT GENERATION", use_container_width=True)

    if submitted:
        errors = utils.validate_prediction_inputs(
            monitored_capacity, programme, planned_maintenance, forced_maintenance, other_reasons
        )
        blocking_errors = [e for e in errors if not e.startswith("Warning")]
        warnings_only = [e for e in errors if e.startswith("Warning")]

        for w in warnings_only:
            st.warning(w)

        if blocking_errors:
            for e in blocking_errors:
                st.error(e)
        else:
            try:
                feature_row = prediction.build_feature_row(
                    station, monitored_capacity, programme,
                    planned_maintenance, forced_maintenance, other_reasons
                )
                model = models['Random_Forest']
                predicted_actual = prediction.predict(model, feature_row)
                predicted_shortfall = predicted_actual - programme
                achievement = (predicted_actual / programme * 100) if programme > 0 else None
                available_capacity = monitored_capacity - (planned_maintenance + forced_maintenance + other_reasons)
                mii = feature_row['Maintenance_Impact_Index'].iloc[0]
                risk = prediction.risk_level(predicted_actual, programme)

                st.success("Prediction complete")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Actual Generation", utils.fmt_mw(predicted_actual))
                c2.metric("Expected Shortfall/Excess", utils.fmt_mw(predicted_shortfall))
                c3.metric("Programme Achievement", utils.fmt_pct(achievement) if achievement is not None else "N/A")

                c4, c5, c6 = st.columns(3)
                c4.metric("Available Capacity", utils.fmt_mw(available_capacity))
                c5.metric("Maintenance Impact Index", utils.fmt_pct(mii))
                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk, "⚪")
                c6.metric("Risk Level", f"{risk_color} {risk}")

                st.divider()
                st.subheader("Why did the model predict this? (main prediction drivers)")
                try:
                    bias, contrib_dict, recon = prediction.explain_prediction(model, feature_row)
                    contrib_series = pd.Series(contrib_dict).sort_values(key=abs, ascending=False)
                    fig = go.Figure(go.Bar(
                        x=contrib_series.values, y=contrib_series.index, orientation='h',
                        marker_color=['#16a34a' if v >= 0 else '#DC2626' for v in contrib_series.values]
                    ))
                    fig.update_layout(title=f"Feature contributions (base value: {bias:.1f} MW)",
                                      xaxis_title="Contribution to prediction (MW)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Positive (green) = pushes prediction up. Negative (red) = pushes prediction down. "
                               "Uses a tree-path contribution decomposition (SHAP substitute — see Phase 10).")
                except Exception as e:
                    st.info(f"Explanation unavailable for this model type: {e}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------
# PAGE: MAINTENANCE ANALYSIS
# ------------------------------------------------------------------
elif page == "🔧 Maintenance Analysis":
    st.title("🔧 Maintenance Analysis")

    if not DATA_OK:
        st.stop()

    comp = analytics.maintenance_composition(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Planned Maintenance Share", utils.fmt_pct(comp['Planned']))
    c2.metric("Forced Maintenance Share", utils.fmt_pct(comp['Forced']))
    c3.metric("Other Reasons Share", utils.fmt_pct(comp['Other']))

    st.divider()
    st.subheader("Station Comparison")
    stations_to_compare = st.multiselect(
        "Select stations to compare (leave empty for top 10 by maintenance)",
        sorted(df['Power_Station'].unique().tolist())
    )

    if stations_to_compare:
        compare_df = station_perf[station_perf['Power_Station'].isin(stations_to_compare)]
    else:
        compare_df = station_perf[station_perf['Records'] >= 50].sort_values(
            'Total_Maintenance_avg', ascending=False).head(10)

    fig = px.bar(compare_df, x='Power_Station', y=['Total_Maintenance_avg', 'Forced_Maintenance_avg'],
                 barmode='group', title='Maintenance Comparison (avg MW per record)')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(compare_df, x='Power_Station', y='Maintenance_Impact_Index_pct',
                  title='Maintenance Impact Index by Station (%)', color_discrete_sequence=['#DC2626'])
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(compare_df[['Power_Station', 'Records', 'Total_Maintenance_avg',
                              'Forced_Maintenance_avg', 'Maintenance_Impact_Index_pct',
                              'Programme_Achievement_pct']], use_container_width=True)

# ------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if model_comparison_df.empty:
        st.warning("Model comparison data not found. Run Phase 8 evaluation first.")
        st.stop()

    st.subheader("Model Comparison Table")
    st.dataframe(model_comparison_df, use_container_width=True)

    best_model_name = model_comparison_df.loc[model_comparison_df['R2'].idxmax(), 'Model']
    st.success(f"🏆 Best model: **{best_model_name}**")

    fig = px.bar(model_comparison_df, x='Model', y='R2', title='R² by Model', color='Model')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(model_comparison_df, x='Model', y='MAE', title='MAE by Model (lower is better)', color='Model')
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Note: 'HistGB (XGBoost substitute)' uses scikit-learn's HistGradientBoostingRegressor "
               "in place of XGBoost, which could not be installed in the offline training sandbox.")

# ------------------------------------------------------------------
# PAGE: EXPLAINABLE AI
# ------------------------------------------------------------------
elif page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI")
    st.caption("Uses a tree-path contribution decomposition as a substitute for SHAP "
               "(unavailable offline) — mathematically verified to reconstruct exact predictions.")

    if feature_importance_df.empty:
        st.warning("Feature importance data not found. Run Phase 9 first.")
    else:
        st.subheader("Global Feature Importance (Random Forest)")
        top10 = feature_importance_df.head(10)
        fig = px.bar(top10, x='importance', y='feature', orientation='h',
                     title='Top 10 Feature Importances')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Individual Prediction Explanation")
    st.caption("Pick a real record from the dataset to see why the model predicted what it did.")

    if DATA_OK and MODELS_OK:
        sample_station = st.selectbox("Station", sorted(df['Power_Station'].unique().tolist()), key='xai_station')
        station_records = df[df['Power_Station'] == sample_station].reset_index(drop=True)
        if not station_records.empty:
            row_idx = st.slider("Record index", 0, len(station_records) - 1, 0)
            row = station_records.iloc[row_idx]
            feature_row = row[MODEL_FEATURE_COLUMNS].to_frame().T

            try:
                model = models['Random_Forest']
                bias, contrib_dict, recon_pred = prediction.explain_prediction(model, feature_row)
                st.write(f"**Actual value:** {row['Actual']:.2f} MW | **Model prediction:** {recon_pred:.2f} MW")

                contrib_series = pd.Series(contrib_dict).sort_values(key=abs, ascending=False)
                fig = go.Figure(go.Bar(
                    x=contrib_series.values, y=contrib_series.index, orientation='h',
                    marker_color=['#16a34a' if v >= 0 else '#DC2626' for v in contrib_series.values]
                ))
                fig.update_layout(title=f"Feature contributions (base value: {bias:.1f} MW)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate explanation: {e}")

# ------------------------------------------------------------------
# PAGE: REPORTS
# ------------------------------------------------------------------
elif page == "📄 Reports":
    st.title("📄 Reports")
    st.caption("Generate a downloadable performance report for a selected station.")

    if not DATA_OK:
        st.stop()

    station = st.selectbox("Select station for report", sorted(df['Power_Station'].unique().tolist()), key='report_station')
    station_df = df[df['Power_Station'] == station]
    station_row = station_perf[station_perf['Power_Station'] == station]

    if st.button("Generate Report"):
        cfg = alerts.AlertConfig()
        station_alerts = alerts.evaluate_dataframe(station_df, cfg)

        report_lines = [
            f"# PowerGenAI Performance Report — {station}",
            "",
            f"**Records analyzed:** {len(station_df)}",
            "",
            "## Generation KPIs",
            f"- Monitored Capacity: {utils.fmt_mw(station_df['Monitored_Capacity'].iloc[0])}",
            f"- Average Programme: {utils.fmt_mw(station_df['Programme'].mean())}",
            f"- Average Actual: {utils.fmt_mw(station_df['Actual'].mean())}",
            f"- Average Shortfall/Excess: {utils.fmt_mw(station_df['Excess_Shortfall'].mean())}",
            "",
            "## Maintenance KPIs",
            f"- Average Total Maintenance: {utils.fmt_mw(station_df['Total_Maintenance'].mean())}",
            f"- Average Forced Maintenance: {utils.fmt_mw(station_df['Forced_Maintenance'].mean())}",
            "",
        ]
        if not station_row.empty:
            r = station_row.iloc[0]
            report_lines += [
                f"- Programme Achievement: {utils.fmt_pct(r['Programme_Achievement_pct'])}",
                f"- Capacity Utilization: {utils.fmt_pct(r['Capacity_Utilization_pct'])}",
                f"- Maintenance Impact Index: {utils.fmt_pct(r['Maintenance_Impact_Index_pct'])}",
                "",
            ]

        report_lines.append("## Alerts Triggered")
        if station_alerts.empty:
            report_lines.append("- No alerts triggered for this station's records.")
        else:
            for alert_type, count in station_alerts['type'].value_counts().items():
                report_lines.append(f"- {alert_type}: {count} record(s)")

        report_text = "\n".join(report_lines)
        st.markdown(report_text)
        st.download_button("⬇️ Download Report (Markdown)", report_text,
                            file_name=f"PowerGenAI_Report_{station}.md", mime="text/markdown")
