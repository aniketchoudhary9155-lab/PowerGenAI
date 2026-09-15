# dashboard/app.py
"""
PowerGenAI — AI-Based Power Generation Forecasting and
Power Station Performance Monitoring System

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prediction
import analytics
import alerts
import utils
from features import MODEL_FEATURE_COLUMNS


# ================================================================
# CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="PowerGenAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    text-align: center;
}

.alert-critical {
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #dc2626;
    background: rgba(220,38,38,.10);
}

.alert-high {
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #f97316;
    background: rgba(249,115,22,.10);
}

.alert-medium {
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #eab308;
    background: rgba(234,179,8,.10);
}

.alert-low {
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #16a34a;
    background: rgba(22,163,74,.10);
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data
def load_data():

    df_path = os.path.join(
        DATA_DIR,
        "powergeneration_features.csv"
    )

    station_path = os.path.join(
        DATA_DIR,
        "station_performance_summary.csv"
    )

    df = pd.read_csv(df_path)
    station_perf = pd.read_csv(station_path)

    return df, station_perf


@st.cache_resource
def load_models():

    models = {}
    errors = {}

    names = [
        "Random_Forest",
        "Linear_Regression",
        "Gradient_Boosting",
        "HistGB_XGBoost_substitute"
    ]

    for name in names:

        path = os.path.join(
            MODELS_DIR,
            f"{name}.joblib"
        )

        if not os.path.exists(path):
            continue

        try:
            models[name] = prediction.load_model(name)

        except Exception as e:
            errors[name] = str(e)

    return models, errors


@st.cache_data
def load_model_comparison():

    path = os.path.join(
        MODELS_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_feature_importance():

    path = os.path.join(
        MODELS_DIR,
        "feature_importance.csv"
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


try:
    df, station_perf = load_data()
    DATA_OK = True

except Exception as e:

    DATA_OK = False
    df = pd.DataFrame()
    station_perf = pd.DataFrame()

    st.error(
        f"Could not load data: {e}"
    )


try:
    models, model_errors = load_models()
    MODELS_OK = "Random_Forest" in models

except Exception:
    models = {}
    model_errors = {}
    MODELS_OK = False


model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_mean(series):

    if len(series) == 0:
        return 0

    return float(series.mean())


def health_score(row):

    achievement = np.clip(
        row.get("Programme_Achievement_pct", 0),
        0,
        100
    )

    utilization = np.clip(
        row.get("Capacity_Utilization_pct", 0),
        0,
        100
    )

    maintenance = np.clip(
        row.get("Maintenance_Impact_Index_pct", 0),
        0,
        100
    )

    score = (
        achievement * 0.40
        + utilization * 0.40
        + (100 - maintenance) * 0.20
    )

    return round(float(np.clip(score, 0, 100)), 1)


def health_status(score):

    if score >= 80:
        return "🟢 Excellent"

    if score >= 60:
        return "🟡 Good"

    if score >= 40:
        return "🟠 Poor"

    return "🔴 Critical"


def risk_from_values(actual, programme):

    if programme <= 0:
        return "LOW"

    achievement = actual / programme * 100

    if achievement >= 90:
        return "LOW"

    if achievement >= 75:
        return "MEDIUM"

    if achievement >= 50:
        return "HIGH"

    return "CRITICAL"


def detect_anomalies(station_df):

    result = station_df.copy()

    if "Actual" not in result.columns:
        result["Anomaly"] = False
        result["Anomaly_Score"] = 0
        return result

    mean = result["Actual"].mean()
    std = result["Actual"].std()

    if std == 0 or np.isnan(std):
        result["Anomaly_Score"] = 0
        result["Anomaly"] = False
        return result

    result["Anomaly_Score"] = (
        (result["Actual"] - mean) / std
    )

    result["Anomaly"] = (
        result["Anomaly_Score"].abs() >= 2
    )

    return result


def create_alerts(data):

    alerts_list = []

    for station in data["Power_Station"].dropna().unique():

        sdf = data[
            data["Power_Station"] == station
        ]

        if sdf.empty:
            continue

        programme = safe_mean(sdf["Programme"])
        actual = safe_mean(sdf["Actual"])
        maintenance = (
            safe_mean(sdf["Total_Maintenance"])
            if "Total_Maintenance" in sdf.columns
            else 0
        )

        achievement = (
            actual / programme * 100
            if programme > 0
            else 100
        )

        if achievement < 50:

            alerts_list.append({
                "Station": station,
                "Severity": "CRITICAL",
                "Alert": "Generation severely below programme",
                "Value": f"{achievement:.1f}% achievement"
            })

        elif achievement < 75:

            alerts_list.append({
                "Station": station,
                "Severity": "HIGH",
                "Alert": "Generation below programme",
                "Value": f"{achievement:.1f}% achievement"
            })

        elif achievement < 90:

            alerts_list.append({
                "Station": station,
                "Severity": "MEDIUM",
                "Alert": "Generation slightly below programme",
                "Value": f"{achievement:.1f}% achievement"
            })

        if programme > 0:

            maintenance_ratio = (
                maintenance / programme * 100
            )

            if maintenance_ratio > 30:

                alerts_list.append({
                    "Station": station,
                    "Severity": "HIGH",
                    "Alert": "High maintenance impact",
                    "Value": f"{maintenance_ratio:.1f}% of programme"
                })

    return pd.DataFrame(alerts_list)


def create_health_table(data):

    rows = []

    for station in data["Power_Station"].dropna().unique():

        sdf = data[
            data["Power_Station"] == station
        ]

        programme = sdf["Programme"].sum()
        actual = sdf["Actual"].sum()

        achievement = (
            actual / programme * 100
            if programme > 0
            else 0
        )

        capacity = safe_mean(
            sdf["Monitored_Capacity"]
        )

        available = safe_mean(
            sdf["Available_Capacity"]
        )

        utilization = (
            available / capacity * 100
            if capacity > 0
            else 0
        )

        maintenance = safe_mean(
            sdf["Total_Maintenance"]
        )

        maintenance_index = (
            maintenance / capacity * 100
            if capacity > 0
            else 0
        )

        row = {
            "Power_Station": station,
            "Programme_Achievement_pct": achievement,
            "Capacity_Utilization_pct": utilization,
            "Maintenance_Impact_Index_pct": maintenance_index
        }

        score = health_score(row)

        rows.append({
            "Power Station": station,
            "Achievement (%)": round(achievement, 2),
            "Utilization (%)": round(utilization, 2),
            "Maintenance Impact (%)": round(
                maintenance_index, 2
            ),
            "Health Score": score,
            "Status": health_status(score)
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "Health Score",
            ascending=False
        )

    return result


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("⚡ PowerGenAI")

st.sidebar.caption(
    "AI-Based Power Generation Forecasting "
    "& Power Station Monitoring"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Executive Dashboard",
        "🏭 Power Station Analysis",
        "🔮 Generation Prediction",
        "🎛️ What-If Simulator",
        "🚨 Alerts & Anomalies",
        "🔧 Maintenance Intelligence",
        "🏆 Station Ranking",
        "🧠 Explainable AI",
        "📈 Model Performance",
        "📄 Reports",
        "ℹ️ AI Methodology"
    ]
)

st.sidebar.divider()

if DATA_OK:

    st.sidebar.metric(
        "Stations",
        df["Power_Station"].nunique()
    )

    st.sidebar.metric(
        "Records",
        len(df)
    )

if MODELS_OK:
    st.sidebar.success("AI Model: Ready")
else:
    st.sidebar.warning("AI Model: Not Available")


# ================================================================
# EXECUTIVE DASHBOARD
# ================================================================

if page == "📊 Executive Dashboard":

    st.title("📊 PowerGenAI Executive Dashboard")

    if not DATA_OK:
        st.stop()

    stations = sorted(
        df["Power_Station"].dropna().unique()
    )

    station_filter = st.selectbox(
        "Select Power Station",
        ["All Stations"] + stations
    )

    if station_filter == "All Stations":
        view_df = df.copy()
    else:
        view_df = df[
            df["Power_Station"] == station_filter
        ]

    total_stations = (
        view_df["Power_Station"].nunique()
    )

    total_capacity = (
        view_df["Monitored_Capacity"].sum()
    )

    total_programme = (
        view_df["Programme"].sum()
    )

    total_actual = (
        view_df["Actual"].sum()
    )

    shortfall = (
        view_df.loc[
            view_df["Excess_Shortfall"] < 0,
            "Excess_Shortfall"
        ].sum()
    )

    achievement = (
        total_actual / total_programme * 100
        if total_programme > 0
        else 0
    )

    total_maintenance = (
        view_df["Total_Maintenance"].sum()
        if "Total_Maintenance" in view_df
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏭 Power Stations",
        total_stations
    )

    c2.metric(
        "⚡ Monitored Capacity",
        f"{total_capacity:,.1f} MW"
    )

    c3.metric(
        "📋 Programme",
        f"{total_programme:,.1f} MW"
    )

    c4.metric(
        "🔌 Actual Generation",
        f"{total_actual:,.1f} MW"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "📉 Shortfall",
        f"{shortfall:,.1f} MW"
    )

    c6.metric(
        "🎯 Achievement",
        f"{achievement:.1f}%"
    )

    c7.metric(
        "🔧 Maintenance",
        f"{total_maintenance:,.1f} MW"
    )

    c8.metric(
        "🏥 Overall Status",
        health_status(
            np.clip(
                achievement,
                0,
                100
            )
        )
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            view_df,
            x="Actual",
            nbins=50,
            title="Actual Generation Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if "Total_Maintenance" in view_df.columns:

            maintenance_values = [
                view_df["Planned_Maintenance"].sum(),
                view_df["Forced_Maintenance"].sum(),
                view_df["Other_Reasons"].sum()
            ]

            fig = px.pie(
                names=[
                    "Planned",
                    "Forced",
                    "Other"
                ],
                values=maintenance_values,
                title="Maintenance Composition"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader(
        "📈 Programme vs Actual Generation"
    )

    trend = view_df.copy()

    if "Date" in trend.columns:

        try:
            trend["Date"] = pd.to_datetime(
                trend["Date"]
            )

            trend = (
                trend.groupby("Date")
                [["Programme", "Actual"]]
                .sum()
                .reset_index()
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=trend["Date"],
                    y=trend["Programme"],
                    name="Programme",
                    mode="lines"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=trend["Date"],
                    y=trend["Actual"],
                    name="Actual",
                    mode="lines"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception:
            st.info(
                "Date column could not be interpreted."
            )

    if station_filter == "All Stations":

        st.subheader(
            "🏭 Top Stations by Generation"
        )

        top = (
            station_perf
            .sort_values(
                "Actual_sum",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            top,
            x="Actual_sum",
            y="Power_Station",
            orientation="h",
            title="Top 15 Stations"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ================================================================
# POWER STATION ANALYSIS
# ================================================================

elif page == "🏭 Power Station Analysis":

    st.title("🏭 Power Station Analysis")

    if not DATA_OK:
        st.stop()

    station = st.selectbox(
        "Select Station",
        sorted(
            df["Power_Station"].unique()
        )
    )

    sdf = df[
        df["Power_Station"] == station
    ].copy()

    if sdf.empty:
        st.warning("No data available.")
        st.stop()

    capacity = safe_mean(
        sdf["Monitored_Capacity"]
    )

    available = safe_mean(
        sdf["Available_Capacity"]
    )

    programme = safe_mean(
        sdf["Programme"]
    )

    actual = safe_mean(
        sdf["Actual"]
    )

    achievement = (
        actual / programme * 100
        if programme > 0
        else 0
    )

    utilization = (
        available / capacity * 100
        if capacity > 0
        else 0
    )

    maintenance = safe_mean(
        sdf["Total_Maintenance"]
    )

    health = health_score({
        "Programme_Achievement_pct":
            achievement,
        "Capacity_Utilization_pct":
            utilization,
        "Maintenance_Impact_Index_pct":
            maintenance / capacity * 100
            if capacity > 0 else 0
    })

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Capacity",
        f"{capacity:.1f} MW"
    )

    c2.metric(
        "Actual",
        f"{actual:.1f} MW"
    )

    c3.metric(
        "Achievement",
        f"{achievement:.1f}%"
    )

    c4.metric(
        "Health Score",
        f"{health}/100"
    )

    st.info(
        f"Station Status: {health_status(health)}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            sdf,
            y=[
                "Programme",
                "Actual"
            ],
            title="Programme vs Actual"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        maintenance_cols = [
            "Planned_Maintenance",
            "Forced_Maintenance",
            "Other_Reasons"
        ]

        available_cols = [
            x for x in maintenance_cols
            if x in sdf.columns
        ]

        if available_cols:

            totals = sdf[
                available_cols
            ].sum()

            fig = px.pie(
                names=totals.index,
                values=totals.values,
                title="Maintenance Breakdown"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("📊 Station Statistics")

    st.dataframe(
        sdf.describe().T,
        use_container_width=True
    )


# ================================================================
# GENERATION PREDICTION
# ================================================================

elif page == "🔮 Generation Prediction":

    st.title("🔮 AI Generation Prediction")

    st.caption(
        "Random Forest predicts Actual Generation "
        "from operating conditions."
    )

    if not DATA_OK or not MODELS_OK:

        st.error(
            "Data or Random Forest model unavailable."
        )

        st.stop()

    station_list = sorted(
        df["Power_Station"].unique()
    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:

            station = st.selectbox(
                "Power Station",
                station_list
            )

            capacity = st.number_input(
                "Monitored Capacity (MW)",
                min_value=0.0,
                value=500.0,
                step=10.0
            )

            programme = st.number_input(
                "Programme Generation (MW)",
                min_value=0.0,
                value=400.0,
                step=10.0
            )

        with c2:

            planned = st.number_input(
                "Planned Maintenance (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

            forced = st.number_input(
                "Forced Maintenance (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

            other = st.number_input(
                "Other Reasons (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

        submit = st.form_submit_button(
            "⚡ PREDICT GENERATION",
            use_container_width=True
        )

    if submit:

        try:

            feature_row = prediction.build_feature_row(
                station,
                capacity,
                programme,
                planned,
                forced,
                other
            )

            model = models[
                "Random_Forest"
            ]

            predicted = prediction.predict(
                model,
                feature_row
            )

            shortfall = predicted - programme

            achievement = (
                predicted / programme * 100
                if programme > 0
                else 0
            )

            available = (
                capacity
                - planned
                - forced
                - other
            )

            risk = risk_from_values(
                predicted,
                programme
            )

            mii = (
                feature_row[
                    "Maintenance_Impact_Index"
                ].iloc[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted Generation",
                f"{predicted:.2f} MW"
            )

            c2.metric(
                "Shortfall / Excess",
                f"{shortfall:.2f} MW"
            )

            c3.metric(
                "Achievement",
                f"{achievement:.2f}%"
            )

            c4, c5, c6 = st.columns(3)

            c4.metric(
                "Available Capacity",
                f"{available:.2f} MW"
            )

            c5.metric(
                "Maintenance Impact",
                f"{mii:.2f}"
            )

            c6.metric(
                "Risk Level",
                risk
            )

            st.divider()

            st.subheader(
                "🧠 Prediction Explanation"
            )

            try:

                bias, contributions, recon = (
                    prediction.explain_prediction(
                        model,
                        feature_row
                    )
                )

                series = (
                    pd.Series(contributions)
                    .sort_values(
                        key=abs,
                        ascending=False
                    )
                )

                fig = go.Figure(
                    go.Bar(
                        x=series.values,
                        y=series.index,
                        orientation="h"
                    )
                )

                fig.update_layout(
                    title=(
                        f"Feature Contributions "
                        f"(Base: {bias:.2f} MW)"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.warning(
                    f"Explanation unavailable: {e}"
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ================================================================
# WHAT-IF SIMULATOR
# ================================================================

elif page == "🎛️ What-If Simulator":

    st.title("🎛️ What-If Power Generation Simulator")

    st.write(
        "Simulate different operating conditions and "
        "observe their effect on predicted generation."
    )

    if not DATA_OK or not MODELS_OK:

        st.error(
            "Random Forest model is required."
        )

        st.stop()

    stations = sorted(
        df["Power_Station"].unique()
    )

    station = st.selectbox(
        "Power Station",
        stations,
        key="whatif_station"
    )

    sdf = df[
        df["Power_Station"] == station
    ]

    default_capacity = safe_mean(
        sdf["Monitored_Capacity"]
    )

    default_programme = safe_mean(
        sdf["Programme"]
    )

    st.subheader(
        "⚙️ Current Operating Condition"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        capacity = st.number_input(
            "Capacity (MW)",
            min_value=0.0,
            value=float(default_capacity)
        )

    with c2:

        programme = st.number_input(
            "Programme (MW)",
            min_value=0.0,
            value=float(default_programme)
        )

    with c3:

        planned = st.number_input(
            "Planned Maintenance (MW)",
            min_value=0.0,
            value=0.0
        )

    c4, c5 = st.columns(2)

    with c4:

        forced = st.number_input(
            "Forced Maintenance (MW)",
            min_value=0.0,
            value=0.0
        )

    with c5:

        other = st.number_input(
            "Other Reasons (MW)",
            min_value=0.0,
            value=0.0
        )

    if st.button(
        "🚀 RUN WHAT-IF SIMULATION",
        use_container_width=True
    ):

        try:

            feature_row = prediction.build_feature_row(
                station,
                capacity,
                programme,
                planned,
                forced,
                other
            )

            predicted = prediction.predict(
                models["Random_Forest"],
                feature_row
            )

            achievement = (
                predicted / programme * 100
                if programme > 0
                else 0
            )

            available = (
                capacity
                - planned
                - forced
                - other
            )

            risk = risk_from_values(
                predicted,
                programme
            )

            st.divider()

            st.subheader(
                "📊 Simulation Result"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "AI Predicted Generation",
                f"{predicted:.2f} MW"
            )

            c2.metric(
                "Available Capacity",
                f"{available:.2f} MW"
            )

            c3.metric(
                "Programme Achievement",
                f"{achievement:.2f}%"
            )

            c4.metric(
                "Risk",
                risk
            )

            st.subheader(
                "📈 Operating Condition Analysis"
            )

            comparison = pd.DataFrame({
                "Parameter": [
                    "Capacity",
                    "Programme",
                    "Planned Maintenance",
                    "Forced Maintenance",
                    "Other Reasons",
                    "Available Capacity",
                    "Predicted Generation"
                ],
                "Value (MW)": [
                    capacity,
                    programme,
                    planned,
                    forced,
                    other,
                    available,
                    predicted
                ]
            })

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            if predicted < programme:

                st.warning(
                    f"⚠️ Expected shortfall: "
                    f"{programme - predicted:.2f} MW"
                )

            else:

                st.success(
                    f"✅ Expected excess generation: "
                    f"{predicted - programme:.2f} MW"
                )

        except Exception as e:

            st.error(
                f"Simulation failed: {e}"
            )


# ================================================================
# ALERTS & ANOMALIES
# ================================================================

elif page == "🚨 Alerts & Anomalies":

    st.title("🚨 Alerts & Anomaly Detection")

    if not DATA_OK:
        st.stop()

    alert_df = create_alerts(df)

    st.subheader("🚨 Operational Alerts")

    if alert_df.empty:

        st.success(
            "No major operational alerts detected."
        )

    else:

        severity_filter = st.multiselect(
            "Filter Severity",
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            default=[
                "CRITICAL",
                "HIGH",
                "MEDIUM"
            ]
        )

        filtered = alert_df[
            alert_df["Severity"].isin(
                severity_filter
            )
        ]

        for _, row in filtered.iterrows():

            severity = row["Severity"].lower()

            st.markdown(
                f"""
                <div class="alert-{severity}">
                <b>{row['Severity']}</b><br>
                <b>Station:</b> {row['Station']}<br>
                <b>Alert:</b> {row['Alert']}<br>
                <b>Value:</b> {row['Value']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "🔎 Automatic Anomaly Detection"
    )

    station = st.selectbox(
        "Select Station for Anomaly Analysis",
        sorted(
            df["Power_Station"].unique()
        )
    )

    sdf = df[
        df["Power_Station"] == station
    ].copy()

    anomaly_df = detect_anomalies(
        sdf
    )

    anomaly_count = int(
        anomaly_df["Anomaly"].sum()
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Records",
        len(anomaly_df)
    )

    c2.metric(
        "Detected Anomalies",
        anomaly_count
    )

    fig = px.scatter(
        anomaly_df,
        y="Actual",
        x=anomaly_df.index,
        color="Anomaly",
        title="Generation Anomaly Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if anomaly_count:

        st.warning(
            f"{anomaly_count} unusual generation "
            f"records detected."
        )

        st.dataframe(
            anomaly_df[
                anomaly_df["Anomaly"]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No statistically significant "
            "generation anomalies detected."
        )


# ================================================================
# MAINTENANCE INTELLIGENCE
# ================================================================

elif page == "🔧 Maintenance Intelligence":

    st.title("🔧 Maintenance Intelligence")

    if not DATA_OK:
        st.stop()

    maintenance_cols = [
        "Planned_Maintenance",
        "Forced_Maintenance",
        "Other_Reasons"
    ]

    totals = {
        col: df[col].sum()
        for col in maintenance_cols
        if col in df.columns
    }

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Planned Maintenance",
        f"{totals.get('Planned_Maintenance', 0):,.1f} MW"
    )

    c2.metric(
        "Forced Maintenance",
        f"{totals.get('Forced_Maintenance', 0):,.1f} MW"
    )

    c3.metric(
        "Other Reasons",
        f"{totals.get('Other_Reasons', 0):,.1f} MW"
    )

    st.divider()

    health_df = create_health_table(
        df
    )

    st.subheader(
        "🏭 Maintenance Risk Ranking"
    )

    if not health_df.empty:

        maintenance_rank = (
            health_df
            .sort_values(
                "Maintenance Impact (%)",
                ascending=False
            )
        )

        fig = px.bar(
            maintenance_rank.head(15),
            x="Power Station",
            y="Maintenance Impact (%)",
            title="Highest Maintenance Impact"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            maintenance_rank,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "💡 Automated Maintenance Recommendations"
    )

    for _, row in health_df.iterrows():

        station = row["Power Station"]
        maintenance = row[
            "Maintenance Impact (%)"
        ]

        achievement = row[
            "Achievement (%)"
        ]

        if maintenance > 30:

            st.error(
                f"🔴 **{station}**: High maintenance impact "
                f"({maintenance:.1f}%). Investigate recurring "
                f"outages and prioritize maintenance planning."
            )

        elif maintenance > 15:

            st.warning(
                f"🟠 **{station}**: Moderate maintenance impact "
                f"({maintenance:.1f}%). Monitor equipment performance."
            )

        elif achievement < 75:

            st.warning(
                f"🟡 **{station}**: Low generation achievement "
                f"({achievement:.1f}%). Investigate operational losses."
            )


# ================================================================
# STATION RANKING
# ================================================================

elif page == "🏆 Station Ranking":

    st.title("🏆 Power Station Performance Ranking")

    if not DATA_OK:
        st.stop()

    health_df = create_health_table(
        df
    )

    if health_df.empty:
        st.warning(
            "Unable to calculate station ranking."
        )
        st.stop()

    top = health_df.iloc[0]

    st.success(
        f"🏆 Best Performing Station: "
        f"**{top['Power Station']}** "
        f"with Health Score "
        f"**{top['Health Score']}/100**"
    )

    st.dataframe(
        health_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        health_df,
        x="Power Station",
        y="Health Score",
        color="Status",
        title="Station Health Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "📊 Performance Comparison"
    )

    selected_metric = st.selectbox(
        "Select KPI",
        [
            "Achievement (%)",
            "Utilization (%)",
            "Maintenance Impact (%)"
        ]
    )

    fig = px.bar(
        health_df.sort_values(
            selected_metric,
            ascending=False
        ),
        x="Power Station",
        y=selected_metric,
        title=selected_metric
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ================================================================
# EXPLAINABLE AI
# ================================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    st.caption(
        "Understand which features influence "
        "Random Forest predictions."
    )

    if not feature_importance_df.empty:

        st.subheader(
            "🌍 Global Feature Importance"
        )

        top10 = (
            feature_importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top10,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Important Features"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance file not found."
        )

    st.divider()

    if DATA_OK and MODELS_OK:

        station = st.selectbox(
            "Station",
            sorted(
                df["Power_Station"].unique()
            )
        )

        sdf = df[
            df["Power_Station"] == station
        ].reset_index(drop=True)

        if not sdf.empty:

            idx = st.slider(
                "Record",
                0,
                len(sdf) - 1,
                0
            )

            row = sdf.iloc[idx]

            feature_row = (
                row[
                    MODEL_FEATURE_COLUMNS
                ]
                .to_frame()
                .T
            )

            try:

                bias, contribution, prediction_value = (
                    prediction.explain_prediction(
                        models["Random_Forest"],
                        feature_row
                    )
                )

                st.write(
                    f"**Actual:** "
                    f"{row['Actual']:.2f} MW"
                )

                st.write(
                    f"**AI Prediction:** "
                    f"{prediction_value:.2f} MW"
                )

                series = (
                    pd.Series(contribution)
                    .sort_values(
                        key=abs,
                        ascending=False
                    )
                )

                fig = go.Figure(
                    go.Bar(
                        x=series.values,
                        y=series.index,
                        orientation="h"
                    )
                )

                fig.update_layout(
                    title=(
                        f"Individual Feature "
                        f"Contributions "
                        f"(Base: {bias:.2f} MW)"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Explanation failed: {e}"
                )


# ================================================================
# MODEL PERFORMANCE
# ================================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    if model_comparison_df.empty:

        st.warning(
            "Model comparison data unavailable."
        )

    else:

        st.dataframe(
            model_comparison_df,
            use_container_width=True,
            hide_index=True
        )

        best = model_comparison_df.loc[
            model_comparison_df["R2"].idxmax()
        ]

        st.success(
            f"🏆 Best Model: **{best['Model']}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                model_comparison_df,
                x="Model",
                y="R2",
                title="R² Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                model_comparison_df,
                x="Model",
                y="MAE",
                title="MAE Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "RMSE" in model_comparison_df.columns:

            fig = px.bar(
                model_comparison_df,
                x="Model",
                y="RMSE",
                title="RMSE Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ================================================================
# REPORTS
# ================================================================

elif page == "📄 Reports":

    st.title("📄 Power Station Performance Reports")

    if not DATA_OK:
        st.stop()

    station = st.selectbox(
        "Select Station",
        sorted(
            df["Power_Station"].unique()
        )
    )

    sdf = df[
        df["Power_Station"] == station
    ]

    programme = sdf["Programme"].sum()
    actual = sdf["Actual"].sum()

    achievement = (
        actual / programme * 100
        if programme > 0
        else 0
    )

    capacity = sdf[
        "Monitored_Capacity"
    ].mean()

    available = sdf[
        "Available_Capacity"
    ].mean()

    utilization = (
        available / capacity * 100
        if capacity > 0
        else 0
    )

    maintenance = (
        sdf["Total_Maintenance"].sum()
        if "Total_Maintenance" in sdf
        else 0
    )

    health = health_score({
        "Programme_Achievement_pct":
            achievement,
        "Capacity_Utilization_pct":
            utilization,
        "Maintenance_Impact_Index_pct":
            maintenance /
            (capacity * len(sdf)) *
            100
            if capacity > 0 else 0
    })

    st.subheader(
        f"📋 {station} Report"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Actual Generation",
        f"{actual:,.1f} MW"
    )

    c2.metric(
        "Programme",
        f"{programme:,.1f} MW"
    )

    c3.metric(
        "Achievement",
        f"{achievement:.1f}%"
    )

    c4.metric(
        "Health Score",
        f"{health}/100"
    )

    st.divider()

    report_df = pd.DataFrame({
        "KPI": [
            "Station",
            "Records",
            "Average Capacity",
            "Average Available Capacity",
            "Total Programme",
            "Total Actual",
            "Programme Achievement",
            "Capacity Utilization",
            "Total Maintenance",
            "Health Score",
            "Status"
        ],
        "Value": [
            station,
            len(sdf),
            f"{capacity:.2f} MW",
            f"{available:.2f} MW",
            f"{programme:.2f} MW",
            f"{actual:.2f} MW",
            f"{achievement:.2f}%",
            f"{utilization:.2f}%",
            f"{maintenance:.2f} MW",
            f"{health}/100",
            health_status(health)
        ]
    })

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Station Report CSV",
        csv_data,
        file_name=(
            f"{station}_PowerGenAI_Report.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    full_csv = sdf.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Complete Station Data",
        full_csv,
        file_name=(
            f"{station}_Complete_Data.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ================================================================
# AI METHODOLOGY
# ================================================================

elif page == "ℹ️ AI Methodology":

    st.title("ℹ️ PowerGenAI — AI Methodology")

    st.subheader(
        "🔄 System Architecture"
    )

    st.code(
        """
        Historical Power Station Data
                    │
                    ▼
        Data Preprocessing
                    │
                    ▼
        Feature Engineering
                    │
                    ▼
        ┌─────────────────────────┐
        │ Machine Learning Models │
        ├─────────────────────────┤
        │ Random Forest           │
        │ Linear Regression       │
        │ Gradient Boosting       │
        │ HistGradientBoosting    │
        └─────────────────────────┘
                    │
                    ▼
        Generation Prediction
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Forecast   Anomaly   Maintenance
          │      Detection   Analysis
          │         │         │
          └─────────┼─────────┘
                    ▼
          Decision Support
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        Alerts    Health     Reports
                  Score
        """,
        language="text"
    )

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        """
        PowerGenAI uses machine learning models to estimate
        actual power generation from operating conditions such
        as monitored capacity, programme generation and
        maintenance-related parameters.
        """
    )

    if MODELS_OK:

        st.success(
            "Random Forest prediction model is available."
        )

    else:

        st.warning(
            "Random Forest model is currently unavailable."
        )

    st.subheader(
        "🎯 Main Objectives"
    )

    objectives = [
        "Predict actual power generation",
        "Monitor power station performance",
        "Identify generation shortfalls",
        "Detect abnormal generation behaviour",
        "Measure maintenance impact",
        "Rank power stations",
        "Provide operational alerts",
        "Explain AI predictions",
        "Support what-if operational decisions",
        "Generate performance reports"
    ]

    for objective in objectives:
        st.write(
            f"✅ {objective}"
        )

    st.subheader(
        "📊 Key Performance Indicators"
    )

    kpis = pd.DataFrame({
        "KPI": [
            "Programme Achievement",
            "Capacity Utilization",
            "Maintenance Impact",
            "Health Score",
            "Generation Shortfall",
            "Prediction Accuracy"
        ],
        "Purpose": [
            "Measures actual generation against programme",
            "Measures utilization of available capacity",
            "Measures effect of maintenance on generation",
            "Overall station performance indicator",
            "Identifies generation below target",
            "Measures machine-learning performance"
        ]
    })

    st.dataframe(
        kpis,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "🧠 Explainable AI"
    )

    st.write(
        """
        The system uses feature contribution analysis to
        identify which operating variables push a prediction
        higher or lower. This helps operators understand the
        reason behind an AI prediction instead of treating the
        model as a black box.
        """
    )

    st.subheader(
        "🚨 Decision Support"
    )

    st.info(
        """
        PowerGenAI combines machine learning, performance
        analytics, anomaly detection, maintenance analysis,
        health scoring and what-if simulation into a unified
        power-station decision-support platform.
        """
    )
