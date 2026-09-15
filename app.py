"""
PowerGenAI
AI-Based Power Generation Forecasting and
Power Station Performance Monitoring System

Run:
streamlit run app.py

or from project root:
streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

# If app.py is inside dashboard/, move one level up
if os.path.basename(CURRENT_DIR).lower() == "dashboard":
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    import prediction
    import analytics
    import alerts
    import utils
    from features import MODEL_FEATURE_COLUMNS

    IMPORTS_OK = True

except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerGenAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #07111f 0%,
            #0b1728 45%,
            #101d31 100%
        );
    }

    /* Main content */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #06101d 0%,
            #0b1828 100%
        );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5edf7;
    }

    /* Titles */
    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        color: #cbd5e1;
    }

    /* KPI cards */
    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(30, 64, 175, 0.25),
            rgba(15, 23, 42, 0.75)
        );
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96,165,250,0.45);
        box-shadow: 0 14px 35px rgba(0,0,0,0.35);
    }

    .metric-title {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #60a5fa;
        margin-top: 5px;
    }

    /* Hero */
    .hero {
        background:
            radial-gradient(
                circle at 80% 20%,
                rgba(59,130,246,0.25),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #0f2745,
                #0a1628
            );
        border: 1px solid rgba(96,165,250,0.2);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 900;
        color: white;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #93c5fd;
        font-size: 1rem;
    }

    /* Section cards */
    .section-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Status badges */
    .status-good {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.14);
        color: #86efac;
        font-weight: 700;
    }

    .status-warning {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(245,158,11,0.14);
        color: #fcd34d;
        font-weight: 700;
    }

    .status-danger {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(239,68,68,0.14);
        color: #fca5a5;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(96,165,250,0.3);
        background: linear-gradient(135deg,#2563eb,#1d4ed8);
        color: white;
        font-weight: 700;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37,99,235,0.3);
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(148,163,184,0.12);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_mw(value):
    """Safe MW formatting."""
    try:
        return f"{float(value):,.2f} MW"
    except Exception:
        return "N/A"


def fmt_pct(value):
    """Safe percentage formatting."""
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "N/A"


def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def risk_badge(risk):
    colors = {
        "LOW": "status-good",
        "MEDIUM": "status-warning",
        "HIGH": "status-warning",
        "CRITICAL": "status-danger"
    }

    css = colors.get(risk, "status-warning")

    st.markdown(
        f'<span class="{css}">{risk}</span>',
        unsafe_allow_html=True
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    features_path = os.path.join(
        DATA_DIR,
        "powergeneration_features.csv"
    )

    station_path = os.path.join(
        DATA_DIR,
        "station_performance_summary.csv"
    )

    df = pd.read_csv(features_path)
    station_perf = pd.read_csv(station_path)

    return df, station_perf


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    model_names = [
        "Random_Forest",
        "Linear_Regression",
        "Gradient_Boosting",
        "HistGB_XGBoost_substitute"
    ]

    models = {}
    errors = {}

    for name in model_names:

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


# ============================================================
# MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    path = os.path.join(
        MODELS_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    path = os.path.join(
        MODELS_DIR,
        "feature_importance.csv"
    )

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# INITIALIZATION
# ============================================================

if not IMPORTS_OK:

    st.error("❌ Project modules could not be imported.")

    st.code(
        IMPORT_ERROR,
        language="text"
    )

    st.info(
        "Check that prediction.py, analytics.py, alerts.py, "
        "utils.py and features.py are present in your project."
    )

    st.stop()


try:

    df, station_perf = load_data()

    DATA_OK = True

except Exception as e:

    DATA_OK = False

    st.error("❌ Data could not be loaded.")

    st.code(str(e))

    st.info(
        "Required files:\n"
        "data/processed/powergeneration_features.csv\n"
        "data/processed/station_performance_summary.csv"
    )

    st.stop()


try:

    models, model_errors = load_models()

    MODELS_OK = "Random_Forest" in models

except Exception as e:

    models = {}
    model_errors = {"General": str(e)}
    MODELS_OK = False


model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:10px 0 20px 0;">
            <div style="font-size:3rem;">⚡</div>
            <div style="font-size:1.7rem;font-weight:900;">
                PowerGenAI
            </div>
            <div style="font-size:0.78rem;color:#94a3b8;">
                Intelligent Power Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "📊 Dashboard",
            "🏭 Power Station Analysis",
            "🔮 Generation Prediction",
            "🎛️ What-If Simulator",
            "🔧 Maintenance Analysis",
            "📈 Model Performance",
            "🧠 Explainable AI",
            "🚨 Alert Center",
            "📄 Reports",
            "📋 Data Explorer"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    if DATA_OK:
        st.markdown(
            '<span class="status-good">● DATA ONLINE</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-danger">● DATA ERROR</span>',
            unsafe_allow_html=True
        )

    if MODELS_OK:
        st.markdown(
            '<span class="status-good">● AI MODEL READY</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-danger">● MODEL OFFLINE</span>',
            unsafe_allow_html=True
        )

    st.divider()

    st.caption("PowerGenAI v1.0")
    st.caption("AI-Based Power Generation Forecasting")


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">⚡ PowerGenAI Control Center</div>
            <div class="hero-subtitle">
                AI-powered generation forecasting, station monitoring,
                maintenance intelligence and operational decision support.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    station_filter = st.selectbox(
        "🏭 Station Filter",
        ["All Stations"] +
        sorted(df["Power_Station"].dropna().unique().tolist())
    )

    if station_filter == "All Stations":
        view_df = df.copy()
    else:
        view_df = df[
            df["Power_Station"] == station_filter
        ].copy()

    if view_df.empty:
        st.warning("No data available for this selection.")
        st.stop()

    total_stations = view_df["Power_Station"].nunique()

    total_capacity = (
        view_df["Monitored_Capacity"].sum()
        if "Monitored_Capacity" in view_df
        else 0
    )

    total_programme = view_df["Programme"].sum()
    total_actual = view_df["Actual"].sum()

    shortfall = (
        view_df.loc[
            view_df["Excess_Shortfall"] < 0,
            "Excess_Shortfall"
        ].sum()
    )

    achievement = (
        total_actual / total_programme * 100
        if total_programme != 0
        else 0
    )

    total_maintenance = (
        view_df["Total_Maintenance"].sum()
        if "Total_Maintenance" in view_df
        else 0
    )

    # KPI GRID

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Power Stations",
            f"{total_stations}",
            "Stations monitored"
        )

    with c2:
        metric_card(
            "Monitored Capacity",
            fmt_mw(total_capacity),
            "Installed/monitored"
        )

    with c3:
        metric_card(
            "Programme",
            fmt_mw(total_programme),
            "Target generation"
        )

    with c4:
        metric_card(
            "Actual Generation",
            fmt_mw(total_actual),
            "Recorded output"
        )

    st.write("")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card(
            "Shortfall",
            fmt_mw(shortfall),
            "Negative = deficit"
        )

    with c6:
        metric_card(
            "Achievement",
            fmt_pct(achievement),
            "Programme achieved"
        )

    with c7:
        metric_card(
            "Maintenance",
            fmt_mw(total_maintenance),
            "Total impact"
        )

    with c8:
        metric_card(
            "Records",
            f"{len(view_df):,}",
            "Data points"
        )

    st.divider()

    # Charts

    col1, col2 = st.columns(2)

    with col1:

        section_title(
            "📈 Actual Generation Distribution",
            "Distribution of recorded generation"
        )

        fig = px.histogram(
            view_df,
            x="Actual",
            nbins=50
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        section_title(
            "🔧 Maintenance Composition",
            "Maintenance category contribution"
        )

        maintenance_columns = [
            "Planned_Maintenance",
            "Forced_Maintenance",
            "Other_Reasons"
        ]

        available = [
            c for c in maintenance_columns
            if c in view_df.columns
        ]

        if available:

            values = view_df[available].sum()

            fig = px.pie(
                values=values.values,
                names=values.index,
                hole=0.55
            )

            fig.update_layout(
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Station leaderboard

    if station_filter == "All Stations":

        st.divider()

        section_title(
            "🏆 Station Performance Leaderboard",
            "Top stations based on total actual generation"
        )

        if not station_perf.empty:

            top = station_perf.sort_values(
                "Actual_sum",
                ascending=False
            ).head(15)

            fig = px.bar(
                top,
                x="Actual_sum",
                y="Power_Station",
                orientation="h",
                text_auto=".2s"
            )

            fig.update_layout(
                template="plotly_dark",
                height=500,
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# PAGE 2 — STATION ANALYSIS
# ============================================================

elif page == "🏭 Power Station Analysis":

    st.title("🏭 Power Station Intelligence")

    station = st.selectbox(
        "Select Power Station",
        sorted(df["Power_Station"].unique())
    )

    station_df = df[
        df["Power_Station"] == station
    ].copy()

    station_row = station_perf[
        station_perf["Power_Station"] == station
    ]

    if station_df.empty:
        st.warning("No data found.")
        st.stop()

    avg_programme = station_df["Programme"].mean()
    avg_actual = station_df["Actual"].mean()
    avg_shortfall = station_df["Excess_Shortfall"].mean()

    achievement = (
        avg_actual / avg_programme * 100
        if avg_programme else 0
    )

    capacity = station_df[
        "Monitored_Capacity"
    ].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Capacity",
            fmt_mw(capacity)
        )

    with c2:
        metric_card(
            "Average Programme",
            fmt_mw(avg_programme)
        )

    with c3:
        metric_card(
            "Average Actual",
            fmt_mw(avg_actual)
        )

    with c4:
        metric_card(
            "Achievement",
            fmt_pct(achievement)
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            station_df.reset_index(),
            y="Actual",
            title="Generation Trend"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        cols = [
            "Planned_Maintenance",
            "Forced_Maintenance",
            "Other_Reasons"
        ]

        cols = [
            c for c in cols
            if c in station_df.columns
        ]

        if cols:

            maintenance = station_df[cols].sum()

            fig = px.bar(
                x=maintenance.index,
                y=maintenance.values,
                title="Maintenance Breakdown"
            )

            fig.update_layout(
                template="plotly_dark",
                height=420
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("📊 Station Statistics")

    stats = station_df[
        [
            "Programme",
            "Actual",
            "Excess_Shortfall",
            "Monitored_Capacity"
        ]
    ].describe().T

    st.dataframe(
        stats.round(2),
        use_container_width=True
    )


# ============================================================
# PAGE 3 — PREDICTION
# ============================================================

elif page == "🔮 Generation Prediction":

    st.title("🔮 AI Generation Prediction")

    st.caption(
        "Predict expected actual generation using the trained Random Forest model."
    )

    if not MODELS_OK:

        st.error(
            "Random Forest model is not available."
        )

        if model_errors:
            st.json(model_errors)

        st.stop()

    stations = sorted(
        df["Power_Station"].unique()
    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:

            station = st.selectbox(
                "🏭 Power Station",
                stations
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
                value=50.0,
                step=1.0
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

        submitted = st.form_submit_button(
            "⚡ RUN AI PREDICTION",
            use_container_width=True
        )

    if submitted:

        try:

            feature_row = prediction.build_feature_row(
                station,
                capacity,
                programme,
                planned,
                forced,
                other
            )

            model = models["Random_Forest"]

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

            available = capacity - (
                planned +
                forced +
                other
            )

            risk = prediction.risk_level(
                predicted,
                programme
            )

            st.success(
                "✅ AI prediction completed successfully."
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "Predicted Generation",
                    fmt_mw(predicted),
                    "AI forecast"
                )

            with c2:
                metric_card(
                    "Expected Shortfall/Excess",
                    fmt_mw(shortfall),
                    "Prediction − programme"
                )

            with c3:
                metric_card(
                    "Programme Achievement",
                    fmt_pct(achievement),
                    "Expected performance"
                )

            c4, c5, c6 = st.columns(3)

            with c4:
                metric_card(
                    "Available Capacity",
                    fmt_mw(available)
                )

            with c5:
                metric_card(
                    "Maintenance",
                    fmt_mw(
                        planned +
                        forced +
                        other
                    )
                )

            with c6:

                st.markdown("### Risk")

                risk_badge(risk)

            # Gauge

            st.divider()

            st.subheader(
                "🎯 Prediction vs Programme"
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=achievement,
                    title={
                        "text": "Programme Achievement (%)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 120]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": 100
                        }
                    }
                )
            )

            gauge.update_layout(
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            # Explanation

            st.subheader(
                "🧠 AI Prediction Drivers"
            )

            try:

                bias, contributions, reconstruction = (
                    prediction.explain_prediction(
                        model,
                        feature_row
                    )
                )

                series = pd.Series(
                    contributions
                ).sort_values(
                    key=abs,
                    ascending=False
                )

                fig = go.Figure(
                    go.Bar(
                        x=series.values,
                        y=series.index,
                        orientation="h"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    title="Feature Contribution Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.info(
                    f"Explanation unavailable: {e}"
                )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# PAGE 4 — WHAT IF
# ============================================================

elif page == "🎛️ What-If Simulator":

    st.title("🎛️ What-If Power Generation Simulator")

    st.caption(
        "Experiment with operational conditions and observe predicted generation."
    )

    if not MODELS_OK:
        st.error("AI model unavailable.")
        st.stop()

    station = st.selectbox(
        "Select Station",
        sorted(df["Power_Station"].unique())
    )

    station_data = df[
        df["Power_Station"] == station
    ]

    base_capacity = float(
        station_data["Monitored_Capacity"].mean()
    )

    base_programme = float(
        station_data["Programme"].mean()
    )

    st.subheader(
        "⚙️ Adjust Operating Conditions"
    )

    c1, c2 = st.columns(2)

    with c1:

        programme = st.slider(
            "Programme (MW)",
            0.0,
            max(base_capacity * 1.2, 1),
            min(base_programme, base_capacity * 1.2)
        )

        planned = st.slider(
            "Planned Maintenance (MW)",
            0.0,
            base_capacity,
            0.0
        )

    with c2:

        forced = st.slider(
            "Forced Maintenance (MW)",
            0.0,
            base_capacity,
            0.0
        )

        other = st.slider(
            "Other Reasons (MW)",
            0.0,
            base_capacity,
            0.0
        )

    try:

        feature_row = prediction.build_feature_row(
            station,
            base_capacity,
            programme,
            planned,
            forced,
            other
        )

        predicted = prediction.predict(
            models["Random_Forest"],
            feature_row
        )

        risk = prediction.risk_level(
            predicted,
            programme
        )

        available = base_capacity - (
            planned +
            forced +
            other
        )

        st.divider()

        st.subheader(
            "📊 Scenario Result"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Predicted Actual",
                fmt_mw(predicted)
            )

        with c2:
            metric_card(
                "Programme",
                fmt_mw(programme)
            )

        with c3:
            metric_card(
                "Available Capacity",
                fmt_mw(available)
            )

        with c4:

            st.markdown("### Risk")
            risk_badge(risk)

        # comparison chart

        chart_df = pd.DataFrame({
            "Metric": [
                "Programme",
                "Predicted Actual",
                "Available Capacity"
            ],
            "MW": [
                programme,
                predicted,
                available
            ]
        })

        fig = px.bar(
            chart_df,
            x="Metric",
            y="MW",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Simulation failed: {e}"
        )


# ============================================================
# PAGE 5 — MAINTENANCE
# ============================================================

elif page == "🔧 Maintenance Analysis":

    st.title("🔧 Maintenance Intelligence")

    columns = [
        "Planned_Maintenance",
        "Forced_Maintenance",
        "Other_Reasons"
    ]

    available = [
        c for c in columns
        if c in df.columns
    ]

    if not available:
        st.warning(
            "Maintenance columns not found."
        )
        st.stop()

    totals = df[available].sum()

    total = totals.sum()

    c1, c2, c3 = st.columns(3)

    for i, col in enumerate(available[:3]):

        share = (
            totals[col] / total * 100
            if total
            else 0
        )

        [c1, c2, c3][i].metric(
            col.replace("_", " "),
            fmt_pct(share)
        )

    st.divider()

    fig = px.pie(
        values=totals.values,
        names=totals.index,
        hole=0.5
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Station maintenance ranking

    if not station_perf.empty:

        st.subheader(
            "🏭 Stations with Highest Maintenance Impact"
        )

        if "Total_Maintenance_avg" in station_perf:

            top = station_perf.sort_values(
                "Total_Maintenance_avg",
                ascending=False
            ).head(15)

            fig = px.bar(
                top,
                x="Total_Maintenance_avg",
                y="Power_Station",
                orientation="h",
                text_auto=".2f"
            )

            fig.update_layout(
                template="plotly_dark",
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# PAGE 6 — MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    if model_comparison_df.empty:

        st.warning(
            "Model comparison file not found."
        )

        st.info(
            "Run your model evaluation/training phase first."
        )

    else:

        st.dataframe(
            model_comparison_df,
            use_container_width=True,
            hide_index=True
        )

        if "R2" in model_comparison_df.columns:

            best = model_comparison_df.loc[
                model_comparison_df["R2"].idxmax()
            ]

            st.success(
                f"🏆 Best model: "
                f"{best['Model']} "
                f"(R² = {best['R2']:.4f})"
            )

            fig = px.bar(
                model_comparison_df,
                x="Model",
                y="R2",
                text_auto=".3f"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "MAE" in model_comparison_df.columns:

            fig = px.bar(
                model_comparison_df,
                x="Model",
                y="MAE",
                text_auto=".2f"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# PAGE 7 — EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    st.caption(
        "Understand which operational features influence model predictions."
    )

    if feature_importance_df.empty:

        st.warning(
            "Feature importance file not found."
        )

    else:

        top = feature_importance_df.head(15)

        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            text_auto=".3f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    if MODELS_OK:

        station = st.selectbox(
            "Select Station",
            sorted(df["Power_Station"].unique()),
            key="xai_station"
        )

        records = df[
            df["Power_Station"] == station
        ].reset_index(drop=True)

        if not records.empty:

            index = st.slider(
                "Select Record",
                0,
                len(records) - 1,
                0
            )

            row = records.iloc[index]

            feature_row = row[
                MODEL_FEATURE_COLUMNS
            ].to_frame().T

            try:

                bias, contributions, prediction_value = (
                    prediction.explain_prediction(
                        models["Random_Forest"],
                        feature_row
                    )
                )

                c1, c2 = st.columns(2)

                with c1:
                    metric_card(
                        "Actual",
                        fmt_mw(row["Actual"])
                    )

                with c2:
                    metric_card(
                        "Model Prediction",
                        fmt_mw(prediction_value)
                    )

                series = pd.Series(
                    contributions
                ).sort_values(
                    key=abs,
                    ascending=False
                )

                fig = px.bar(
                    x=series.values,
                    y=series.index,
                    orientation="h"
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"XAI explanation failed: {e}"
                )


# ============================================================
# PAGE 8 — ALERT CENTER
# ============================================================

elif page == "🚨 Alert Center":

    st.title("🚨 Power Station Alert Center")

    st.caption(
        "Identify records requiring operational attention."
    )

    try:

        cfg = alerts.AlertConfig()

        all_alerts = alerts.evaluate_dataframe(
            df,
            cfg
        )

        if all_alerts.empty:

            st.success(
                "✅ No critical alerts detected."
            )

        else:

            st.warning(
                f"⚠️ {len(all_alerts)} alert records detected."
            )

            st.dataframe(
                all_alerts,
                use_container_width=True,
                hide_index=True
            )

            if "type" in all_alerts.columns:

                counts = (
                    all_alerts["type"]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "Alert Type",
                    "Count"
                ]

                fig = px.bar(
                    counts,
                    x="Alert Type",
                    y="Count",
                    text_auto=True
                )

                fig.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Alert system failed: {e}"
        )


# ============================================================
# PAGE 9 — REPORTS
# ============================================================

elif page == "📄 Reports":

    st.title("📄 Automated Performance Reports")

    station = st.selectbox(
        "Select Station",
        sorted(df["Power_Station"].unique()),
        key="report_station"
    )

    station_df = df[
        df["Power_Station"] == station
    ]

    if st.button(
        "📑 Generate Performance Report",
        use_container_width=True
    ):

        try:

            cfg = alerts.AlertConfig()

            station_alerts = alerts.evaluate_dataframe(
                station_df,
                cfg
            )

            programme = station_df["Programme"].mean()
            actual = station_df["Actual"].mean()

            achievement = (
                actual / programme * 100
                if programme
                else 0
            )

            report = f"""
# PowerGenAI Performance Report

## Station

{station}

## Generation Performance

Records analyzed: {len(station_df)}

Average Programme: {fmt_mw(programme)}

Average Actual Generation: {fmt_mw(actual)}

Average Shortfall/Excess:
{fmt_mw(station_df["Excess_Shortfall"].mean())}

Programme Achievement:
{fmt_pct(achievement)}

## Maintenance

Average Planned Maintenance:
{fmt_mw(station_df["Planned_Maintenance"].mean())}

Average Forced Maintenance:
{fmt_mw(station_df["Forced_Maintenance"].mean())}

Average Other Reasons:
{fmt_mw(station_df["Other_Reasons"].mean())}

## Alerts

Total Alerts:
{len(station_alerts)}

---

Generated by PowerGenAI
AI-Based Power Generation Forecasting and Monitoring System
"""

            st.markdown(report)

            st.download_button(
                "⬇️ Download Report",
                report,
                file_name=f"PowerGenAI_{station}_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Report generation failed: {e}"
            )


# ============================================================
# PAGE 10 — DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.title("📋 Power Generation Data Explorer")

    st.caption(
        "Explore the processed dataset used by PowerGenAI."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Rows",
            f"{len(df):,}"
        )

    with c2:
        metric_card(
            "Columns",
            f"{len(df.columns):,}"
        )

    with c3:
        metric_card(
            "Stations",
            f"{df['Power_Station'].nunique():,}"
        )

    st.divider()

    station = st.selectbox(
        "Filter Station",
        ["All"] +
        sorted(df["Power_Station"].unique())
    )

    if station != "All":

        filtered = df[
            df["Power_Station"] == station
        ]

    else:

        filtered = df

    search = st.text_input(
        "🔎 Search within data"
    )

    if search:

        mask = filtered.astype(str).apply(
            lambda col:
            col.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(axis=1)

        filtered = filtered[mask]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        csv,
        "PowerGenAI_filtered_data.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>
    <div style="
        text-align:center;
        color:#64748b;
        font-size:0.8rem;
        padding:15px;
    ">
        ⚡ <b>PowerGenAI</b> —
        AI-Based Power Generation Forecasting &
        Power Station Performance Monitoring System
        <br>
        Built for intelligent energy analytics and decision support.
    </div>
    """,
    unsafe_allow_html=True
)
