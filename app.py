"""
===============================================================
POWERGENAI
AI-Based Power Generation Forecasting & Power Station
Performance Monitoring System

Professional Streamlit Dashboard

Run:
    streamlit run app.py

or:
    streamlit run dashboard/app.py
===============================================================
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ===============================================================
# PROJECT PATH
# ===============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(CURRENT_DIR).lower() == "dashboard":
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

sys.path.insert(0, PROJECT_ROOT)


# ===============================================================
# PROJECT IMPORTS
# ===============================================================

import prediction
import analytics
import alerts
import utils

from features import MODEL_FEATURE_COLUMNS


# ===============================================================
# PAGE CONFIG
# ===============================================================

st.set_page_config(
    page_title="PowerGenAI | Power Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===============================================================
# PATHS
# ===============================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ===============================================================
# PROFESSIONAL UI CSS
# ===============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {

    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(37,99,235,0.16),
            transparent 27%
        ),

        radial-gradient(
            circle at 95% 0%,
            rgba(6,182,212,0.12),
            transparent 25%
        ),

        linear-gradient(
            135deg,
            #050b14,
            #071525 45%,
            #06111f
        );

    color: #e6eef8;
}


.main .block-container {

    max-width: 1550px;

    padding-top: 1.2rem;
    padding-bottom: 4rem;

}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #071321 0%,
            #091a2d 50%,
            #06111e 100%
        );

    border-right:
        1px solid rgba(148,163,184,0.14);

}


section[data-testid="stSidebar"] * {

    color: #dbeafe;

}


section[data-testid="stSidebar"] .stRadio label {

    border-radius: 10px;

}


/* =========================================================
   HEADINGS
========================================================= */

h1 {

    color: #ffffff !important;

    font-weight: 850 !important;

    letter-spacing: -0.8px;

}


h2 {

    color: #eaf4ff !important;

    font-weight: 800 !important;

}


h3 {

    color: #dceeff !important;

    font-weight: 750 !important;

}


/* =========================================================
   HERO
========================================================= */

.pg-hero {

    position: relative;

    overflow: hidden;

    padding: 30px 32px;

    margin-bottom: 25px;

    border-radius: 24px;

    background:

        linear-gradient(
            135deg,
            rgba(30,64,175,0.50),
            rgba(8,47,73,0.55)
        );

    border:
        1px solid rgba(96,165,250,0.24);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.28);

}


.pg-hero:after {

    content: "⚡";

    position: absolute;

    right: 40px;
    top: 5px;

    font-size: 110px;

    opacity: 0.08;

}


.pg-hero-title {

    font-size: 38px;

    font-weight: 900;

    color: #ffffff;

    letter-spacing: -1px;

}


.pg-hero-subtitle {

    color: #a8c3da;

    font-size: 15px;

    margin-top: 6px;

}


.pg-status-row {

    display: flex;

    gap: 9px;

    flex-wrap: wrap;

    margin-top: 18px;

}


.pg-status {

    padding: 7px 13px;

    border-radius: 999px;

    background:
        rgba(3,15,30,0.62);

    border:
        1px solid rgba(148,163,184,0.18);

    font-size: 11px;

    font-weight: 800;

    letter-spacing: .4px;

}


.green {

    color: #86efac;

}


.blue {

    color: #93c5fd;

}


.yellow {

    color: #fde68a;

}


.red {

    color: #fca5a5;

}


/* =========================================================
   KPI CARDS
========================================================= */

.pg-kpi {

    min-height: 145px;

    padding: 21px;

    border-radius: 18px;

    background:

        linear-gradient(
            145deg,
            rgba(15,31,51,0.96),
            rgba(8,20,35,0.96)
        );

    border:
        1px solid rgba(148,163,184,0.13);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);

    transition:
        transform .2s ease,
        border .2s ease;

}


.pg-kpi:hover {

    transform: translateY(-3px);

    border:
        1px solid rgba(56,189,248,0.35);

}


.pg-kpi-label {

    color: #88a4bd;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;

    font-weight: 800;

}


.pg-kpi-value {

    color: #ffffff;

    font-size: 29px;

    font-weight: 900;

    margin-top: 8px;

}


.pg-kpi-sub {

    color: #68829a;

    font-size: 11px;

    margin-top: 5px;

}


/* =========================================================
   SECTION
========================================================= */

.pg-section {

    margin-top: 27px;

    margin-bottom: 15px;

    padding-left: 13px;

    border-left:
        4px solid #38bdf8;

}


.pg-section-title {

    color: #e9f5ff;

    font-size: 19px;

    font-weight: 850;

}


.pg-section-sub {

    color: #7893aa;

    font-size: 12px;

    margin-top: 3px;

}


/* =========================================================
   HEALTH SCORE
========================================================= */

.health-card {

    padding: 20px;

    border-radius: 18px;

    background:
        rgba(10,25,42,0.82);

    border:
        1px solid rgba(148,163,184,0.13);

}


.health-score {

    font-size: 42px;

    font-weight: 900;

    color: #ffffff;

}


.health-label {

    color: #8ba6bd;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: .8px;

}


/* =========================================================
   ALERTS
========================================================= */

.pg-alert {

    padding: 14px 16px;

    margin: 7px 0;

    border-radius: 13px;

    background:
        rgba(15,23,42,0.78);

    border:
        1px solid rgba(148,163,184,0.12);

}


.pg-alert-critical {

    border-left:
        4px solid #ef4444;

}


.pg-alert-high {

    border-left:
        4px solid #f97316;

}


.pg-alert-medium {

    border-left:
        4px solid #eab308;

}


.pg-alert-low {

    border-left:
        4px solid #22c55e;

}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {

    min-height: 43px;

    border-radius: 11px;

    font-weight: 800;

    border:
        1px solid rgba(96,165,250,0.30);

}


.stDownloadButton > button {

    min-height: 43px;

    border-radius: 11px;

    font-weight: 800;

}


/* =========================================================
   FOOTER
========================================================= */

.pg-footer {

    margin-top: 45px;

    padding-top: 18px;

    border-top:
        1px solid rgba(148,163,184,0.12);

    text-align: center;

    color: #58738b;

    font-size: 11px;

}

</style>
""",
    unsafe_allow_html=True
)


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def kpi(label, value, subtitle=""):

    return f"""
    <div class="pg-kpi">

        <div class="pg-kpi-label">
            {label}
        </div>

        <div class="pg-kpi-value">
            {value}
        </div>

        <div class="pg-kpi-sub">
            {subtitle}
        </div>

    </div>
    """


def section(title, subtitle=""):

    st.markdown(
        f"""
        <div class="pg-section">

            <div class="pg-section-title">
                {title}
            </div>

            <div class="pg-section-sub">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def hero(title, subtitle, statuses):

    pills = ""

    for text, css in statuses:

        pills += f"""
        <span class="pg-status {css}">
            {text}
        </span>
        """

    st.markdown(
        f"""
        <div class="pg-hero">

            <div class="pg-hero-title">
                {title}
            </div>

            <div class="pg-hero-subtitle">
                {subtitle}
            </div>

            <div class="pg-status-row">
                {pills}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def dark_plot(fig, height=430):

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(8,20,35,0.35)",

        font=dict(
            color="#cbd5e1"
        ),

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        ),

        height=height,

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def risk_icon(risk):

    mapping = {

        "LOW": "🟢",

        "MEDIUM": "🟡",

        "HIGH": "🟠",

        "CRITICAL": "🔴"

    }

    return f"{mapping.get(risk, '⚪')} {risk}"


def safe_pct(value):

    try:

        if pd.isna(value):
            return "—"

        return f"{float(value):.1f}%"

    except Exception:

        return "—"


def station_health(row):

    """
    Composite station health score.

    Uses:
    - Programme achievement
    - Capacity utilization
    - Maintenance impact

    This is an application-level indicator,
    not a new ML model.
    """

    achievement = float(
        row.get("Programme_Achievement_pct", 0)
    )

    utilization = float(
        row.get("Capacity_Utilization_pct", 0)
    )

    maintenance = float(
        row.get("Maintenance_Impact_Index_pct", 0)
    )

    achievement_score = np.clip(
        achievement,
        0,
        100
    )

    utilization_score = np.clip(
        utilization,
        0,
        100
    )

    maintenance_score = 100 - np.clip(
        maintenance,
        0,
        100
    )

    score = (

        0.45 * achievement_score

        +

        0.35 * utilization_score

        +

        0.20 * maintenance_score

    )

    return round(
        float(np.clip(score, 0, 100)),
        1
    )


# ===============================================================
# LOAD DATA
# ===============================================================

@st.cache_data
def load_data():

    data_file = os.path.join(
        DATA_DIR,
        "powergeneration_features.csv"
    )

    station_file = os.path.join(
        DATA_DIR,
        "station_performance_summary.csv"
    )

    df = pd.read_csv(
        data_file
    )

    station_perf = pd.read_csv(
        station_file
    )

    return df, station_perf


# ===============================================================
# LOAD MODELS
# ===============================================================

@st.cache_resource
def load_models():

    models = {}

    errors = {}

    model_names = [

        "Random_Forest",

        "Linear_Regression",

        "Gradient_Boosting",

        "HistGB_XGBoost_substitute"

    ]

    for name in model_names:

        path = os.path.join(
            MODELS_DIR,
            f"{name}.joblib"
        )

        if not os.path.exists(path):
            continue

        try:

            models[name] = prediction.load_model(
                name
            )

        except Exception as e:

            errors[name] = str(e)

    return models, errors


# ===============================================================
# LOAD MODEL COMPARISON
# ===============================================================

@st.cache_data
def load_model_comparison():

    path = os.path.join(
        MODELS_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


# ===============================================================
# LOAD FEATURE IMPORTANCE
# ===============================================================

@st.cache_data
def load_feature_importance():

    path = os.path.join(
        MODELS_DIR,
        "feature_importance.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


# ===============================================================
# DATA STATUS
# ===============================================================

try:

    df, station_perf = load_data()

    DATA_OK = True

except Exception as e:

    DATA_OK = False

    df = pd.DataFrame()

    station_perf = pd.DataFrame()

    st.error(
        f"⚠️ Could not load processed data: {e}"
    )


# ===============================================================
# MODEL STATUS
# ===============================================================

try:

    models, model_errors = load_models()

    MODELS_OK = (
        "Random_Forest"
        in models
    )

except Exception as e:

    models = {}

    model_errors = {
        "General": str(e)
    }

    MODELS_OK = False


model_comparison_df = load_model_comparison()

feature_importance_df = load_feature_importance()


# ===============================================================
# SIDEBAR
# ===============================================================

st.sidebar.markdown(
    """
    <div style="
        padding:10px 0 22px 0;
    ">

        <div style="
            font-size:29px;
            font-weight:900;
            color:white;
        ">
            ⚡ PowerGenAI
        </div>

        <div style="
            font-size:10px;
            color:#6f8ba3;
            letter-spacing:1.2px;
            margin-top:4px;
        ">
            AI POWER ANALYTICS PLATFORM
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
    <div style="
        font-size:10px;
        font-weight:800;
        letter-spacing:1px;
        color:#607991;
        margin-bottom:7px;
    ">
        NAVIGATION
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "",
    [

        "📊 Dashboard",

        "🏭 Power Station Analysis",

        "🔮 Generation Prediction",

        "🎛️ What-If Simulator",

        "🚨 Alerts & Risk",

        "🔧 Maintenance Analysis",

        "📈 Model Performance",

        "🧠 Explainable AI",

        "📄 Reports",

    ]
)


st.sidebar.divider()


st.sidebar.markdown(
    """
    <div style="
        font-size:10px;
        font-weight:800;
        letter-spacing:1px;
        color:#607991;
        margin-bottom:10px;
    ">
        SYSTEM STATUS
    </div>
    """,
    unsafe_allow_html=True
)


if DATA_OK:

    st.sidebar.success(
        f"🟢 DATA ONLINE · {len(df):,} records"
    )

else:

    st.sidebar.error(
        "🔴 DATA OFFLINE"
    )


if MODELS_OK:

    st.sidebar.success(
        "🟢 AI MODEL READY"
    )

else:

    st.sidebar.warning(
        "🟡 AI MODEL OFFLINE"
    )


st.sidebar.divider()


st.sidebar.caption(
    "PowerGenAI v1.0"
)

st.sidebar.caption(
    "AI Forecasting • Monitoring • XAI"
)


# ===============================================================
# DASHBOARD
# ===============================================================

if page == "📊 Dashboard":

    hero(

        "⚡ PowerGenAI Command Center",

        "AI-Based Power Generation Forecasting & "
        "Power Station Performance Monitoring",

        [

            (
                "🟢 DATA ONLINE"
                if DATA_OK
                else
                "🔴 DATA ERROR",

                "green"
                if DATA_OK
                else
                "red"
            ),

            (
                "🟢 AI MODEL READY"
                if MODELS_OK
                else
                "🟡 AI MODEL OFFLINE",

                "green"
                if MODELS_OK
                else
                "yellow"
            ),

            (
                "🔵 ANALYTICS ACTIVE",
                "blue"
            )

        ]

    )


    if not DATA_OK:

        st.stop()


    station_filter = st.selectbox(

        "🏭 Filter by Power Station",

        [
            "All Stations"
        ]
        +
        sorted(
            df[
                "Power_Station"
            ]
            .dropna()
            .unique()
            .tolist()
        )

    )


    if station_filter == "All Stations":

        view_df = df

    else:

        view_df = df[
            df[
                "Power_Station"
            ]
            ==
            station_filter
        ]


    if view_df.empty:

        st.warning(
            "No records match the selected filter."
        )

        st.stop()


    # -----------------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------------

    total_stations = (
        view_df[
            "Power_Station"
        ]
        .nunique()
    )


    total_capacity = (
        analytics.total_monitored_capacity(
            view_df
        )
    )


    total_programme = (
        view_df[
            "Programme"
        ]
        .sum()
    )


    total_actual = (
        view_df[
            "Actual"
        ]
        .sum()
    )


    total_shortfall = (

        view_df.loc[
            view_df[
                "Excess_Shortfall"
            ]
            < 0,

            "Excess_Shortfall"

        ]
        .sum()

    )


    achievement = (

        total_actual
        /
        total_programme
        *
        100

        if total_programme > 0
        else np.nan

    )


    total_maintenance = (
        view_df[
            "Total_Maintenance"
        ]
        .sum()
    )


    # -----------------------------------------------------------
    # KPI ROW
    # -----------------------------------------------------------

    section(
        "Executive Overview",
        "High-level operational performance indicators"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.markdown(
        kpi(
            "Power Stations",
            f"{total_stations}",
            "Monitored stations"
        ),
        unsafe_allow_html=True
    )


    c2.markdown(
        kpi(
            "Monitored Capacity",
            utils.fmt_mw(
                total_capacity
            ),
            "MW"
        ),
        unsafe_allow_html=True
    )


    c3.markdown(
        kpi(
            "Programme Generation",
            utils.fmt_mw(
                total_programme
            ),
            "Total programmed generation"
        ),
        unsafe_allow_html=True
    )


    c4.markdown(
        kpi(
            "Actual Generation",
            utils.fmt_mw(
                total_actual
            ),
            "Total actual generation"
        ),
        unsafe_allow_html=True
    )


    c5, c6, c7, c8 = st.columns(4)


    c5.markdown(
        kpi(
            "Shortfall",
            utils.fmt_mw(
                total_shortfall
            ),
            "Negative generation gap"
        ),
        unsafe_allow_html=True
    )


    c6.markdown(
        kpi(
            "Programme Achievement",
            utils.fmt_pct(
                achievement
            ),
            "Actual / Programme"
        ),
        unsafe_allow_html=True
    )


    c7.markdown(
        kpi(
            "Maintenance Impact",
            utils.fmt_mw(
                total_maintenance
            ),
            "Total maintenance"
        ),
        unsafe_allow_html=True
    )


    # -----------------------------------------------------------
    # STATION HEALTH
    # -----------------------------------------------------------

    if (
        not station_perf.empty
        and
        "Programme_Achievement_pct"
        in station_perf.columns
    ):

        health_values = []

        for _, row in station_perf.iterrows():

            health_values.append(
                station_health(row)
            )


        average_health = (
            np.mean(
                health_values
            )
            if health_values
            else 0
        )


    else:

        average_health = 0


    c8.markdown(
        kpi(
            "Fleet Health",
            f"{average_health:.1f}/100",
            "Composite station health"
        ),
        unsafe_allow_html=True
    )


    # -----------------------------------------------------------
    # CHARTS
    # -----------------------------------------------------------

    section(
        "Generation Intelligence",
        "Visual analysis of actual generation and maintenance"
    )


    col1, col2 = st.columns(2)


    with col1:

        fig = px.histogram(

            view_df,

            x="Actual",

            nbins=55,

            title="Actual Generation Distribution"

        )

        dark_plot(
            fig
        )


    with col2:

        comp = (
            analytics
            .maintenance_composition(
                view_df
            )
        )


        fig = px.pie(

            names=list(
                comp.keys()
            ),

            values=list(
                comp.values()
            ),

            hole=0.55,

            title="Maintenance Composition"

        )

        dark_plot(
            fig
        )


    # -----------------------------------------------------------
    # TOP STATIONS
    # -----------------------------------------------------------

    if station_filter == "All Stations":

        section(
            "🏆 Station Leaderboard",
            "Top power stations ranked by total actual generation"
        )


        if not station_perf.empty:

            top = (
                station_perf
                .sort_values(
                    "Actual_sum",
                    ascending=False
                )
                .head(12)
                .copy()
            )


            fig = px.bar(

                top.sort_values(
                    "Actual_sum"
                ),

                x="Actual_sum",

                y="Power_Station",

                orientation="h",

                title="Top Performing Stations"

            )


            dark_plot(
                fig,
                500
            )


    # -----------------------------------------------------------
    # QUICK INSIGHTS
    # -----------------------------------------------------------

    section(
        "💡 Automated Insights",
        "Quick observations generated from available station data"
    )


    insight_cols = st.columns(3)


    if not station_perf.empty:

        best_station = (
            station_perf
            .sort_values(
                "Programme_Achievement_pct",
                ascending=False
            )
            .iloc[0]
        )


        worst_station = (
            station_perf
            .sort_values(
                "Programme_Achievement_pct",
                ascending=True
            )
            .iloc[0]
        )


        highest_maintenance = (
            station_perf
            .sort_values(
                "Maintenance_Impact_Index_pct",
                ascending=False
            )
            .iloc[0]
        )


        insight_cols[0].success(
            "🏆 **Best Achievement**\n\n"
            f"{best_station['Power_Station']} — "
            f"{safe_pct(best_station['Programme_Achievement_pct'])}"
        )


        insight_cols[1].warning(
            "⚠️ **Lowest Achievement**\n\n"
            f"{worst_station['Power_Station']} — "
            f"{safe_pct(worst_station['Programme_Achievement_pct'])}"
        )


        insight_cols[2].error(
            "🔧 **Highest Maintenance Impact**\n\n"
            f"{highest_maintenance['Power_Station']} — "
            f"{safe_pct(highest_maintenance['Maintenance_Impact_Index_pct'])}"
        )


    # -----------------------------------------------------------
    # DOWNLOAD DATA
    # -----------------------------------------------------------

    section(
        "📥 Data Export",
        "Download the currently selected dataset"
    )


    csv_data = view_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(

        "⬇️ Download Filtered Data (CSV)",

        csv_data,

        file_name="PowerGenAI_filtered_data.csv",

        mime="text/csv",

        use_container_width=True

    )


# ===============================================================
# POWER STATION ANALYSIS
# ===============================================================

elif page == "🏭 Power Station Analysis":

    hero(

        "🏭 Power Station Intelligence",

        "Detailed station-level operational performance analysis.",

        [

            (
                "🔵 PERFORMANCE ANALYTICS",
                "blue"
            ),

            (
                "🟢 STATION MONITORING",
                "green"
            )

        ]

    )


    if not DATA_OK:

        st.stop()


    stations = sorted(

        df[
            "Power_Station"
        ]
        .dropna()
        .unique()
        .tolist()

    )


    station = st.selectbox(
        "Select Power Station",
        stations
    )


    station_df = df[
        df[
            "Power_Station"
        ]
        ==
        station
    ]


    station_row = station_perf[
        station_perf[
            "Power_Station"
        ]
        ==
        station
    ]


    if station_df.empty:

        st.warning(
            "No data available."
        )

        st.stop()


    section(
        "Station Performance",
        f"Operational profile of {station}"
    )


    capacity = analytics.total_monitored_capacity(
        station_df
    )


    avg_actual = (
        station_df[
            "Actual"
        ]
        .mean()
    )


    avg_programme = (
        station_df[
            "Programme"
        ]
        .mean()
    )


    avg_available = (
        station_df[
            "Available_Capacity"
        ]
        .mean()
    )


    avg_maintenance = (
        station_df[
            "Total_Maintenance"
        ]
        .mean()
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.markdown(
        kpi(
            "Capacity",
            utils.fmt_mw(
                capacity
            ),
            "Monitored capacity"
        ),
        unsafe_allow_html=True
    )


    c2.markdown(
        kpi(
            "Actual",
            utils.fmt_mw(
                avg_actual
            ),
            "Average generation"
        ),
        unsafe_allow_html=True
    )


    c3.markdown(
        kpi(
            "Programme",
            utils.fmt_mw(
                avg_programme
            ),
            "Average programme"
        ),
        unsafe_allow_html=True
    )


    c4.markdown(
        kpi(
            "Maintenance",
            utils.fmt_mw(
                avg_maintenance
            ),
            "Average maintenance"
        ),
        unsafe_allow_html=True
    )


    if not station_row.empty:

        r = station_row.iloc[0]

        health = station_health(
            r
        )


        section(
            "❤️ Station Health",
            "Composite performance indicator"
        )


        h1, h2, h3, h4 = st.columns(4)


        h1.markdown(
            kpi(
                "Health Score",
                f"{health:.1f}/100",
                "Composite station health"
            ),
            unsafe_allow_html=True
        )


        h2.markdown(
            kpi(
                "Achievement",
                utils.fmt_pct(
                    r[
                        "Programme_Achievement_pct"
                    ]
                ),
                "Programme performance"
            ),
            unsafe_allow_html=True
        )


        h3.markdown(
            kpi(
                "Utilization",
                utils.fmt_pct(
                    r[
                        "Capacity_Utilization_pct"
                    ]
                ),
                "Capacity usage"
            ),
            unsafe_allow_html=True
        )


        h4.markdown(
            kpi(
                "Maintenance Impact",
                utils.fmt_pct(
                    r[
                        "Maintenance_Impact_Index_pct"
                    ]
                ),
                "Maintenance pressure"
            ),
            unsafe_allow_html=True
        )


    section(
        "📈 Actual vs Programme",
        "Historical station generation profile"
    )


    temp = (
        station_df
        .reset_index(
            drop=True
        )
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=list(
                range(
                    len(temp)
                )
            ),
            y=temp[
                "Actual"
            ],
            mode="lines",
            name="Actual"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=list(
                range(
                    len(temp)
                )
            ),
            y=temp[
                "Programme"
            ],
            mode="lines",
            name="Programme",
            line=dict(
                dash="dash"
            )
        )
    )


    fig.update_layout(
        title="Generation Performance"
    )


    dark_plot(
        fig,
        460
    )


    col1, col2 = st.columns(2)


    with col1:

        maintenance = station_df[
            [
                "Planned_Maintenance",
                "Forced_Maintenance",
                "Other_Reasons"
            ]
        ].sum()


        fig = px.pie(

            names=maintenance.index,

            values=maintenance.values,

            hole=0.5,

            title="Maintenance Breakdown"

        )


        dark_plot(
            fig
        )


    with col2:

        scatter_df = station_df.copy()


        fig = px.scatter(

            scatter_df,

            x="Available_Capacity",

            y="Actual",

            size="Programme",

            hover_data=[
                "Programme",
                "Excess_Shortfall"
            ],

            title="Available Capacity vs Actual Generation"

        )


        dark_plot(
            fig
        )


# ===============================================================
# GENERATION PREDICTION
# ===============================================================

elif page == "🔮 Generation Prediction":

    hero(

        "🔮 AI Generation Prediction",

        "Predict actual power generation using the trained Random Forest model.",

        [

            (
                "🟢 RANDOM FOREST",
                "green"
            ),

            (
                "🔵 AI FORECASTING",
                "blue"
            ),

            (
                "🧠 XAI AVAILABLE",
                "blue"
            )

        ]

    )


    if not DATA_OK or not MODELS_OK:

        st.error(
            "Data or Random Forest model is unavailable."
        )

        st.stop()


    stations = sorted(

        df[
            "Power_Station"
        ]
        .dropna()
        .unique()
        .tolist()

    )


    section(
        "⚙️ Operating Conditions",
        "Provide expected operating conditions"
    )


    with st.form(
        "prediction_form"
    ):

        col1, col2 = st.columns(2)


        with col1:

            station = st.selectbox(
                "Power Station",
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


        with col2:

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

        errors = utils.validate_prediction_inputs(

            capacity,

            programme,

            planned,

            forced,

            other

        )


        blocking = [
            e
            for e in errors
            if not e.startswith(
                "Warning"
            )
        ]


        warnings = [
            e
            for e in errors
            if e.startswith(
                "Warning"
            )
        ]


        for warning in warnings:

            st.warning(
                warning
            )


        if blocking:

            for error in blocking:

                st.error(
                    error
                )

        else:

            try:

                feature_row = (
                    prediction
                    .build_feature_row(
                        station,
                        capacity,
                        programme,
                        planned,
                        forced,
                        other
                    )
                )


                model = models[
                    "Random_Forest"
                ]


                predicted = prediction.predict(
                    model,
                    feature_row
                )


                shortfall = (
                    predicted
                    -
                    programme
                )


                achievement = (

                    predicted
                    /
                    programme
                    *
                    100

                    if programme > 0
                    else None

                )


                available = (

                    capacity
                    -
                    planned
                    -
                    forced
                    -
                    other

                )


                mii = (
                    feature_row[
                        "Maintenance_Impact_Index"
                    ]
                    .iloc[0]
                )


                risk = prediction.risk_level(
                    predicted,
                    programme
                )


                st.success(
                    "✅ AI prediction completed successfully."
                )


                section(
                    "⚡ Prediction Result",
                    "AI-generated generation forecast"
                )


                r1, r2, r3 = st.columns(3)


                r1.markdown(
                    kpi(
                        "Predicted Generation",
                        utils.fmt_mw(
                            predicted
                        ),
                        "AI forecast"
                    ),
                    unsafe_allow_html=True
                )


                r2.markdown(
                    kpi(
                        "Shortfall / Excess",
                        utils.fmt_mw(
                            shortfall
                        ),
                        "Compared with programme"
                    ),
                    unsafe_allow_html=True
                )


                r3.markdown(
                    kpi(
                        "Achievement",
                        utils.fmt_pct(
                            achievement
                        )
                        if achievement is not None
                        else "N/A",
                        "Predicted programme achievement"
                    ),
                    unsafe_allow_html=True
                )


                r4, r5, r6 = st.columns(3)


                r4.markdown(
                    kpi(
                        "Available Capacity",
                        utils.fmt_mw(
                            available
                        ),
                        "After maintenance"
                    ),
                    unsafe_allow_html=True
                )


                r5.markdown(
                    kpi(
                        "Maintenance Impact",
                        utils.fmt_pct(
                            mii
                        ),
                        "Maintenance impact index"
                    ),
                    unsafe_allow_html=True
                )


                r6.markdown(
                    kpi(
                        "Risk Level",
                        risk_icon(
                            risk
                        ),
                        "AI operational risk"
                    ),
                    unsafe_allow_html=True
                )


                # XAI
                section(
                    "🧠 Why did the model predict this?",
                    "Main prediction drivers"
                )


                try:

                    bias, contributions, reconstructed = (
                        prediction
                        .explain_prediction(
                            model,
                            feature_row
                        )
                    )


                    series = (
                        pd.Series(
                            contributions
                        )
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
                            f"· Base Value: "
                            f"{bias:.1f} MW"
                        ),

                        xaxis_title=(
                            "Contribution to prediction (MW)"
                        )

                    )


                    dark_plot(
                        fig,
                        520
                    )


                    st.caption(
                        "Positive contribution increases "
                        "the prediction; negative contribution "
                        "reduces it."
                    )


                except Exception as e:

                    st.info(
                        f"XAI explanation unavailable: {e}"
                    )


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ===============================================================
# WHAT-IF SIMULATOR
# ===============================================================

elif page == "🎛️ What-If Simulator":

    hero(

        "🎛️ What-If Scenario Simulator",

        "Test operational scenarios and instantly see their predicted impact.",

        [

            (
                "🔵 INTERACTIVE",
                "blue"
            ),

            (
                "🟢 AI POWERED",
                "green"
            ),

            (
                "🟡 SCENARIO ANALYSIS",
                "yellow"
            )

        ]

    )


    if not DATA_OK or not MODELS_OK:

        st.error(
            "Data or Random Forest model unavailable."
        )

        st.stop()


    stations = sorted(

        df[
            "Power_Station"
        ]
        .dropna()
        .unique()
        .tolist()

    )


    station = st.selectbox(
        "🏭 Simulation Station",
        stations
    )


    station_data = df[
        df[
            "Power_Station"
        ]
        ==
        station
    ]


    base_capacity = float(
        station_data[
            "Monitored_Capacity"
        ]
        .mean()
    )


    base_programme = float(
        station_data[
            "Programme"
        ]
        .mean()
    )


    section(
        "🎚️ Operational Controls",
        "Move the sliders and explore the AI response"
    )


    c1, c2 = st.columns(2)


    with c1:

        sim_programme = st.slider(

            "Programme Generation (MW)",

            0.0,

            max(
                base_capacity * 1.2,
                1.0
            ),

            min(
                base_programme,
                base_capacity * 1.2
            )

        )


        sim_planned = st.slider(

            "Planned Maintenance (MW)",

            0.0,

            max(
                base_capacity,
                1.0
            ),

            0.0

        )


    with c2:

        sim_forced = st.slider(

            "Forced Maintenance (MW)",

            0.0,

            max(
                base_capacity,
                1.0
            ),

            0.0

        )


        sim_other = st.slider(

            "Other Reasons (MW)",

            0.0,

            max(
                base_capacity,
                1.0
            ),

            0.0

        )


    try:

        feature_row = (
            prediction
            .build_feature_row(
                station,
                base_capacity,
                sim_programme,
                sim_planned,
                sim_forced,
                sim_other
            )
        )


        model = models[
            "Random_Forest"
        ]


        simulated = prediction.predict(
            model,
            feature_row
        )


        simulated_shortfall = (
            simulated
            -
            sim_programme
        )


        simulated_achievement = (

            simulated
            /
            sim_programme
            *
            100

            if sim_programme > 0
            else 0

        )


        available = (

            base_capacity
            -
            sim_planned
            -
            sim_forced
            -
            sim_other

        )


        risk = prediction.risk_level(

            simulated,

            sim_programme

        )


        section(
            "⚡ Scenario Result",
            f"AI response for {station}"
        )


        a, b, c = st.columns(3)


        a.markdown(
            kpi(
                "Predicted Generation",
                utils.fmt_mw(
                    simulated
                ),
                "Scenario output"
            ),
            unsafe_allow_html=True
        )


        b.markdown(
            kpi(
                "Shortfall / Excess",
                utils.fmt_mw(
                    simulated_shortfall
                ),
                "Scenario gap"
            ),
            unsafe_allow_html=True
        )


        c.markdown(
            kpi(
                "Risk",
                risk_icon(
                    risk
                ),
                "Scenario risk"
            ),
            unsafe_allow_html=True
        )


        d, e = st.columns(2)


        d.markdown(
            kpi(
                "Available Capacity",
                utils.fmt_mw(
                    available
                ),
                "After simulated maintenance"
            ),
            unsafe_allow_html=True
        )


        e.markdown(
            kpi(
                "Achievement",
                utils.fmt_pct(
                    simulated_achievement
                ),
                "Scenario achievement"
            ),
            unsafe_allow_html=True
        )


        scenario_df = pd.DataFrame({

            "Metric": [

                "Programme",

                "Predicted Actual",

                "Available Capacity"

            ],

            "MW": [

                sim_programme,

                simulated,

                available

            ]

        })


        fig = px.bar(

            scenario_df,

            x="Metric",

            y="MW",

            title="Scenario Snapshot"

        )


        dark_plot(
            fig,
            400
        )


    except Exception as e:

        st.error(
            f"Simulation failed: {e}"
        )


# ===============================================================
# ALERTS & RISK
# ===============================================================

elif page == "🚨 Alerts & Risk":

    hero(

        "🚨 Alerts & Risk Center",

        "Identify stations requiring operational attention.",

        [

            (
                "🔴 CRITICAL",
                "red"
            ),

            (
                "🟠 HIGH",
                "yellow"
            ),

            (
                "🟢 MONITORING ACTIVE",
                "green"
            )

        ]

    )


    if not DATA_OK:

        st.stop()


    cfg = alerts.AlertConfig()


    all_alerts = []


    stations = (
        df[
            "Power_Station"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    for station in stations:

        station_data = df[
            df[
                "Power_Station"
            ]
            ==
            station
        ]


        try:

            result = alerts.evaluate_dataframe(
                station_data,
                cfg
            )


            if not result.empty:

                temp = result.copy()

                temp[
                    "Power_Station"
                ] = station

                all_alerts.append(
                    temp
                )

        except Exception:
            pass


    if all_alerts:

        alert_df = pd.concat(
            all_alerts,
            ignore_index=True
        )


        counts = (
            alert_df[
                "type"
            ]
            .value_counts()
        )


        section(
            "Alert Summary",
            "Detected operational conditions"
        )


        cols = st.columns(
            min(
                4,
                max(
                    1,
                    len(counts)
                )
            )
        )


        for i, (name, count) in enumerate(
            counts.items()
        ):

            cols[
                i % len(cols)
            ].markdown(

                kpi(
                    str(name),
                    str(count),
                    "Alert records"
                ),

                unsafe_allow_html=True

            )


        st.divider()


        st.dataframe(

            alert_df,

            use_container_width=True,

            hide_index=True

        )


    else:

        st.success(
            "🟢 No operational alerts were detected."
        )


# ===============================================================
# MAINTENANCE
# ===============================================================

elif page == "🔧 Maintenance Analysis":

    hero(

        "🔧 Maintenance Intelligence",

        "Analyse planned, forced and other maintenance impacts.",

        [

            (
                "🔵 MAINTENANCE ANALYTICS",
                "blue"
            ),

            (
                "🟠 RISK MONITORING",
                "yellow"
            )

        ]

    )


    if not DATA_OK:

        st.stop()


    comp = (
        analytics
        .maintenance_composition(
            df
        )
    )


    c1, c2, c3 = st.columns(3)


    c1.markdown(
        kpi(
            "Planned Maintenance",
            utils.fmt_pct(
                comp["Planned"]
            ),
            "Share of maintenance"
        ),
        unsafe_allow_html=True
    )


    c2.markdown(
        kpi(
            "Forced Maintenance",
            utils.fmt_pct(
                comp["Forced"]
            ),
            "Share of maintenance"
        ),
        unsafe_allow_html=True
    )


    c3.markdown(
        kpi(
            "Other Reasons",
            utils.fmt_pct(
                comp["Other"]
            ),
            "Share of maintenance"
        ),
        unsafe_allow_html=True
    )


    stations_to_compare = st.multiselect(

        "Select stations",

        sorted(
            df[
                "Power_Station"
            ]
            .dropna()
            .unique()
            .tolist()
        )

    )


    if stations_to_compare:

        compare_df = station_perf[
            station_perf[
                "Power_Station"
            ]
            .isin(
                stations_to_compare
            )
        ]

    else:

        compare_df = (
            station_perf[
                station_perf[
                    "Records"
                ]
                >= 50
            ]
            .sort_values(
                "Total_Maintenance_avg",
                ascending=False
            )
            .head(10)
        )


    section(
        "Maintenance Comparison",
        "Station-level maintenance behaviour"
    )


    fig = px.bar(

        compare_df,

        x="Power_Station",

        y=[
            "Total_Maintenance_avg",
            "Forced_Maintenance_avg"
        ],

        barmode="group",

        title="Maintenance Comparison"

    )


    dark_plot(
        fig
    )


    fig2 = px.bar(

        compare_df,

        x="Power_Station",

        y="Maintenance_Impact_Index_pct",

        title="Maintenance Impact Index"

    )


    dark_plot(
        fig2
    )


    st.dataframe(

        compare_df[

            [

                "Power_Station",

                "Records",

                "Total_Maintenance_avg",

                "Forced_Maintenance_avg",

                "Maintenance_Impact_Index_pct",

                "Programme_Achievement_pct"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )


# ===============================================================
# MODEL PERFORMANCE
# ===============================================================

elif page == "📈 Model Performance":

    hero(

        "📈 AI Model Performance Lab",

        "Compare machine-learning models used for generation forecasting.",

        [

            (
                "🔵 MODEL EVALUATION",
                "blue"
            ),

            (
                "🟢 PERFORMANCE ANALYTICS",
                "green"
            )

        ]

    )


    if model_comparison_df.empty:

        st.warning(
            "Model comparison data not found."
        )

        st.stop()


    best_model = (
        model_comparison_df
        .loc[
            model_comparison_df[
                "R2"
            ]
            .idxmax(),
            "Model"
        ]
    )


    best_r2 = (
        model_comparison_df[
            "R2"
        ]
        .max()
    )


    lowest_mae = (
        model_comparison_df[
            "MAE"
        ]
        .min()
    )


    c1, c2, c3 = st.columns(3)


    c1.markdown(
        kpi(
            "Best Model",
            best_model,
            "Highest R²"
        ),
        unsafe_allow_html=True
    )


    c2.markdown(
        kpi(
            "Best R²",
            f"{best_r2:.4f}",
            "Higher is better"
        ),
        unsafe_allow_html=True
    )


    c3.markdown(
        kpi(
            "Lowest MAE",
            f"{lowest_mae:.4f}",
            "Lower is better"
        ),
        unsafe_allow_html=True
    )


    section(
        "Model Comparison",
        "Detailed evaluation results"
    )


    st.dataframe(

        model_comparison_df,

        use_container_width=True,

        hide_index=True

    )


    c1, c2 = st.columns(2)


    with c1:

        fig = px.bar(

            model_comparison_df,

            x="Model",

            y="R2",

            title="R² Comparison"

        )

        dark_plot(
            fig
        )


    with c2:

        fig = px.bar(

            model_comparison_df,

            x="Model",

            y="MAE",

            title="MAE Comparison"

        )

        dark_plot(
            fig
        )


    st.info(

        "HistGradientBoosting is used as the "
        "XGBoost substitute in the current "
        "offline training environment."

    )


# ===============================================================
# EXPLAINABLE AI
# ===============================================================

elif page == "🧠 Explainable AI":

    hero(

        "🧠 Explainable AI",

        "Understand why the model makes a particular prediction.",

        [

            (
                "🧠 FEATURE IMPORTANCE",
                "blue"
            ),

            (
                "🔍 PREDICTION DRIVERS",
                "green"
            )

        ]

    )


    if not feature_importance_df.empty:

        section(
            "Global Feature Importance",
            "Most influential features in the Random Forest"
        )


        top10 = (
            feature_importance_df
            .head(10)
        )


        fig = px.bar(

            top10.sort_values(
                "importance"
            ),

            x="importance",

            y="feature",

            orientation="h",

            title="Top 10 Feature Importances"

        )


        dark_plot(
            fig,
            500
        )


    else:

        st.warning(
            "Feature importance file not found."
        )


    if DATA_OK and MODELS_OK:

        section(
            "Individual Prediction Explanation",
            "Select a real record from the dataset"
        )


        station = st.selectbox(

            "Power Station",

            sorted(
                df[
                    "Power_Station"
                ]
                .dropna()
                .unique()
                .tolist()
            ),

            key="xai_station"

        )


        records = (
            df[
                df[
                    "Power_Station"
                ]
                ==
                station
            ]
            .reset_index(
                drop=True
            )
        )


        if not records.empty:

            index = st.slider(

                "Record Index",

                0,

                len(records) - 1,

                0

            )


            row = records.iloc[
                index
            ]


            feature_row = (
                row[
                    MODEL_FEATURE_COLUMNS
                ]
                .to_frame()
                .T
            )


            try:

                model = models[
                    "Random_Forest"
                ]


                bias, contribution, prediction_value = (
                    prediction
                    .explain_prediction(
                        model,
                        feature_row
                    )
                )


                c1, c2 = st.columns(2)


                c1.markdown(

                    kpi(
                        "Actual Generation",
                        f"{row['Actual']:.2f} MW",
                        "Observed value"
                    ),

                    unsafe_allow_html=True

                )


                c2.markdown(

                    kpi(
                        "Model Prediction",
                        f"{prediction_value:.2f} MW",
                        "AI prediction"
                    ),

                    unsafe_allow_html=True

                )


                series = (
                    pd.Series(
                        contribution
                    )
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
                        f"Prediction Drivers "
                        f"· Base Value: "
                        f"{bias:.2f} MW"
                    ),

                    xaxis_title=(
                        "Contribution (MW)"
                    )

                )


                dark_plot(
                    fig,
                    520
                )


            except Exception as e:

                st.error(
                    f"Could not generate XAI explanation: {e}"
                )


# ===============================================================
# REPORTS
# ===============================================================

elif page == "📄 Reports":

    hero(

        "📄 Power Station Reports",

        "Generate a professional station-level performance report.",

        [

            (
                "📄 REPORT GENERATOR",
                "blue"
            ),

            (
                "⬇️ DOWNLOADABLE",
                "green"
            )

        ]

    )


    if not DATA_OK:

        st.stop()


    stations = sorted(

        df[
            "Power_Station"
        ]
        .dropna()
        .unique()
        .tolist()

    )


    station = st.selectbox(

        "Select Power Station",

        stations,

        key="report_station"

    )


    station_df = df[
        df[
            "Power_Station"
        ]
        ==
        station
    ]


    station_row = station_perf[
        station_perf[
            "Power_Station"
        ]
        ==
        station
    ]


    if st.button(

        "📄 GENERATE PERFORMANCE REPORT",

        use_container_width=True

    ):

        cfg = alerts.AlertConfig()


        try:

            station_alerts = (
                alerts
                .evaluate_dataframe(
                    station_df,
                    cfg
                )
            )

        except Exception:

            station_alerts = pd.DataFrame()


        report = [

            "# PowerGenAI Performance Report",

            "",

            f"## Power Station: {station}",

            "",

            f"Records analysed: {len(station_df)}",

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

            ""

        ]


        if not station_row.empty:

            r = station_row.iloc[
                0
            ]


            report.extend(

                [

                    f"- Programme Achievement: {utils.fmt_pct(r['Programme_Achievement_pct'])}",

                    f"- Capacity Utilization: {utils.fmt_pct(r['Capacity_Utilization_pct'])}",

                    f"- Maintenance Impact Index: {utils.fmt_pct(r['Maintenance_Impact_Index_pct'])}",

                    ""

                ]

            )


        report.append(
            "## Alerts"
        )


        if station_alerts.empty:

            report.append(
                "- No alerts triggered."
            )

        else:

            for alert_type, count in (
                station_alerts[
                    "type"
                ]
                .value_counts()
                .items()
            ):

                report.append(

                    f"- {alert_type}: "
                    f"{count} record(s)"

                )


        report_text = "\n".join(
            report
        )


        st.markdown(
            report_text
        )


        st.download_button(

            "⬇️ DOWNLOAD REPORT",

            report_text,

            file_name=(
                f"PowerGenAI_"
                f"{station}_Report.md"
            ),

            mime="text/markdown",

            use_container_width=True

        )


# ===============================================================
# FOOTER
# ===============================================================

st.markdown(

    """
    <div class="pg-footer">

        ⚡ <b>PowerGenAI</b>

        <br>

        AI-Based Power Generation Forecasting
        • Power Station Monitoring
        • Predictive Analytics
        • Explainable AI

        <br><br>

        Major Project Dashboard

    </div>
    """,

    unsafe_allow_html=True

)
