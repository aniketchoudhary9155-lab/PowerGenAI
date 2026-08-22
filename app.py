"""
PowerGenAI
AI-Based Power Generation Forecasting & Power Station Performance Monitoring

Run:
    streamlit run app.py

IMPORTANT:
This file expects the existing PowerGenAI project structure:

PowerGenAI/
│
├── app.py
├── prediction.py
├── analytics.py
├── alerts.py
├── utils.py
├── features.py
│
├── data/
│   └── processed/
│       ├── powergeneration_features.csv
│       └── station_performance_summary.csv
│
└── models/
    ├── Random_Forest.joblib
    ├── Linear_Regression.joblib
    ├── Gradient_Boosting.joblib
    ├── HistGB_XGBoost_substitute.joblib
    ├── model_comparison.csv
    └── feature_importance.csv
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ================================================================
# PATH FIX
# ================================================================
# DO NOT change __file__ to _file_

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import prediction
    import analytics
    import alerts
    import utils
    from features import MODEL_FEATURE_COLUMNS
except Exception as e:
    st.error("PowerGenAI internal modules could not be imported.")
    st.code(str(e))
    st.stop()


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="PowerGenAI | Smart Power Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# PREMIUM CSS
# ================================================================

st.markdown(
    """
<style>

/* --------------------------------------------------------------
   GLOBAL
-------------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(37,99,235,0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(124,58,237,0.12), transparent 30%),
        linear-gradient(135deg, #050816 0%, #0b1120 45%, #111827 100%);
    color: #f8fafc;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}


/* --------------------------------------------------------------
   SIDEBAR
-------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(15,23,42,0.98),
            rgba(2,6,23,0.98)
        );
    border-right: 1px solid rgba(148,163,184,0.15);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}


/* --------------------------------------------------------------
   SIDEBAR BRAND
-------------------------------------------------------------- */

.sidebar-brand {
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.25),
            rgba(124,58,237,0.18)
        );

    border: 1px solid rgba(96,165,250,0.25);

    box-shadow:
        0 10px 40px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.08);

    animation: fadeIn 0.8s ease;
}

.sidebar-logo {
    font-size: 32px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-top: 5px;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #94a3b8;
}


/* --------------------------------------------------------------
   HERO
-------------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;

    padding: 38px;
    border-radius: 28px;
    margin-bottom: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.28),
            rgba(124,58,237,0.22),
            rgba(15,23,42,0.75)
        );

    border: 1px solid rgba(147,197,253,0.20);

    box-shadow:
        0 20px 70px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    animation: fadeSlide 0.8s ease;
}

.hero::before {
    content: "";
    position: absolute;

    width: 260px;
    height: 260px;

    right: -80px;
    top: -100px;

    border-radius: 50%;

    background: rgba(59,130,246,0.20);

    filter: blur(40px);

    animation: floating 6s ease-in-out infinite;
}

.hero::after {
    content: "";
    position: absolute;

    width: 180px;
    height: 180px;

    left: -80px;
    bottom: -100px;

    border-radius: 50%;

    background: rgba(168,85,247,0.15);

    filter: blur(35px);

    animation: floating 7s ease-in-out infinite reverse;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-icon {
    font-size: 55px;
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    margin: 5px 0;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #93c5fd,
            #c4b5fd
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 16px;
}


/* --------------------------------------------------------------
   KPI CARDS
-------------------------------------------------------------- */

.kpi-card {
    padding: 22px;
    min-height: 130px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.80),
            rgba(15,23,42,0.65)
        );

    border: 1px solid rgba(148,163,184,0.16);

    box-shadow:
        0 12px 40px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.05);

    backdrop-filter: blur(20px);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease,
        border-color 0.3s ease;

    animation: fadeSlide 0.7s ease;
}

.kpi-card:hover {
    transform: translateY(-7px) scale(1.015);

    border-color:
        rgba(96,165,250,0.40);

    box-shadow:
        0 20px 50px rgba(37,99,235,0.18);
}

.kpi-icon {
    font-size: 28px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    margin-top: 6px;
}


/* --------------------------------------------------------------
   SECTION HEADERS
-------------------------------------------------------------- */

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #94a3b8;
    margin-bottom: 20px;
}


/* --------------------------------------------------------------
   GLASS PANEL
-------------------------------------------------------------- */

.glass {
    padding: 24px;

    border-radius: 22px;

    background:
        rgba(15,23,42,0.62);

    border:
        1px solid rgba(148,163,184,0.15);

    backdrop-filter:
        blur(18px);

    box-shadow:
        0 15px 45px rgba(0,0,0,0.22),
        inset 0 1px 0 rgba(255,255,255,0.05);

    animation:
        fadeIn 0.7s ease;
}


/* --------------------------------------------------------------
   STATUS
-------------------------------------------------------------- */

.status-online {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 8px 14px;

    border-radius: 999px;

    background: rgba(34,197,94,0.12);

    border:
        1px solid rgba(34,197,94,0.25);

    color: #86efac;

    font-size: 12px;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;

    background: #22c55e;

    border-radius: 50%;

    box-shadow:
        0 0 10px #22c55e;

    animation:
        pulse 1.6s infinite;
}


/* --------------------------------------------------------------
   BUTTONS
-------------------------------------------------------------- */

.stButton > button {
    border-radius: 14px !important;

    border:
        1px solid rgba(96,165,250,0.30) !important;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.85),
            rgba(79,70,229,0.85)
        ) !important;

    color: white !important;

    font-weight: 700 !important;

    transition:
        all 0.25s ease !important;

    box-shadow:
        0 8px 25px rgba(37,99,235,0.20) !important;
}

.stButton > button:hover {
    transform:
        translateY(-3px)
        scale(1.01);

    box-shadow:
        0 14px 35px rgba(37,99,235,0.35) !important;
}


/* --------------------------------------------------------------
   INPUTS
-------------------------------------------------------------- */

.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div,
.stSlider {
    border-radius: 12px;
}


/* --------------------------------------------------------------
   DATAFRAME
-------------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}


/* --------------------------------------------------------------
   ANIMATIONS
-------------------------------------------------------------- */

@keyframes fadeIn {
    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }
}

@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes floating {
    0%, 100% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(20px);
    }
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(34,197,94,0.6);
    }

    70% {
        box-shadow: 0 0 0 10px rgba(34,197,94,0);
    }

    100% {
        box-shadow: 0 0 0 0 rgba(34,197,94,0);
    }
}


/* --------------------------------------------------------------
   FOOTER
-------------------------------------------------------------- */

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 10px 10px;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# DIRECTORIES
# ================================================================

DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data
def load_data():
    data_path = os.path.join(
        DATA_DIR,
        "powergeneration_features.csv"
    )

    station_path = os.path.join(
        DATA_DIR,
        "station_performance_summary.csv"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing file:\n{data_path}"
        )

    if not os.path.exists(station_path):
        raise FileNotFoundError(
            f"Missing file:\n{station_path}"
        )

    df = pd.read_csv(data_path)
    station_perf = pd.read_csv(station_path)

    return df, station_perf


@st.cache_resource
def load_models():

    models = {}
    errors = {}

    model_names = [
        "Random_Forest",
        "Linear_Regression",
        "Gradient_Boosting",
        "HistGB_XGBoost_substitute",
    ]

    for name in model_names:

        path = os.path.join(
            MODELS_DIR,
            f"{name}.joblib"
        )

        if not os.path.exists(path):
            errors[name] = "Model file not found."
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

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_feature_importance():

    path = os.path.join(
        MODELS_DIR,
        "feature_importance.csv"
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


# ================================================================
# LOAD EVERYTHING
# ================================================================

try:

    df, station_perf = load_data()

    DATA_OK = True

except Exception as e:

    DATA_OK = False

    st.error("⚠️ PowerGenAI data could not be loaded.")

    st.code(str(e))

    st.info(
        "Check that data/processed contains "
        "powergeneration_features.csv and "
        "station_performance_summary.csv."
    )

    st.stop()


try:

    models, model_errors = load_models()

    MODELS_OK = "Random_Forest" in models

except Exception as e:

    models = {}
    model_errors = {}

    MODELS_OK = False

    st.error("⚠️ Model loading failed.")

    st.code(str(e))


model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()


# ================================================================
# GLOBAL METRICS
# ================================================================

total_stations = df["Power_Station"].nunique()

total_actual = df["Actual"].sum()

total_programme = df["Programme"].sum()

total_shortfall = df.loc[
    df["Excess_Shortfall"] < 0,
    "Excess_Shortfall"
].sum()

avg_achievement = (
    total_actual / total_programme * 100
    if total_programme > 0
    else 0
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.markdown(
    """
<div class="sidebar-brand">

<div class="sidebar-logo">⚡</div>

<div class="sidebar-title">
PowerGenAI
</div>

<div class="sidebar-subtitle">
Smart Power Generation Intelligence
</div>

</div>
""",
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    """
<div class="status-online">
<span class="status-dot"></span>
SYSTEM ONLINE
</div>
""",
    unsafe_allow_html=True,
)


st.sidebar.write("")


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard",
        "🏭 Power Station Analysis",
        "🔮 Generation Prediction",
        "🎛️ What-If Simulator",
        "🔧 Maintenance Analysis",
        "📈 Model Performance",
        "🧠 Explainable AI",
        "📄 Reports",
    ],
)


st.sidebar.divider()


st.sidebar.markdown(
    """
### ⚡ PowerGenAI

AI-powered monitoring and forecasting platform.

**Core Technologies**
- Machine Learning
- Random Forest
- Gradient Boosting
- Predictive Analytics
- Explainable AI
- Interactive Visualization

**Project Status**
🟢 Data Engine  
🟢 ML Engine  
🟢 Analytics Engine  
🟢 Dashboard  
""",
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def hero(title, subtitle, icon="⚡"):

    st.markdown(
        f"""
<div class="hero">

<div class="hero-content">

<div class="hero-icon">
{icon}
</div>

<div class="hero-title">
{title}
</div>

<div class="hero-subtitle">
{subtitle}
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )


def kpi(icon, label, value):

    st.markdown(
        f"""
<div class="kpi-card">

<div class="kpi-icon">
{icon}
</div>

<div class="kpi-label">
{label}
</div>

<div class="kpi-value">
{value}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):

    st.markdown(
        f"""
<div class="section-title">
{title}
</div>

<div class="section-subtitle">
{subtitle}
</div>
""",
        unsafe_allow_html=True,
    )


# ================================================================
# DASHBOARD
# ================================================================

if page == "📊 Dashboard":

    hero(
        "PowerGenAI Command Center",
        "AI-powered real-time style monitoring and generation intelligence",
        "⚡",
    )

    # ------------------------------------------------------------
    # TOP KPIs
    # ------------------------------------------------------------

    cols = st.columns(4)

    with cols[0]:
        kpi(
            "🏭",
            "Power Stations",
            f"{total_stations:,}",
        )

    with cols[1]:
        kpi(
            "⚡",
            "Total Actual",
            utils.fmt_mw(total_actual),
        )

    with cols[2]:
        kpi(
            "🎯",
            "Programme",
            utils.fmt_mw(total_programme),
        )

    with cols[3]:
        kpi(
            "📈",
            "Achievement",
            utils.fmt_pct(avg_achievement),
        )

    st.write("")

    # ------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------

    station_filter = st.selectbox(
        "🔎 Select Power Station",
        [
            "All Stations"
        ]
        + sorted(
            df["Power_Station"]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    if station_filter == "All Stations":

        view_df = df

    else:

        view_df = df[
            df["Power_Station"] == station_filter
        ]

    # ------------------------------------------------------------
    # SECOND KPI ROW
    # ------------------------------------------------------------

    total_capacity = analytics.total_monitored_capacity(
        view_df
    )

    maintenance = view_df[
        "Total_Maintenance"
    ].sum()

    cols = st.columns(4)

    with cols[0]:

        kpi(
            "🔋",
            "Monitored Capacity",
            utils.fmt_mw(total_capacity),
        )

    with cols[1]:

        kpi(
            "🛠️",
            "Maintenance",
            utils.fmt_mw(maintenance),
        )

    with cols[2]:

        kpi(
            "📉",
            "Shortfall",
            utils.fmt_mw(total_shortfall),
        )

    with cols[3]:

        kpi(
            "📋",
            "Records",
            f"{len(view_df):,}",
        )

    st.write("")

    # ------------------------------------------------------------
    # CHARTS
    # ------------------------------------------------------------

    section(
        "Generation Intelligence",
        "Interactive overview of station generation behaviour",
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            view_df,
            x="Actual",
            nbins=50,
            title="⚡ Actual Generation Distribution",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        comp = analytics.maintenance_composition(
            view_df
        )

        fig = px.pie(
            names=list(comp.keys()),
            values=list(comp.values()),
            hole=0.55,
            title="🛠️ Maintenance Composition",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ------------------------------------------------------------
    # STATION RANKING
    # ------------------------------------------------------------

    if station_filter == "All Stations":

        top15 = (
            station_perf
            .sort_values(
                "Actual_sum",
                ascending=False,
            )
            .head(15)
        )

        fig = px.bar(
            top15,
            x="Actual_sum",
            y="Power_Station",
            orientation="h",
            title="🏆 Top 15 Stations by Actual Generation",
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis={
                "categoryorder":
                "total ascending"
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ================================================================
# POWER STATION ANALYSIS
# ================================================================

elif page == "🏭 Power Station Analysis":

    hero(
        "Power Station Intelligence",
        "Deep-dive operational analysis for individual generating stations",
        "🏭",
    )

    station = st.selectbox(
        "Select Station",
        sorted(
            df["Power_Station"]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    station_df = df[
        df["Power_Station"] == station
    ]

    station_row = station_perf[
        station_perf["Power_Station"] == station
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi(
            "🔋",
            "Capacity",
            utils.fmt_mw(
                analytics.total_monitored_capacity(
                    station_df
                )
            ),
        )

    with c2:
        kpi(
            "📋",
            "Records",
            f"{len(station_df):,}",
        )

    with c3:
        kpi(
            "⚡",
            "Average Actual",
            utils.fmt_mw(
                station_df["Actual"].mean()
            ),
        )

    with c4:
        kpi(
            "🎯",
            "Average Programme",
            utils.fmt_mw(
                station_df["Programme"].mean()
            ),
        )

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        fig = px.line(
            station_df.reset_index(),
            y="Actual",
            title="⚡ Actual Generation Trend",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        maint = station_df[
            [
                "Planned_Maintenance",
                "Forced_Maintenance",
                "Other_Reasons",
            ]
        ].sum()

        fig = px.pie(
            names=maint.index,
            values=maint.values,
            hole=0.5,
            title="🛠️ Maintenance Breakdown",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if not station_row.empty:

        r = station_row.iloc[0]

        st.subheader("Station Performance Indicators")

        c1, c2, c3 = st.columns(3)

        with c1:
            kpi(
                "🎯",
                "Programme Achievement",
                utils.fmt_pct(
                    r["Programme_Achievement_pct"]
                ),
            )

        with c2:
            kpi(
                "🔋",
                "Capacity Utilization",
                utils.fmt_pct(
                    r["Capacity_Utilization_pct"]
                ),
            )

        with c3:
            kpi(
                "🛠️",
                "Maintenance Impact",
                utils.fmt_pct(
                    r["Maintenance_Impact_Index_pct"]
                ),
            )


# ================================================================
# GENERATION PREDICTION
# ================================================================

elif page == "🔮 Generation Prediction":

    hero(
        "AI Generation Predictor",
        "Predict expected actual generation using the trained Random Forest model",
        "🔮",
    )

    if not MODELS_OK:

        st.error(
            "Random Forest model is not available."
        )

        st.info(
            "Check the models/Random_Forest.joblib file."
        )

        st.stop()

    stations = sorted(
        df["Power_Station"]
        .dropna()
        .unique()
        .tolist()
    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:

            station = st.selectbox(
                "🏭 Power Station",
                stations,
            )

            monitored_capacity = st.number_input(
                "🔋 Monitored Capacity (MW)",
                min_value=0.0,
                value=500.0,
                step=10.0,
            )

            programme = st.number_input(
                "🎯 Programme Generation (MW)",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )

        with c2:

            planned = st.number_input(
                "🛠️ Planned Maintenance (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )

            forced = st.number_input(
                "🚨 Forced Maintenance (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )

            other = st.number_input(
                "📌 Other Reasons (MW)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "⚡ RUN AI PREDICTION",
            use_container_width=True,
        )

    if submitted:

        with st.spinner(
            "AI model is analysing operating conditions..."
        ):

            time.sleep(1)

            try:

                errors = utils.validate_prediction_inputs(
                    monitored_capacity,
                    programme,
                    planned,
                    forced,
                    other,
                )

                blocking = [
                    e
                    for e in errors
                    if not e.startswith("Warning")
                ]

                for e in errors:

                    if e.startswith("Warning"):
                        st.warning(e)

                if blocking:

                    for e in blocking:
                        st.error(e)

                else:

                    feature_row = prediction.build_feature_row(
                        station,
                        monitored_capacity,
                        programme,
                        planned,
                        forced,
                        other,
                    )

                    model = models[
                        "Random_Forest"
                    ]

                    predicted = prediction.predict(
                        model,
                        feature_row,
                    )

                    shortfall = (
                        predicted - programme
                    )

                    achievement = (
                        predicted
                        / programme
                        * 100
                        if programme > 0
                        else None
                    )

                    available = (
                        monitored_capacity
                        - (
                            planned
                            + forced
                            + other
                        )
                    )

                    mii = feature_row[
                        "Maintenance_Impact_Index"
                    ].iloc[0]

                    risk = prediction.risk_level(
                        predicted,
                        programme,
                    )

                    st.success(
                        "✅ AI Prediction Completed"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        kpi(
                            "⚡",
                            "Predicted Actual",
                            utils.fmt_mw(
                                predicted
                            ),
                        )

                    with c2:
                        kpi(
                            "📉",
                            "Shortfall / Excess",
                            utils.fmt_mw(
                                shortfall
                            ),
                        )

                    with c3:
                        kpi(
                            "🎯",
                            "Achievement",
                            utils.fmt_pct(
                                achievement
                            )
                            if achievement is not None
                            else "N/A",
                        )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        kpi(
                            "🔋",
                            "Available Capacity",
                            utils.fmt_mw(
                                available
                            ),
                        )

                    with c2:
                        kpi(
                            "🛠️",
                            "Maintenance Impact",
                            utils.fmt_pct(mii),
                        )

                    with c3:

                        emoji = {
                            "LOW": "🟢",
                            "MEDIUM": "🟡",
                            "HIGH": "🟠",
                            "CRITICAL": "🔴",
                        }.get(
                            risk,
                            "⚪",
                        )

                        kpi(
                            "🚦",
                            "Risk Level",
                            f"{emoji} {risk}",
                        )

                    # ------------------------------------------------
                    # EXPLANATION
                    # ------------------------------------------------

                    st.divider()

                    section(
                        "🧠 AI Decision Explanation",
                        "Understand which operating factors influenced the prediction",
                    )

                    try:

                        bias, contributions, recon = (
                            prediction.explain_prediction(
                                model,
                                feature_row,
                            )
                        )

                        series = (
                            pd.Series(contributions)
                            .sort_values(
                                key=abs,
                                ascending=False,
                            )
                        )

                        fig = go.Figure(
                            go.Bar(
                                x=series.values,
                                y=series.index,
                                orientation="h",
                            )
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            title=(
                                f"Feature Contributions "
                                f"(Base: {bias:.2f} MW)"
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    except Exception as e:

                        st.info(
                            f"Explanation unavailable: {e}"
                        )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(str(e))


# ================================================================
# WHAT IF SIMULATOR
# ================================================================

elif page == "🎛️ What-If Simulator":

    hero(
        "What-If Power Simulator",
        "Experiment with operational conditions and instantly see predicted outcomes",
        "🎛️",
    )

    if not MODELS_OK:

        st.error(
            "Random Forest model unavailable."
        )

        st.stop()

    stations = sorted(
        df["Power_Station"]
        .dropna()
        .unique()
        .tolist()
    )

    station = st.selectbox(
        "🏭 Simulation Station",
        stations,
    )

    station_df = df[
        df["Power_Station"] == station
    ]

    base_capacity = float(
        station_df[
            "Monitored_Capacity"
        ].mean()
    )

    base_programme = float(
        station_df[
            "Programme"
        ].mean()
    )

    st.subheader(
        "🎚️ Adjust Operating Conditions"
    )

    c1, c2 = st.columns(2)

    with c1:

        programme = st.slider(
            "Programme Generation (MW)",
            0.0,
            max(
                base_capacity * 1.2,
                1.0,
            ),
            min(
                base_programme,
                max(
                    base_capacity * 1.2,
                    1.0,
                ),
            ),
        )

        planned = st.slider(
            "Planned Maintenance (MW)",
            0.0,
            max(base_capacity, 1.0),
            0.0,
        )

    with c2:

        forced = st.slider(
            "Forced Maintenance (MW)",
            0.0,
            max(base_capacity, 1.0),
            0.0,
        )

        other = st.slider(
            "Other Reasons (MW)",
            0.0,
            max(base_capacity, 1.0),
            0.0,
        )

    try:

        feature_row = prediction.build_feature_row(
            station,
            base_capacity,
            programme,
            planned,
            forced,
            other,
        )

        predicted = prediction.predict(
            models["Random_Forest"],
            feature_row,
        )

        risk = prediction.risk_level(
            predicted,
            programme,
        )

        available = (
            base_capacity
            - planned
            - forced
            - other
        )

        st.divider()

        section(
            "Simulation Results",
            "Live model response to your selected scenario",
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi(
                "⚡",
                "Predicted Generation",
                utils.fmt_mw(predicted),
            )

        with c2:
            kpi(
                "🔋",
                "Available Capacity",
                utils.fmt_mw(available),
            )

        with c3:
            kpi(
                "📊",
                "Difference",
                utils.fmt_mw(
                    predicted - programme
                ),
            )

        with c4:

            emoji = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "CRITICAL": "🔴",
            }.get(
                risk,
                "⚪",
            )

            kpi(
                "🚦",
                "Risk",
                f"{emoji} {risk}",
            )

        # Comparison chart

        chart_df = pd.DataFrame(
            {
                "Metric": [
                    "Programme",
                    "Predicted",
                    "Available Capacity",
                ],
                "MW": [
                    programme,
                    predicted,
                    available,
                ],
            }
        )

        fig = px.bar(
            chart_df,
            x="Metric",
            y="MW",
            title="Scenario Comparison",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Simulation failed: {e}"
        )


# ================================================================
# MAINTENANCE
# ================================================================

elif page == "🔧 Maintenance Analysis":

    hero(
        "Maintenance Intelligence",
        "Identify maintenance patterns and operational impact",
        "🔧",
    )

    comp = analytics.maintenance_composition(
        df
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi(
            "🟢",
            "Planned",
            utils.fmt_pct(
                comp["Planned"]
            ),
        )

    with c2:
        kpi(
            "🔴",
            "Forced",
            utils.fmt_pct(
                comp["Forced"]
            ),
        )

    with c3:
        kpi(
            "🟡",
            "Other",
            utils.fmt_pct(
                comp["Other"]
            ),
        )

    st.write("")

    stations = st.multiselect(
        "Select stations",
        sorted(
            df["Power_Station"]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    if stations:

        compare_df = station_perf[
            station_perf[
                "Power_Station"
            ].isin(stations)
        ]

    else:

        compare_df = (
            station_perf
            .sort_values(
                "Total_Maintenance_avg",
                ascending=False,
            )
            .head(10)
        )

    fig = px.bar(
        compare_df,
        x="Power_Station",
        y=[
            "Total_Maintenance_avg",
            "Forced_Maintenance_avg",
        ],
        barmode="group",
        title="Maintenance Comparison",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    fig2 = px.bar(
        compare_df,
        x="Power_Station",
        y="Maintenance_Impact_Index_pct",
        title="Maintenance Impact Index",
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    st.dataframe(
        compare_df,
        use_container_width=True,
    )


# ================================================================
# MODEL PERFORMANCE
# ================================================================

elif page == "📈 Model Performance":

    hero(
        "Machine Learning Laboratory",
        "Compare trained models and evaluate prediction quality",
        "📈",
    )

    if model_comparison_df.empty:

        st.warning(
            "model_comparison.csv not found."
        )

        st.stop()

    st.dataframe(
        model_comparison_df,
        use_container_width=True,
    )

    best_model = model_comparison_df.loc[
        model_comparison_df["R2"].idxmax(),
        "Model",
    ]

    st.success(
        f"🏆 Best Performing Model: {best_model}"
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            model_comparison_df,
            x="Model",
            y="R2",
            title="R² Score",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        fig = px.bar(
            model_comparison_df,
            x="Model",
            y="MAE",
            title="Mean Absolute Error",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ================================================================
# EXPLAINABLE AI
# ================================================================

elif page == "🧠 Explainable AI":

    hero(
        "Explainable Artificial Intelligence",
        "Understand why the Random Forest model makes its prediction",
        "🧠",
    )

    if not feature_importance_df.empty:

        top10 = (
            feature_importance_df
            .head(10)
        )

        fig = px.bar(
            top10,
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance",
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis={
                "categoryorder":
                "total ascending"
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    if MODELS_OK:

        station = st.selectbox(
            "Select Station",
            sorted(
                df["Power_Station"]
                .dropna()
                .unique()
                .tolist()
            ),
            key="xai_station",
        )

        station_records = (
            df[
                df["Power_Station"] == station
            ]
            .reset_index(drop=True)
        )

        if not station_records.empty:

            idx = st.slider(
                "Select Record",
                0,
                len(station_records) - 1,
                0,
            )

            row = station_records.iloc[
                idx
            ]

            feature_row = (
                row[
                    MODEL_FEATURE_COLUMNS
                ]
                .to_frame()
                .T
            )

            try:

                bias, contributions, reconstructed = (
                    prediction.explain_prediction(
                        models[
                            "Random_Forest"
                        ],
                        feature_row,
                    )
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    kpi(
                        "📌",
                        "Actual",
                        f"{row['Actual']:.2f} MW",
                    )

                with c2:
                    kpi(
                        "🤖",
                        "Prediction",
                        f"{reconstructed:.2f} MW",
                    )

                with c3:
                    kpi(
                        "🎯",
                        "Error",
                        f"{abs(row['Actual'] - reconstructed):.2f} MW",
                    )

                series = (
                    pd.Series(
                        contributions
                    )
                    .sort_values(
                        key=abs,
                        ascending=False,
                    )
                )

                fig = go.Figure(
                    go.Bar(
                        x=series.values,
                        y=series.index,
                        orientation="h",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    title="Individual Prediction Drivers",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Could not generate explanation: {e}"
                )


# ================================================================
# REPORTS
# ================================================================

elif page == "📄 Reports":

    hero(
        "Automated Performance Reports",
        "Generate station-wise operational intelligence reports",
        "📄",
    )

    station = st.selectbox(
        "Select Station",
        sorted(
            df["Power_Station"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="report_station",
    )

    station_df = df[
        df["Power_Station"] == station
    ]

    station_row = station_perf[
        station_perf["Power_Station"] == station
    ]

    if st.button(
        "📄 GENERATE REPORT",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating intelligent report..."
        ):

            time.sleep(0.8)

            cfg = alerts.AlertConfig()

            station_alerts = (
                alerts.evaluate_dataframe(
                    station_df,
                    cfg,
                )
            )

            lines = [
                f"# PowerGenAI Performance Report — {station}",
                "",
                f"**Records analysed:** {len(station_df)}",
                "",
                "## Generation KPIs",
                "",
                f"- Average Programme: {utils.fmt_mw(station_df['Programme'].mean())}",
                f"- Average Actual: {utils.fmt_mw(station_df['Actual'].mean())}",
                f"- Average Shortfall/Excess: {utils.fmt_mw(station_df['Excess_Shortfall'].mean())}",
                "",
                "## Maintenance KPIs",
                "",
                f"- Average Total Maintenance: {utils.fmt_mw(station_df['Total_Maintenance'].mean())}",
                f"- Average Forced Maintenance: {utils.fmt_mw(station_df['Forced_Maintenance'].mean())}",
                "",
            ]

            if not station_row.empty:

                r = station_row.iloc[0]

                lines += [
                    "## Station Performance",
                    "",
                    f"- Programme Achievement: {utils.fmt_pct(r['Programme_Achievement_pct'])}",
                    f"- Capacity Utilization: {utils.fmt_pct(r['Capacity_Utilization_pct'])}",
                    f"- Maintenance Impact Index: {utils.fmt_pct(r['Maintenance_Impact_Index_pct'])}",
                    "",
                ]

            lines.append(
                "## Alerts"
            )

            if station_alerts.empty:

                lines.append(
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

                    lines.append(
                        f"- {alert_type}: {count} record(s)"
                    )

            report = "\n".join(
                lines
            )

            st.markdown(
                report
            )

            st.download_button(
                "⬇️ DOWNLOAD MARKDOWN REPORT",
                report,
                file_name=(
                    f"PowerGenAI_Report_"
                    f"{station}.md"
                ),
                mime="text/markdown",
                use_container_width=True,
            )


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    """
<div class="footer">

⚡ <b>PowerGenAI</b> · AI-Based Power Generation
Forecasting & Performance Monitoring

<br><br>

Machine Learning · Predictive Analytics ·
Explainable AI · Interactive Visualization

</div>
""",
    unsafe_allow_html=True,
)
