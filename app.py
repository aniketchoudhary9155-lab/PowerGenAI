"""
=============================================================
POWERGENAI V2
AI-Based Power Generation Forecasting & Monitoring System
=============================================================

Run:
    streamlit run app.py

=============================================================
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(CURRENT_DIR).lower() == "dashboard":
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 2. PROJECT IMPORTS
# ============================================================

try:
    import prediction
    import analytics
    import alerts
    import utils
    from features import MODEL_FEATURE_COLUMNS

    IMPORT_OK = True
    IMPORT_ERROR = ""

except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


# ============================================================
# 3. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerGenAI | AI Power Control Room",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 4. PREMIUM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(37,99,235,0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6,182,212,0.12),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(124,58,237,0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #071426 45%,
                #020617 100%
            );

        color: #e2e8f0;
    }


    /* =====================================================
       ANIMATED BACKGROUND
       ===================================================== */

    .stApp::before {
        content: "";
        position: fixed;
        width: 500px;
        height: 500px;
        left: -180px;
        top: 15%;
        background: rgba(37,99,235,0.08);
        border-radius: 50%;
        filter: blur(100px);
        animation: floatOrb 12s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: "";
        position: fixed;
        width: 450px;
        height: 450px;
        right: -180px;
        bottom: 10%;
        background: rgba(6,182,212,0.07);
        border-radius: 50%;
        filter: blur(100px);
        animation: floatOrb2 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes floatOrb {

        0%,100% {
            transform: translate(0,0) scale(1);
        }

        50% {
            transform: translate(100px,-60px) scale(1.2);
        }

    }

    @keyframes floatOrb2 {

        0%,100% {
            transform: translate(0,0) scale(1);
        }

        50% {
            transform: translate(-100px,50px) scale(1.15);
        }

    }


    /* =====================================================
       MAIN CONTAINER
       ===================================================== */

    .main .block-container {
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        position: relative;
        z-index: 1;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                rgba(2,6,23,0.98),
                rgba(7,20,38,0.96)
            );

        border-right: 1px solid rgba(148,163,184,0.10);

    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }


    /* =====================================================
       SIDEBAR BRAND
       ===================================================== */

    .brand {

        text-align: center;

        padding: 12px 5px 20px;

    }

    .brand-icon {

        font-size: 3.2rem;

        display: inline-block;

        animation: pulsePower 2s infinite;

        filter:
            drop-shadow(0 0 10px rgba(59,130,246,0.8));

    }

    @keyframes pulsePower {

        0%,100% {
            transform: scale(1);
            filter:
                drop-shadow(0 0 8px rgba(59,130,246,0.6));
        }

        50% {
            transform: scale(1.12);
            filter:
                drop-shadow(0 0 25px rgba(59,130,246,1));
        }

    }

    .brand-name {

        font-size: 1.65rem;
        font-weight: 900;
        color: white;

    }

    .brand-sub {

        font-size: 0.72rem;
        color: #64748b;

    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {

        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                135deg,
                rgba(15,42,75,0.88),
                rgba(8,20,37,0.78)
            );

        border:
            1px solid rgba(96,165,250,0.20);

        border-radius: 26px;

        padding: 32px;

        margin-bottom: 25px;

        box-shadow:
            0 20px 60px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.05);

        backdrop-filter: blur(20px);

    }

    .hero::after {

        content: "";

        position: absolute;

        width: 350px;
        height: 350px;

        right: -120px;
        top: -180px;

        background:
            radial-gradient(
                circle,
                rgba(59,130,246,0.30),
                transparent 65%
            );

        animation: heroGlow 7s ease-in-out infinite;

    }

    @keyframes heroGlow {

        0%,100% {
            transform: scale(1);
            opacity: 0.7;
        }

        50% {
            transform: scale(1.25);
            opacity: 1;
        }

    }

    .hero-title {

        font-size: 2.65rem;
        font-weight: 900;
        color: white;
        letter-spacing: -1.5px;

    }

    .hero-subtitle {

        color: #93c5fd;
        margin-top: 5px;
        font-size: 1rem;

    }


    /* =====================================================
       LIVE INDICATOR
       ===================================================== */

    .live {

        display: inline-flex;

        align-items: center;

        gap: 8px;

        padding: 7px 13px;

        border-radius: 999px;

        background:
            rgba(34,197,94,0.10);

        border:
            1px solid rgba(34,197,94,0.25);

        color: #86efac;

        font-size: 0.75rem;

        font-weight: 800;

        margin-top: 15px;

    }

    .live-dot {

        width: 9px;
        height: 9px;

        background: #22c55e;

        border-radius: 50%;

        box-shadow:
            0 0 0 0 rgba(34,197,94,0.7);

        animation: livePulse 1.5s infinite;

    }

    @keyframes livePulse {

        0% {
            box-shadow:
                0 0 0 0 rgba(34,197,94,0.7);
        }

        70% {
            box-shadow:
                0 0 0 9px rgba(34,197,94,0);
        }

        100% {
            box-shadow:
                0 0 0 0 rgba(34,197,94,0);
        }

    }


    /* =====================================================
       GLASS KPI
       ===================================================== */

    .glass-card {

        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.70),
                rgba(15,23,42,0.55)
            );

        border:
            1px solid rgba(148,163,184,0.12);

        border-radius: 20px;

        padding: 20px;

        min-height: 130px;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.04);

        backdrop-filter: blur(18px);

        transition:
            transform 0.30s ease,
            border-color 0.30s ease,
            box-shadow 0.30s ease;

        animation: cardAppear 0.65s ease both;

    }

    .glass-card:hover {

        transform:
            translateY(-7px)
            scale(1.015);

        border-color:
            rgba(96,165,250,0.38);

        box-shadow:
            0 20px 45px rgba(0,0,0,0.38),
            0 0 25px rgba(37,99,235,0.08);

    }

    @keyframes cardAppear {

        from {
            opacity: 0;
            transform: translateY(15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }

    .metric-label {

        color: #94a3b8;

        font-size: 0.72rem;

        text-transform: uppercase;

        letter-spacing: 1px;

        font-weight: 700;

    }

    .metric-number {

        color: #f8fafc;

        font-size: 1.7rem;

        font-weight: 900;

        margin-top: 8px;

    }

    .metric-info {

        color: #60a5fa;

        font-size: 0.72rem;

        margin-top: 7px;

    }


    /* =====================================================
       SECTION
       ===================================================== */

    .section {

        background:
            rgba(15,23,42,0.58);

        border:
            1px solid rgba(148,163,184,0.10);

        border-radius: 20px;

        padding: 20px;

        margin: 12px 0;

        backdrop-filter: blur(16px);

    }


    /* =====================================================
       STATUS
       ===================================================== */

    .status-good {

        display: inline-block;

        background:
            rgba(34,197,94,0.12);

        color: #86efac;

        border:
            1px solid rgba(34,197,94,0.20);

        border-radius: 999px;

        padding: 6px 12px;

        font-size: 0.75rem;

        font-weight: 800;

    }

    .status-warning {

        display: inline-block;

        background:
            rgba(245,158,11,0.12);

        color: #fcd34d;

        border:
            1px solid rgba(245,158,11,0.20);

        border-radius: 999px;

        padding: 6px 12px;

        font-size: 0.75rem;

        font-weight: 800;

    }

    .status-danger {

        display: inline-block;

        background:
            rgba(239,68,68,0.12);

        color: #fca5a5;

        border:
            1px solid rgba(239,68,68,0.20);

        border-radius: 999px;

        padding: 6px 12px;

        font-size: 0.75rem;

        font-weight: 800;

    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {

        border-radius: 12px !important;

        border:
            1px solid rgba(96,165,250,0.25) !important;

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #1d4ed8
            ) !important;

        color: white !important;

        font-weight: 800 !important;

        transition:
            all 0.25s ease !important;

    }

    .stButton > button:hover {

        transform:
            translateY(-3px);

        box-shadow:
            0 10px 30px rgba(37,99,235,0.35);

    }


    /* =====================================================
       INPUTS
       ===================================================== */

    div[data-baseweb="select"] > div {

        background:
            rgba(15,23,42,0.75) !important;

        border-radius: 12px !important;

        border:
            1px solid rgba(148,163,184,0.15) !important;

    }

    input {

        background:
            rgba(15,23,42,0.65) !important;

        color: white !important;

    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {

        transition: all 0.25s ease;

    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        color: #64748b;

        font-size: 0.75rem;

        padding: 25px;

        margin-top: 40px;

    }


    /* =====================================================
       SCROLLBAR
       ===================================================== */

    ::-webkit-scrollbar {

        width: 7px;

    }

    ::-webkit-scrollbar-track {

        background: #020617;

    }

    ::-webkit-scrollbar-thumb {

        background: #1e40af;

        border-radius: 20px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def safe_mw(value):

    try:
        return f"{float(value):,.2f} MW"
    except Exception:
        return "N/A"


def safe_pct(value):

    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "N/A"


def glass_metric(
    title,
    value,
    subtitle="",
    icon="⚡"
):

    st.markdown(
        f"""
        <div class="glass-card">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
            ">

                <div class="metric-label">
                    {title}
                </div>

                <div style="
                    font-size:1.5rem;
                ">
                    {icon}
                </div>

            </div>

            <div class="metric-number">
                {value}
            </div>

            <div class="metric-info">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def risk_html(risk):

    risk = str(risk).upper()

    if risk == "LOW":

        return (
            '<span class="status-good">'
            '🟢 LOW RISK'
            '</span>'
        )

    if risk == "MEDIUM":

        return (
            '<span class="status-warning">'
            '🟡 MEDIUM RISK'
            '</span>'
        )

    if risk == "HIGH":

        return (
            '<span class="status-warning">'
            '🟠 HIGH RISK'
            '</span>'
        )

    return (
        '<span class="status-danger">'
        '🔴 CRITICAL RISK'
        '</span>'
    )


def section_header(
    title,
    subtitle=""
):

    st.markdown(
        f"""
        <div style="
            margin:18px 0 10px 0;
        ">

            <div style="
                font-size:1.25rem;
                font-weight:800;
                color:#f8fafc;
            ">
                {title}
            </div>

            <div style="
                font-size:0.78rem;
                color:#64748b;
                margin-top:3px;
            ">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    features_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "powergeneration_features.csv"
    )

    station_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "station_performance_summary.csv"
    )

    df = pd.read_csv(features_file)

    station_perf = pd.read_csv(
        station_file
    )

    return df, station_perf


# ============================================================
# 7. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    names = [
        "Random_Forest",
        "Linear_Regression",
        "Gradient_Boosting",
        "HistGB_XGBoost_substitute"
    ]

    loaded = {}
    errors = {}

    for name in names:

        path = os.path.join(
            PROJECT_ROOT,
            "models",
            f"{name}.joblib"
        )

        if not os.path.exists(path):
            continue

        try:

            loaded[name] = prediction.load_model(
                name
            )

        except Exception as e:

            errors[name] = str(e)

    return loaded, errors


# ============================================================
# 8. OTHER MODEL FILES
# ============================================================

@st.cache_data
def load_model_comparison():

    path = os.path.join(
        PROJECT_ROOT,
        "models",
        "model_comparison.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_feature_importance():

    path = os.path.join(
        PROJECT_ROOT,
        "models",
        "feature_importance.csv"
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# 9. INITIALIZE
# ============================================================

if not IMPORT_OK:

    st.error(
        "❌ PowerGenAI modules could not be imported."
    )

    st.code(
        IMPORT_ERROR,
        language="text"
    )

    st.stop()


try:

    df, station_perf = load_data()

    DATA_OK = True

except Exception as e:

    DATA_OK = False

    st.error(
        "❌ PowerGenAI data could not be loaded."
    )

    st.code(
        str(e),
        language="text"
    )

    st.info(
        "Check your data/processed folder."
    )

    st.stop()


try:

    models, model_errors = load_models()

    MODELS_OK = (
        "Random_Forest" in models
    )

except Exception as e:

    models = {}

    model_errors = {
        "System": str(e)
    }

    MODELS_OK = False


model_comparison_df = (
    load_model_comparison()
)

feature_importance_df = (
    load_feature_importance()
)


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                ⚡
            </div>

            <div class="brand-name">
                PowerGenAI
            </div>

            <div class="brand-sub">
                AI POWER CONTROL ROOM
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### 🧭 Navigation"
    )

    page = st.radio(
        "Select Module",
        [
            "📊 Command Center",
            "🏭 Station Intelligence",
            "🔮 AI Prediction",
            "🎛️ What-If Simulator",
            "🔧 Maintenance AI",
            "📈 Model Performance",
            "🧠 Explainable AI",
            "🚨 Alert Center",
            "📄 Smart Reports",
            "📋 Data Explorer"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### 🖥️ System Status"
    )

    st.markdown(
        '<span class="status-good">'
        '● SYSTEM ONLINE'
        '</span>',
        unsafe_allow_html=True
    )

    if MODELS_OK:

        st.markdown(
            '<span class="status-good">'
            '● AI MODEL READY'
            '</span>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<span class="status-danger">'
            '● AI MODEL OFFLINE'
            '</span>',
            unsafe_allow_html=True
        )

    st.divider()

    if st.button(
        "🔄 Refresh System",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    st.caption(
        "PowerGenAI v2.0"
    )

    st.caption(
        "AI-Based Power Generation Forecasting"
    )


# ============================================================
# 11. COMMAND CENTER
# ============================================================

if page == "📊 Command Center":

    now = datetime.now().strftime(
        "%d %b %Y • %I:%M:%S %p"
    )

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-title">
                ⚡ PowerGenAI Command Center
            </div>

            <div class="hero-subtitle">
                Intelligent power generation forecasting,
                performance monitoring & operational analytics
            </div>

            <div class="live">
                <span class="live-dot"></span>
                LIVE MONITORING • {now}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    station_filter = st.selectbox(
        "🏭 Monitor Station",
        ["All Stations"] +
        sorted(
            df["Power_Station"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if station_filter == "All Stations":

        view = df.copy()

    else:

        view = df[
            df["Power_Station"] ==
            station_filter
        ].copy()

    if view.empty:

        st.warning(
            "No records available."
        )

        st.stop()

    stations = view[
        "Power_Station"
    ].nunique()

    programme = view[
        "Programme"
    ].sum()

    actual = view[
        "Actual"
    ].sum()

    shortfall = view.loc[
        view["Excess_Shortfall"] < 0,
        "Excess_Shortfall"
    ].sum()

    maintenance = (
        view["Total_Maintenance"].sum()
        if "Total_Maintenance" in view.columns
        else 0
    )

    achievement = (
        actual / programme * 100
        if programme
        else 0
    )

    capacity = (
        view["Monitored_Capacity"].sum()
        if "Monitored_Capacity" in view.columns
        else 0
    )

    # --------------------------------------------------------
    # KPI ROW 1
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        glass_metric(
            "Power Stations",
            f"{stations}",
            "Active monitored stations",
            "🏭"
        )

    with c2:

        glass_metric(
            "Total Capacity",
            safe_mw(capacity),
            "Monitored generation capacity",
            "⚡"
        )

    with c3:

        glass_metric(
            "Programme",
            safe_mw(programme),
            "Target generation",
            "🎯"
        )

    with c4:

        glass_metric(
            "Actual Generation",
            safe_mw(actual),
            "Recorded generation",
            "🔋"
        )

    st.write("")

    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        glass_metric(
            "Achievement",
            safe_pct(achievement),
            "Programme achievement",
            "📈"
        )

    with c2:

        glass_metric(
            "Shortfall",
            safe_mw(shortfall),
            "Generation deficit",
            "⚠️"
        )

    with c3:

        glass_metric(
            "Maintenance",
            safe_mw(maintenance),
            "Total maintenance impact",
            "🔧"
        )

    with c4:

        glass_metric(
            "Data Records",
            f"{len(view):,}",
            "Operational records",
            "📊"
        )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE GAUGE
    # --------------------------------------------------------

    section_header(
        "🎯 Overall Station Performance",
        "AI monitoring overview"
    )

    gauge_col, insight_col = st.columns(
        [1.1, 1]
    )

    with gauge_col:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=min(
                    max(achievement, 0),
                    120
                ),
                number={
                    "suffix": "%"
                },
                title={
                    "text":
                    "Programme Achievement"
                },
                gauge={
                    "axis": {
                        "range": [0, 120]
                    },
                    "bar": {
                        "thickness": 0.25
                    },
                    "threshold": {
                        "line": {
                            "width": 5
                        },
                        "value": 100
                    }
                }
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    with insight_col:

        st.markdown(
            """
            <div class="section">

                <h3>🤖 AI Operational Insight</h3>

                <p>
                PowerGenAI continuously evaluates generation,
                programme achievement and maintenance impact
                to support operational decision-making.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if achievement >= 100:

            st.success(
                "🟢 Generation is meeting or exceeding the programme."
            )

        elif achievement >= 90:

            st.warning(
                "🟡 Generation is slightly below the programme."
            )

        else:

            st.error(
                "🔴 Significant generation shortfall detected."
            )

        if maintenance > 0:

            st.info(
                "🔧 Maintenance activity is contributing "
                "to available-capacity reduction."
            )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        section_header(
            "📈 Generation Distribution",
            "Actual generation frequency"
        )

        fig = px.histogram(
            view,
            x="Actual",
            nbins=55
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

        section_header(
            "🔧 Maintenance Composition",
            "Maintenance contribution by category"
        )

        maint_cols = [
            "Planned_Maintenance",
            "Forced_Maintenance",
            "Other_Reasons"
        ]

        maint_cols = [
            x for x in maint_cols
            if x in view.columns
        ]

        if maint_cols:

            values = view[
                maint_cols
            ].sum()

            fig = px.pie(
                names=values.index,
                values=values.values,
                hole=0.58
            )

            fig.update_layout(
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # LEADERBOARD
    # --------------------------------------------------------

    if station_filter == "All Stations":

        st.divider()

        section_header(
            "🏆 Station Performance Leaderboard",
            "Top 15 stations by total actual generation"
        )

        if not station_perf.empty:

            if "Actual_sum" in station_perf.columns:

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
                    text_auto=".2s"
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=520,
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
# 12. STATION INTELLIGENCE
# ============================================================

elif page == "🏭 Station Intelligence":

    st.title(
        "🏭 Power Station Intelligence"
    )

    station = st.selectbox(
        "Select Power Station",
        sorted(
            df["Power_Station"].unique()
        )
    )

    data = df[
        df["Power_Station"] ==
        station
    ].copy()

    summary = station_perf[
        station_perf["Power_Station"] ==
        station
    ]

    if data.empty:

        st.warning(
            "No data found."
        )

        st.stop()

    avg_programme = data[
        "Programme"
    ].mean()

    avg_actual = data[
        "Actual"
    ].mean()

    avg_shortfall = data[
        "Excess_Shortfall"
    ].mean()

    avg_capacity = data[
        "Monitored_Capacity"
    ].mean()

    achievement = (
        avg_actual /
        avg_programme *
        100
        if avg_programme
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        glass_metric(
            "Capacity",
            safe_mw(avg_capacity),
            "Average monitored capacity",
            "⚡"
        )

    with c2:

        glass_metric(
            "Programme",
            safe_mw(avg_programme),
            "Average target",
            "🎯"
        )

    with c3:

        glass_metric(
            "Actual",
            safe_mw(avg_actual),
            "Average generation",
            "🔋"
        )

    with c4:

        glass_metric(
            "Achievement",
            safe_pct(achievement),
            "Station performance",
            "📈"
        )

    st.divider()

    # Trend

    c1, c2 = st.columns(2)

    with c1:

        section_header(
            "📈 Generation Trend",
            station
        )

        chart_data = data.reset_index()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=chart_data["Programme"],
                mode="lines",
                name="Programme"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=chart_data["Actual"],
                mode="lines",
                name="Actual"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        section_header(
            "🔧 Maintenance Profile",
            station
        )

        cols = [
            "Planned_Maintenance",
            "Forced_Maintenance",
            "Other_Reasons"
        ]

        cols = [
            x for x in cols
            if x in data.columns
        ]

        if cols:

            maintenance = data[
                cols
            ].sum()

            fig = px.bar(
                x=maintenance.index,
                y=maintenance.values,
                text_auto=".2f"
            )

            fig.update_layout(
                template="plotly_dark",
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Summary

    if not summary.empty:

        st.divider()

        section_header(
            "📊 Station KPI Summary"
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 13. AI PREDICTION
# ============================================================

elif page == "🔮 AI Prediction":

    st.title(
        "🔮 AI Generation Prediction"
    )

    st.caption(
        "Random Forest based generation forecasting engine"
    )

    if not MODELS_OK:

        st.error(
            "❌ Random Forest model is unavailable."
        )

        if model_errors:

            st.json(
                model_errors
            )

        st.stop()

    stations = sorted(
        df["Power_Station"].unique()
    )

    with st.form(
        "ai_prediction_form"
    ):

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

        submit = st.form_submit_button(
            "⚡ RUN AI FORECAST",
            use_container_width=True
        )

    if submit:

        with st.spinner(
            "🤖 AI engine analysing operating conditions..."
        ):

            try:

                time.sleep(0.8)

                feature_row = (
                    prediction.build_feature_row(
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

                predicted = (
                    prediction.predict(
                        model,
                        feature_row
                    )
                )

                shortfall = (
                    predicted -
                    programme
                )

                achievement = (
                    predicted /
                    programme *
                    100
                    if programme
                    else 0
                )

                available = (
                    capacity -
                    planned -
                    forced -
                    other
                )

                risk = (
                    prediction.risk_level(
                        predicted,
                        programme
                    )
                )

                st.success(
                    "✅ AI forecast generated successfully."
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    glass_metric(
                        "Predicted Generation",
                        safe_mw(predicted),
                        "AI forecast",
                        "🔮"
                    )

                with c2:

                    glass_metric(
                        "Expected Difference",
                        safe_mw(shortfall),
                        "Prediction − programme",
                        "📊"
                    )

                with c3:

                    glass_metric(
                        "Achievement",
                        safe_pct(achievement),
                        "Expected programme achievement",
                        "🎯"
                    )

                st.write("")

                c1, c2, c3 = st.columns(3)

                with c1:

                    glass_metric(
                        "Available Capacity",
                        safe_mw(available),
                        "After maintenance",
                        "⚡"
                    )

                with c2:

                    glass_metric(
                        "Maintenance Load",
                        safe_mw(
                            planned +
                            forced +
                            other
                        ),
                        "Total maintenance",
                        "🔧"
                    )

                with c3:

                    st.markdown(
                        "### Risk Assessment"
                    )

                    st.markdown(
                        risk_html(risk),
                        unsafe_allow_html=True
                    )

                # Gauge

                st.divider()

                section_header(
                    "🎯 AI Performance Gauge"
                )

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=min(
                            max(
                                achievement,
                                0
                            ),
                            120
                        ),
                        number={
                            "suffix": "%"
                        },
                        title={
                            "text":
                            "Expected Achievement"
                        },
                        gauge={
                            "axis": {
                                "range":
                                [0, 120]
                            },
                            "bar": {
                                "thickness":
                                0.25
                            },
                            "threshold": {
                                "line": {
                                    "width": 5
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

                section_header(
                    "🧠 Why did the AI predict this?",
                    "Feature contribution analysis"
                )

                try:

                    (
                        bias,
                        contributions,
                        reconstruction
                    ) = prediction.explain_prediction(
                        model,
                        feature_row
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
                        template="plotly_dark",
                        height=500,
                        title=(
                            f"Prediction Drivers "
                            f"(Base: {bias:.2f} MW)"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                except Exception as e:

                    st.info(
                        "Detailed explanation unavailable."
                    )

            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.exception(e)


# ============================================================
# 14. WHAT-IF SIMULATOR
# ============================================================

elif page == "🎛️ What-If Simulator":

    st.title(
        "🎛️ What-If Scenario Simulator"
    )

    st.caption(
        "Change operating conditions and immediately evaluate AI-predicted generation."
    )

    if not MODELS_OK:

        st.error(
            "AI model unavailable."
        )

        st.stop()

    station = st.selectbox(
        "🏭 Select Station",
        sorted(
            df["Power_Station"].unique()
        )
    )

    station_data = df[
        df["Power_Station"] ==
        station
    ]

    base_capacity = float(
        station_data[
            "Monitored_Capacity"
        ].mean()
    )

    base_programme = float(
        station_data[
            "Programme"
        ].mean()
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        programme = st.slider(
            "🎯 Programme (MW)",
            0.0,
            max(
                base_capacity * 1.2,
                1
            ),
            min(
                base_programme,
                base_capacity * 1.2
            )
        )

        planned = st.slider(
            "🔧 Planned Maintenance",
            0.0,
            max(base_capacity, 1.0),
            0.0
        )

    with c2:

        forced = st.slider(
            "🚨 Forced Maintenance",
            0.0,
            max(base_capacity, 1.0),
            0.0
        )

        other = st.slider(
            "⚙️ Other Reasons",
            0.0,
            max(base_capacity, 1.0),
            0.0
        )

    try:

        features = (
            prediction.build_feature_row(
                station,
                base_capacity,
                programme,
                planned,
                forced,
                other
            )
        )

        predicted = prediction.predict(
            models["Random_Forest"],
            features
        )

        risk = prediction.risk_level(
            predicted,
            programme
        )

        available = (
            base_capacity -
            planned -
            forced -
            other
        )

        st.divider()

        section_header(
            "📊 Scenario Output",
            "AI simulated operational result"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            glass_metric(
                "Predicted Actual",
                safe_mw(predicted),
                "Scenario generation",
                "🔮"
            )

        with c2:

            glass_metric(
                "Programme",
                safe_mw(programme),
                "Target",
                "🎯"
            )

        with c3:

            glass_metric(
                "Available Capacity",
                safe_mw(available),
                "After maintenance",
                "⚡"
            )

        with c4:

            st.markdown(
                "### Scenario Risk"
            )

            st.markdown(
                risk_html(risk),
                unsafe_allow_html=True
            )

        chart_df = pd.DataFrame(
            {
                "Metric": [
                    "Programme",
                    "Predicted",
                    "Available"
                ],
                "MW": [
                    programme,
                    predicted,
                    available
                ]
            }
        )

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
            f"Scenario simulation failed: {e}"
        )


# ============================================================
# 15. MAINTENANCE AI
# ============================================================

elif page == "🔧 Maintenance AI":

    st.title(
        "🔧 Maintenance Intelligence"
    )

    st.caption(
        "Analyse maintenance patterns and identify high-impact stations."
    )

    cols = [
        "Planned_Maintenance",
        "Forced_Maintenance",
        "Other_Reasons"
    ]

    cols = [
        x for x in cols
        if x in df.columns
    ]

    if not cols:

        st.warning(
            "Maintenance data unavailable."
        )

        st.stop()

    totals = df[
        cols
    ].sum()

    total = totals.sum()

    c1, c2, c3 = st.columns(3)

    for i, col in enumerate(
        cols[:3]
    ):

        share = (
            totals[col] /
            total *
            100
            if total
            else 0
        )

        [c1, c2, c3][i].markdown(
            f"""
            <div class="glass-card">

                <div class="metric-label">
                    {col.replace("_"," ")}
                </div>

                <div class="metric-number">
                    {share:.2f}%
                </div>

                <div class="metric-info">
                    Maintenance share
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    fig = px.pie(
        values=totals.values,
        names=totals.index,
        hole=0.58
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if (
        not station_perf.empty
        and
        "Total_Maintenance_avg"
        in station_perf.columns
    ):

        section_header(
            "🏭 Maintenance Impact Ranking"
        )

        top = (
            station_perf
            .sort_values(
                "Total_Maintenance_avg",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            top,
            x="Total_Maintenance_avg",
            y="Power_Station",
            orientation="h",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
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
# 16. MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Machine Learning Performance"
    )

    if model_comparison_df.empty:

        st.warning(
            "model_comparison.csv was not found."
        )

        st.stop()

    st.dataframe(
        model_comparison_df,
        use_container_width=True,
        hide_index=True
    )

    if "R2" in model_comparison_df.columns:

        best_index = (
            model_comparison_df[
                "R2"
            ].idxmax()
        )

        best = model_comparison_df.loc[
            best_index
        ]

        st.success(
            f"🏆 Best Model: "
            f"{best['Model']} | "
            f"R² = {best['R2']:.4f}"
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
# 17. EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI"
    )

    st.caption(
        "Understand what drives PowerGenAI predictions."
    )

    if not feature_importance_df.empty:

        section_header(
            "🌍 Global Feature Importance",
            "Most influential features in the Random Forest model"
        )

        top = (
            feature_importance_df
            .head(15)
        )

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
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    if MODELS_OK:

        station = st.selectbox(
            "🏭 Station",
            sorted(
                df["Power_Station"].unique()
            ),
            key="xai_station"
        )

        records = df[
            df["Power_Station"] ==
            station
        ].reset_index(
            drop=True
        )

        if not records.empty:

            idx = st.slider(
                "📍 Select Record",
                0,
                len(records) - 1,
                0
            )

            row = records.iloc[
                idx
            ]

            try:

                feature_row = row[
                    MODEL_FEATURE_COLUMNS
                ].to_frame().T

                (
                    bias,
                    contributions,
                    reconstructed
                ) = prediction.explain_prediction(
                    models["Random_Forest"],
                    feature_row
                )

                c1, c2 = st.columns(2)

                with c1:

                    glass_metric(
                        "Actual",
                        safe_mw(
                            row["Actual"]
                        ),
                        "Observed generation",
                        "🔋"
                    )

                with c2:

                    glass_metric(
                        "AI Prediction",
                        safe_mw(
                            reconstructed
                        ),
                        "Model output",
                        "🤖"
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
                    template="plotly_dark",
                    height=550,
                    title="Prediction Contribution Map"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"XAI error: {e}"
                )


# ============================================================
# 18. ALERT CENTER
# ============================================================

elif page == "🚨 Alert Center":

    st.title(
        "🚨 Intelligent Alert Center"
    )

    st.caption(
        "Automatic detection of operational performance issues."
    )

    try:

        config = alerts.AlertConfig()

        alert_df = alerts.evaluate_dataframe(
            df,
            config
        )

        if alert_df.empty:

            st.markdown(
                """
                <div class="section"
                     style="text-align:center;padding:50px;">

                    <div style="
                        font-size:4rem;
                    ">
                        🟢
                    </div>

                    <h2>
                        All Systems Normal
                    </h2>

                    <p>
                        No operational alerts were detected
                        in the current dataset.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.error(
                f"🚨 {len(alert_df)} alerts detected."
            )

            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True
            )

            if "type" in alert_df.columns:

                counts = (
                    alert_df[
                        "type"
                    ]
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
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Alert engine error: {e}"
        )


# ============================================================
# 19. SMART REPORTS
# ============================================================

elif page == "📄 Smart Reports":

    st.title(
        "📄 Smart Performance Reports"
    )

    station = st.selectbox(
        "🏭 Select Station",
        sorted(
            df["Power_Station"].unique()
        ),
        key="report_station"
    )

    data = df[
        df["Power_Station"] ==
        station
    ]

    if st.button(
        "📑 GENERATE AI REPORT",
        use_container_width=True
    ):

        try:

            config = alerts.AlertConfig()

            alert_df = (
                alerts.evaluate_dataframe(
                    data,
                    config
                )
            )

            programme = (
                data["Programme"].mean()
            )

            actual = (
                data["Actual"].mean()
            )

            achievement = (
                actual /
                programme *
                100
                if programme
                else 0
            )

            report = f"""
# ⚡ PowerGenAI Performance Report

## Power Station

**{station}**

---

## 📊 Generation Performance

Records Analysed:
**{len(data)}**

Average Programme:
**{safe_mw(programme)}**

Average Actual Generation:
**{safe_mw(actual)}**

Average Shortfall / Excess:
**{safe_mw(data["Excess_Shortfall"].mean())}**

Programme Achievement:
**{safe_pct(achievement)}**

---

## 🔧 Maintenance Performance

Planned Maintenance:
**{safe_mw(data["Planned_Maintenance"].mean())}**

Forced Maintenance:
**{safe_mw(data["Forced_Maintenance"].mean())}**

Other Reasons:
**{safe_mw(data["Other_Reasons"].mean())}**

---

## 🚨 Alerts

Total Alerts:
**{len(alert_df)}**

---

## 🤖 PowerGenAI Summary

PowerGenAI analysed operational generation,
programme performance and maintenance conditions
for the selected station.

---

Generated by PowerGenAI V2
AI-Based Power Generation Forecasting &
Power Station Performance Monitoring System
"""

            st.markdown(
                report
            )

            st.download_button(
                "⬇️ DOWNLOAD REPORT",
                report,
                file_name=(
                    f"PowerGenAI_"
                    f"{station}_Report.md"
                ),
                mime="text/markdown",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Report generation failed: {e}"
            )


# ============================================================
# 20. DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.title(
        "📋 Power Generation Data Explorer"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        glass_metric(
            "Rows",
            f"{len(df):,}",
            "Dataset records",
            "📊"
        )

    with c2:

        glass_metric(
            "Features",
            f"{len(df.columns):,}",
            "Available variables",
            "🧬"
        )

    with c3:

        glass_metric(
            "Stations",
            f"{df['Power_Station'].nunique():,}",
            "Monitored stations",
            "🏭"
        )

    st.divider()

    station = st.selectbox(
        "🏭 Station Filter",
        ["All"] +
        sorted(
            df["Power_Station"].unique()
        )
    )

    if station == "All":

        filtered = df.copy()

    else:

        filtered = df[
            df["Power_Station"] ==
            station
        ].copy()

    search = st.text_input(
        "🔎 Search dataset"
    )

    if search:

        mask = filtered.astype(
            str
        ).apply(
            lambda column:
            column.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(
            axis=1
        )

        filtered = filtered[
            mask
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ DOWNLOAD FILTERED DATA",
        csv,
        "PowerGenAI_filtered_data.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# 21. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        ⚡ <b style="color:#93c5fd;">
        PowerGenAI V2
        </b>

        <br><br>

        AI-Based Power Generation Forecasting &
        Power Station Performance Monitoring System

        <br>

        <span style="color:#475569;">
        Intelligent Analytics • Predictive Monitoring •
        Explainable AI • Operational Decision Support
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
