"""Streamlit web app for salary prediction."""

import sys
from pathlib import Path

# Ensure the 'developer_salary_prediction' directory is on sys.path
# so that `from src.*` imports work regardless of working directory
# (Streamlit Cloud runs from the repo root).
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from src.infer import predict_salary, get_local_currency, valid_categories
from src.schema import SalaryInput

# Page configuration
st.set_page_config(
    page_title="Developer Salary Predictor | AI-Powered Predictions",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Mode CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════
   PALETTE
   bg-deep:     #0d1117      (near-black body)
   bg-surface:  #161b22      (card / sidebar)
   bg-elevated: #1c2333      (inputs, hover)
   border:      #30363d      (subtle edges)
   gold:        #f0b429      (primary accent – money)
   gold-dim:    #9e7a19      (muted gold)
   emerald:     #10b981      (success / positive)
   rose:        #f43f5e      (error)
   text:        #e6edf3      (body copy)
   text-muted:  #8b949e      (secondary text)
   ═══════════════════════════════════════════════════════════════════ */

/* ── Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after {
    box-sizing: border-box;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
.main .block-container { 
    max-width: 1200px; 
    padding: clamp(1rem, 3vw, 2rem);
    width: 100%;
}
[data-testid="stBottomBlockContainer"] { background: #0d1117 !important; }

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Typography ───────────────────────────────────────────────────── */
h1, h2, h3, h4, p, span, li, label, div {
    color: #e6edf3 !important;
}
h1 {
    font-weight: 800 !important;
    font-size: clamp(1.8rem, 5vw, 2.6rem) !important;
    letter-spacing: -0.03em;
    text-align: center;
    line-height: 1.2 !important;
}
h2 {
    font-weight: 700 !important;
    border-left: 4px solid #f0b429;
    padding-left: 0.9rem;
    margin-top: 1.8rem !important;
    font-size: clamp(1.3rem, 3.5vw, 1.75rem) !important;
}
h3 { 
    font-weight: 600 !important; 
    color: #f0b429 !important; 
    font-size: clamp(1.1rem, 2.5vw, 1.4rem) !important;
}

.subtitle {
    text-align: center;
    color: #8b949e !important;
    font-size: clamp(0.95rem, 2.5vw, 1.15rem);
    margin-bottom: 1.6rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    padding: 0 1rem;
}

/* ── Hero Feature Cards ───────────────────────────────────────────── */
.hero-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: clamp(1rem, 3vw, 1.6rem) clamp(0.8rem, 2.5vw, 1.4rem);
    text-align: center;
    transition: border-color 0.3s, transform 0.3s;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.hero-card:hover {
    border-color: #f0b429;
    transform: translateY(-4px);
}
.hero-card .icon { 
    font-size: clamp(1.5rem, 4vw, 2rem); 
    margin-bottom: 0.5rem; 
}
.hero-card .card-title {
    color: #f0b429 !important;
    font-weight: 700;
    font-size: clamp(1rem, 2vw, 1.1rem);
    margin-bottom: 0.35rem;
}
.hero-card .card-desc {
    color: #8b949e !important;
    font-size: clamp(0.85rem, 1.8vw, 0.92rem);
    line-height: 1.45;
}

/* ── Columns / Cards ──────────────────────────────────────────────── */
[data-testid="column"] {
    background: #161b22 !important;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: clamp(1rem, 2.5vw, 1.4rem) !important;
    margin: 0.35rem;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #0d1117 !important;
    border-right: 1px solid #1c2333 !important;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] hr {
    border-color: #30363d !important;
    opacity: 0.6;
}
[data-testid="stSidebar"] code {
    color: #f0b429 !important;
    background: #1c2333 !important;
}
[data-testid="stSidebar"] table {
    font-size: clamp(0.8rem, 1.5vw, 0.9rem) !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161b22;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #30363d;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 10px;
    font-weight: 600;
    padding: clamp(0.5rem, 1.5vw, 0.6rem) clamp(0.8rem, 2.5vw, 1.4rem);
    border: none !important;
    font-size: clamp(0.85rem, 1.8vw, 1rem);
    white-space: nowrap;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #1c2333 !important;
    color: #f0b429 !important;
    box-shadow: 0 0 12px rgba(240,180,41,0.12);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Inputs ───────────────────────────────────────────────────────── */
[data-testid="stSelectbox"],
[data-testid="stNumberInput"] {
    background: transparent !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #1c2333 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    min-height: 44px;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
    border-color: #f0b429 !important;
    box-shadow: 0 0 0 2px rgba(240,180,41,0.18) !important;
}
/* dropdown menu */
[data-baseweb="menu"], [data-baseweb="popover"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    max-height: 60vh;
    overflow-y: auto;
}
[data-baseweb="menu"] li { 
    color: #e6edf3 !important; 
    min-height: 44px;
    padding: 0.75rem 1rem;
    font-size: clamp(0.9rem, 2vw, 1rem);
}
[data-baseweb="menu"] li:hover { background: #1c2333 !important; }
[data-baseweb="menu"] li[aria-selected="true"] { background: #1c2333 !important; }

/* Input labels */
label {
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    margin-bottom: 0.5rem !important;
}

/* ── Primary Button ───────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #f0b429 0%, #d4990a 100%) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: clamp(0.75rem, 2vw, 0.85rem) clamp(1.5rem, 4vw, 2.4rem) !important;
    font-size: clamp(1rem, 2.2vw, 1.1rem) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 24px rgba(240,180,41,0.30);
    transition: all 0.25s ease;
    min-height: 48px;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 32px rgba(240,180,41,0.45) !important;
    filter: brightness(1.08);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Metrics ──────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: clamp(1.5rem, 5vw, 2.4rem) !important;
    font-weight: 800 !important;
    color: #10b981 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-weight: 500;
    font-size: clamp(0.85rem, 1.8vw, 1rem) !important;
}

/* ── Success / Info / Error Alerts ─────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    padding: clamp(0.8rem, 2vw, 1rem) !important;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
}
/* success */
div[data-testid="stAlert"][data-baseweb*="positive"],
.element-container .stAlert:has([data-testid="stMarkdownContainer"]) {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.30) !important;
    color: #34d399 !important;
}
/* info */
div[data-testid="stAlert"]:not([data-baseweb*="positive"]):not([data-baseweb*="negative"]) {
    background: rgba(240,180,41,0.06) !important;
    border: 1px solid rgba(240,180,41,0.20) !important;
}
/* error */
div[data-testid="stAlert"][data-baseweb*="negative"] {
    background: rgba(244,63,94,0.08) !important;
    border: 1px solid rgba(244,63,94,0.30) !important;
    color: #fb7185 !important;
}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg,
        transparent 0%, #30363d 20%, #f0b429 50%, #30363d 80%, transparent 100%) !important;
    margin: clamp(1rem, 4vw, 2rem) 0 !important;
    opacity: 0.6;
}

/* ── Footer ───────────────────────────────────────────────────────── */
.footer-dark {
    text-align: center;
    margin-top: clamp(2rem, 5vw, 3rem);
    padding: clamp(1rem, 3vw, 1.5rem) 0;
    border-top: 1px solid #30363d;
}
.footer-dark p {
    color: #484f58 !important;
    font-size: clamp(0.8rem, 1.6vw, 0.88rem) !important;
    margin: 0.25rem 0;
}
.footer-dark strong { color: #8b949e !important; }

/* ── Spinner ──────────────────────────────────────────────────────── */
.stSpinner > div > div { 
    border-top-color: #f0b429 !important; 
    width: clamp(40px, 8vw, 60px) !important;
    height: clamp(40px, 8vw, 60px) !important;
}

/* ── Summary card ─────────────────────────────────────────────────── */
.summary-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: clamp(1rem, 2.5vw, 1.2rem) clamp(1rem, 3vw, 1.4rem);
}
.summary-card li {
    color: #8b949e !important;
    padding: 0.15rem 0;
    font-size: clamp(0.85rem, 1.8vw, 0.95rem);
    line-height: 1.6;
}
.summary-card strong { 
    color: #e6edf3 !important; 
    font-size: clamp(0.85rem, 1.8vw, 0.95rem);
}
.summary-card ul {
    padding-left: 1.2rem;
    margin: 0;
}

/* ── Misc ─────────────────────────────────────────────────────────── */
[data-testid="stCaption"] { 
    color: #484f58 !important; 
    font-size: clamp(0.75rem, 1.5vw, 0.85rem) !important;
}
code { 
    color: #f0b429 !important; 
    background: #1c2333 !important; 
    border-radius: 4px; 
    padding: 0.15rem 0.4rem;
    font-size: clamp(0.8rem, 1.6vw, 0.9rem);
}

/* subtle glow on metric hover */
[data-testid="stMetric"] {
    transition: box-shadow 0.3s;
    border-radius: 12px;
    padding: clamp(0.3rem, 1vw, 0.5rem);
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 0 20px rgba(16,185,129,0.12);
}

/* animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-container { animation: fadeUp 0.45s ease-out; }

/* ═══════════════════════════════════════════════════════════════════
   RESPONSIVE MEDIA QUERIES
   ═══════════════════════════════════════════════════════════════════ */

/* Mobile devices (portrait, up to 600px) */
@media only screen and (max-width: 600px) {
    .main .block-container {
        padding: 1rem 0.5rem !important;
    }
    
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.3rem !important;
        margin-top: 1rem !important;
    }
    
    .subtitle {
        font-size: 0.95rem !important;
        padding: 0 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Stack columns vertically on mobile */
    [data-testid="column"] {
        margin: 0.5rem 0 !important;
        padding: 1rem !important;
    }
    
    /* Hero cards more compact */
    .hero-card {
        padding: 1rem 0.8rem;
        min-height: 120px;
        margin-bottom: 0.5rem;
    }
    
    .hero-card .icon {
        font-size: 1.5rem;
    }
    
    .hero-card .card-title {
        font-size: 1rem;
    }
    
    .hero-card .card-desc {
        font-size: 0.85rem;
    }
    
    /* Tabs take full width */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        padding: 0.5rem 0.8rem;
    }
    
    /* Button adjustments */
    .stButton > button {
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
    }
    
    /* Metrics smaller on mobile */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    
    /* Summary cards */
    .summary-card {
        padding: 1rem !important;
        margin-bottom: 1rem;
    }
    
    .summary-card li {
        font-size: 0.85rem;
    }
    
    /* Sidebar adjustments for mobile */
    [data-testid="stSidebar"] {
        max-width: 280px !important;
    }
    
    /* Footer text smaller */
    .footer-dark p {
        font-size: 0.8rem !important;
    }
    
    /* Inputs touch-friendly */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        min-height: 48px;
        font-size: 1rem !important;
    }
    
    [data-baseweb="menu"] li {
        min-height: 48px;
        padding: 0.85rem 1rem;
    }
}

/* Tablets (601px - 900px) */
@media only screen and (min-width: 601px) and (max-width: 900px) {
    .main .block-container {
        padding: 1.5rem 1rem !important;
    }
    
    h1 {
        font-size: 2.2rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    .subtitle {
        font-size: 1.05rem !important;
    }
    
    .hero-card {
        padding: 1.3rem 1.1rem;
    }
    
    [data-testid="column"] {
        padding: 1.2rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        padding: 0.55rem 1.2rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
}

/* Small desktops (901px - 1200px) */
@media only screen and (min-width: 901px) and (max-width: 1200px) {
    .main .block-container {
        max-width: 1100px;
        padding: 1.8rem !important;
    }
    
    h1 {
        font-size: 2.4rem !important;
    }
    
    .hero-card {
        padding: 1.5rem 1.3rem;
    }
}

/* Large screens (1201px+) */
@media only screen and (min-width: 1201px) {
    .main .block-container {
        max-width: 1200px;
        padding: 2rem !important;
    }
    
    .hero-card {
        padding: 1.6rem 1.4rem;
    }
}

/* Landscape mobile devices */
@media only screen and (max-height: 600px) and (orientation: landscape) {
    .main .block-container {
        padding: 0.5rem !important;
    }
    
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    .subtitle {
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem;
    }
    
    .hero-card {
        padding: 0.8rem 0.6rem;
        min-height: 100px;
    }
    
    .hero-card .icon {
        font-size: 1.3rem;
        margin-bottom: 0.3rem;
    }
}

/* High DPI displays */
@media only screen and (-webkit-min-device-pixel-ratio: 2),
       only screen and (min-resolution: 192dpi) {
    body {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
}

/* Print styles */
@media print {
    .stButton,
    [data-testid="stSidebar"],
    .footer-dark {
        display: none !important;
    }
    
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("<h1>💰 Developer Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Predict compensation using ML models trained on Stack Overflow survey data</p>",
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""<div class='hero-card'>
        <div class='icon'>🎯</div>
        <div class='card-title'>Accurate</div>
        <div class='card-desc'>XGBoost gradient boosting trained on real developer survey data</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""<div class='hero-card'>
        <div class='icon'>🌍</div>
        <div class='card-title'>Global</div>
        <div class='card-desc'>{len(valid_categories['Country'])}+ countries with local currency conversion</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    st.markdown("""<div class='hero-card'>
        <div class='icon'>⚡</div>
        <div class='card-title'>Instant</div>
        <div class='card-desc'>Get your personalized salary estimate in seconds</div>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 About")
    st.markdown("---")

    st.markdown("""
**XGBoost** model trained on the Stack Overflow Developer Survey.

#### Prediction Factors
| | Factor |
|---|---|
| 🌍 | Country |
| 💻 | Coding experience |
| 👔 | Work experience |
| 🎓 | Education |
| 🔧 | Developer type |
| 🏢 | Industry |
| 👤 | Age |
| 👥 | IC / Manager |
    """)

    st.info("💡 Results are estimates based on survey averages.")

    st.markdown("---")
    st.markdown("#### Coverage")

    coverage_data = {
        "🌍 Countries": len(valid_categories['Country']),
        "🎓 Education": len(valid_categories['EdLevel']),
        "👨‍💻 Dev Types": len(valid_categories['DevType']),
        "🏢 Industries": len(valid_categories['Industry']),
        "📅 Age Ranges": len(valid_categories['Age']),
        "👥 Roles": len(valid_categories['ICorPM']),
    }

    for label, count in coverage_data.items():
        st.markdown(f"**{label}:** `{count}`")

    st.caption("Only values from the training set are available.")

    st.markdown("---")
    st.markdown("#### Tech Stack")
    st.markdown("Streamlit · XGBoost · Pydantic · Pandas")

# Main input form
st.markdown("---")
st.header("🔍 Enter Developer Information")
st.markdown("Fill in the details below to get your salary prediction")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "💼 Professional Info", "🎯 Generate Prediction"])

# Get valid categories from training
valid_countries = valid_categories["Country"]
valid_education_levels = valid_categories["EdLevel"]
valid_dev_types = valid_categories["DevType"]
valid_industries = valid_categories["Industry"]
valid_ages = valid_categories["Age"]
valid_icorpm = valid_categories["ICorPM"]

# Set default values (if available)
default_country = (
    "United States of America"
    if "United States of America" in valid_countries
    else valid_countries[0]
)
default_education = (
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
    if "Bachelor's degree (B.A., B.S., B.Eng., etc.)" in valid_education_levels
    else valid_education_levels[0]
)
default_dev_type = (
    "Developer, back-end"
    if "Developer, back-end" in valid_dev_types
    else valid_dev_types[0]
)
default_industry = (
    "Software Development"
    if "Software Development" in valid_industries
    else valid_industries[0]
)
default_age = "25-34 years old" if "25-34 years old" in valid_ages else valid_ages[0]
default_icorpm = (
    "Individual contributor"
    if "Individual contributor" in valid_icorpm
    else valid_icorpm[0]
)

# Tab 1: Personal Information
with tab1:
    st.markdown("### 📍 Location & Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox(
            "🌍 Country",
            options=valid_countries,
            index=valid_countries.index(default_country),
            help="Your country of residence (impacts salary significantly)",
        )
        
        age = st.selectbox(
            "👤 Age Range",
            options=valid_ages,
            index=valid_ages.index(default_age),
            help="Your current age range",
        )
    
    with col2:
        education = st.selectbox(
            "🎓 Education Level",
            options=valid_education_levels,
            index=valid_education_levels.index(default_education),
            help="Your highest level of education completed",
        )
        
        ic_or_pm = st.selectbox(
            "👥 Role Type",
            options=valid_icorpm,
            index=valid_icorpm.index(default_icorpm),
            help="Are you an individual contributor or people manager?",
        )

# Tab 2: Professional Information
with tab2:
    st.markdown("### 💼 Experience & Specialization")
    
    col3, col4 = st.columns(2)
    
    with col3:
        years = st.number_input(
            "💻 Total Years of Coding",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Including education, how many years have you been coding?",
        )
        
        dev_type = st.selectbox(
            "🔧 Developer Type",
            options=valid_dev_types,
            index=valid_dev_types.index(default_dev_type),
            help="Your primary developer role or specialization",
        )
    
    with col4:
        work_exp = st.number_input(
            "👔 Years of Professional Experience",
            min_value=0,
            max_value=50,
            value=3,
            step=1,
            help="Years of professional work experience (not including education)",
        )
        
        industry = st.selectbox(
            "🏢 Industry",
            options=valid_industries,
            index=valid_industries.index(default_industry),
            help="The industry sector you work in",
        )

# Tab 3: Prediction
with tab3:
    st.markdown("### 🎯 Ready to Predict?")
    st.markdown("Review your information and hit the button below.")
    
    # Display summary
    st.markdown("#### 📋 Summary")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""<div class='summary-card'>
        <ul>
            <li><strong>Country:</strong> {country}</li>
            <li><strong>Age:</strong> {age}</li>
            <li><strong>Education:</strong> {education}</li>
            <li><strong>Role:</strong> {ic_or_pm}</li>
        </ul></div>""", unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""<div class='summary-card'>
        <ul>
            <li><strong>Coding Years:</strong> {years}</li>
            <li><strong>Work Exp:</strong> {work_exp} yrs</li>
            <li><strong>Dev Type:</strong> {dev_type}</li>
            <li><strong>Industry:</strong> {industry}</li>
        </ul></div>""", unsafe_allow_html=True)
    
    st.markdown("---")

    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict My Salary", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Create input model
            input_data = SalaryInput(
                country=country,
                years_code=years,
                work_exp=work_exp,
                education_level=education,
                dev_type=dev_type,
                industry=industry,
                age=age,
                ic_or_pm=ic_or_pm,
            )

            # Make prediction
            with st.spinner("🤖 AI is analyzing your profile..."):
                salary = predict_salary(input_data)

            # Display result with animation
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.balloons()
            st.success("✅ Prediction Complete!")
            
            st.markdown("### 💵 Your Predicted Salary")

            # Show USD and local currency side by side
            local = get_local_currency(country, salary)
            if local and local["code"] != "USD":
                col_usd, col_local = st.columns(2)
                with col_usd:
                    st.metric(
                        label="💵 Annual Salary (USD)",
                        value=f"${salary:,.0f}",
                        help="Predicted annual compensation in US Dollars",
                    )
                with col_local:
                    st.metric(
                        label=f"💰 Annual Salary ({local['code']})",
                        value=f"{local['salary_local']:,.0f} {local['code']}",
                        help=f"Converted using survey rate: 1 USD = {local['rate']} {local['code']} ({local['name']})",
                    )
                
                # Additional insights
                st.markdown("---")
                st.markdown("#### 📊 Additional Insights")
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    monthly_usd = salary / 12
                    st.metric("📅 Monthly (USD)", f"${monthly_usd:,.0f}")
                
                with insight_col2:
                    hourly_usd = salary / (52 * 40)  # Assuming 40 hours/week
                    st.metric("⏰ Hourly (USD)", f"${hourly_usd:,.0f}")
                
                with insight_col3:
                    if local:
                        monthly_local = local['salary_local'] / 12
                        st.metric(f"📅 Monthly ({local['code']})", f"{monthly_local:,.0f}")
            else:
                st.metric(
                    label="💵 Estimated Annual Salary",
                    value=f"${salary:,.0f}",
                    help="Predicted annual compensation in USD",
                )
                
                # Additional insights for USD only
                st.markdown("---")
                st.markdown("#### 📊 Salary Breakdown")
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    monthly_usd = salary / 12
                    st.metric("📅 Monthly", f"${monthly_usd:,.0f}")
                
                with insight_col2:
                    hourly_usd = salary / (52 * 40)
                    st.metric("⏰ Hourly", f"${hourly_usd:,.0f}")
                
                with insight_col3:
                    weekly_usd = salary / 52
                    st.metric("📆 Weekly", f"${weekly_usd:,.0f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Disclaimer
            st.info("ℹ️ **Note:** This prediction is based on survey data and represents an estimate. Actual salaries may vary based on company size, specific skills, location within country, and other factors not captured in this model.")

        except FileNotFoundError:
            st.error(
                """
                ❌ **Model Not Found!** 
                
                Please train the model first by running:
                ```bash
                python src/train.py
                ```
                """
            )
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            st.exception(e)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class='footer-dark'>
    <p><strong>Developer Salary Predictor</strong></p>
    <p>Streamlit · Stack Overflow Survey · XGBoost</p>
    <p style='margin-top:0.5rem;'>© 2026 — built for developers</p>
</div>
""", unsafe_allow_html=True)

