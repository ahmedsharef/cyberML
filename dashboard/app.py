"""
app.py
──────
Streamlit entry point.

Run:
    streamlit run dashboard/app.py

Pages are in dashboard/pages/ and are auto-discovered by Streamlit's
multi-page mechanism (filenames prefixed with a number for ordering).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.db import init_db

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CyberShield — ML Cyberattack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure DB tables exist ────────────────────────────────────────────────────
init_db()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar nav */
    [data-testid="stSidebarNav"] { padding-top: 1rem; }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Alert banner */
    .alert-banner {
        background: #d32f2f;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Severity badges */
    .badge-high   { background:#d32f2f; color:white; padding:3px 10px; border-radius:12px; font-size:12px; }
    .badge-medium { background:#f57c00; color:white; padding:3px 10px; border-radius:12px; font-size:12px; }
    .badge-low    { background:#388e3c; color:white; padding:3px 10px; border-radius:12px; font-size:12px; }
    .badge-none   { background:#757575; color:white; padding:3px 10px; border-radius:12px; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Home / Overview (landing page shown when no page is selected) ─────────────
st.title("🛡️ CyberShield — ML Cyberattack Detection System")
st.markdown(
    """
    Welcome to the **Admin Dashboard**. Use the sidebar to navigate between pages:

    | Page | Description |
    |------|-------------|
    | 📊 Overview | KPI summary cards |
    | 📡 Live Monitoring | Simulated real-time traffic feed |
    | 🥧 Attack Distribution | Donut chart of attack categories |
    | 📈 Traffic Over Time | Line chart of traffic volume |
    | 🚨 Alerts & Reports | Recent alerts table with filters |
    | 🤖 Model Performance | Metrics and comparison report |
    | ⚙️ Settings | Thresholds, model selection, feed speed |

    ---
    **Quick start:**
    1. Add your dataset files to `data/raw/` (see `data/download_instructions.md`)
    2. Run `python -m src.train_models` to train all models
    3. Come back here and explore the dashboard
    """
)

# Live alert count in sidebar
from src.db import get_open_alerts_count
open_alerts = get_open_alerts_count()
if open_alerts > 0:
    st.sidebar.error(f"🚨 {open_alerts} open alert{'s' if open_alerts != 1 else ''}")
else:
    st.sidebar.success("✅ No open alerts")
