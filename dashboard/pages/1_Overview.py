"""
Overview / Home page
────────────────────
KPI cards: Total Traffic, Attacks Detected, Normal Traffic, Model Accuracy.
All numbers come from the SQLite DB and the saved best model metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go

from src.db import get_kpi_stats, get_attack_distribution, init_db
from src.config import MODELS_DIR, REPORTS_DIR

init_db()

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Overview")
st.caption("Live KPIs pulled from the detection database and trained model metrics.")

# ── KPI Stats ─────────────────────────────────────────────────────────────────
stats = get_kpi_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌐 Total Traffic",    f"{stats['total']:,}")
col2.metric("🚨 Attacks Detected", f"{stats['attacks']:,}",
            delta=f"{stats['attack_rate']:.1f}% attack rate",
            delta_color="inverse")
col3.metric("✅ Normal Traffic",   f"{stats['normal']:,}")

# Model accuracy from reports
def _read_best_accuracy() -> str:
    best_path = MODELS_DIR / "best_model.txt"
    if not best_path.exists():
        return "Not trained yet"
    model_name = best_path.read_text().strip()
    report_path = REPORTS_DIR / "model_comparison_report.md"
    if not report_path.exists():
        return "Report missing"
    # Parse the table for the best model's accuracy
    for line in report_path.read_text().split("\n"):
        if line.startswith(f"| {model_name} "):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 2:
                return parts[2]   # accuracy column
    return "n/a"

col4.metric("🤖 Model Accuracy", _read_best_accuracy())

st.divider()

# ── Attack distribution mini-chart ────────────────────────────────────────────
dist = get_attack_distribution()

if dist:
    st.subheader("Attack Type Breakdown (all time)")
    labels  = [d["attack_type"] for d in dist]
    values  = [d["count"]       for d in dist]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        textinfo="label+percent",
        marker_colors=[
            "#ef5350","#ab47bc","#42a5f5","#26a69a",
            "#ffca28","#8d6e63","#78909c","#66bb6a",
        ],
    ))
    fig.update_layout(
        showlegend=True,
        height=380,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No traffic data yet. Run the live monitoring feed or upload a CSV.")

# ── Recent activity summary ───────────────────────────────────────────────────
st.divider()
st.subheader("Recent Activity")
from src.db import get_recent_traffic
import pandas as pd

rows = get_recent_traffic(20)
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(
        df[["timestamp","source_ip","attack_type","severity","confidence"]]
          .rename(columns={"timestamp":"Time","source_ip":"Source IP",
                           "attack_type":"Attack Type","severity":"Severity",
                           "confidence":"Confidence"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No records in the database yet.")
