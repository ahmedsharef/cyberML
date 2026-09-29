"""
Attack Distribution page
────────────────────────
Donut chart of detected attack types from the DB, plus a bar breakdown.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import get_attack_distribution, get_recent_traffic, init_db

init_db()
st.set_page_config(page_title="Attack Distribution", page_icon="🥧", layout="wide")
st.title("🥧 Attack Distribution")
st.caption("All attack classifications logged in the database.")

dist = get_attack_distribution()

if not dist:
    st.info("No attack data yet. Start the live monitoring feed or classify some traffic first.")
    st.stop()

df_dist = pd.DataFrame(dist)
df_dist = df_dist.sort_values("count", ascending=False)

# Colour palette matching the project slides
COLORS = {
    "ddos":          "#ef5350",
    "malware":       "#ab47bc",
    "phishing":      "#42a5f5",
    "ransomware":    "#ff7043",
    "sql_injection": "#26a69a",
    "mitm":          "#ffca28",
    "other":         "#8d6e63",
    "normal":        "#66bb6a",
}
palette = [COLORS.get(t, "#90a4ae") for t in df_dist["attack_type"]]

col_left, col_right = st.columns(2)

# ── Donut chart ───────────────────────────────────────────────────────────────
with col_left:
    st.subheader("Attack Category Donut")
    fig_donut = go.Figure(go.Pie(
        labels=df_dist["attack_type"],
        values=df_dist["count"],
        hole=0.50,
        textinfo="label+percent",
        marker=dict(colors=palette),
        pull=[0.05 if i == 0 else 0 for i in range(len(df_dist))],
    ))
    fig_donut.update_layout(
        height=420,
        margin=dict(t=20,b=20,l=20,r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        showlegend=True,
        legend=dict(orientation="v", x=1.02),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Bar chart ─────────────────────────────────────────────────────────────────
with col_right:
    st.subheader("Attack Count by Type")
    fig_bar = px.bar(
        df_dist,
        x="attack_type", y="count",
        color="attack_type",
        color_discrete_map=COLORS,
        labels={"attack_type": "Attack Type", "count": "Count"},
        text_auto=True,
    )
    fig_bar.update_layout(
        height=420,
        showlegend=False,
        margin=dict(t=20,b=40,l=40,r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Severity breakdown ────────────────────────────────────────────────────────
st.subheader("Severity Breakdown")
rows = get_recent_traffic(5000)
if rows:
    df_all  = pd.DataFrame(rows)
    df_atk  = df_all[df_all["is_attack"] == True]
    if not df_atk.empty:
        sev_counts = df_atk["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        sev_colors = {"high": "#d32f2f", "medium": "#f57c00", "low": "#388e3c"}
        fig_sev = px.bar(
            sev_counts, x="severity", y="count",
            color="severity",
            color_discrete_map=sev_colors,
            text_auto=True,
            labels={"severity":"Severity Level","count":"Count"},
        )
        fig_sev.update_layout(
            height=300, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
        )
        st.plotly_chart(fig_sev, use_container_width=True)

# Raw counts table
st.subheader("Raw Counts")
st.dataframe(
    df_dist.rename(columns={"attack_type": "Attack Type", "count": "Count"}),
    use_container_width=True, hide_index=True,
)
