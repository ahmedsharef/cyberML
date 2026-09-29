"""
Traffic Over Time page
──────────────────────
Line chart of traffic volume (total vs attacks) over time, pulled from the DB.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db import get_traffic_over_time, init_db

init_db()
st.set_page_config(page_title="Traffic Over Time", page_icon="📈", layout="wide")
st.title("📈 Traffic Over Time")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    window_hours = st.select_slider(
        "Time window",
        options=[1, 3, 6, 12, 24, 48, 168],
        value=24,
        format_func=lambda h: f"{h}h" if h < 24 else (f"{h//24}d" if h >= 24 else f"{h}h"),
    )
with col2:
    auto_refresh = st.checkbox("Auto-refresh (every 10s)", value=False)

if auto_refresh:
    import time as _time
    st.caption("🔄 Auto-refreshing…")
    _time.sleep(10)
    st.rerun()

# ── Data ──────────────────────────────────────────────────────────────────────
data = get_traffic_over_time(hours=window_hours)

if not data:
    st.info(
        "No traffic data in the selected window. "
        "Start the Live Monitoring feed to generate data."
    )
    st.stop()

df = pd.DataFrame(data)
df["bucket"]  = pd.to_datetime(df["bucket"])
df["normal"]  = df["total"] - df["attacks"]

# ── Main line chart ───────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["bucket"], y=df["total"],
    name="Total Traffic",
    line=dict(color="#42a5f5", width=2),
    fill="tozeroy", fillcolor="rgba(66,165,245,0.1)",
))
fig.add_trace(go.Scatter(
    x=df["bucket"], y=df["attacks"],
    name="Attacks",
    line=dict(color="#ef5350", width=2),
    fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
))
fig.add_trace(go.Scatter(
    x=df["bucket"], y=df["normal"],
    name="Normal",
    line=dict(color="#66bb6a", width=1.5, dash="dot"),
))

fig.update_layout(
    height=450,
    hovermode="x unified",
    margin=dict(t=20, b=40, l=50, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    legend=dict(orientation="h", y=-0.2),
    xaxis_title="Time (UTC)",
    yaxis_title="Packet Count",
    xaxis=dict(showgrid=True, gridcolor="#333"),
    yaxis=dict(showgrid=True, gridcolor="#333"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"Summary — last {window_hours}h")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",  int(df["total"].sum()))
c2.metric("Total Attacks",  int(df["attacks"].sum()))
c3.metric("Peak Traffic",   int(df["total"].max()),    help="Max per-minute count")
c4.metric("Peak Attacks",   int(df["attacks"].max()),  help="Max attacks in one minute")

# ── Attack rate trend ─────────────────────────────────────────────────────────
df["attack_rate"] = (df["attacks"] / df["total"].replace(0, 1) * 100).round(2)

fig2 = go.Figure(go.Scatter(
    x=df["bucket"], y=df["attack_rate"],
    mode="lines+markers",
    line=dict(color="#ffca28", width=2),
    name="Attack Rate (%)",
    fill="tozeroy", fillcolor="rgba(255,202,40,0.1)",
))
fig2.update_layout(
    height=250,
    margin=dict(t=20, b=40, l=50, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    xaxis_title="Time (UTC)",
    yaxis_title="Attack Rate (%)",
    xaxis=dict(showgrid=True, gridcolor="#333"),
    yaxis=dict(showgrid=True, gridcolor="#333", range=[0, 100]),
)
st.subheader("Attack Rate Over Time (%)")
st.plotly_chart(fig2, use_container_width=True)
