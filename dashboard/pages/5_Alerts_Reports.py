"""
Alerts & Detection Reports page
────────────────────────────────
Table of recent alerts with filtering by severity / attack type / status.
Allows acknowledging / resolving alerts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.db import get_recent_alerts, update_alert_status, init_db

init_db()
st.set_page_config(page_title="Alerts & Reports", page_icon="🚨", layout="wide")
st.title("🚨 Alerts & Detection Reports")

# ── Load alerts ───────────────────────────────────────────────────────────────
alerts = get_recent_alerts(200)

if not alerts:
    st.info("No alerts yet. Alerts are generated automatically when attacks are detected.")
    st.stop()

df = pd.DataFrame(alerts)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sev_filter  = st.multiselect("Severity",    ["high","medium","low"],
                                  default=["high","medium","low"])
    type_filter = st.multiselect("Attack Type", sorted(df["attack_type"].unique()),
                                  default=sorted(df["attack_type"].unique()))
    stat_filter = st.multiselect("Status",      ["open","acknowledged","resolved"],
                                  default=["open","acknowledged"])
    search_ip   = st.text_input("Filter by Source IP (partial match)")

# Apply filters
mask = (
    df["severity"].isin(sev_filter) &
    df["attack_type"].isin(type_filter) &
    df["status"].isin(stat_filter)
)
if search_ip:
    mask = mask & df["source_ip"].str.contains(search_ip, na=False)

df_filtered = df[mask].copy()

st.caption(f"Showing {len(df_filtered)} of {len(df)} alerts.")

# ── Alert banner for open high-severity ───────────────────────────────────────
open_high = df[(df["severity"] == "high") & (df["status"] == "open")]
if not open_high.empty:
    st.markdown(
        f'<div class="alert-banner">⚠️ {len(open_high)} HIGH-severity alert(s) require attention!</div>',
        unsafe_allow_html=True,
    )

# ── Severity colour helper ─────────────────────────────────────────────────────
def _badge(sev: str) -> str:
    mapping = {
        "high":   '<span class="badge-high">HIGH</span>',
        "medium": '<span class="badge-medium">MEDIUM</span>',
        "low":    '<span class="badge-low">LOW</span>',
    }
    return mapping.get(sev, sev)

# ── Main table ────────────────────────────────────────────────────────────────
st.subheader("Alert Log")

# Format for display
display_df = df_filtered[[
    "id","timestamp","source_ip","dest_ip","attack_type","severity","status","notes"
]].copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

st.dataframe(
    display_df.rename(columns={
        "id":"ID", "timestamp":"Time", "source_ip":"Source IP",
        "dest_ip":"Dest IP", "attack_type":"Attack Type",
        "severity":"Severity", "status":"Status", "notes":"Notes",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Alert action panel ────────────────────────────────────────────────────────
st.subheader("Update Alert Status")

col1, col2, col3 = st.columns(3)
alert_id_input = col1.number_input("Alert ID", min_value=1, step=1, value=int(df_filtered["id"].iloc[0]) if len(df_filtered) else 1)
new_status     = col2.selectbox("New Status", ["acknowledged","resolved","open"])

if col3.button("Update", use_container_width=True):
    update_alert_status(alert_id_input, new_status)
    st.success(f"Alert #{alert_id_input} updated to '{new_status}'.")
    st.rerun()

st.divider()

# ── Summary stats ─────────────────────────────────────────────────────────────
st.subheader("Alert Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Open",         int((df["status"] == "open").sum()))
c2.metric("Acknowledged", int((df["status"] == "acknowledged").sum()))
c3.metric("Resolved",     int((df["status"] == "resolved").sum()))

# Attack-type breakdown of alerts
by_type = df.groupby("attack_type").size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(
    by_type.rename(columns={"attack_type":"Attack Type","count":"Alert Count"}),
    use_container_width=True, hide_index=True,
)
