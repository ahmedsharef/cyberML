"""
Alert & Response page
─────────────────────
Shows open high-severity alerts with a "Security Alert" banner,
recent notifications log, and a manual threat injection button for demo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.db import (
    get_open_alerts_count,
    get_recent_alerts,
    get_recent_traffic,
    init_db,
    log_traffic,
    update_alert_status,
)
from src.alerting import process_alert
from src.predict import load_pipeline, predict

init_db()
st.set_page_config(page_title="Alert & Response", page_icon="⚡", layout="wide")
st.title("⚡ Alert & Response")

# ── Open alert banner ─────────────────────────────────────────────────────────
open_count = get_open_alerts_count()

if open_count > 0:
    st.markdown(
        f'<div style="background:#d32f2f;color:white;padding:12px 16px;'
        f'border-radius:8px;font-weight:bold;font-size:16px;margin-bottom:1rem;">'
        f'🚨 SECURITY ALERT — {open_count} unresolved attack alert(s) detected!</div>',
        unsafe_allow_html=True,
    )
else:
    st.success("✅ All clear — no open alerts at this time.")

st.divider()

# ── Recent alerts ─────────────────────────────────────────────────────────────
st.subheader("Recent Alerts")
alerts = get_recent_alerts(50)

if alerts:
    df_alerts = pd.DataFrame(alerts)
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
    # Highlight open/high
    def _style_row(row):
        if row["status"] == "open" and row["severity"] == "high":
            return ["background-color: rgba(211,47,47,0.2)"] * len(row)
        elif row["status"] == "open":
            return ["background-color: rgba(245,124,0,0.1)"] * len(row)
        return [""] * len(row)

    styled = (
        df_alerts[[
            "id","timestamp","attack_type","severity","source_ip","dest_ip","status","notes"
        ]]
        .rename(columns={
            "id":"ID","timestamp":"Time","attack_type":"Attack Type",
            "severity":"Severity","source_ip":"Source IP","dest_ip":"Dest IP",
            "status":"Status","notes":"Notes"
        })
    )
    st.dataframe(styled.style.apply(_style_row, axis=1), use_container_width=True, hide_index=True)
else:
    st.info("No alerts in the database.")

st.divider()

# ── Manual threat injection (demo button) ─────────────────────────────────────
st.subheader("🎯 Inject Simulated Threat (Demo)")
st.caption(
    "Use this to simulate a detected attack and demonstrate the full alert pipeline "
    "without needing live traffic."
)

@st.cache_resource
def _load_bundle():
    try:
        return load_pipeline()
    except Exception:
        return None

bundle = _load_bundle()

col1, col2, col3 = st.columns(3)
sim_attack_type = col1.selectbox(
    "Attack Type",
    ["ddos","malware","ransomware","phishing","sql_injection","mitm","other"],
)
sim_source_ip = col2.text_input("Source IP", value="192.168.1.100")
sim_confidence = col3.slider("Confidence", 0.0, 1.0, 0.92)

if st.button("🚨 Inject Alert", use_container_width=False):
    # Build a fake prediction matching the selected type
    fake_pred = {
        "is_attack":   True,
        "attack_type": sim_attack_type,
        "confidence":  sim_confidence,
    }
    fake_record = {
        "source_ip":  sim_source_ip,
        "dest_ip":    "10.0.0.1",
        "protocol_type": "tcp",
    }
    tid = log_traffic(fake_record, fake_pred)
    aid = process_alert(tid, fake_pred, fake_record)

    st.error(
        f"🚨 Alert #{aid} generated — "
        f"**{sim_attack_type.upper()}** from {sim_source_ip} "
        f"(confidence: {sim_confidence:.0%})"
    )
    st.rerun()

st.divider()

# ── Notification log ──────────────────────────────────────────────────────────
st.subheader("Notification Log")

from sqlalchemy.orm import Session as _Sess
from src.db import _get_engine, AlertNotification

with _Sess(bind=_get_engine()) as sess:
    notifs = sess.query(AlertNotification).order_by(
        AlertNotification.timestamp.desc()
    ).limit(50).all()
    notif_rows = [
        {
            "ID":        n.id,
            "Time":      n.timestamp.isoformat() if n.timestamp else "",
            "Alert ID":  n.alert_id,
            "Channel":   n.channel,
            "Status":    n.status,
            "Message":   n.message[:80],
        }
        for n in notifs
    ]

if notif_rows:
    st.dataframe(pd.DataFrame(notif_rows), use_container_width=True, hide_index=True)
else:
    st.info("No notifications logged yet.")

# ── Bulk resolve ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bulk Resolve")
if st.button("✅ Acknowledge all open alerts"):
    open_alerts = [a for a in get_recent_alerts(500) if a["status"] == "open"]
    for a in open_alerts:
        update_alert_status(a["id"], "acknowledged")
    st.success(f"Acknowledged {len(open_alerts)} alerts.")
    st.rerun()
