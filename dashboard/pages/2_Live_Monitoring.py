"""
Live Monitoring page
────────────────────
Simulates a live traffic feed by replaying test-set rows at a configurable
speed. Each row is classified, logged to the DB, and displayed in a streaming
table with colour-coded severity.

The user can also upload their own CSV for batch classification.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.db import init_db, log_traffic
from src.alerting import process_alert
from src.config import LIVE_FEED_SPEED, DATA_PROCESSED_DIR
from src.predict import load_pipeline, predict_batch_fast

init_db()
st.set_page_config(page_title="Live Monitoring", page_icon="📡", layout="wide")
st.title("📡 Live Monitoring")

# ── Session state ─────────────────────────────────────────────────────────────
if "live_results" not in st.session_state:
    st.session_state.live_results = []
if "feed_running" not in st.session_state:
    st.session_state.feed_running = False

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Feed Controls")
    feed_speed = st.slider("Rows per second", 1, 20, LIVE_FEED_SPEED)
    max_rows   = st.slider("Max rows to process", 50, 2000, 200)
    start_btn  = st.button("▶ Start Live Feed", use_container_width=True)
    stop_btn   = st.button("⏹ Stop", use_container_width=True)

if stop_btn:
    st.session_state.feed_running = False

# ── Load model pipeline (cached) ──────────────────────────────────────────────
@st.cache_resource
def _load():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None

bundle = _load()

# ── Upload section ────────────────────────────────────────────────────────────
st.subheader("Upload CSV for batch classification")
uploaded = st.file_uploader(
    "Upload a CSV file matching the dataset schema", type=["csv"]
)

if uploaded:
    df_up = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df_up):,} rows × {df_up.shape[1]} columns")

    if bundle is None:
        st.error("No trained model found. Run `python -m src.train_models` first.")
    else:
        if st.button("🔍 Classify uploaded data"):
            with st.spinner("Running predictions..."):
                from src.predict import predict_batch
                results_df = predict_batch(df_up, bundle)

            st.success(f"Done — {results_df['is_attack'].sum()} attacks detected out of {len(results_df)} records.")
            st.dataframe(
                results_df[["is_attack","attack_type","confidence"]]
                          .rename(columns={"is_attack":"Is Attack",
                                           "attack_type":"Attack Type",
                                           "confidence":"Confidence"}),
                use_container_width=True,
            )
            # Download button
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button("⬇ Download results CSV", csv_bytes,
                               "results.csv", "text/csv")

st.divider()

# ── Live feed ─────────────────────────────────────────────────────────────────
st.subheader("Live Traffic Stream")

if bundle is None:
    st.warning("⚠️ No trained model found. Run `python -m src.train_models` first, then refresh.")
    st.stop()

# Load test set for simulation
X_test_path = DATA_PROCESSED_DIR / "X_test.npy"
y_test_path = DATA_PROCESSED_DIR / "y_bin_test.npy"

if not X_test_path.exists():
    st.warning("Processed test data not found. Run training first.")
    st.stop()

X_test  = np.load(X_test_path)
y_true  = np.load(y_test_path)

# Apply feature selection if stored in meta
meta = bundle.preprocessing_meta
selected = meta.get("selected_features")
feature_names = meta["feature_names"]
if selected is not None:
    indices = [feature_names.index(f) for f in selected if f in feature_names]
    X_test  = X_test[:, indices]

# Counters placeholder
kpi_placeholder   = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

SEVERITY_COLORS = {"high": "🔴", "medium": "🟠", "low": "🟡", "none": "🟢"}

def _render_table(rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows[-100:])   # show last 100
    df["sev"] = df["severity"].map(SEVERITY_COLORS).fillna("⚪")
    df["display"] = df["sev"] + " " + df["attack_type"].str.upper()
    table_placeholder.dataframe(
        df[["timestamp","source_ip","display","confidence","is_attack"]]
          .rename(columns={"timestamp":"Time","source_ip":"Source IP",
                           "display":"Detection","confidence":"Confidence",
                           "is_attack":"Attack?"}),
        use_container_width=True,
        hide_index=True,
    )

def _render_kpis(rows: list[dict]):
    total   = len(rows)
    attacks = sum(1 for r in rows if r["is_attack"])
    kpi_placeholder.columns(1)
    c1, c2, c3 = kpi_placeholder.columns(3)
    c1.metric("Total Seen",       total)
    c2.metric("Attacks",          attacks,  delta=f"{attacks/total*100:.1f}%" if total else "0%")
    c3.metric("Normal",           total - attacks)

def _rolling_chart(rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Group by batches of 10
    df["batch"] = np.arange(len(df)) // 10
    grouped = df.groupby("batch").agg(
        total=("is_attack", "count"),
        attacks=("is_attack", "sum"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["batch"], y=grouped["total"],
                             name="Total", line=dict(color="#42a5f5")))
    fig.add_trace(go.Scatter(x=grouped["batch"], y=grouped["attacks"],
                             name="Attacks", line=dict(color="#ef5350")))
    fig.update_layout(
        height=250, margin=dict(t=10,b=30,l=40,r=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
        legend=dict(orientation="h"),
        xaxis_title="Batch (10 rows)", yaxis_title="Count",
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

if start_btn:
    st.session_state.feed_running = True
    st.session_state.live_results = []

if st.session_state.feed_running:
    import random, string

    def _fake_ip():
        return ".".join(str(random.randint(1, 254)) for _ in range(4))

    n = min(max_rows, len(X_test))
    indices = np.random.choice(len(X_test), size=n, replace=False)
    delay   = 1.0 / feed_speed

    for i, idx in enumerate(indices):
        if not st.session_state.feed_running:
            break

        x_row = X_test[idx:idx+1]
        res_df = predict_batch_fast(x_row, bundle)
        result = res_df.iloc[0]

        from src.config import SEVERITY_MAPPING
        severity = SEVERITY_MAPPING.get(result["attack_type"], "none")

        row_data = {
            "timestamp":   pd.Timestamp.utcnow().isoformat(),
            "source_ip":   _fake_ip(),
            "dest_ip":     _fake_ip(),
            "is_attack":   bool(result["is_attack"]),
            "attack_type": result["attack_type"],
            "confidence":  round(float(result["confidence"]), 4),
            "severity":    severity,
        }
        st.session_state.live_results.append(row_data)

        # Write to DB
        record   = {"source_ip": row_data["source_ip"], "dest_ip": row_data["dest_ip"]}
        pred_out = {"is_attack": row_data["is_attack"],
                    "attack_type": row_data["attack_type"],
                    "confidence": row_data["confidence"]}
        tid = log_traffic(record, pred_out)
        if row_data["is_attack"]:
            process_alert(tid, pred_out, record)

        # Render
        _render_kpis(st.session_state.live_results)
        _render_table(st.session_state.live_results)
        _rolling_chart(st.session_state.live_results)

        time.sleep(delay)

    st.session_state.feed_running = False
    st.success(f"✅ Feed complete — processed {len(st.session_state.live_results)} records.")
else:
    # Show existing results if any
    if st.session_state.live_results:
        _render_kpis(st.session_state.live_results)
        _render_table(st.session_state.live_results)
        _rolling_chart(st.session_state.live_results)
    else:
        st.info("Press ▶ Start Live Feed in the sidebar to begin simulation.")
