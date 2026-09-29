"""
Model Performance page
──────────────────────
Shows the accuracy gauge for the best model and renders the full
model_comparison_report.md with embedded confusion matrix / ROC images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")
st.title("🤖 Model Performance")

# ── Best model info ───────────────────────────────────────────────────────────
best_path = MODELS_DIR / "best_model.txt"

if not best_path.exists():
    st.warning(
        "No trained model found. Run `python -m src.train_models` from the "
        "project root, then refresh this page."
    )
    st.stop()

best_model_name = best_path.read_text().strip()
st.success(f"Best model: **{best_model_name}**")

# ── Parse accuracy from report ────────────────────────────────────────────────
def _parse_metrics_from_report() -> dict:
    """Pull the best model's row from the comparison table."""
    report_path = REPORTS_DIR / "model_comparison_report.md"
    if not report_path.exists():
        return {}
    metrics = {}
    for line in report_path.read_text().split("\n"):
        if line.startswith(f"| {best_model_name} "):
            parts = [p.strip() for p in line.split("|")]
            # Table: Model | Accuracy | Precision | Recall | Weighted F1 | Macro F1 | ROC-AUC
            if len(parts) >= 8:
                try:
                    metrics = {
                        "accuracy":    float(parts[2]),
                        "precision":   float(parts[3]),
                        "recall":      float(parts[4]),
                        "weighted_f1": float(parts[5]),
                        "macro_f1":    float(parts[6]),
                        "roc_auc":     parts[7] if parts[7] != "n/a" else None,
                    }
                except (ValueError, IndexError):
                    pass
            break
    return metrics

metrics = _parse_metrics_from_report()

# ── KPI row ───────────────────────────────────────────────────────────────────
if metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",    f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Precision",   f"{metrics['precision']*100:.2f}%")
    c3.metric("Recall",      f"{metrics['recall']*100:.2f}%")
    c4.metric("Weighted F1", f"{metrics['weighted_f1']*100:.2f}%")
    c5.metric("Macro F1",    f"{metrics['macro_f1']*100:.2f}%")

    # ── Accuracy Gauge ────────────────────────────────────────────────────────
    st.subheader("Accuracy Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=metrics["accuracy"] * 100,
        number={"suffix": "%", "font": {"size": 36}},
        delta={"reference": 90, "increasing": {"color": "#66bb6a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar": {"color": "#42a5f5"},
            "steps": [
                {"range": [0,  60], "color": "#d32f2f"},
                {"range": [60, 80], "color": "#f57c00"},
                {"range": [80, 100],"color": "#388e3c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 95,
            },
        },
        title={"text": f"Best Model Accuracy — {best_model_name}"},
    ))
    fig_gauge.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(t=30, b=20, l=30, r=30),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ── Comparison figures ────────────────────────────────────────────────────────
st.subheader("Model Comparison Charts")

wf1_img  = FIGURES_DIR / "model_comparison_weighted_f1.png"
acc_img   = FIGURES_DIR / "model_comparison_accuracy.png"

if wf1_img.exists() and acc_img.exists():
    col1, col2 = st.columns(2)
    col1.image(str(wf1_img), caption="Weighted F1 by Model", use_column_width=True)
    col2.image(str(acc_img), caption="Accuracy by Model",    use_column_width=True)
else:
    st.info("Comparison charts not found — run training first.")

st.divider()

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("Confusion Matrices (Multi-class)")

model_names = ["random_forest","decision_tree","svm","xgboost","mlp"]
cols = st.columns(min(3, len(model_names)))

for i, name in enumerate(model_names):
    cm_img = FIGURES_DIR / f"cm_{name}_multi.png"
    if cm_img.exists():
        cols[i % 3].image(str(cm_img), caption=name, use_column_width=True)

st.divider()

# ── Full report markdown ──────────────────────────────────────────────────────
st.subheader("Full Comparison Report")
report_path = REPORTS_DIR / "model_comparison_report.md"
if report_path.exists():
    with st.expander("📄 View full report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.info("Report not yet generated. Run training first.")
