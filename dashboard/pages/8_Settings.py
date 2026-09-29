"""
System Settings page
────────────────────
Configure alert thresholds, choose the active model, and toggle feed speed.
Changes are written to .env (in the project root) so they persist across
sessions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import joblib

from src.config import (
    MODELS_DIR,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
    LIVE_FEED_SPEED,
    DATASET,
    PROJECT_ROOT,
)

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ System Settings")
st.caption("Changes are saved to `.env` in the project root and take effect on next restart.")

# ── Discover available trained models ─────────────────────────────────────────
def _available_models() -> list[str]:
    binary_files = list(MODELS_DIR.glob("*_binary.joblib"))
    return sorted(set(f.stem.replace("_binary", "") for f in binary_files))

available = _available_models()
best_path  = MODELS_DIR / "best_model.txt"
current_best = best_path.read_text().strip() if best_path.exists() else (available[0] if available else "none")

st.divider()

# ── Model selection ───────────────────────────────────────────────────────────
st.subheader("🤖 Active Model")

if not available:
    st.warning("No trained models found. Run `python -m src.train_models` first.")
else:
    selected_model = st.selectbox(
        "Choose the active model for the prediction pipeline",
        available,
        index=available.index(current_best) if current_best in available else 0,
    )
    if st.button("Set as Active Model"):
        best_path.write_text(selected_model)
        st.success(f"Active model set to **{selected_model}**.")
        # Clear the cached pipeline
        st.cache_resource.clear()

st.divider()

# ── Threshold settings ────────────────────────────────────────────────────────
st.subheader("🚦 Alert Severity Thresholds")
st.caption(
    "Confidence scores below MEDIUM_THRESHOLD generate low-severity alerts; "
    "above HIGH_THRESHOLD generate high-severity. Between = medium."
)

col1, col2 = st.columns(2)
new_high   = col1.slider("HIGH threshold",   0.5, 1.0, SEVERITY_HIGH_THRESHOLD,   step=0.01)
new_medium = col2.slider("MEDIUM threshold", 0.3, 0.9, SEVERITY_MEDIUM_THRESHOLD, step=0.01)

if new_high <= new_medium:
    st.error("HIGH threshold must be greater than MEDIUM threshold.")

st.divider()

# ── Feed speed ────────────────────────────────────────────────────────────────
st.subheader("📡 Live Feed Speed")
new_speed = st.slider("Rows per second (default)", 1, 30, LIVE_FEED_SPEED)

st.divider()

# ── Dataset selection ─────────────────────────────────────────────────────────
st.subheader("📂 Dataset")
new_dataset = st.radio(
    "Active dataset",
    ["nsl_kdd", "unsw_nb15"],
    index=0 if DATASET == "nsl_kdd" else 1,
    help="Change requires re-running the training pipeline.",
)

if new_dataset != DATASET:
    st.warning(
        "⚠️ Changing the dataset requires re-running `python -m src.train_models` "
        "to regenerate processed data and model files."
    )

def _write_env(high: float, medium: float, speed: int, dataset: str) -> None:
    """Overwrite the .env file with updated values, preserving SMTP keys."""
    env_path = PROJECT_ROOT / ".env"

    # Read existing content if present
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    # Update only the keys we manage
    existing["DATASET"]                  = dataset
    existing["LIVE_FEED_SPEED"]          = str(speed)
    existing["SEVERITY_HIGH_THRESHOLD"]  = str(high)
    existing["SEVERITY_MEDIUM_THRESHOLD"]= str(medium)

    lines = []
    for k, v in existing.items():
        lines.append(f"{k}={v}")

    env_path.write_text("\n".join(lines) + "\n")


st.divider()

# ── Save to .env ──────────────────────────────────────────────────────────────
if st.button("💾 Save Settings to .env", use_container_width=False):
    if new_high <= new_medium:
        st.error("Cannot save — HIGH threshold must be greater than MEDIUM threshold.")
    else:
        _write_env(new_high, new_medium, new_speed, new_dataset)
        st.success(
            ".env updated. Restart the Streamlit app (`Ctrl+C` then `streamlit run dashboard/app.py`) "
            "for changes to take effect."
        )

# ── Current .env display ──────────────────────────────────────────────────────
st.subheader("Current .env Values")
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    # Show but mask passwords
    masked_lines = []
    for line in env_path.read_text().splitlines():
        if any(secret in line.upper() for secret in ["PASSWORD","SECRET","KEY","TOKEN"]):
            k, _, _ = line.partition("=")
            masked_lines.append(f"{k}=****")
        else:
            masked_lines.append(line)
    st.code("\n".join(masked_lines), language="ini")
else:
    st.info("No .env file found. Copy `.env.example` to `.env` and fill in your values.")
