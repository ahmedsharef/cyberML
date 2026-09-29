"""
config.py
─────────
Central configuration loader. Reads .env and label_mapping.yaml,
exposes typed constants used by every other module.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# ── Locate project root (one level above src/) ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present (silently skip if missing — CI / Docker may inject vars)
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET: str = os.getenv("DATASET", "nsl_kdd").lower().strip()
assert DATASET in ("nsl_kdd", "unsw_nb15"), (
    f"DATASET must be 'nsl_kdd' or 'unsw_nb15', got '{DATASET}'"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = Path(os.getenv("DATA_RAW_DIR",       PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
MODELS_DIR         = Path(os.getenv("MODELS_DIR",         PROJECT_ROOT / "models"))
REPORTS_DIR        = PROJECT_ROOT / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
DB_PATH            = Path(os.getenv("DB_PATH",            PROJECT_ROOT / "data" / "cyberattack.db"))
CONFIG_DIR         = PROJECT_ROOT / "config"

# Ensure writable dirs exist at import time
for _d in (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Label mapping ─────────────────────────────────────────────────────────────
with open(CONFIG_DIR / "label_mapping.yaml", "r") as _f:
    _mapping_yaml = yaml.safe_load(_f)

TARGET_CATEGORIES: list[str] = _mapping_yaml["target_categories"]
LABEL_MAPPING: dict[str, str] = _mapping_yaml[DATASET]   # raw label → target category
SEVERITY_MAPPING: dict[str, str] = _mapping_yaml["severity"]

# ── Live feed ─────────────────────────────────────────────────────────────────
LIVE_FEED_SPEED: int = int(os.getenv("LIVE_FEED_SPEED", 2))

# ── Alert thresholds ──────────────────────────────────────────────────────────
SEVERITY_HIGH_THRESHOLD:   float = float(os.getenv("SEVERITY_HIGH_THRESHOLD",   0.85))
SEVERITY_MEDIUM_THRESHOLD: float = float(os.getenv("SEVERITY_MEDIUM_THRESHOLD", 0.60))

# ── SMTP (optional) ───────────────────────────────────────────────────────────
SMTP_HOST:     str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT:     int = int(os.getenv("SMTP_PORT", 587))
SMTP_USER:     str = os.getenv("SMTP_USER", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO:str = os.getenv("ALERT_EMAIL_TO", "")

# ── NSL-KDD column definitions ────────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
NSL_KDD_DROP        = ["difficulty"]   # not a feature, only used in NSL-KDD

# ── UNSW-NB15 column definitions ──────────────────────────────────────────────
UNSW_NB15_CATEGORICAL = ["proto", "service", "state"]
UNSW_NB15_DROP        = ["id", "srcip", "sport", "dstip", "dsport"]  # identifiers, not features
UNSW_NB15_LABEL_COL   = "attack_cat"    # multi-class label
UNSW_NB15_BINARY_COL  = "label"         # binary label (0/1)
