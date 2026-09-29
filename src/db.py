"""
db.py
─────
SQLite database layer (via SQLAlchemy Core).
Handles schema creation and all CRUD operations for:
    • traffic_log   — every classified record
    • alerts        — attack detections that crossed severity threshold
    • alert_notifications — log of email/SMS send attempts

Public API
──────────
    init_db()
    log_traffic(record_dict, prediction_dict) → int  (row id)
    log_alert(traffic_log_id, attack_type, severity, source_ip) → int
    log_notification(alert_id, channel, status, message) → int
    get_recent_traffic(limit) → list[dict]
    get_recent_alerts(limit) → list[dict]
    get_kpi_stats() → dict
    get_attack_distribution() → list[dict]
    get_traffic_over_time(hours) → list[dict]
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DB_PATH

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Engine & Session factory
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE = None
_SessionFactory = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return _ENGINE


def _get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine())
    return _SessionFactory()


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TrafficLog(Base):
    __tablename__ = "traffic_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
    source_ip    = Column(String(50), default="0.0.0.0")
    dest_ip      = Column(String(50), default="0.0.0.0")
    protocol     = Column(String(20), default="unknown")
    is_attack    = Column(Integer, default=0)           # 0 or 1
    attack_type  = Column(String(50), default="normal")
    confidence   = Column(Float, default=0.0)
    severity     = Column(String(20), default="none")   # none/low/medium/high
    raw_features = Column(Text, default="")             # JSON snapshot (optional)


class Alert(Base):
    __tablename__ = "alerts"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime, default=datetime.utcnow, index=True)
    traffic_log_id  = Column(Integer, default=0)
    attack_type     = Column(String(50))
    severity        = Column(String(20))
    source_ip       = Column(String(50), default="0.0.0.0")
    dest_ip         = Column(String(50), default="0.0.0.0")
    status          = Column(String(20), default="open")  # open/acknowledged/resolved
    notes           = Column(Text, default="")


class AlertNotification(Base):
    __tablename__ = "alert_notifications"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_id  = Column(Integer, default=0)
    channel   = Column(String(20))   # "email" or "sms"
    status    = Column(String(20))   # "sent" or "failed" or "simulated"
    message   = Column(Text, default="")


# ─────────────────────────────────────────────────────────────────────────────
# Schema init
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(_get_engine())
    logger.info("Database initialised at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────────────────────────────────────

def log_traffic(
    record: dict,
    prediction: dict,
    timestamp: Optional[datetime] = None,
) -> int:
    """
    Insert one traffic record + its classification result.

    Parameters
    ----------
    record     : Raw feature dict (may include source_ip, dest_ip, protocol)
    prediction : Output of predict() — {is_attack, attack_type, confidence}

    Returns
    -------
    Integer row id of the inserted row.
    """
    from src.config import SEVERITY_MAPPING
    import json

    severity = SEVERITY_MAPPING.get(prediction.get("attack_type", "normal"), "none")

    row = TrafficLog(
        timestamp   = timestamp or datetime.utcnow(),
        source_ip   = str(record.get("source_ip", record.get("srcip", "0.0.0.0"))),
        dest_ip     = str(record.get("dest_ip",   record.get("dstip",  "0.0.0.0"))),
        protocol    = str(record.get("protocol_type", record.get("proto", "unknown"))),
        is_attack   = int(prediction.get("is_attack", False)),
        attack_type = str(prediction.get("attack_type", "normal")),
        confidence  = float(prediction.get("confidence", 0.0)),
        severity    = severity,
        raw_features= json.dumps({k: v for k, v in list(record.items())[:10]}),
    )

    with _get_session() as session:
        session.add(row)
        session.commit()
        return row.id


def log_alert(
    traffic_log_id: int,
    attack_type: str,
    severity: str,
    source_ip: str = "0.0.0.0",
    dest_ip: str = "0.0.0.0",
    notes: str = "",
) -> int:
    """Insert an alert row. Returns the alert id."""
    row = Alert(
        timestamp      = datetime.utcnow(),
        traffic_log_id = traffic_log_id,
        attack_type    = attack_type,
        severity       = severity,
        source_ip      = source_ip,
        dest_ip        = dest_ip,
        notes          = notes,
    )
    with _get_session() as session:
        session.add(row)
        session.commit()
        return row.id


def log_notification(
    alert_id: int,
    channel: str,
    status: str,
    message: str = "",
) -> int:
    """Log an alert notification attempt."""
    row = AlertNotification(
        alert_id  = alert_id,
        channel   = channel,
        status    = status,
        message   = message,
    )
    with _get_session() as session:
        session.add(row)
        session.commit()
        return row.id


def update_alert_status(alert_id: int, status: str) -> None:
    """Update an alert's status (open → acknowledged → resolved)."""
    with _get_session() as session:
        alert = session.get(Alert, alert_id)
        if alert:
            alert.status = status
            session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_traffic(limit: int = 100) -> list[dict]:
    """Return the most recent `limit` traffic log rows as dicts."""
    with _get_session() as session:
        rows = (
            session.query(TrafficLog)
            .order_by(TrafficLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [_traffic_to_dict(r) for r in rows]


def get_recent_alerts(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` alerts as dicts."""
    with _get_session() as session:
        rows = (
            session.query(Alert)
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [_alert_to_dict(r) for r in rows]


def get_kpi_stats() -> dict:
    """
    Return KPI numbers for the dashboard Overview page.
    {total, attacks, normal, attack_rate, latest_timestamp}
    """
    with _get_session() as session:
        total   = session.query(func.count(TrafficLog.id)).scalar() or 0
        attacks = session.query(func.count(TrafficLog.id)).filter(TrafficLog.is_attack == 1).scalar() or 0
        normal  = total - attacks
        latest_row = (
            session.query(TrafficLog.timestamp)
            .order_by(TrafficLog.timestamp.desc())
            .first()
        )
        latest = latest_row[0].isoformat() if latest_row else "n/a"
        return {
            "total":            total,
            "attacks":          attacks,
            "normal":           normal,
            "attack_rate":      round(attacks / total * 100, 2) if total else 0.0,
            "latest_timestamp": latest,
        }


def get_attack_distribution() -> list[dict]:
    """
    Return attack type counts for the donut/pie chart.
    Only includes attack rows (is_attack=1).
    """
    with _get_session() as session:
        rows = (
            session.query(
                TrafficLog.attack_type,
                func.count(TrafficLog.id).label("count"),
            )
            .filter(TrafficLog.is_attack == 1)
            .group_by(TrafficLog.attack_type)
            .all()
        )
        return [{"attack_type": r.attack_type, "count": r.count} for r in rows]


def get_traffic_over_time(hours: int = 24) -> list[dict]:
    """
    Return traffic volume bucketed by minute for the past `hours` hours.
    Returns a list of {bucket: datetime_str, total: int, attacks: int}.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    with _get_session() as session:
        # SQLite strftime bucketing by minute
        rows = session.execute(
            text(
                """
                SELECT
                    strftime('%Y-%m-%dT%H:%M:00', timestamp) AS bucket,
                    COUNT(*) AS total,
                    SUM(is_attack) AS attacks
                FROM traffic_log
                WHERE timestamp >= :since
                GROUP BY bucket
                ORDER BY bucket
                """
            ),
            {"since": since.isoformat()},
        ).fetchall()
        return [
            {"bucket": r[0], "total": r[1], "attacks": r[2] or 0}
            for r in rows
        ]


def get_open_alerts_count() -> int:
    with _get_session() as session:
        return session.query(func.count(Alert.id)).filter(Alert.status == "open").scalar() or 0


# ─────────────────────────────────────────────────────────────────────────────
# Row → dict helpers
# ─────────────────────────────────────────────────────────────────────────────

def _traffic_to_dict(r: TrafficLog) -> dict:
    return {
        "id":          r.id,
        "timestamp":   r.timestamp.isoformat() if r.timestamp else "",
        "source_ip":   r.source_ip,
        "dest_ip":     r.dest_ip,
        "protocol":    r.protocol,
        "is_attack":   bool(r.is_attack),
        "attack_type": r.attack_type,
        "confidence":  r.confidence,
        "severity":    r.severity,
    }


def _alert_to_dict(r: Alert) -> dict:
    return {
        "id":             r.id,
        "timestamp":      r.timestamp.isoformat() if r.timestamp else "",
        "traffic_log_id": r.traffic_log_id,
        "attack_type":    r.attack_type,
        "severity":       r.severity,
        "source_ip":      r.source_ip,
        "dest_ip":        r.dest_ip,
        "status":         r.status,
        "notes":          r.notes,
    }
