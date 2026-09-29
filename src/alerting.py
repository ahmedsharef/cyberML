"""
alerting.py
───────────
Handles the full alerting flow for detected attacks:
    1. Compute severity from attack type + confidence
    2. Write alert row to the DB
    3. Attempt to send email notification (real if SMTP creds are set in .env,
       simulated otherwise)
    4. Log the notification attempt back to the DB

Public API
──────────
    process_alert(traffic_log_id, prediction, record) → alert_id | None
    send_email_alert(alert_id, attack_type, severity, source_ip)
    get_severity(attack_type, confidence) → str
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from src.config import (
    ALERT_EMAIL_TO,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MAPPING,
    SEVERITY_MEDIUM_THRESHOLD,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
)
from src.db import log_alert, log_notification

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Severity resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_severity(attack_type: str, confidence: float) -> str:
    """
    Determine alert severity.

    Rules
    ─────
    - "none"   if not an attack
    - From label_mapping.yaml base severity, then downgrade if confidence is low.
    - High confidence (≥ SEVERITY_HIGH_THRESHOLD) keeps the base level.
    - Medium confidence (≥ SEVERITY_MEDIUM_THRESHOLD) downgrades high→medium.
    - Low confidence downgrades everything by one level.
    """
    if attack_type == "normal":
        return "none"

    base = SEVERITY_MAPPING.get(attack_type, "low")

    severity_order = ["none", "low", "medium", "high"]
    base_idx = severity_order.index(base) if base in severity_order else 1

    if confidence >= SEVERITY_HIGH_THRESHOLD:
        return severity_order[base_idx]
    elif confidence >= SEVERITY_MEDIUM_THRESHOLD:
        # Downgrade by 1 step
        return severity_order[max(1, base_idx - 1)]
    else:
        # Downgrade by 2 steps but keep at least "low"
        return severity_order[max(1, base_idx - 2)]


# ─────────────────────────────────────────────────────────────────────────────
# Alert orchestration
# ─────────────────────────────────────────────────────────────────────────────

def process_alert(
    traffic_log_id: int,
    prediction: dict,
    record: dict,
) -> Optional[int]:
    """
    Full alerting flow for one classified traffic record.

    Parameters
    ----------
    traffic_log_id : Row id in traffic_log table (from db.log_traffic)
    prediction     : predict() output dict
    record         : Original feature dict (for source_ip etc.)

    Returns
    -------
    alert_id if an alert was created, None if no alert needed (normal traffic).
    """
    if not prediction.get("is_attack"):
        return None

    attack_type = prediction["attack_type"]
    confidence  = prediction["confidence"]
    severity    = get_severity(attack_type, confidence)

    if severity == "none":
        return None

    source_ip = str(record.get("source_ip", record.get("srcip", "0.0.0.0")))
    dest_ip   = str(record.get("dest_ip",   record.get("dstip",  "0.0.0.0")))

    notes = (
        f"Detected at {datetime.utcnow().isoformat()} UTC. "
        f"Confidence: {confidence:.2%}. "
        f"Model: two-stage pipeline."
    )

    alert_id = log_alert(
        traffic_log_id=traffic_log_id,
        attack_type=attack_type,
        severity=severity,
        source_ip=source_ip,
        dest_ip=dest_ip,
        notes=notes,
    )

    logger.info(
        "Alert #%d created: %s (%s) from %s",
        alert_id, attack_type, severity, source_ip,
    )

    # Attempt email notification
    send_email_alert(alert_id, attack_type, severity, source_ip, confidence)

    return alert_id


# ─────────────────────────────────────────────────────────────────────────────
# Email notification
# ─────────────────────────────────────────────────────────────────────────────

def send_email_alert(
    alert_id: int,
    attack_type: str,
    severity: str,
    source_ip: str,
    confidence: float = 0.0,
) -> None:
    """
    Send a security alert email.

    If SMTP credentials are configured in .env, sends a real email.
    Otherwise, logs a simulated notification to the DB.
    """
    subject = f"[SECURITY ALERT] {severity.upper()} — {attack_type.upper()} Detected"
    body = _build_email_body(alert_id, attack_type, severity, source_ip, confidence)

    # Check if we have real credentials
    if SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL_TO:
        _send_real_email(alert_id, subject, body)
    else:
        # Simulated — just log it
        log_notification(
            alert_id=alert_id,
            channel="email",
            status="simulated",
            message=f"[SIMULATED] To: {ALERT_EMAIL_TO or 'not set'} | {subject}",
        )
        logger.info(
            "Alert #%d: email simulated (no SMTP credentials in .env)", alert_id
        )


def _send_real_email(alert_id: int, subject: str, body: str) -> None:
    """Attempt to send via SMTP. Logs success or failure to DB."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_EMAIL_TO
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [ALERT_EMAIL_TO], msg.as_string())

        log_notification(
            alert_id=alert_id,
            channel="email",
            status="sent",
            message=f"Sent to {ALERT_EMAIL_TO}: {subject}",
        )
        logger.info("Alert #%d: email sent to %s", alert_id, ALERT_EMAIL_TO)

    except Exception as exc:
        log_notification(
            alert_id=alert_id,
            channel="email",
            status="failed",
            message=f"Error: {exc}",
        )
        logger.error("Alert #%d: email failed — %s", alert_id, exc)


def _build_email_body(
    alert_id: int,
    attack_type: str,
    severity: str,
    source_ip: str,
    confidence: float,
) -> str:
    severity_color = {"high": "#d32f2f", "medium": "#f57c00", "low": "#388e3c"}.get(
        severity, "#757575"
    )
    return f"""
    <html><body style="font-family: Arial, sans-serif; padding: 20px;">
      <h2 style="color:{severity_color};">⚠ Security Alert #{alert_id}</h2>
      <table style="border-collapse:collapse; width:100%; max-width:500px;">
        <tr><td style="padding:8px; font-weight:bold;">Attack Type</td>
            <td style="padding:8px;">{attack_type.upper()}</td></tr>
        <tr style="background:#f5f5f5;">
          <td style="padding:8px; font-weight:bold;">Severity</td>
          <td style="padding:8px; color:{severity_color}; font-weight:bold;">
            {severity.upper()}</td></tr>
        <tr><td style="padding:8px; font-weight:bold;">Source IP</td>
            <td style="padding:8px;">{source_ip}</td></tr>
        <tr style="background:#f5f5f5;">
          <td style="padding:8px; font-weight:bold;">Confidence</td>
          <td style="padding:8px;">{confidence:.2%}</td></tr>
        <tr><td style="padding:8px; font-weight:bold;">Time (UTC)</td>
            <td style="padding:8px;">{datetime.utcnow().isoformat()}</td></tr>
      </table>
      <p style="margin-top:20px; color:#555; font-size:12px;">
        Generated by ML-Based Cyberattack Detection System
      </p>
    </body></html>
    """
