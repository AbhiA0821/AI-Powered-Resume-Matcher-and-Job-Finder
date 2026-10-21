import os
import json
import sqlite3
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("hireagent.rapidapi")

# Determine workspace base directory (root of repository)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DEFAULT_DB_PATH = os.environ.get("SQLITE_DB_PATH") or os.path.join(BASE_DIR, "data", "resume_matcher.db")
DEFAULT_SNAPSHOT_PATH = os.environ.get("JSEARCH_USAGE_PATH") or os.path.join(BASE_DIR, "data", "jsearch_usage.json")

MONTHLY_LIMIT = 200

# Threshold configurations: (count, key, severity, message)
THRESHOLDS = [
    (170, "85%", "WARNING", "JSearch API usage has reached 85% (170/200 requests) for the current month."),
    (180, "90%", "HIGH WARNING", "JSearch API usage has reached 90% (180/200 requests) for the current month."),
    (190, "95%", "CRITICAL WARNING", "JSearch API usage has reached 95% (190/200 requests) for the current month."),
    (200, "100%", "STOP", "JSearch API usage has reached 100% (200/200 requests) for the current month. Quota exhausted - API requests blocked.")
]

class JSearchUsageTracker:
    """Thread-safe, restart-persistent monthly JSearch API usage tracker.
    
    Persists counts in the project's SQLite database (data/resume_matcher.db)
    and maintains an optional JSON snapshot for immediate inspection.
    Enforces a strict 200 requests/month hard limit with warning thresholds at
    85% (170), 90% (180), 95% (190), and 100% (200).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        monthly_limit: int = MONTHLY_LIMIT,
        snapshot_path: Optional[str] = None
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.monthly_limit = monthly_limit
        self.snapshot_path = snapshot_path or DEFAULT_SNAPSHOT_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jsearch_usage (
                        month TEXT PRIMARY KEY,
                        requests_used INTEGER NOT NULL DEFAULT 0,
                        warnings_triggered TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def get_current_month() -> str:
        """Returns the current calendar month in YYYY-MM format."""
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _get_usage_unlocked(self, target_month: str) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT month, requests_used, warnings_triggered, updated_at FROM jsearch_usage WHERE month = ?",
                (target_month,)
            )
            row = cursor.fetchone()
            if not row:
                now_iso = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO jsearch_usage (month, requests_used, warnings_triggered, updated_at) VALUES (?, 0, '[]', ?)",
                    (target_month, now_iso)
                )
                conn.commit()
                requests_used = 0
                warnings_triggered = []
                updated_at = now_iso
            else:
                requests_used = row["requests_used"]
                try:
                    warnings_triggered = json.loads(row["warnings_triggered"] or "[]")
                except Exception:
                    warnings_triggered = []
                updated_at = row["updated_at"]
        finally:
            conn.close()

        requests_remaining = max(0, self.monthly_limit - requests_used)
        percentage_used = round((requests_used / self.monthly_limit) * 100, 2) if self.monthly_limit > 0 else 100.0

        status = "OK"
        active_warning = None
        if requests_used >= self.monthly_limit:
            status = "STOP"
            active_warning = "JSearch API usage has reached 100% (200/200 requests) for the current month. Quota exhausted - API requests blocked."
        elif requests_used >= 190:
            status = "CRITICAL WARNING"
            active_warning = "JSearch API usage has reached 95% (190/200 requests) for the current month."
        elif requests_used >= 180:
            status = "HIGH WARNING"
            active_warning = "JSearch API usage has reached 90% (180/200 requests) for the current month."
        elif requests_used >= 170:
            status = "WARNING"
            active_warning = "JSearch API usage has reached 85% (170/200 requests) for the current month."

        result = {
            "requests_used": requests_used,
            "requests_remaining": requests_remaining,
            "percentage_used": percentage_used,
            "current_month": target_month,
            "monthly_limit": self.monthly_limit,
            "status": status,
            "active_warning": active_warning,
            "warnings_triggered": warnings_triggered,
            "is_blocked": requests_used >= self.monthly_limit,
            "updated_at": updated_at
        }
        self._write_snapshot(result)
        return result

    def get_usage(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Returns the current usage metrics and status for the specified or current month."""
        target_month = month or self.get_current_month()
        with self._lock:
            return self._get_usage_unlocked(target_month)

    def can_make_request(self, month: Optional[str] = None) -> bool:
        """Checks if a new request is permitted under the monthly quota."""
        target_month = month or self.get_current_month()
        with self._lock:
            usage = self._get_usage_unlocked(target_month)
            return usage["requests_used"] < self.monthly_limit

    def acquire_request_slot(self, month: Optional[str] = None) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Atomically checks and reserves 1 request slot.
        
        Returns:
            Tuple of (allowed: bool, usage_info: dict, new_warning_message: Optional[str])
        If usage is already at monthly_limit (200), returns (False, usage_info, None).
        """
        target_month = month or self.get_current_month()
        with self._lock:
            usage = self._get_usage_unlocked(target_month)
            current_used = usage["requests_used"]

            # Quota hard stop: NEVER exceed internal monthly_limit
            if current_used >= self.monthly_limit:
                logger.error(
                    f"JSearch API monthly quota of {self.monthly_limit} reached for {target_month}. "
                    "Outbound request blocked."
                )
                return False, usage, None

            new_used = current_used + 1
            now_iso = datetime.now(timezone.utc).isoformat()
            warnings_triggered: List[str] = list(usage["warnings_triggered"])
            new_warning_msg: Optional[str] = None

            # Check thresholds:
            # 85% (170) -> WARNING
            # 90% (180) -> HIGH WARNING
            # 95% (190) -> CRITICAL WARNING
            # 100% (200) -> STOP
            for threshold_val, threshold_key, level, msg in THRESHOLDS:
                if new_used >= threshold_val and threshold_key not in warnings_triggered:
                    warnings_triggered.append(threshold_key)
                    new_warning_msg = msg
                    if level == "STOP":
                        logger.error(msg)
                    else:
                        logger.warning(msg)

            # Persist update atomically
            conn = self._get_connection()
            try:
                conn.execute(
                    "UPDATE jsearch_usage SET requests_used = ?, warnings_triggered = ?, updated_at = ? WHERE month = ?",
                    (new_used, json.dumps(warnings_triggered), now_iso, target_month)
                )
                conn.commit()
            finally:
                conn.close()

            updated_usage = self._get_usage_unlocked(target_month)
            self._write_snapshot(updated_usage)

            return True, updated_usage, new_warning_msg

    def _write_snapshot(self, usage: Dict[str, Any]) -> None:
        """Mirrors the usage status to JSON for transparency."""
        try:
            if self.snapshot_path and self.snapshot_path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(self.snapshot_path)), exist_ok=True)
                with open(self.snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(usage, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not write usage snapshot: {e}")

    def set_requests_used(self, count: int, month: Optional[str] = None, warnings: Optional[List[str]] = None) -> None:
        """Sets usage state directly (for test fixtures and administration)."""
        target_month = month or self.get_current_month()
        now_iso = datetime.now(timezone.utc).isoformat()
        warn_list = warnings if warnings is not None else []
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO jsearch_usage (month, requests_used, warnings_triggered, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(month) DO UPDATE SET requests_used=excluded.requests_used, "
                    "warnings_triggered=excluded.warnings_triggered, updated_at=excluded.updated_at",
                    (target_month, count, json.dumps(warn_list), now_iso)
                )
                conn.commit()
            finally:
                conn.close()
            self._write_snapshot(self._get_usage_unlocked(target_month))

    def reset_month(self, month: Optional[str] = None) -> None:
        """Resets the usage tracking for a given month."""
        target_month = month or self.get_current_month()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM jsearch_usage WHERE month = ?", (target_month,))
                conn.commit()
            finally:
                conn.close()

# Module-level default singleton
_default_tracker: Optional[JSearchUsageTracker] = None

def get_usage_tracker(db_path: Optional[str] = None) -> JSearchUsageTracker:
    global _default_tracker
    if _default_tracker is None or db_path is not None:
        if db_path is not None:
            return JSearchUsageTracker(db_path=db_path)
        _default_tracker = JSearchUsageTracker()
    return _default_tracker

def get_jsearch_quota_status(month: Optional[str] = None) -> Dict[str, Any]:
    """Helper for FastAPI, Flask, and diagnostic endpoints to query real-time quota status."""
    return get_usage_tracker().get_usage(month=month)
