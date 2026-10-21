import os
import time
import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from app.services.job_sources.usage_tracker import JSearchUsageTracker
from app.services.job_sources.rapidapi_job_service import (
    search_live_jobs,
    get_usage_tracker,
    _JOB_CACHE
)

@pytest.fixture
def temp_tracker(tmp_path):
    """Provides an isolated SQLite-backed tracker instance for test independence."""
    db_file = str(tmp_path / "test_usage.db")
    snapshot_file = str(tmp_path / "test_usage.json")
    return JSearchUsageTracker(db_path=db_file, snapshot_path=snapshot_file)

def test_request_count_increments_only_for_actual_api_calls(temp_tracker):
    """Verifies that the request count increments by 1 when an actual outbound API call is made."""
    with patch("app.services.job_sources.rapidapi_job_service.get_usage_tracker", return_value=temp_tracker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"job_id": "test_1", "job_title": "Python Dev", "employer_name": "Acme"}]}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            jobs = search_live_jobs(query="UniqueDevQuery_1", location="India")
            assert len(jobs) == 1
            assert mock_get.call_count == 1
            usage = temp_tracker.get_usage()
            assert usage["requests_used"] == 1
            assert usage["requests_remaining"] == 199
            assert usage["percentage_used"] == 0.5

def test_cached_requests_do_not_increment_counter(temp_tracker):
    """Verifies that queries served from cache within TTL do NOT increment the usage counter."""
    with patch("app.services.job_sources.rapidapi_job_service.get_usage_tracker", return_value=temp_tracker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"job_id": "test_cached_1", "job_title": "Django Dev", "employer_name": "Acme"}]}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            # First call -> Live HTTP request (counter increments to 1)
            jobs1 = search_live_jobs(query="CachedDevQuery_2", location="India")
            assert len(jobs1) == 1
            assert mock_get.call_count == 1
            assert temp_tracker.get_usage()["requests_used"] == 1

            # Second call with same parameters -> Cache hit (no HTTP call, counter remains 1)
            jobs2 = search_live_jobs(query="CachedDevQuery_2", location="India")
            assert len(jobs2) == 1
            assert mock_get.call_count == 1  # Still 1
            assert temp_tracker.get_usage()["requests_used"] == 1  # Still 1

def test_threshold_169_no_85_warning(temp_tracker):
    """Verifies that at 169 requests (84.5%), no 85% warning is generated."""
    temp_tracker.set_requests_used(168)
    allowed, usage, warn_msg = temp_tracker.acquire_request_slot()

    assert allowed is True
    assert usage["requests_used"] == 169
    assert usage["percentage_used"] == 84.5
    assert usage["status"] == "OK"
    assert usage["active_warning"] is None
    assert warn_msg is None
    assert "85%" not in usage["warnings_triggered"]

def test_threshold_170_85_warning(temp_tracker):
    """Verifies that at exactly 170 requests (85%), the 85% warning is triggered with exact message."""
    temp_tracker.set_requests_used(169)
    allowed, usage, warn_msg = temp_tracker.acquire_request_slot()

    assert allowed is True
    assert usage["requests_used"] == 170
    assert usage["percentage_used"] == 85.0
    assert usage["status"] == "WARNING"
    expected_msg = "JSearch API usage has reached 85% (170/200 requests) for the current month."
    assert warn_msg == expected_msg
    assert usage["active_warning"] == expected_msg
    assert "85%" in usage["warnings_triggered"]

def test_threshold_180_90_warning(temp_tracker):
    """Verifies that at 180 requests (90%), the 90% HIGH WARNING is triggered."""
    temp_tracker.set_requests_used(179, warnings=["85%"])
    allowed, usage, warn_msg = temp_tracker.acquire_request_slot()

    assert allowed is True
    assert usage["requests_used"] == 180
    assert usage["percentage_used"] == 90.0
    assert usage["status"] == "HIGH WARNING"
    expected_msg = "JSearch API usage has reached 90% (180/200 requests) for the current month."
    assert warn_msg == expected_msg
    assert usage["active_warning"] == expected_msg
    assert "90%" in usage["warnings_triggered"]

def test_threshold_190_95_warning(temp_tracker):
    """Verifies that at 190 requests (95%), the 95% CRITICAL WARNING is triggered."""
    temp_tracker.set_requests_used(189, warnings=["85%", "90%"])
    allowed, usage, warn_msg = temp_tracker.acquire_request_slot()

    assert allowed is True
    assert usage["requests_used"] == 190
    assert usage["percentage_used"] == 95.0
    assert usage["status"] == "CRITICAL WARNING"
    expected_msg = "JSearch API usage has reached 95% (190/200 requests) for the current month."
    assert warn_msg == expected_msg
    assert usage["active_warning"] == expected_msg
    assert "95%" in usage["warnings_triggered"]

def test_threshold_200_api_blocked(temp_tracker):
    """Verifies that at 200/200 requests (100%), the API is strictly blocked from making outbound calls."""
    temp_tracker.set_requests_used(200, warnings=["85%", "90%", "95%", "100%"])
    
    # 1. Tracker level check
    allowed, usage, warn_msg = temp_tracker.acquire_request_slot()
    assert allowed is False
    assert usage["requests_used"] == 200
    assert usage["is_blocked"] is True
    assert usage["status"] == "STOP"

    # 2. Service level check: Outbound HTTP request must NEVER be made
    with patch("app.services.job_sources.rapidapi_job_service.get_usage_tracker", return_value=temp_tracker):
        with patch("requests.get") as mock_get:
            jobs = search_live_jobs(query="BlockedQuery", location="India")
            assert jobs == []
            assert mock_get.call_count == 0  # Zero HTTP calls made!

def test_threshold_200_fallback_used(temp_tracker):
    """Verifies that when quota is exhausted, search_live_jobs returns empty list enabling clean fallback."""
    temp_tracker.set_requests_used(200)
    with patch("app.services.job_sources.rapidapi_job_service.get_usage_tracker", return_value=temp_tracker):
        jobs = search_live_jobs(query="AnyDevRole", location="India")
        assert jobs == []

def test_warning_is_not_duplicated(temp_tracker):
    """Verifies that warnings are generated ONLY ONCE per month for each threshold."""
    temp_tracker.set_requests_used(169)

    # 1st call: crosses to 170 -> warning generated
    allowed1, usage1, warn_msg1 = temp_tracker.acquire_request_slot()
    assert usage1["requests_used"] == 170
    assert warn_msg1 is not None

    # 2nd call: crosses to 171 -> NO duplicate warning
    allowed2, usage2, warn_msg2 = temp_tracker.acquire_request_slot()
    assert usage2["requests_used"] == 171
    assert warn_msg2 is None

    # 3rd call: crosses to 172 -> NO duplicate warning
    allowed3, usage3, warn_msg3 = temp_tracker.acquire_request_slot()
    assert usage3["requests_used"] == 172
    assert warn_msg3 is None

    # Threshold "85%" appears exactly once in warnings_triggered list
    assert usage3["warnings_triggered"].count("85%") == 1

def test_monthly_reset_works(temp_tracker):
    """Verifies that counter automatically resets to 0 when calendar month changes."""
    month_oct = "2026-10"
    month_nov = "2026-11"

    # In October, 200 requests were consumed
    temp_tracker.set_requests_used(200, month=month_oct, warnings=["85%", "90%", "95%", "100%"])
    oct_usage = temp_tracker.get_usage(month=month_oct)
    assert oct_usage["requests_used"] == 200
    assert oct_usage["is_blocked"] is True

    # November begins -> month changes
    nov_usage = temp_tracker.get_usage(month=month_nov)
    assert nov_usage["requests_used"] == 0
    assert nov_usage["requests_remaining"] == 200
    assert nov_usage["percentage_used"] == 0.0
    assert nov_usage["is_blocked"] is False
    assert nov_usage["status"] == "OK"
    assert nov_usage["warnings_triggered"] == []

    # Can make request in November
    allowed, usage, _ = temp_tracker.acquire_request_slot(month=month_nov)
    assert allowed is True
    assert usage["requests_used"] == 1

    # October remains untouched in database history
    assert temp_tracker.get_usage(month=month_oct)["requests_used"] == 200

def test_restart_persistence_behavior(tmp_path):
    """Verifies that usage counter survives application restarts by persisting in SQLite."""
    db_file = str(tmp_path / "persistent_usage.db")
    snapshot_file = str(tmp_path / "persistent_usage.json")

    # Instance 1: Initial run records 80 requests
    tracker1 = JSearchUsageTracker(db_path=db_file, snapshot_path=snapshot_file)
    tracker1.set_requests_used(80)
    assert tracker1.get_usage()["requests_used"] == 80

    # Simulate restart by instantiating a brand new tracker pointing to same file
    tracker2 = JSearchUsageTracker(db_path=db_file, snapshot_path=snapshot_file)
    reloaded_usage = tracker2.get_usage()
    assert reloaded_usage["requests_used"] == 80
    assert reloaded_usage["requests_remaining"] == 120
    assert reloaded_usage["percentage_used"] == 40.0

    # Acquire one more request on restarted instance
    allowed, usage, _ = tracker2.acquire_request_slot()
    assert allowed is True
    assert usage["requests_used"] == 81

def test_counter_never_exceeds_200(temp_tracker):
    """Verifies concurrency safety: simultaneous requests cannot exceed the internal 200 threshold."""
    temp_tracker.set_requests_used(195)

    def attempt_request(_):
        allowed, _, _ = temp_tracker.acquire_request_slot()
        return allowed

    # Launch 15 concurrent attempts when only 5 slots remain
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(attempt_request, range(15)))

    # Exactly 5 should succeed, 10 should be rejected
    assert results.count(True) == 5
    assert results.count(False) == 10

    # Final counter MUST be strictly 200, never 201+
    final_usage = temp_tracker.get_usage()
    assert final_usage["requests_used"] == 200
    assert final_usage["requests_remaining"] == 0
    assert final_usage["is_blocked"] is True
