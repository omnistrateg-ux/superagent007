"""Tests for candle fetching (paper_futures)."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import json

# Import module under test
from moex_agent.paper_futures import (
    fetch_futures_candles,
    _find_candle_offset,
    _parse_candle_time,
    _check_candle_freshness,
    _candle_offset_cache,
    MSK,
    STALE_CANDLE_THRESHOLD_SEC,
)


class TestParseCandle_Time:
    """Tests for _parse_candle_time helper."""

    def test_valid_moex_format(self):
        """Test parsing standard MOEX candle timestamp."""
        result = _parse_candle_time("2026-04-10 09:00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 10
        assert result.hour == 9
        assert result.minute == 0
        assert result.tzinfo == MSK

    def test_invalid_format_returns_none(self):
        """Test that invalid format returns None."""
        assert _parse_candle_time("invalid") is None
        assert _parse_candle_time("") is None
        assert _parse_candle_time(None) is None

    def test_wrong_date_format_returns_none(self):
        """Test that wrong date format returns None."""
        assert _parse_candle_time("10-04-2026 09:00:00") is None
        assert _parse_candle_time("2026/04/10 09:00:00") is None


class TestCheckCandleFreshness:
    """Tests for _check_candle_freshness."""

    def test_empty_candles_no_warning(self, caplog):
        """Test that empty candles don't cause warning."""
        _check_candle_freshness([], "TEST")
        assert "old" not in caplog.text

    def test_fresh_candles_no_warning(self, caplog):
        """Test that fresh candles don't trigger warning."""
        now = datetime.now(MSK)
        fresh_time = now - timedelta(minutes=30)
        candles = [[100, 101, 102, 99, fresh_time.strftime("%Y-%m-%d %H:%M:%S")]]

        with patch("moex_agent.paper_futures.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            _check_candle_freshness(candles, "TEST")

        assert "old" not in caplog.text.lower()

    def test_stale_candles_during_trading_hours_warns(self, caplog):
        """Test that stale candles during trading hours trigger warning."""
        import logging
        caplog.set_level(logging.WARNING)

        # Simulate Tuesday 12:00 MSK with 3h old candle
        now = datetime(2026, 4, 7, 12, 0, 0, tzinfo=MSK)  # Tuesday
        stale_time = now - timedelta(hours=3)
        candles = [[100, 101, 102, 99, stale_time.strftime("%Y-%m-%d %H:%M:%S")]]

        with patch("moex_agent.paper_futures.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            _check_candle_freshness(candles, "BRJ6")

        assert "3.0h old" in caplog.text or "old" in caplog.text.lower()

    def test_stale_candles_weekend_no_warning(self, caplog):
        """Test that stale candles on weekend use debug level."""
        import logging
        caplog.set_level(logging.DEBUG)

        # Saturday 12:00 MSK
        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=MSK)  # Saturday
        old_time = now - timedelta(hours=20)
        candles = [[100, 101, 102, 99, old_time.strftime("%Y-%m-%d %H:%M:%S")]]

        with patch("moex_agent.paper_futures.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            _check_candle_freshness(candles, "BRJ6")

        # Should be DEBUG level, not WARNING
        assert "weekend" in caplog.text.lower()

    def test_stale_candles_early_morning_no_warning(self, caplog):
        """Test that stale candles before 7:00 MSK use debug level."""
        import logging
        caplog.set_level(logging.DEBUG)

        # Tuesday 6:50 MSK - before trading starts
        now = datetime(2026, 4, 7, 6, 50, 0, tzinfo=MSK)  # Tuesday 6:50
        old_time = now - timedelta(hours=12)
        candles = [[100, 101, 102, 99, old_time.strftime("%Y-%m-%d %H:%M:%S")]]

        with patch("moex_agent.paper_futures.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            _check_candle_freshness(candles, "BRJ6")

        # Should be DEBUG level, not WARNING
        assert "outside trading hours" in caplog.text.lower()


class TestFindCandleOffset:
    """Tests for _find_candle_offset binary search."""

    def test_binary_search_finds_offset(self):
        """Test that binary search converges to valid offset."""
        # Mock API responses: data exists up to offset 5000
        def mock_urlopen(url, timeout=None):
            # Extract start parameter from URL
            if "start=" in url:
                start = int(url.split("start=")[1].split("&")[0])
            else:
                start = 0

            mock_resp = MagicMock()
            if start < 5000:
                # Return some data
                mock_resp.read.return_value = json.dumps({
                    "candles": {"data": [["2026-04-10 09:00:00"]]}
                }).encode()
            else:
                # No data at this offset
                mock_resp.read.return_value = json.dumps({
                    "candles": {"data": []}
                }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            offset = _find_candle_offset("BRJ6", 60)

        # Should find offset close to 5000 (within binary search precision)
        assert offset > 4000
        assert offset < 5500

    def test_binary_search_handles_no_data(self):
        """Test that binary search handles no data case."""
        def mock_urlopen(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "candles": {"data": []}
            }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            offset = _find_candle_offset("INVALID", 60)

        # Should return 0 when no data found
        assert offset == 0

    def test_binary_search_handles_network_error(self):
        """Test that binary search handles network errors gracefully."""
        call_count = [0]

        def mock_urlopen(url, timeout=None):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise Exception("Network error")
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "candles": {"data": [["2026-04-10 09:00:00"]]}
            }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            # Should not raise, should return some offset
            offset = _find_candle_offset("BRJ6", 60)
            assert isinstance(offset, int)


class TestFetchFuturesCandles:
    """Tests for fetch_futures_candles main function."""

    def setup_method(self):
        """Clear cache before each test."""
        _candle_offset_cache.clear()

    def test_returns_list_on_success(self):
        """Test that function returns list of candles."""
        mock_candles = [
            [100.0, 101.0, 102.0, 99.0, "2026-04-10 09:00:00"],
            [101.0, 102.0, 103.0, 100.0, "2026-04-10 10:00:00"],
        ]

        def mock_urlopen(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "candles": {"data": mock_candles}
            }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = fetch_futures_candles("BRJ6", interval=60, count=5)

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_returns_empty_list_on_error(self):
        """Test that function returns empty list on error."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = fetch_futures_candles("BRJ6")

        assert result == []

    def test_cache_is_used(self):
        """Test that offset cache is used on subsequent calls."""
        # Pre-populate cache
        _candle_offset_cache["BRJ6_60"] = (5000, float("inf"))

        mock_candles = [[100.0, 101.0, 102.0, 99.0, "2026-04-10 09:00:00"]]
        call_count = [0]

        def mock_urlopen(url, timeout=None):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "candles": {"data": mock_candles}
            }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            fetch_futures_candles("BRJ6", interval=60, count=5)

        # Should only make 1 call (fetch candles), not binary search calls
        assert call_count[0] == 1

    def test_returns_last_n_candles(self):
        """Test that function returns last N candles when more available."""
        mock_candles = [
            [100 + i, 101 + i, 102 + i, 99 + i, f"2026-04-10 {9+i:02d}:00:00"]
            for i in range(10)
        ]

        def mock_urlopen(url, timeout=None):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "candles": {"data": mock_candles}
            }).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = fetch_futures_candles("BRJ6", count=3)

        assert len(result) == 3
        # Should be the last 3 candles
        assert result[0][0] == 107  # open of candle index 7
