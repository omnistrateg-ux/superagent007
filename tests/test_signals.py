"""Tests for signals module with new filters."""
import pytest
from datetime import datetime, timezone

from moex_agent.anomaly import AnomalyResult, Direction
from moex_agent.signals import (
    SignalFilter,
    filter_signal,
    get_liquidity_tier,
    check_spread_for_horizon,
    check_liquidity_tier,
    check_opening_session,
    TIER_1_TICKERS,
    TIER_3_TICKERS,
)


def test_liquidity_tiers():
    """Test liquidity tier classification."""
    assert get_liquidity_tier("SBER") == 1
    assert get_liquidity_tier("GAZP") == 1
    assert get_liquidity_tier("YDEX") == 2
    assert get_liquidity_tier("TRNFP") == 3
    assert get_liquidity_tier("UNKNOWN") == 3  # Unknown = Tier 3


def test_spread_check_for_short_horizons():
    """Test that short horizons require tight spreads."""
    config = SignalFilter()

    # 5m horizon with wide spread - should fail
    ok, reason = check_spread_for_horizon(30.0, "5m", config)
    assert not ok
    assert "5m" in reason

    # 5m horizon with tight spread - should pass
    ok, reason = check_spread_for_horizon(20.0, "5m", config)
    assert ok

    # 1h horizon with wider spread - should pass
    ok, reason = check_spread_for_horizon(80.0, "1h", config)
    assert ok


def test_tier3_short_restriction():
    """Test that SHORT is restricted for Tier 3 tickers."""
    config = SignalFilter(restrict_short_tier3=True)

    # Tier 3 SHORT - should fail
    ok, reason = check_liquidity_tier("TRNFP", Direction.SHORT, config)
    assert not ok
    assert "Tier 3" in reason

    # Tier 3 LONG - should pass
    ok, reason = check_liquidity_tier("TRNFP", Direction.LONG, config)
    assert ok

    # Tier 1 SHORT - should pass
    ok, reason = check_liquidity_tier("SBER", Direction.SHORT, config)
    assert ok


def test_opening_session_filter():
    """Test opening session filter (first 15 minutes)."""
    config = SignalFilter(skip_opening_session=True, opening_session_minutes=15)

    # 10:05 Moscow = 07:05 UTC - should fail (within first 15 min)
    ts = datetime(2024, 1, 15, 7, 5, tzinfo=timezone.utc)
    ok, reason = check_opening_session(ts, config)
    assert not ok
    assert "Opening session" in reason

    # 10:20 Moscow = 07:20 UTC - should pass
    ts = datetime(2024, 1, 15, 7, 20, tzinfo=timezone.utc)
    ok, reason = check_opening_session(ts, config)
    assert ok


def test_filter_signal_with_all_filters():
    """Test full filter_signal with all new filters."""
    anomaly = AnomalyResult(
        secid="TRNFP",  # Tier 3
        score=2.0,
        direction=Direction.SHORT,  # SHORT on Tier 3 = fail
        z_ret_5m=2.0,
        z_vol_5m=1.5,
        ret_5m=0.02,
        turnover_5m=2_000_000,
        spread_bps=30.0,  # Wide spread
        volume_spike=1.5,
    )

    features = {
        "rsi_14": 50.0,
        "macd": 0.1,
        "macd_signal": 0.05,
        "macd_hist": 0.05,
        "bb_position": 0.5,
        "adx": 25.0,
    }

    config = SignalFilter()
    passed, reasons = filter_signal(
        anomaly, features, config,
        horizon="5m",
        timestamp=datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),  # Not opening
    )

    # Should fail due to Tier 3 SHORT restriction
    assert not passed
    assert any("Tier 3" in r for r in reasons)


def test_filter_signal_passes_all():
    """Test that valid signal passes all filters."""
    anomaly = AnomalyResult(
        secid="SBER",  # Tier 1
        score=2.0,
        direction=Direction.LONG,
        z_ret_5m=2.0,
        z_vol_5m=1.5,
        ret_5m=0.02,
        turnover_5m=5_000_000,
        spread_bps=10.0,  # Tight spread
        volume_spike=2.0,
    )

    features = {
        "rsi_14": 45.0,  # Good for LONG
        "macd": 0.1,
        "macd_signal": 0.05,
        "macd_hist": 0.05,  # Positive histogram
        "bb_position": 0.5,  # Middle of bands
        "adx": 30.0,  # Good trend strength
    }

    config = SignalFilter()
    passed, reasons = filter_signal(
        anomaly, features, config,
        horizon="5m",
        timestamp=datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),
    )

    assert passed
    assert "All rules passed" in reasons
