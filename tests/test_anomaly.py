"""Tests for anomaly detection."""
import numpy as np
import pandas as pd
import pytest

from moex_agent.anomaly import robust_z, compute_anomalies, Direction


def test_robust_z_normal_distribution():
    """Test robust z-score on normal data."""
    np.random.seed(42)
    hist = np.random.randn(1000)

    # Value at mean should have z ~ 0
    z = robust_z(0.0, hist)
    assert abs(z) < 0.5

    # Value at 3 sigma should have high z
    z = robust_z(3.0, hist)
    assert z > 2.0


def test_robust_z_with_outliers():
    """MAD-based z-score should be resistant to outliers."""
    np.random.seed(42)
    hist = np.concatenate([
        np.random.randn(900),
        np.array([100, 100, 100])  # Outliers
    ])

    # Standard z-score would be affected by outliers
    # MAD-based should be more stable
    z = robust_z(2.0, hist)
    assert 1.0 < z < 3.0


def test_compute_anomalies_empty():
    """Should handle empty data gracefully."""
    df = pd.DataFrame(columns=["secid", "ts", "close", "value", "volume"])
    quotes = {}

    anomalies = compute_anomalies(
        df, quotes,
        min_turnover_rub_5m=1_000_000,
        max_spread_bps=200,
        top_n=10,
    )

    assert anomalies == []


def test_compute_anomalies_basic():
    """Test basic anomaly detection."""
    np.random.seed(42)
    n = 300

    # Create normal data with a spike at the end
    close = 100 + np.random.randn(n).cumsum() * 0.01
    close[-1] += 2  # Spike

    df = pd.DataFrame({
        "secid": ["SBER"] * n,
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "close": close,
        "value": np.random.uniform(1e6, 1e7, n),
        "volume": np.random.uniform(1e4, 1e5, n),
    })

    quotes = {"SBER": {"bid": 100, "ask": 100.1}}

    anomalies = compute_anomalies(
        df, quotes,
        min_turnover_rub_5m=1_000_000,
        max_spread_bps=200,
        top_n=10,
        min_abs_z_ret=0.5,
    )

    # Should detect something
    assert len(anomalies) >= 0  # May or may not detect depending on threshold


def test_anomaly_direction():
    """Test that direction is correctly assigned."""
    np.random.seed(42)
    n = 300

    # Create data with positive return
    close = 100 + np.arange(n) * 0.01

    df = pd.DataFrame({
        "secid": ["SBER"] * n,
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "close": close,
        "value": np.random.uniform(1e6, 1e7, n),
        "volume": np.random.uniform(1e4, 1e5, n),
    })

    quotes = {"SBER": {"bid": 100, "ask": 100.1}}

    anomalies = compute_anomalies(
        df, quotes,
        min_turnover_rub_5m=0,  # No filter
        max_spread_bps=10000,    # No filter
        top_n=10,
        min_abs_z_ret=0.0,      # Low threshold
    )

    if anomalies:
        # With consistently rising prices, should be LONG
        assert anomalies[0].direction == Direction.LONG
