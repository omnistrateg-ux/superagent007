"""Tests for feature computation."""
import numpy as np
import pandas as pd
import pytest

from moex_agent.features import FEATURE_COLS, build_feature_frame, compute_rsi, compute_atr


def test_feature_cols_count():
    """Verify we have exactly 30 features."""
    assert len(FEATURE_COLS) == 30


def test_feature_cols_unique():
    """Verify all feature names are unique."""
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_compute_rsi_range():
    """RSI should be between 0 and 100."""
    close = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103] * 5)
    rsi = compute_rsi(close, period=14)
    valid = rsi.dropna()
    assert all(0 <= v <= 100 for v in valid)


def test_build_feature_frame():
    """Test feature frame building."""
    # Create sample data
    np.random.seed(42)
    n = 200
    data = {
        "secid": ["SBER"] * n,
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": 100 + np.random.randn(n).cumsum(),
        "high": 101 + np.random.randn(n).cumsum(),
        "low": 99 + np.random.randn(n).cumsum(),
        "close": 100 + np.random.randn(n).cumsum(),
        "value": np.random.uniform(1e6, 1e7, n),
        "volume": np.random.uniform(1e4, 1e5, n),
    }
    df = pd.DataFrame(data)

    # Fix high/low
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.1
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.1

    feats = build_feature_frame(df)

    # Should have all feature columns
    for col in FEATURE_COLS:
        assert col in feats.columns, f"Missing column: {col}"

    # Should have secid and ts
    assert "secid" in feats.columns
    assert "ts" in feats.columns


def test_build_feature_frame_multiple_tickers():
    """Test with multiple tickers."""
    np.random.seed(42)
    n = 100

    dfs = []
    for ticker in ["SBER", "GAZP"]:
        data = {
            "secid": [ticker] * n,
            "ts": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": 100 + np.random.randn(n).cumsum(),
            "high": 101 + np.random.randn(n).cumsum(),
            "low": 99 + np.random.randn(n).cumsum(),
            "close": 100 + np.random.randn(n).cumsum(),
            "value": np.random.uniform(1e6, 1e7, n),
            "volume": np.random.uniform(1e4, 1e5, n),
        }
        df = pd.DataFrame(data)
        df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.1
        df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.1
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    feats = build_feature_frame(combined)

    assert set(feats["secid"].unique()) == {"SBER", "GAZP"}
