"""
MOEX Agent v2.5 Phase 3 Feature Integration

Combines all Phase 3 features into unified feature pipeline:
1. Cross-asset features (relative strength, basis, lead-lag)
2. Calendar features (session phase, tax period, events)
3. External feeds (CME/ICE overnight, VIX, risk sentiment)

Usage:
    from moex_agent.phase3_features import Phase3Features

    p3 = Phase3Features()
    features = p3.get_all_features(
        ticker="SBER",
        df=candle_df,
        index_df=index_df,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar_features import (
    CalendarFeatures,
    get_calendar_features,
    CALENDAR_FEATURE_COLS,
)
from .cross_asset import (
    CrossAssetFeatures,
    get_cross_asset_features,
    CROSS_ASSET_FEATURE_COLS,
)
from .external_feeds import (
    ExternalFeeds,
    get_external_feeds,
    LEAD_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


# All Phase 3 feature columns
PHASE3_FEATURE_COLS = CROSS_ASSET_FEATURE_COLS + CALENDAR_FEATURE_COLS + LEAD_FEATURE_COLS


@dataclass
class Phase3State:
    """Container for all Phase 3 features."""

    # Cross-asset
    cross_features: Dict[str, float]

    # Calendar
    calendar_features: Dict[str, float]

    # External leads
    lead_features: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        """Combine all features into single dict."""
        result = {}
        result.update(self.cross_features)
        result.update(self.calendar_features)
        result.update(self.lead_features)
        return result

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.to_dict())


class Phase3Features:
    """
    Unified interface for all Phase 3 features.
    """

    def __init__(self):
        self.cross_asset = get_cross_asset_features()
        self.calendar = get_calendar_features()
        self.external = get_external_feeds()

        # Cache for external data (expensive to fetch)
        self._external_cache: Optional[Dict[str, float]] = None
        self._external_cache_time: Optional[datetime] = None
        self._external_cache_ttl_min = 5

    def get_all_features(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
        futures_price: Optional[float] = None,
        spot_price: Optional[float] = None,
        imbalance: float = 0.0,
    ) -> Phase3State:
        """
        Get all Phase 3 features for a ticker at given timestamp.

        Args:
            ticker: Ticker symbol
            ticker_df: DataFrame with ticker OHLCV and returns
            index_df: DataFrame with IMOEX returns (optional)
            timestamp: Time for calendar features (default: now)
            futures_price: For basis calculation (optional)
            spot_price: For basis calculation (optional)
            imbalance: Order book imbalance for cross signal

        Returns:
            Phase3State with all features
        """
        timestamp = timestamp or datetime.now()

        # 1. Cross-asset features
        cross_dict = self._get_cross_features(
            ticker, ticker_df, index_df, futures_price, spot_price, imbalance
        )

        # 2. Calendar features
        cal_state = self.calendar.get_features(timestamp, ticker)
        calendar_dict = cal_state.to_dict()

        # 3. External lead features (cached)
        lead_dict = self._get_external_features()

        return Phase3State(
            cross_features=cross_dict,
            calendar_features=calendar_dict,
            lead_features=lead_dict,
        )

    def _get_cross_features(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        index_df: Optional[pd.DataFrame],
        futures_price: Optional[float],
        spot_price: Optional[float],
        imbalance: float,
    ) -> Dict[str, float]:
        """Extract cross-asset features."""
        # Get return series
        if "r_1m" in ticker_df.columns:
            ticker_returns = ticker_df["r_1m"].dropna()
        elif "close" in ticker_df.columns:
            ticker_returns = ticker_df["close"].pct_change().dropna()
        else:
            ticker_returns = pd.Series(dtype=float)

        # Index returns
        if index_df is not None:
            if "r_1m" in index_df.columns:
                index_returns = index_df["r_1m"].dropna()
            elif "close" in index_df.columns:
                index_returns = index_df["close"].pct_change().dropna()
            else:
                index_returns = pd.Series(dtype=float)
        else:
            # Use ticker returns as proxy (beta=1)
            index_returns = ticker_returns

        # Get cross-asset state
        cross_state = self.cross_asset.get_features(
            ticker=ticker,
            ticker_returns=ticker_returns,
            index_returns=index_returns,
            futures_price=futures_price,
            spot_price=spot_price,
            imbalance=imbalance,
        )

        return cross_state.to_dict()

    def _get_external_features(self) -> Dict[str, float]:
        """Get external lead features with caching."""
        now = datetime.now()

        # Check cache
        if (
            self._external_cache is not None
            and self._external_cache_time is not None
            and (now - self._external_cache_time).total_seconds() < self._external_cache_ttl_min * 60
        ):
            return self._external_cache

        # Fetch fresh data
        try:
            signals = self.external.get_lead_features()
            self._external_cache = signals.to_dict()
            self._external_cache_time = now
        except Exception as e:
            logger.warning(f"Error fetching external feeds: {e}")
            # Return defaults
            self._external_cache = {col: 0.0 for col in LEAD_FEATURE_COLS}
            self._external_cache["lead_vix"] = 20.0  # Default VIX
            self._external_cache_time = now

        return self._external_cache

    def add_features_to_df(
        self,
        df: pd.DataFrame,
        ticker: str,
        index_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add Phase 3 features to existing DataFrame.

        Args:
            df: DataFrame with OHLCV data and timestamp index
            ticker: Ticker symbol
            index_df: Index DataFrame (optional)

        Returns:
            DataFrame with Phase 3 features added
        """
        df = df.copy()

        # Get current features (external data is point-in-time)
        lead_dict = self._get_external_features()

        # Add lead features (constant for all rows - point-in-time snapshot)
        for col, val in lead_dict.items():
            df[col] = val

        # Add calendar features per row
        for idx in df.index:
            if isinstance(idx, pd.Timestamp):
                ts = idx.to_pydatetime()
            else:
                ts = datetime.now()

            cal_state = self.calendar.get_features(ts, ticker)
            for col, val in cal_state.to_dict().items():
                if col not in df.columns:
                    df[col] = np.nan
                df.loc[idx, col] = val

        # Add cross-asset features (rolling calculation)
        cross_dict = self._get_cross_features(ticker, df, index_df, None, None, 0.0)
        for col, val in cross_dict.items():
            df[col] = val  # Most recent value for all rows

        return df


# Singleton
_phase3: Optional[Phase3Features] = None


def get_phase3_features() -> Phase3Features:
    """Get or create global Phase3Features instance."""
    global _phase3
    if _phase3 is None:
        _phase3 = Phase3Features()
    return _phase3


def build_phase3_features(
    ticker: str,
    df: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience function to add all Phase 3 features to a DataFrame.

    Args:
        ticker: Ticker symbol
        df: DataFrame with OHLCV data
        index_df: Index returns (optional)

    Returns:
        DataFrame with Phase 3 features added
    """
    p3 = get_phase3_features()
    return p3.add_features_to_df(df, ticker, index_df)
