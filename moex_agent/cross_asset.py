"""
MOEX Agent v2.5 Cross-Asset Features

Phase 3: Cross-asset correlations, lead-lag, and basis features.

Features:
1. Relative strength vs index
2. Residual returns (alpha)
3. Sector momentum
4. Futures-equity lead
5. Basis z-score (futures vs spot)
6. Basis × microstructure cross signal

Usage:
    from moex_agent.cross_asset import CrossAssetFeatures

    cross = CrossAssetFeatures()
    features = cross.get_features(ticker, df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Sector definitions
SECTORS = {
    "oil_gas": ["LKOH", "ROSN", "TATN", "GAZP", "NVTK", "SNGS", "SIBN"],
    "banks": ["SBER", "VTBR", "TCSG", "CBOM"],
    "metals": ["GMKN", "PLZL", "NLMK", "CHMF", "MAGN"],
    "tech": ["YNDX", "OZON", "VKCO"],
    "telecom": ["MTSS", "RTKM"],
    "retail": ["MGNT", "X5", "FIVE"],
}

# Ticker to sector mapping
TICKER_SECTOR = {}
for sector, tickers in SECTORS.items():
    for ticker in tickers:
        TICKER_SECTOR[ticker] = sector

# Futures-spot pairs
FUTURES_SPOT_PAIRS = {
    "RI": "RTS",   # RTS Index (calculated)
    "MX": "IMOEX", # MOEX Index
    "BR": "BZ",    # ICE Brent
}


@dataclass
class CrossAssetState:
    """Container for cross-asset features."""
    # Relative strength
    relative_strength_5m: float = 0.0
    relative_strength_30m: float = 0.0
    relative_strength_1h: float = 0.0

    # Residual return
    residual_return: float = 0.0
    beta_to_index: float = 1.0

    # Sector
    sector_momentum: float = 0.0
    sector_rank: int = 0  # 1 = strongest

    # Lead-lag
    futures_lead: float = 0.0

    # Basis
    basis_raw: float = 0.0
    basis_zscore: float = 0.0

    # Cross signals
    basis_micro_cross: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cross_rel_strength_5m": self.relative_strength_5m,
            "cross_rel_strength_30m": self.relative_strength_30m,
            "cross_rel_strength_1h": self.relative_strength_1h,
            "cross_residual_return": self.residual_return,
            "cross_beta": self.beta_to_index,
            "cross_sector_momentum": self.sector_momentum,
            "cross_sector_rank": float(self.sector_rank),
            "cross_futures_lead": self.futures_lead,
            "cross_basis_raw": self.basis_raw,
            "cross_basis_zscore": self.basis_zscore,
            "cross_basis_micro": self.basis_micro_cross,
        }


class CrossAssetFeatures:
    """
    Calculate cross-asset features for trading signals.
    """

    # Rolling window for beta calculation
    BETA_WINDOW = 20  # 20 days

    # Basis z-score parameters
    BASIS_HISTORY_DAYS = 60

    def __init__(self):
        # Cache for beta estimates
        self._beta_cache: Dict[str, Tuple[datetime, float]] = {}

        # Basis history for z-score
        self._basis_history: Dict[str, List[float]] = {}

    def get_features(
        self,
        ticker: str,
        ticker_returns: pd.Series,
        index_returns: pd.Series,
        sector_returns: Optional[Dict[str, pd.Series]] = None,
        futures_return: Optional[float] = None,
        spot_price: Optional[float] = None,
        futures_price: Optional[float] = None,
        carry: float = 0.0,
        imbalance: float = 0.0,
    ) -> CrossAssetState:
        """
        Calculate all cross-asset features for a ticker.

        Args:
            ticker: Ticker symbol
            ticker_returns: Recent returns for ticker
            index_returns: IMOEX returns (same period)
            sector_returns: Dict of sector -> returns series
            futures_return: RI/MX return for lead-lag
            spot_price: Spot price (for basis)
            futures_price: Futures price (for basis)
            carry: Expected carry (annualized)
            imbalance: Order book imbalance (for cross signal)

        Returns:
            CrossAssetState with all features
        """
        state = CrossAssetState()

        # 1. Relative strength
        state.relative_strength_5m = self._relative_strength(ticker_returns, index_returns, 5)
        state.relative_strength_30m = self._relative_strength(ticker_returns, index_returns, 30)
        state.relative_strength_1h = self._relative_strength(ticker_returns, index_returns, 60)

        # 2. Residual return
        state.beta_to_index = self._estimate_beta(ticker, ticker_returns, index_returns)
        state.residual_return = self._residual_return(
            ticker_returns, index_returns, state.beta_to_index
        )

        # 3. Sector momentum
        if sector_returns:
            state.sector_momentum, state.sector_rank = self._sector_momentum(
                ticker, sector_returns
            )

        # 4. Futures lead
        if futures_return is not None:
            ticker_return = ticker_returns.iloc[-1] if len(ticker_returns) > 0 else 0.0
            state.futures_lead = futures_return - ticker_return

        # 5. Basis z-score
        if spot_price and futures_price and spot_price > 0:
            state.basis_raw = (futures_price / spot_price - 1) - carry
            state.basis_zscore = self._basis_zscore(ticker, state.basis_raw)

        # 6. Basis × microstructure cross
        state.basis_micro_cross = state.basis_zscore * imbalance

        return state

    def _relative_strength(
        self,
        ticker_returns: pd.Series,
        index_returns: pd.Series,
        window: int,
    ) -> float:
        """
        Calculate relative strength vs index.

        ticker_return - index_return over window.
        Positive = ticker outperforming.
        """
        if len(ticker_returns) < window or len(index_returns) < window:
            return 0.0

        ticker_ret = ticker_returns.iloc[-window:].sum()
        index_ret = index_returns.iloc[-window:].sum()

        return float(ticker_ret - index_ret)

    def _estimate_beta(
        self,
        ticker: str,
        ticker_returns: pd.Series,
        index_returns: pd.Series,
    ) -> float:
        """
        Estimate beta from rolling regression.

        Uses cache with daily refresh.
        """
        today = datetime.now().date()

        # Check cache
        if ticker in self._beta_cache:
            cached_date, cached_beta = self._beta_cache[ticker]
            if cached_date == today:
                return cached_beta

        # Need at least BETA_WINDOW observations
        if len(ticker_returns) < self.BETA_WINDOW or len(index_returns) < self.BETA_WINDOW:
            return 1.0

        try:
            # Align series
            tr = ticker_returns.iloc[-self.BETA_WINDOW:]
            ir = index_returns.iloc[-self.BETA_WINDOW:]

            # Simple OLS beta = cov(ticker, index) / var(index)
            cov = np.cov(tr.values, ir.values)[0, 1]
            var = np.var(ir.values)

            if var > 1e-10:
                beta = float(cov / var)
                beta = np.clip(beta, 0.1, 3.0)  # Reasonable bounds
            else:
                beta = 1.0

            self._beta_cache[ticker] = (today, beta)
            return beta

        except Exception:
            return 1.0

    def _residual_return(
        self,
        ticker_returns: pd.Series,
        index_returns: pd.Series,
        beta: float,
    ) -> float:
        """
        Calculate residual return (alpha).

        residual = ticker_return - beta × index_return
        Positive residual = specific demand for the stock.
        """
        if len(ticker_returns) == 0 or len(index_returns) == 0:
            return 0.0

        ticker_ret = ticker_returns.iloc[-1]
        index_ret = index_returns.iloc[-1]

        return float(ticker_ret - beta * index_ret)

    def _sector_momentum(
        self,
        ticker: str,
        sector_returns: Dict[str, pd.Series],
    ) -> Tuple[float, int]:
        """
        Calculate sector momentum and rank.

        Returns:
            (sector_momentum, sector_rank)
            momentum = cumulative return of ticker's sector
            rank = 1 (strongest) to N (weakest)
        """
        sector = TICKER_SECTOR.get(ticker)
        if not sector or sector not in sector_returns:
            return 0.0, 0

        # Calculate cumulative returns for each sector
        sector_cum_returns = {}
        for s, returns in sector_returns.items():
            if len(returns) >= 60:  # 1h
                sector_cum_returns[s] = returns.iloc[-60:].sum()
            elif len(returns) > 0:
                sector_cum_returns[s] = returns.sum()

        if not sector_cum_returns:
            return 0.0, 0

        # Sort by return (descending)
        sorted_sectors = sorted(sector_cum_returns.items(), key=lambda x: x[1], reverse=True)

        # Find rank of ticker's sector
        rank = 1
        for s, _ in sorted_sectors:
            if s == sector:
                break
            rank += 1

        momentum = sector_cum_returns.get(sector, 0.0)

        return float(momentum), rank

    def _basis_zscore(self, ticker: str, basis_raw: float) -> float:
        """
        Calculate basis z-score.

        z = (basis - mean) / std

        |z| > 2 suggests mean-reversion opportunity.
        """
        # Update history
        if ticker not in self._basis_history:
            self._basis_history[ticker] = []

        self._basis_history[ticker].append(basis_raw)

        # Keep only recent history
        max_len = self.BASIS_HISTORY_DAYS * 530  # ~530 1m bars per day
        if len(self._basis_history[ticker]) > max_len:
            self._basis_history[ticker] = self._basis_history[ticker][-max_len:]

        history = self._basis_history[ticker]

        if len(history) < 100:
            return 0.0

        mean = np.mean(history)
        std = np.std(history)

        if std < 1e-10:
            return 0.0

        return float((basis_raw - mean) / std)

    # === Convenience methods ===

    def get_futures_equity_lead(
        self,
        ri_return_5m: float,
        sber_return_5m: float,
    ) -> float:
        """
        Calculate RI → SBER lead signal.

        RI/MX move before equities.
        If RI is up and SBER hasn't moved → SBER likely to follow.

        Args:
            ri_return_5m: RI futures return (last 5 min)
            sber_return_5m: SBER return (last 5 min)

        Returns:
            Lead signal (positive = RI ahead, SBER should follow)
        """
        return float(ri_return_5m - sber_return_5m)

    def get_basis_for_futures(
        self,
        ticker: str,
        futures_price: float,
        spot_price: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.15,  # ~15% for RUB
    ) -> Tuple[float, float]:
        """
        Calculate futures basis and z-score.

        Args:
            ticker: Futures ticker (RI, MX, BR)
            futures_price: Current futures price
            spot_price: Current spot/index price
            days_to_expiry: Days until expiration
            risk_free_rate: Annualized risk-free rate

        Returns:
            (basis_raw, basis_zscore)
        """
        if spot_price <= 0 or days_to_expiry <= 0:
            return 0.0, 0.0

        # Expected carry (fair value)
        time_to_expiry = days_to_expiry / 365.0
        carry = (1 + risk_free_rate) ** time_to_expiry - 1

        # Actual basis
        basis_raw = (futures_price / spot_price - 1) - carry
        basis_zscore = self._basis_zscore(ticker, basis_raw)

        return float(basis_raw), float(basis_zscore)

    def basis_micro_cross_signal(
        self,
        basis_zscore: float,
        imbalance: float,
    ) -> float:
        """
        Calculate basis × microstructure cross signal.

        When basis is anomalous AND order book pressure aligns → strong signal.

        Example:
            basis_z = -2 (futures cheap vs spot)
            imbalance > 0 (buyers dominate)
            → signal = -2 × 0.3 = -0.6 → strong LONG for futures

        Args:
            basis_zscore: Basis z-score (negative = futures cheap)
            imbalance: Order book imbalance (positive = bid pressure)

        Returns:
            Cross signal (interpret with sign)
        """
        # Note: negative basis_z × positive imbalance = negative value
        # But this means "cheap futures + buying pressure" = LONG
        # So for LONG signal, we want: -basis_z × imbalance > 0

        return float(-basis_zscore * imbalance)


# === Rollover utilities ===

def days_to_expiry(expiry_date: datetime, today: Optional[datetime] = None) -> int:
    """Calculate days until futures expiration."""
    if today is None:
        today = datetime.now()

    delta = expiry_date - today
    return max(0, delta.days)


def should_roll(days_remaining: int, threshold: int = 3) -> bool:
    """
    Check if position should be rolled to next contract.

    Don't open new positions with < threshold days to expiry.
    """
    return days_remaining < threshold


def volume_ratio_near_far(near_volume: float, far_volume: float) -> float:
    """
    Calculate liquidity ratio between near and far contracts.

    Returns:
        near / (near + far). < 0.4 means liquidity moved to far contract.
    """
    total = near_volume + far_volume
    if total <= 0:
        return 0.5

    return float(near_volume / total)


# Feature column names
CROSS_ASSET_FEATURE_COLS = [
    "cross_rel_strength_5m",
    "cross_rel_strength_30m",
    "cross_rel_strength_1h",
    "cross_residual_return",
    "cross_beta",
    "cross_sector_momentum",
    "cross_sector_rank",
    "cross_futures_lead",
    "cross_basis_raw",
    "cross_basis_zscore",
    "cross_basis_micro",
]


# Singleton
_cross_asset: Optional[CrossAssetFeatures] = None


def get_cross_asset_features() -> CrossAssetFeatures:
    """Get or create global CrossAssetFeatures instance."""
    global _cross_asset
    if _cross_asset is None:
        _cross_asset = CrossAssetFeatures()
    return _cross_asset
