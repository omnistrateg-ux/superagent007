#!/usr/bin/env python3
"""
Full Feature Model Training (77 features)

Trains LightGBM + CatBoost on all feature categories:
- Base TA (30 features)
- Anomaly (5 features)
- Cross-Asset (11 features)
- Calendar (22 features)
- External/Lead (9 features)

Total: 77 features

Usage:
    python -m moex_agent.train_full_features
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train_full_features")

MSK = timezone(timedelta(hours=3))


# =============================================================================
# FEATURE DEFINITIONS (77 total)
# =============================================================================

BASE_TA_COLS = [
    # Returns (5)
    "r_1m", "r_5m", "r_10m", "r_30m", "r_60m",
    # Turnover (3)
    "turn_1m", "turn_5m", "turn_10m",
    # ATR & VWAP (2)
    "atr_14", "dist_vwap_atr",
    # RSI (2)
    "rsi_14", "rsi_7",
    # MACD (3)
    "macd", "macd_signal", "macd_hist",
    # Bollinger Bands (2)
    "bb_position", "bb_width",
    # Stochastic (2)
    "stoch_k", "stoch_d",
    # ADX (1)
    "adx",
    # OBV (1)
    "obv_change",
    # Momentum (2)
    "momentum_10", "momentum_30",
    # Volatility (2)
    "volatility_10", "volatility_30",
    # Moving averages (3)
    "price_sma20_ratio", "price_sma50_ratio", "sma20_sma50_ratio",
    # Volume (1)
    "volume_sma_ratio",
    # Extra (1)
    "hl_range",
]

ANOMALY_COLS = [
    "anomaly_z_ret_5m",
    "anomaly_z_vol_5m",
    "anomaly_score",
    "anomaly_volume_spike",
    "anomaly_direction",
]

CROSS_ASSET_COLS = [
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

CALENDAR_COLS = [
    "cal_session_phase",
    "cal_time_normalized",
    "cal_is_evening",
    "cal_day_of_week",
    "cal_is_monday",
    "cal_is_friday",
    "cal_is_month_start",
    "cal_is_month_end",
    "cal_is_quarter_end",
    "cal_is_tax_period",
    "cal_tax_day",
    "cal_is_expiry_week",
    "cal_is_expiry_day",
    "cal_days_to_expiry",
    "cal_days_to_divcut",
    "cal_div_yield",
    "cal_is_post_divcut",
    "cal_is_cbr_day",
    "cal_is_opec_day",
    "cal_is_fed_day",
    "cal_event_risk_mult",
    "cal_liquidity_expected",
]

EXTERNAL_COLS = [
    "lead_brent_overnight",
    "lead_brent_5m",
    "lead_sp500_overnight",
    "lead_vix",
    "lead_vix_change",
    "lead_ng_overnight",
    "lead_gold_overnight",
    "lead_usdrub_change",
    "lead_risk_sentiment",
]

ALL_FEATURE_COLS = BASE_TA_COLS + ANOMALY_COLS + CROSS_ASSET_COLS + CALENDAR_COLS + EXTERNAL_COLS
assert len(ALL_FEATURE_COLS) == 77, f"Expected 77 features, got {len(ALL_FEATURE_COLS)}"


# =============================================================================
# MOEX ISS DATA FETCHER
# =============================================================================

def find_active_contracts(base: str) -> List[str]:
    """Find active contract symbols for a base symbol."""
    try:
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
            f"?iss.meta=off&iss.only=securities,marketdata"
            f"&securities.columns=SECID,SHORTNAME"
            f"&marketdata.columns=SECID,OPENPOSITION,VOLTODAY"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())

        sec_rows = data.get("securities", {}).get("data", [])
        md_rows = data.get("marketdata", {}).get("data", [])

        # Get OI from marketdata
        oi_map = {}
        for row in md_rows:
            if row[0]:
                oi_map[row[0]] = row[1] or 0

        # Find matching contracts
        candidates = []
        for row in sec_rows:
            secid = row[0]
            if secid and secid.startswith(base):
                oi = oi_map.get(secid, 0)
                candidates.append((secid, oi))

        # Sort by OI
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:3]]
    except Exception as e:
        log.warning(f"Error finding contracts for {base}: {e}")
        return []


def fetch_candles_1m(secid: str, days: int = 90) -> pd.DataFrame:
    """Fetch 1-minute candles from MOEX ISS."""
    end_date = datetime.now(MSK).date()
    start_date = end_date - timedelta(days=days)

    all_candles = []
    current_start = str(start_date)

    log.info(f"  Fetching 1m candles for {secid}...")

    while True:
        try:
            url = (
                f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}/candles.json"
                f"?interval=1&from={current_start}&till={end_date}"
                f"&iss.meta=off&candles.columns=open,close,high,low,begin,value,volume"
            )
            data = json.loads(urllib.request.urlopen(url, timeout=30).read())
            batch = data.get("candles", {}).get("data", [])

            if not batch:
                break

            for row in batch:
                all_candles.append({
                    "open": row[0],
                    "close": row[1],
                    "high": row[2],
                    "low": row[3],
                    "ts": row[4],
                    "value": row[5] or 0,
                    "volume": row[6] or 0,
                })

            # Pagination
            last_ts = batch[-1][4]
            try:
                last_dt = datetime.fromisoformat(last_ts.replace(" ", "T"))
                current_start = (last_dt + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
            except:
                break

            if len(batch) < 500:
                break

        except Exception as e:
            log.warning(f"Error fetching candles: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["ts"] = pd.to_datetime(df["ts"])
    df["secid"] = secid
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    log.info(f"    Got {len(df)} candles")
    return df


def load_futures_data(tickers: List[str], days: int = 90) -> pd.DataFrame:
    """Load futures data for multiple tickers."""
    all_dfs = []

    for ticker in tickers:
        log.info(f"Loading data for {ticker}...")
        contracts = find_active_contracts(ticker)

        if not contracts:
            log.warning(f"  No contracts found for {ticker}")
            continue

        for secid in contracts[:2]:  # Most liquid 2 contracts
            df = fetch_candles_1m(secid, days=days)
            if not df.empty:
                df["ticker"] = ticker
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_atr(g: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator."""
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands position and width."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    position = (close - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (close + 1e-9)
    return position, width


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator."""
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-9)
    d = k.rolling(3).mean()
    return k, d


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period, adjust=False).mean()


def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_z(values: pd.Series, window: int = 500) -> pd.Series:
    """Rolling robust z-score using MAD."""
    result = pd.Series(index=values.index, dtype=float)

    for i in range(len(values)):
        if i < 50:
            result.iloc[i] = 0.0
            continue

        start = max(0, i - window)
        hist = values.iloc[start:i].dropna().values

        if len(hist) < 30:
            result.iloc[i] = 0.0
            continue

        med = np.median(hist)
        mad = _mad(hist)

        if mad < 1e-12:
            result.iloc[i] = 0.0
        else:
            result.iloc[i] = (values.iloc[i] - med) / (1.4826 * mad)

    return result


def build_base_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build base TA features (30 features)."""
    g = df.copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    volume = g["volume"].astype(float)
    value = g["value"].astype(float)

    feat = pd.DataFrame(index=g.index)

    # Returns
    feat["r_1m"] = close.pct_change(1)
    feat["r_5m"] = close.pct_change(5)
    feat["r_10m"] = close.pct_change(10)
    feat["r_30m"] = close.pct_change(30)
    feat["r_60m"] = close.pct_change(60)

    # Turnover
    feat["turn_1m"] = value
    feat["turn_5m"] = value.rolling(5).sum()
    feat["turn_10m"] = value.rolling(10).sum()

    # ATR & VWAP
    feat["atr_14"] = compute_atr(g, 14)
    vwap_30 = value.rolling(30).sum() / (volume.rolling(30).sum() + 1e-9)
    feat["dist_vwap_atr"] = (close - vwap_30) / (feat["atr_14"] + 1e-9)

    # RSI
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["rsi_7"] = compute_rsi(close, 7)

    # MACD
    macd_line, signal_line, histogram = compute_macd(close)
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"] = histogram

    # Bollinger Bands
    bb_pos, bb_w = compute_bollinger(close)
    feat["bb_position"] = bb_pos
    feat["bb_width"] = bb_w

    # Stochastic
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d

    # ADX
    feat["adx"] = compute_adx(high, low, close)

    # OBV
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_mean = obv.rolling(100, min_periods=10).mean().abs() + 1e-9
    feat["obv_change"] = obv.diff(10) / obv_mean

    # Momentum
    feat["momentum_10"] = close / close.shift(10) - 1
    feat["momentum_30"] = close / close.shift(30) - 1

    # Volatility
    feat["volatility_10"] = close.pct_change().rolling(10).std()
    feat["volatility_30"] = close.pct_change().rolling(30).std()

    # Moving averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    feat["price_sma20_ratio"] = close / (sma_20 + 1e-9)
    feat["price_sma50_ratio"] = close / (sma_50 + 1e-9)
    feat["sma20_sma50_ratio"] = sma_20 / (sma_50 + 1e-9)

    # Volume
    feat["volume_sma_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    # High-Low range
    feat["hl_range"] = (high - low) / (close + 1e-9)

    return feat


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build anomaly features (5 features)."""
    g = df.copy()
    close = g["close"].astype(float)
    volume = g["volume"].astype(float)
    value = g["value"].astype(float)

    feat = pd.DataFrame(index=g.index)

    # 5-minute returns and turnover
    r5 = close.pct_change(5)
    turn5 = value.rolling(5).sum()

    # Robust z-scores
    feat["anomaly_z_ret_5m"] = robust_z(r5, window=500)
    feat["anomaly_z_vol_5m"] = robust_z(turn5, window=500)

    # Volume spike
    vol_avg = volume.rolling(100, min_periods=10).mean()
    feat["anomaly_volume_spike"] = volume / (vol_avg + 1e-9)

    # Anomaly score
    abs_z_ret = feat["anomaly_z_ret_5m"].abs()
    vol_bonus = 0.3 * feat["anomaly_z_vol_5m"].clip(0, 4)
    spike_bonus = 0.2 * (feat["anomaly_volume_spike"] - 1).clip(0, 5)
    feat["anomaly_score"] = abs_z_ret + vol_bonus + spike_bonus

    # Direction
    feat["anomaly_direction"] = np.where(r5 > 0, 1, np.where(r5 < 0, -1, 0))

    return feat


def build_cross_asset_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build cross-asset features (11 features) - simplified for futures."""
    n = len(df)
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    returns = close.pct_change()

    # Relative strength (self-relative for futures)
    feat["cross_rel_strength_5m"] = returns.rolling(5).sum()
    feat["cross_rel_strength_30m"] = returns.rolling(30).sum()
    feat["cross_rel_strength_1h"] = returns.rolling(60).sum()

    # Residual return (alpha) - simplified
    feat["cross_residual_return"] = returns - returns.rolling(60).mean()

    # Beta (volatility ratio)
    vol_short = returns.rolling(20).std()
    vol_long = returns.rolling(60).std()
    feat["cross_beta"] = vol_short / (vol_long + 1e-9)

    # Sector momentum (use own momentum for futures)
    feat["cross_sector_momentum"] = returns.rolling(60).sum()
    feat["cross_sector_rank"] = 0.5  # Placeholder

    # Futures lead (use lagged returns)
    feat["cross_futures_lead"] = returns.shift(1) - returns.shift(5)

    # Basis features (placeholder - would need spot data)
    ema_20 = close.ewm(span=20).mean()
    feat["cross_basis_raw"] = (close - ema_20) / (ema_20 + 1e-9)
    feat["cross_basis_zscore"] = robust_z(feat["cross_basis_raw"], window=200)
    feat["cross_basis_micro"] = feat["cross_basis_zscore"] * feat["cross_rel_strength_5m"]

    return feat


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build calendar features (22 features)."""
    ts = pd.to_datetime(df["ts"])
    n = len(df)
    feat = pd.DataFrame(index=df.index)

    # Session phase (encoded)
    hour = ts.dt.hour
    minute = ts.dt.minute
    time_mins = hour * 60 + minute

    # Session phase encoding
    def get_session_phase(t):
        if 410 <= t < 420: return 0.1   # Morning auction
        if 420 <= t < 600: return 0.2   # Morning session
        if 600 <= t < 615: return 0.3   # Opening drive
        if 615 <= t < 690: return 0.4   # Morning active
        if 690 <= t < 780: return 0.5   # Midday
        if 780 <= t < 840: return 0.6   # Lunch
        if 840 <= t < 845: return 0.0   # Clearing
        if 845 <= t < 960: return 0.7   # Afternoon
        if 960 <= t < 1120: return 0.8  # Pre-close
        if 1120 <= t < 1130: return 0.9 # Closing auction
        if 1145 <= t < 1430: return 1.0 # Evening session
        return 0.0

    feat["cal_session_phase"] = time_mins.apply(get_session_phase)

    # Time normalized (0 = 10:00, 1 = 18:40)
    feat["cal_time_normalized"] = ((time_mins - 600) / 520).clip(0, 1)

    # Evening session
    feat["cal_is_evening"] = ((time_mins >= 1145) & (time_mins < 1430)).astype(float)

    # Day of week
    dow = ts.dt.dayofweek
    feat["cal_day_of_week"] = dow / 4.0
    feat["cal_is_monday"] = (dow == 0).astype(float)
    feat["cal_is_friday"] = (dow == 4).astype(float)

    # Month effects
    day = ts.dt.day
    feat["cal_is_month_start"] = (day <= 3).astype(float)
    feat["cal_is_month_end"] = (day >= 28).astype(float)

    month = ts.dt.month
    feat["cal_is_quarter_end"] = (month.isin([3, 6, 9, 12]) & (day >= 25)).astype(float)

    # Tax period (20-25)
    feat["cal_is_tax_period"] = ((day >= 20) & (day <= 25)).astype(float)
    feat["cal_tax_day"] = np.where((day >= 20) & (day <= 25), (day - 19) / 6.0, 0.0)

    # Expiration (3rd Thursday of March, June, Sep, Dec)
    # Simplified: estimate days to expiry
    feat["cal_is_expiry_week"] = 0.0  # Would need expiry calendar
    feat["cal_is_expiry_day"] = 0.0
    feat["cal_days_to_expiry"] = 0.5  # Placeholder

    # Dividends (not applicable for futures)
    feat["cal_days_to_divcut"] = 1.0
    feat["cal_div_yield"] = 0.0
    feat["cal_is_post_divcut"] = 0.0

    # Events (simplified)
    feat["cal_is_cbr_day"] = 0.0
    feat["cal_is_opec_day"] = 0.0
    feat["cal_is_fed_day"] = 0.0

    # Risk/liquidity multipliers
    feat["cal_event_risk_mult"] = 1.0 - 0.5 * feat["cal_is_evening"]
    feat["cal_liquidity_expected"] = 1.0 - 0.7 * feat["cal_is_evening"] - 0.3 * feat["cal_is_friday"]

    return feat


def build_external_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build external/lead features (9 features) - simplified using lagged returns."""
    n = len(df)
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    returns = close.pct_change()

    # Lead signals (use lagged own returns as proxy)
    # In production, would fetch from yfinance
    feat["lead_brent_overnight"] = returns.shift(60) if ticker == "BR" else 0.0
    feat["lead_brent_5m"] = returns.shift(5) if ticker == "BR" else 0.0
    feat["lead_sp500_overnight"] = returns.shift(60) if ticker in ["RI", "MX"] else 0.0

    # VIX proxy (volatility)
    vol_5m = returns.rolling(5).std() * np.sqrt(252 * 78)  # Annualized
    feat["lead_vix"] = 20 + vol_5m * 100  # Proxy VIX
    feat["lead_vix_change"] = feat["lead_vix"].pct_change(60)

    # Natural gas
    feat["lead_ng_overnight"] = returns.shift(60) if ticker == "NG" else 0.0

    # Gold (placeholder)
    feat["lead_gold_overnight"] = 0.0

    # USD/RUB change (placeholder)
    feat["lead_usdrub_change"] = 0.0

    # Risk sentiment (based on volatility)
    feat["lead_risk_sentiment"] = -feat["lead_vix_change"].clip(-0.1, 0.1) * 10

    return feat


def build_all_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build all 77 features for a ticker."""
    log.info(f"  Building features for {ticker}...")

    # Build each category
    base_ta = build_base_ta_features(df)
    anomaly = build_anomaly_features(df)
    cross_asset = build_cross_asset_features(df, ticker)
    calendar = build_calendar_features(df)
    external = build_external_features(df, ticker)

    # Combine
    features = pd.concat([base_ta, anomaly, cross_asset, calendar, external], axis=1)

    # Add metadata
    features["ts"] = df["ts"].values
    features["close"] = df["close"].values
    features["ticker"] = ticker

    # Replace inf
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def create_target(df: pd.DataFrame, horizon_mins: int = 10) -> pd.Series:
    """Create binary target: price after horizon > entry price."""
    close = df["close"].astype(float)
    future_price = close.shift(-horizon_mins)
    # 1 if price goes up, 0 otherwise
    target = (future_price > close).astype(int)
    return target


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lightgbm(X_train, y_train, X_test, y_test, feature_names: List[str]) -> Tuple[Any, Dict]:
    """Train LightGBM classifier."""
    try:
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "random_state": 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "win_rate": y_pred.mean(),
        }

        # Feature importance
        importance = dict(zip(feature_names, model.feature_importances_))

        return model, {"metrics": metrics, "importance": importance}

    except (ImportError, OSError) as e:
        log.warning(f"LightGBM not available: {e}")
        return None, {}


def train_catboost(X_train, y_train, X_test, y_test, feature_names: List[str]) -> Tuple[Any, Dict]:
    """Train CatBoost classifier."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not installed. Run: pip install catboost")
        return None, {}

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="AUC",
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "win_rate": y_pred.mean(),
    }

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))

    return model, {"metrics": metrics, "importance": importance}


def print_feature_importance(importance: Dict[str, float], top_n: int = 20):
    """Print top N features by importance."""
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for v in importance.values())

    print(f"\n{'='*60}")
    print(f"TOP-{top_n} FEATURES BY IMPORTANCE")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Feature':<35} {'Importance':>10} {'%':>8}")
    print("-" * 60)

    for i, (feat, imp) in enumerate(sorted_imp[:top_n], 1):
        pct = imp / total * 100 if total > 0 else 0
        print(f"{i:<5} {feat:<35} {imp:>10.2f} {pct:>7.1f}%")

    return sorted_imp


def filter_low_importance_features(
    importance: Dict[str, float],
    threshold_pct: float = 1.0
) -> List[str]:
    """Return features with importance >= threshold."""
    total = sum(v for v in importance.values())
    if total == 0:
        return list(importance.keys())

    selected = []
    noise = []

    for feat, imp in importance.items():
        pct = imp / total * 100
        if pct >= threshold_pct:
            selected.append(feat)
        else:
            noise.append((feat, pct))

    print(f"\n{'='*60}")
    print(f"NOISE FEATURES (importance < {threshold_pct}%)")
    print(f"{'='*60}")
    for feat, pct in sorted(noise, key=lambda x: x[1]):
        print(f"  {feat}: {pct:.2f}%")
    print(f"\nRemoved {len(noise)} noise features")
    print(f"Keeping {len(selected)} features")

    return selected


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("FULL FEATURE MODEL TRAINING (77 features)")
    print("=" * 70)

    # Configuration
    TICKERS = ["BR", "MX", "RI", "NG"]
    HORIZON_MINS = 60  # 1 hour horizon
    DAYS = 90
    TRAIN_RATIO = 0.7

    # 1. Load data
    print("\n[1/8] Loading futures data...")
    df_raw = load_futures_data(TICKERS, days=DAYS)

    if df_raw.empty:
        log.error("No data loaded. Check MOEX ISS API availability.")
        return

    print(f"  Total raw candles: {len(df_raw):,}")

    # 2. Build features
    print("\n[2/8] Building 77 features...")
    all_features = []

    for ticker in df_raw["ticker"].unique():
        ticker_df = df_raw[df_raw["ticker"] == ticker].copy().reset_index(drop=True)
        if len(ticker_df) < 200:
            log.warning(f"  Skipping {ticker}: insufficient data ({len(ticker_df)} candles)")
            continue

        features = build_all_features(ticker_df, ticker)
        all_features.append(features)

    if not all_features:
        log.error("No features built. Check data.")
        return

    df_features = pd.concat(all_features, ignore_index=True)
    print(f"  Total feature rows: {len(df_features):,}")

    # 3. Create target
    print("\n[3/8] Creating binary target (10m horizon)...")
    df_features["target"] = create_target(df_features, horizon_mins=HORIZON_MINS)

    # Drop rows with NaN
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df_features.columns]
    df_clean = df_features.dropna(subset=feature_cols + ["target"])
    print(f"  Clean rows: {len(df_clean):,} (dropped {len(df_features) - len(df_clean):,} NaN rows)")

    # 4. Train/test split (time-based)
    print("\n[4/8] Splitting data (70% train / 30% test)...")
    split_idx = int(len(df_clean) * TRAIN_RATIO)

    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # 5. Train LightGBM
    print("\n[5/8] Training LightGBM...")
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_test, y_test, feature_cols)

    if lgb_model and lgb_results:
        print(f"  Accuracy: {lgb_results['metrics']['accuracy']:.4f}")
        print(f"  AUC:      {lgb_results['metrics']['auc']:.4f}")
        print(f"  Precision:{lgb_results['metrics']['precision']:.4f}")
        print(f"  Recall:   {lgb_results['metrics']['recall']:.4f}")
    else:
        print("  LightGBM skipped (library issue)")

    # 6. Train CatBoost
    print("\n[6/8] Training CatBoost...")
    cb_model, cb_results = train_catboost(X_train, y_train, X_test, y_test, feature_cols)

    if cb_model:
        print(f"  Accuracy: {cb_results['metrics']['accuracy']:.4f}")
        print(f"  AUC:      {cb_results['metrics']['auc']:.4f}")
        print(f"  Precision:{cb_results['metrics']['precision']:.4f}")
        print(f"  Recall:   {cb_results['metrics']['recall']:.4f}")

    # 7. Feature importance analysis
    print("\n[7/8] Feature importance analysis...")

    # Average importance from both models (use CatBoost if LightGBM failed)
    combined_importance = {}
    for feat in feature_cols:
        lgb_imp = lgb_results.get("importance", {}).get(feat, 0) if lgb_results else 0
        cb_imp = cb_results.get("importance", {}).get(feat, 0) if cb_results else 0
        if lgb_imp > 0 and cb_imp > 0:
            combined_importance[feat] = (lgb_imp + cb_imp) / 2
        else:
            combined_importance[feat] = max(lgb_imp, cb_imp)

    sorted_importance = print_feature_importance(combined_importance, top_n=20)

    # Filter noise
    selected_features = filter_low_importance_features(combined_importance, threshold_pct=1.0)

    # 8. Retrain without noise
    print("\n[8/8] Retraining without noise features...")

    X_train_filtered = train_df[selected_features].values
    X_test_filtered = test_df[selected_features].values

    print(f"\n--- LightGBM (filtered: {len(selected_features)} features) ---")
    lgb_model_filtered, lgb_results_filtered = train_lightgbm(
        X_train_filtered, y_train, X_test_filtered, y_test, selected_features
    )
    if lgb_model_filtered and lgb_results_filtered:
        lgb_orig_acc = lgb_results.get('metrics', {}).get('accuracy', 0) if lgb_results else 0
        lgb_orig_auc = lgb_results.get('metrics', {}).get('auc', 0) if lgb_results else 0
        print(f"  Accuracy: {lgb_results_filtered['metrics']['accuracy']:.4f} (was {lgb_orig_acc:.4f})")
        print(f"  AUC:      {lgb_results_filtered['metrics']['auc']:.4f} (was {lgb_orig_auc:.4f})")
    else:
        print("  LightGBM skipped")
        lgb_results_filtered = {}

    print(f"\n--- CatBoost (filtered: {len(selected_features)} features) ---")
    cb_model_filtered, cb_results_filtered = train_catboost(
        X_train_filtered, y_train, X_test_filtered, y_test, selected_features
    )
    if cb_model_filtered:
        print(f"  Accuracy: {cb_results_filtered['metrics']['accuracy']:.4f} (was {cb_results['metrics']['accuracy']:.4f})")
        print(f"  AUC:      {cb_results_filtered['metrics']['auc']:.4f} (was {cb_results['metrics']['auc']:.4f})")

    # Save best model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Determine best model
    lgb_auc = lgb_results_filtered.get("metrics", {}).get("auc", 0) if lgb_results_filtered else 0
    cb_auc = cb_results_filtered.get("metrics", {}).get("auc", 0) if cb_results_filtered else 0

    if cb_auc >= lgb_auc and cb_model_filtered:
        best_model = cb_model_filtered
        best_name = "catboost"
        best_results = cb_results_filtered
    elif lgb_model_filtered:
        best_model = lgb_model_filtered
        best_name = "lightgbm"
        best_results = lgb_results_filtered
    else:
        best_model = cb_model_filtered
        best_name = "catboost"
        best_results = cb_results_filtered

    if best_model:
        model_path = models_dir / f"model_full_77feat_{best_name}_{HORIZON_MINS}m.joblib"
        joblib.dump({
            "model": best_model,
            "features": selected_features,
            "metrics": best_results["metrics"],
            "importance": best_results["importance"],
            "all_features": feature_cols,
            "tickers": TICKERS,
            "horizon_mins": HORIZON_MINS,
            "trained_at": datetime.now().isoformat(),
        }, model_path)
        print(f"\n✓ Best model saved: {model_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total features:     77 → {len(selected_features)} (after noise removal)")
    print(f"Train samples:      {len(X_train):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Best model:         {best_name.upper()}")
    print(f"Best AUC:           {max(lgb_auc, cb_auc):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
