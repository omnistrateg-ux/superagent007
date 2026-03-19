"""
MOEX Agent v2 Mean Reversion Strategy

Labels and features for mean reversion to VWAP.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("moex_agent.mean_reversion")

# Mean reversion feature columns
MR_FEATURE_COLS = [
    "z_score",           # (price - VWAP) / rolling_std
    "rsi_14",            # RSI indicator
    "bb_position",       # Position within Bollinger Bands
    "volume_spike",      # volume / avg_volume_20
    "dist_vwap_pct",     # Distance from VWAP in %
]


def compute_vwap(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    Compute Volume Weighted Average Price.

    VWAP = cumsum(price * volume) / cumsum(volume)
    We use rolling VWAP over period bars.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]

    vwap = pv.rolling(period).sum() / df["volume"].rolling(period).sum()
    return vwap


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    return sma, upper, lower


def build_mr_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Build mean reversion features for each candle.

    Args:
        candles: DataFrame with [secid, ts, open, high, low, close, volume, value]

    Returns:
        DataFrame with [secid, ts, z_score, rsi_14, bb_position, volume_spike, dist_vwap_pct]
    """
    results = []

    for secid, g in candles.groupby("secid", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)

        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float)

        # VWAP and rolling std
        vwap = compute_vwap(g, period=100)
        rolling_std = close.rolling(100).std()

        # Z-score: how many std away from VWAP
        z_score = (close - vwap) / (rolling_std + 1e-10)

        # RSI
        rsi = compute_rsi(close, period=14)

        # Bollinger Bands position
        _, bb_upper, bb_lower = compute_bollinger_bands(close, period=20, num_std=2.0)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Volume spike
        avg_volume = volume.rolling(20).mean()
        volume_spike = volume / (avg_volume + 1e-10)

        # Distance from VWAP in %
        dist_vwap_pct = (close - vwap) / (vwap + 1e-10) * 100

        df = pd.DataFrame({
            "secid": secid,
            "ts": g["ts"],
            "z_score": z_score,
            "rsi_14": rsi,
            "bb_position": bb_position,
            "volume_spike": volume_spike,
            "dist_vwap_pct": dist_vwap_pct,
            "vwap": vwap,
            "close": close,
        })

        results.append(df)

    return pd.concat(results, ignore_index=True)


def make_mr_labels(
    candles: pd.DataFrame,
    z_threshold: float = 1.5,
    stop_pct: float = 1.0,
    max_bars: int = 60,
    fee_bps: float = 8.0,
) -> pd.DataFrame:
    """
    Create mean reversion labels.

    LONG signal (1):
        - Entry when z_score < -z_threshold (price below VWAP - 1.5σ)
        - Win if price returns to VWAP before hitting stop (-1%)

    SHORT signal (-1):
        - Entry when z_score > z_threshold (price above VWAP + 1.5σ)
        - Win if price returns to VWAP before hitting stop (+1%)

    No signal (0): z_score between -z_threshold and +z_threshold

    Args:
        candles: DataFrame with candle data
        z_threshold: Minimum |z_score| to trigger signal
        stop_pct: Stop loss percentage
        max_bars: Maximum bars to wait for mean reversion
        fee_bps: Round-trip fee in basis points

    Returns:
        DataFrame with [secid, ts, y_mr, signal_type]
        y_mr: 1 = win, 0 = loss/no signal
        signal_type: 1 = LONG, -1 = SHORT, 0 = no signal
    """
    # First build features to get z_score and VWAP
    features = build_mr_features(candles)

    fee = fee_bps / 10000.0

    all_labels = []

    for secid, g in features.groupby("secid", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)

        close = g["close"].values
        vwap = g["vwap"].values
        z_score = g["z_score"].values
        n = len(g)

        labels = np.zeros(n, dtype=int)
        signal_types = np.zeros(n, dtype=int)

        for i in range(n - max_bars):
            z = z_score[i]

            if pd.isna(z) or pd.isna(vwap[i]):
                continue

            # Check for LONG signal (price below VWAP - 1.5σ)
            if z < -z_threshold:
                signal_types[i] = 1  # LONG
                entry = close[i]
                target = vwap[i]  # Return to VWAP
                stop = entry * (1 - stop_pct/100 - fee)

                # Check if price returns to VWAP before stop
                for j in range(i + 1, min(i + max_bars + 1, n)):
                    high_j = close[j]  # Using close as proxy for high
                    low_j = close[j]   # Using close as proxy for low

                    # For LONG: check if we hit VWAP (target)
                    if close[j] >= target:
                        labels[i] = 1  # Win
                        break
                    # Check stop
                    if close[j] <= stop:
                        labels[i] = 0  # Loss
                        break

            # Check for SHORT signal (price above VWAP + 1.5σ)
            elif z > z_threshold:
                signal_types[i] = -1  # SHORT
                entry = close[i]
                target = vwap[i]  # Return to VWAP
                stop = entry * (1 + stop_pct/100 + fee)

                # Check if price returns to VWAP before stop
                for j in range(i + 1, min(i + max_bars + 1, n)):
                    # For SHORT: check if we hit VWAP (target)
                    if close[j] <= target:
                        labels[i] = 1  # Win
                        break
                    # Check stop
                    if close[j] >= stop:
                        labels[i] = 0  # Loss
                        break

        result = pd.DataFrame({
            "secid": secid,
            "ts": g["ts"],
            "y_mr": labels,
            "signal_type": signal_types,
        })

        all_labels.append(result)

    return pd.concat(all_labels, ignore_index=True)


def train_mr_model(
    candles: pd.DataFrame,
    train_days: int = 120,
) -> Tuple[object, dict]:
    """
    Train mean reversion model using LogisticRegression L1.

    Args:
        candles: Full candle data
        train_days: Number of days for training

    Returns:
        (model_package, metrics)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    import joblib
    from pathlib import Path

    candles = candles.copy()
    candles["ts"] = pd.to_datetime(candles["ts"], utc=True)

    # Calculate cutoff
    min_date = candles["ts"].min()
    cutoff = min_date + pd.Timedelta(days=train_days)

    train_candles = candles[candles["ts"] < cutoff].copy()
    logger.info(f"Training on {len(train_candles):,} candles (first {train_days} days)")

    # Build features
    logger.info("Building mean reversion features...")
    features = build_mr_features(train_candles)

    # Build labels
    logger.info("Creating mean reversion labels...")
    labels = make_mr_labels(train_candles, z_threshold=1.5, stop_pct=1.0, max_bars=60)

    # Merge
    df = features.merge(labels[["secid", "ts", "y_mr", "signal_type"]], on=["secid", "ts"])

    # Filter to only rows with signals
    df_signals = df[df["signal_type"] != 0].copy()
    df_signals = df_signals.dropna(subset=MR_FEATURE_COLS)

    logger.info(f"Signal rows: {len(df_signals):,}")
    logger.info(f"LONG signals: {(df_signals['signal_type'] == 1).sum():,}")
    logger.info(f"SHORT signals: {(df_signals['signal_type'] == -1).sum():,}")
    logger.info(f"Wins: {df_signals['y_mr'].sum():,} ({df_signals['y_mr'].mean()*100:.1f}%)")

    if len(df_signals) < 100:
        logger.error("Not enough signals for training!")
        return None, {}

    # Prepare arrays
    X = df_signals[MR_FEATURE_COLS].to_numpy(dtype=float)
    y = df_signals["y_mr"].to_numpy(dtype=int)

    # Walk-forward validation
    logger.info("Walk-forward validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    wf_results = []

    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.1,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()

        # Calculate win rate on predicted "wins"
        pred_wins = y_pred == 1
        if pred_wins.sum() > 0:
            actual_wins = y_test[pred_wins].mean()
            wf_results.append(actual_wins * 100)

    if wf_results:
        wf_wr = np.mean(wf_results)
        logger.info(f"WF Win Rate: {wf_wr:.1f}%")
    else:
        wf_wr = 0

    # Final model training
    logger.info("Training final model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.1,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Check coefficients
    n_nonzero = np.sum(model.coef_ != 0)
    logger.info(f"Non-zero coefficients: {n_nonzero}/{len(MR_FEATURE_COLS)}")

    # Feature importance
    for name, coef in zip(MR_FEATURE_COLS, model.coef_[0]):
        logger.info(f"  {name}: {coef:.4f}")

    # Predictions stats
    y_proba = model.predict_proba(X_scaled)[:, 1]
    logger.info(f"Max probability: {y_proba.max():.2f}")
    logger.info(f"Mean probability: {y_proba.mean():.2f}")

    # Package
    model_package = {
        "model": model,
        "scaler": scaler,
        "feature_names": MR_FEATURE_COLS,
    }

    metrics = {
        "wf_win_rate": wf_wr,
        "n_signals": len(df_signals),
        "base_win_rate": df_signals["y_mr"].mean() * 100,
        "n_nonzero_coef": int(n_nonzero),
        "max_probability": float(y_proba.max()),
    }

    # Save model
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model_mr.joblib"
    joblib.dump(model_package, model_path)
    logger.info(f"Model saved: {model_path}")

    return model_package, metrics


def backtest_mr(
    candles: pd.DataFrame,
    model_package: dict,
    train_days: int = 120,
    z_threshold: float = 5.0,  # Very extreme deviation (5σ event)
    stop_pct: float = 0.2,     # Very tight stop
    max_bars: int = 1440,      # Full day - wait for mean reversion
    p_threshold: float = 0.5,
    fee_bps: float = 8.0,
    overshoot_pct: float = 0.3,
) -> dict:
    """
    Backtest mean reversion strategy on OOS data.

    Args:
        candles: Full candle data
        model_package: Trained model package
        train_days: Days used for training (for cutoff)
        z_threshold: Z-score threshold for signals
        stop_pct: Stop loss percentage
        max_bars: Maximum bars to hold
        p_threshold: Probability threshold for taking trades
        fee_bps: Round-trip fee in basis points

    Returns:
        Dict with backtest results
    """
    model = model_package["model"]
    scaler = model_package["scaler"]

    candles = candles.copy()
    candles["ts"] = pd.to_datetime(candles["ts"], utc=True)

    # Filter to test period
    min_date = candles["ts"].min()
    cutoff = min_date + pd.Timedelta(days=train_days)

    test_candles = candles[candles["ts"] >= cutoff].copy()
    logger.info(f"Test period: {cutoff.date()} onwards ({len(test_candles):,} candles)")

    # Build features
    logger.info("Building test features...")
    features = build_mr_features(test_candles)
    features = features.dropna(subset=MR_FEATURE_COLS)

    fee = fee_bps / 10000.0
    trades = []

    # Process each ticker
    unique_tickers = test_candles["secid"].unique()
    logger.info(f"Processing {len(unique_tickers)} tickers...")

    for ticker_idx, secid in enumerate(unique_tickers, 1):
        ticker_candles = test_candles[test_candles["secid"] == secid].sort_values("ts").reset_index(drop=True)
        ticker_features = features[features["secid"] == secid].sort_values("ts").reset_index(drop=True)

        if len(ticker_candles) < max_bars + 100:
            continue

        # Cooldown
        last_trade_idx = -max_bars

        for i in range(100, len(ticker_features) - max_bars, 5):  # Step by 5
            # Cooldown check
            if i - last_trade_idx < max_bars:
                continue

            row = ticker_features.iloc[i]
            dist_pct = row["dist_vwap_pct"]  # Actual % distance from VWAP

            if pd.isna(dist_pct):
                continue

            # Check for signal - use actual % distance from VWAP
            # Signal when price is >0.8% away from VWAP
            dist_threshold = 1.2  # 1.2% away from VWAP

            if dist_pct < -dist_threshold:
                signal_type = 1  # LONG - price below VWAP by 1%+
            elif dist_pct > dist_threshold:
                signal_type = -1  # SHORT - price above VWAP by 1%+
            else:
                continue

            # Get model prediction
            X = row[MR_FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0, 1]

            if prob < p_threshold:
                continue

            # Simulate trade
            entry_price = row["close"]
            vwap = row["vwap"]

            if signal_type == 1:  # LONG
                # Target: VWAP (pure mean reversion)
                target = vwap
                stop = entry_price * (1 - stop_pct/100)
            else:  # SHORT
                # Target: VWAP (pure mean reversion)
                target = vwap
                stop = entry_price * (1 + stop_pct/100)

            # Find exit
            exit_price = entry_price
            exit_reason = "ttl"
            ts_entry = row["ts"]

            # Get candle data for simulation
            candle_idx = ticker_candles[ticker_candles["ts"] == ts_entry].index
            if len(candle_idx) == 0:
                continue
            candle_start = candle_idx[0]

            for j in range(1, min(max_bars + 1, len(ticker_candles) - candle_start)):
                bar = ticker_candles.iloc[candle_start + j]
                high = bar["high"]
                low = bar["low"]

                if signal_type == 1:  # LONG
                    # Check stop first
                    if low <= stop:
                        exit_price = stop
                        exit_reason = "stop"
                        break
                    # Check target (VWAP)
                    if high >= target:
                        exit_price = target
                        exit_reason = "take"
                        break
                else:  # SHORT
                    # Check stop first
                    if high >= stop:
                        exit_price = stop
                        exit_reason = "stop"
                        break
                    # Check target (VWAP)
                    if low <= target:
                        exit_price = target
                        exit_reason = "take"
                        break

                exit_price = bar["close"]

            # Calculate PnL
            if signal_type == 1:  # LONG
                pnl_pct = (exit_price - entry_price) / entry_price - fee
            else:  # SHORT
                pnl_pct = (entry_price - exit_price) / entry_price - fee

            pnl = entry_price * pnl_pct
            is_win = pnl > 0

            trades.append({
                "timestamp": str(ts_entry),
                "secid": secid,
                "direction": "LONG" if signal_type == 1 else "SHORT",
                "z_score": dist_pct,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "vwap": vwap,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "probability": prob,
                "is_win": is_win,
                "exit_reason": exit_reason,
            })

            last_trade_idx = i

        if ticker_idx % 5 == 0:
            logger.info(f"[{ticker_idx}/{len(unique_tickers)}] {secid}, trades: {len(trades)}")

    # Calculate metrics
    if not trades:
        logger.warning("No trades!")
        return {"total_trades": 0}

    wins = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if not t["is_win"]]
    pnl_list = [t["pnl"] for t in trades]

    total_pnl = sum(pnl_list)
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1

    win_rate = len(wins) / len(trades) * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    sharpe = np.mean(pnl_list) / (np.std(pnl_list) + 1e-9) if len(pnl_list) > 1 else 0

    # Exit breakdown
    take_exits = [t for t in trades if t["exit_reason"] == "take"]
    stop_exits = [t for t in trades if t["exit_reason"] == "stop"]
    ttl_exits = [t for t in trades if t["exit_reason"] == "ttl"]

    # Direction breakdown
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    long_wins = [t for t in long_trades if t["is_win"]]
    short_wins = [t for t in short_trades if t["is_win"]]

    print("\n" + "=" * 60)
    print("MEAN REVERSION BACKTEST RESULTS")
    print("=" * 60)
    print(f"Test period:      {cutoff.date()} onwards")
    print(f"Tickers:          {len(unique_tickers)}")
    print(f"Z threshold:      {z_threshold}")
    print(f"P threshold:      {p_threshold}")
    print("-" * 60)
    print(f"Total Trades:     {len(trades)}")
    print(f"Wins/Losses:      {len(wins)} / {len(losses)}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total PnL:        {total_pnl:+,.2f}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Sharpe:           {sharpe:.2f}")
    print(f"Avg Win:          {gross_profit/len(wins) if wins else 0:+,.2f}")
    print(f"Avg Loss:         {gross_loss/len(losses) if losses else 0:,.2f}")

    # Also show in percentage terms
    avg_win_pct = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    avg_loss_pct = np.mean([abs(t["pnl_pct"]) for t in losses]) * 100 if losses else 0
    print(f"Avg Win %:        {avg_win_pct:+.3f}%")
    print(f"Avg Loss %:       {avg_loss_pct:.3f}%")
    print("-" * 60)
    print(f"Direction breakdown:")
    print(f"  LONG:  {len(long_trades)} trades, {len(long_wins)} wins ({len(long_wins)/len(long_trades)*100 if long_trades else 0:.1f}%)")
    print(f"  SHORT: {len(short_trades)} trades, {len(short_wins)} wins ({len(short_wins)/len(short_trades)*100 if short_trades else 0:.1f}%)")
    print("-" * 60)
    print(f"Exit breakdown:")
    print(f"  Take (VWAP):    {len(take_exits)} ({len(take_exits)/len(trades)*100:.1f}%)")
    print(f"  Stop loss:      {len(stop_exits)} ({len(stop_exits)/len(trades)*100:.1f}%)")
    print(f"  TTL (time out): {len(ttl_exits)} ({len(ttl_exits)/len(trades)*100:.1f}%)")
    print("=" * 60 + "\n")

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "long_trades": len(long_trades),
        "long_win_rate": len(long_wins) / len(long_trades) * 100 if long_trades else 0,
        "short_trades": len(short_trades),
        "short_win_rate": len(short_wins) / len(short_trades) * 100 if short_trades else 0,
        "trades": trades,
    }


def run_mr_pipeline(
    tickers: list = None,
    train_days: int = 120,
    p_threshold: float = 0.5,
    stop_pct: float = 0.5,
    overshoot_pct: float = 0.3,
):
    """
    Run full mean reversion pipeline: train + backtest.
    """
    import pandas as pd
    from .config import load_config
    from .storage import connect

    config = load_config()
    conn = connect(config.sqlite_path)

    # Build ticker filter
    if tickers:
        ticker_list = ",".join(f"'{t}'" for t in tickers)
        ticker_filter = f"AND secid IN ({ticker_list})"
    else:
        ticker_filter = ""

    # Load candles
    logger.info("Loading candles...")
    q = f"""
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1 {ticker_filter}
    ORDER BY secid, ts
    """
    candles = pd.read_sql_query(q, conn)
    conn.close()

    logger.info(f"Loaded {len(candles):,} candles")

    # Train
    logger.info("=" * 60)
    logger.info("TRAINING MEAN REVERSION MODEL")
    logger.info("=" * 60)
    model_package, metrics = train_mr_model(candles, train_days=train_days)

    if model_package is None:
        return

    # Backtest
    logger.info("=" * 60)
    logger.info("BACKTESTING ON OOS DATA")
    logger.info("=" * 60)
    results = backtest_mr(
        candles,
        model_package,
        train_days=train_days,
        p_threshold=p_threshold,
        stop_pct=stop_pct,
        overshoot_pct=overshoot_pct,
    )

    return results
