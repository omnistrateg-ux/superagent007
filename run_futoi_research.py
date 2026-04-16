#!/usr/bin/env python3
"""
FUTOI Positioning Research Pipeline

Uses FUTOI (Futures Open Interest by client group) as positioning dataset.
Tests whether YUR (institutional) vs FIZ (retail) positioning predicts next-day returns.

Hypotheses:
- H1_REVERSAL: Extreme YUR net position → next-day reversal
- H2_CONTINUATION: YUR position change (delta) → next-day continuation
- H3_DIVERGENCE: YUR/FIZ disagreement → direction signal

Data:
- FUTOI: 2020-2025, daily snapshots at 23:50 MSK
- Prices: Daily OHLCV from MOEX ISS (front-month contract)

Split:
- Discovery: 2020-01-01 to 2023-12-31
- Holdout: 2024-01-01 to 2025-12-30

Falsification:
- Reversed signal (should produce PF < 1.0)
- Shuffled labels (destroys temporal structure)
- With costs (5 bps round-trip)
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings('ignore')

# Config
DATA_DIR = Path("data")
FUTOI_FILE = DATA_DIR / "futoi_futures" / "futoi_all.parquet"
PRICES_CACHE = DATA_DIR / "prices_daily_cache.parquet"

TICKERS = ['BR', 'RI', 'MX', 'NG', 'Si']
DISCOVERY_END = '2023-12-31'
HOLDOUT_START = '2024-01-01'

COST_BPS = 5  # 5 basis points round-trip

# Kill thresholds
MIN_PF = 1.2  # Profit factor threshold
MAX_PVAL = 0.10  # Statistical significance threshold
MIN_TRADES = 50  # Minimum trades for validity


def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# PRICE DATA
# =============================================================================

def fetch_contract_history(secid: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch full history for a specific futures contract from MOEX ISS."""
    url = f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities/{secid}.json'
    all_data = []
    start = 0
    limit = 500

    while True:
        params = {
            'iss.meta': 'off',
            'iss.only': 'history',
            'start': start,
            'limit': limit,
            'from': start_date,
            'till': end_date,
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            data = r.json()
            if 'history' not in data or not data['history']['data']:
                break
            cols = data['history']['columns']
            rows = data['history']['data']
            all_data.extend(rows)
            if len(rows) < limit:
                break
            start += limit
        except Exception as e:
            break

    if all_data:
        return pd.DataFrame(all_data, columns=cols)
    return pd.DataFrame()


def get_all_contracts(asset_code: str, years: list[int]) -> list[str]:
    """Generate all possible contract codes for an asset."""
    months = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    contracts = []
    for year in years:
        year_digit = year % 10  # 2020 -> 0, 2021 -> 1, etc.
        for m in months:
            contracts.append(f'{asset_code}{m}{year_digit}')
    return contracts


def build_price_dataset(dates: list[str], tickers: list[str]) -> pd.DataFrame:
    """Build daily price dataset for specified tickers and dates using bulk fetch."""

    if PRICES_CACHE.exists():
        log(f"Loading cached prices from {PRICES_CACHE}")
        df = pd.read_parquet(PRICES_CACHE)
        cached_dates = set(df['date'].unique())
        needed_dates = set(dates)
        if needed_dates.issubset(cached_dates):
            return df[df['ticker'].isin(tickers)]
        log(f"  Cache incomplete, rebuilding...")

    # Bulk fetch all contract histories
    start_date = min(dates)
    end_date = max(dates)
    years = list(range(2020, 2026))

    all_records = []

    for ticker in tickers:
        log(f"  Fetching {ticker} contracts...")
        contracts = get_all_contracts(ticker, years)

        contract_data = []
        for i, contract in enumerate(contracts):
            df_contract = fetch_contract_history(contract, start_date, end_date)
            if not df_contract.empty:
                df_contract['ticker'] = ticker
                df_contract['contract'] = contract
                contract_data.append(df_contract)

            if (i + 1) % 20 == 0:
                log(f"    {i+1}/{len(contracts)} contracts checked")

        if contract_data:
            df_ticker = pd.concat(contract_data, ignore_index=True)
            df_ticker = df_ticker.rename(columns={
                'TRADEDATE': 'date',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'VOLUME': 'volume',
                'OPENPOSITION': 'oi',
            })

            # For each date, pick most liquid contract
            for date_str in df_ticker['date'].unique():
                day_data = df_ticker[df_ticker['date'] == date_str]
                valid = day_data[day_data['volume'].notna() & (day_data['volume'] > 0)]
                if valid.empty:
                    continue
                best = valid.loc[valid['volume'].idxmax()]
                all_records.append({
                    'date': best['date'],
                    'ticker': ticker,
                    'open': best['open'],
                    'high': best['high'],
                    'low': best['low'],
                    'close': best['close'],
                    'volume': best['volume'],
                    'oi': best['oi'],
                    'contract': best['contract'],
                })

            log(f"    {ticker}: {len([r for r in all_records if r['ticker'] == ticker])} daily prices")

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_parquet(PRICES_CACHE, index=False)
        log(f"  Cached {len(df)} price records")
        return df[df['ticker'].isin(tickers)]

    return pd.DataFrame()


# =============================================================================
# FUTOI FEATURES
# =============================================================================

def load_futoi() -> pd.DataFrame:
    """Load FUTOI data and clean it."""
    log(f"Loading FUTOI from {FUTOI_FILE}")
    df = pd.read_parquet(FUTOI_FILE)

    # Keep only relevant tickers
    df = df[df['ticker'].isin(TICKERS)].copy()

    # Rename columns for clarity
    df = df.rename(columns={'tradedate': 'date'})

    log(f"  Loaded {len(df):,} FUTOI records for {df['ticker'].nunique()} tickers")
    log(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    log(f"  Client groups: {df['clgroup'].unique().tolist()}")

    return df


def build_positioning_features(futoi: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily positioning features from FUTOI data.

    Features per ticker per day:
    - yur_net: YUR net position (pos column)
    - fiz_net: FIZ net position
    - yur_long_ratio: YUR long / (long + short)
    - fiz_long_ratio: FIZ long / (long + short)
    - yur_net_pct: YUR net position as % of total OI
    - yur_delta: YUR net change from prior day
    - yur_fiz_divergence: sign(YUR net) != sign(FIZ net)
    - yur_concentration: YUR traders / total traders
    """
    log("Building positioning features...")

    # Pivot to get YUR and FIZ side by side
    records = []

    for (date, ticker), group in futoi.groupby(['date', 'ticker']):
        yur = group[group['clgroup'] == 'YUR']
        fiz = group[group['clgroup'] == 'FIZ']

        if yur.empty or fiz.empty:
            continue

        yur = yur.iloc[0]
        fiz = fiz.iloc[0]

        # Net positions
        yur_net = yur['pos']
        fiz_net = fiz['pos']

        # Long ratios
        yur_long_total = abs(yur['pos_long']) + abs(yur['pos_short'])
        fiz_long_total = abs(fiz['pos_long']) + abs(fiz['pos_short'])

        yur_long_ratio = abs(yur['pos_long']) / yur_long_total if yur_long_total > 0 else 0.5
        fiz_long_ratio = abs(fiz['pos_long']) / fiz_long_total if fiz_long_total > 0 else 0.5

        # Total OI (approximation)
        total_oi = abs(yur_net) + abs(fiz_net)
        yur_net_pct = yur_net / total_oi if total_oi > 0 else 0

        # Trader concentration
        yur_traders = yur['pos_long_num'] + yur['pos_short_num']
        fiz_traders = fiz['pos_long_num'] + fiz['pos_short_num']
        total_traders = yur_traders + fiz_traders
        yur_concentration = yur_traders / total_traders if total_traders > 0 else 0.5

        # Divergence (YUR and FIZ disagree on direction)
        divergence = 1 if (yur_net > 0) != (fiz_net > 0) else 0

        records.append({
            'date': date,
            'ticker': ticker,
            'yur_net': yur_net,
            'fiz_net': fiz_net,
            'yur_long_ratio': yur_long_ratio,
            'fiz_long_ratio': fiz_long_ratio,
            'yur_net_pct': yur_net_pct,
            'yur_concentration': yur_concentration,
            'yur_fiz_divergence': divergence,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Add delta features (change from prior day)
    df['yur_net_delta'] = df.groupby('ticker')['yur_net'].diff()
    df['yur_long_ratio_delta'] = df.groupby('ticker')['yur_long_ratio'].diff()

    # Add z-scores for extreme detection
    for col in ['yur_net', 'yur_net_pct', 'yur_long_ratio']:
        df[f'{col}_zscore'] = df.groupby('ticker')[col].transform(
            lambda x: (x - x.rolling(60, min_periods=20).mean()) /
                      x.rolling(60, min_periods=20).std()
        )

    log(f"  Built {len(df):,} positioning records with {len(df.columns)} features")

    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Join positioning features with next-day returns and generate signals.
    """
    log("Generating signals...")

    # Compute next-day returns from prices
    prices = prices.sort_values(['ticker', 'date']).copy()
    prices['next_ret'] = prices.groupby('ticker')['close'].pct_change().shift(-1)
    prices['next_open'] = prices.groupby('ticker')['open'].shift(-1)
    prices['next_close'] = prices.groupby('ticker')['close'].shift(-1)

    # Open-to-close return for more realistic execution
    prices['next_ret_otc'] = (prices['next_close'] - prices['next_open']) / prices['next_open']

    # Merge
    df = features.merge(
        prices[['date', 'ticker', 'close', 'next_ret', 'next_ret_otc']],
        on=['date', 'ticker'],
        how='inner'
    )

    # Signal: H1_REVERSAL - Extreme YUR positioning (z-score > 2) predicts reversal
    df['sig_h1_reversal'] = np.where(
        df['yur_net_pct_zscore'] > 2, -1,  # YUR very long → expect down
        np.where(df['yur_net_pct_zscore'] < -2, 1, 0)  # YUR very short → expect up
    )

    # Signal: H2_CONTINUATION - YUR flow (delta) predicts continuation
    df['sig_h2_continuation'] = np.where(
        df['yur_net_delta'] > 0, 1,  # YUR adding longs → expect up
        np.where(df['yur_net_delta'] < 0, -1, 0)  # YUR adding shorts → expect down
    )

    # Signal: H3_DIVERGENCE - When YUR/FIZ disagree, follow YUR
    df['sig_h3_divergence'] = np.where(
        (df['yur_fiz_divergence'] == 1) & (df['yur_net'] > 0), 1,
        np.where((df['yur_fiz_divergence'] == 1) & (df['yur_net'] < 0), -1, 0)
    )

    # Signal: H4_EXTREME_LONG_RATIO - Extreme YUR long ratio
    df['sig_h4_long_ratio'] = np.where(
        df['yur_long_ratio_zscore'] > 2, -1,  # Crowded long → reversal
        np.where(df['yur_long_ratio_zscore'] < -2, 1, 0)
    )

    log(f"  Generated signals for {len(df):,} date-ticker pairs")

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def backtest_signal(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str = 'next_ret_otc',
    cost_bps: float = 0,
    reverse: bool = False,
    shuffle: bool = False,
) -> dict:
    """
    Backtest a signal column.

    Returns:
        dict with metrics: trades, win_rate, pf, sharpe, total_ret, p_val
    """
    data = df[[signal_col, return_col]].dropna().copy()

    if shuffle:
        # Destroy temporal structure by shuffling returns
        data[return_col] = np.random.permutation(data[return_col].values)

    # Filter to actual signals
    data = data[data[signal_col] != 0].copy()

    if len(data) < MIN_TRADES:
        return {'trades': len(data), 'valid': False}

    # Apply signal direction
    signal = data[signal_col].values
    if reverse:
        signal = -signal

    # Compute strategy returns with costs
    cost = cost_bps / 10000
    strat_ret = signal * data[return_col].values - cost

    # Metrics
    trades = len(data)
    wins = (strat_ret > 0).sum()
    losses = (strat_ret < 0).sum()
    win_rate = wins / trades if trades > 0 else 0

    gross_profit = strat_ret[strat_ret > 0].sum()
    gross_loss = abs(strat_ret[strat_ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    total_ret = strat_ret.sum()

    # Statistical significance: t-test against zero
    t_stat, p_val = stats.ttest_1samp(strat_ret, 0)

    return {
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'pf': pf,
        'sharpe': sharpe,
        'total_ret': total_ret,
        't_stat': t_stat,
        'p_val': p_val / 2,  # One-tailed
        'valid': True,
    }


def evaluate_hypothesis(
    df: pd.DataFrame,
    signal_col: str,
    ticker: str,
    hypothesis_name: str,
) -> dict:
    """
    Full hypothesis evaluation with discovery/holdout split and falsification.
    """
    data = df[df['ticker'] == ticker].copy()

    # Split
    discovery = data[data['date'] <= DISCOVERY_END]
    holdout = data[data['date'] >= HOLDOUT_START]

    results = {
        'hypothesis': hypothesis_name,
        'ticker': ticker,
        'signal': signal_col,
    }

    # Discovery (in-sample)
    disc_res = backtest_signal(discovery, signal_col)
    results['disc_trades'] = disc_res['trades']
    results['disc_pf'] = disc_res.get('pf', np.nan)
    results['disc_sharpe'] = disc_res.get('sharpe', np.nan)
    results['disc_pval'] = disc_res.get('p_val', np.nan)
    results['disc_ret'] = disc_res.get('total_ret', np.nan)

    # Check if discovery passes thresholds
    if not disc_res.get('valid', False):
        results['verdict'] = 'INSUFFICIENT_DATA'
        return results

    if disc_res['pf'] < MIN_PF:
        results['verdict'] = f'KILLED_DISCOVERY (PF={disc_res["pf"]:.2f} < {MIN_PF})'
        return results

    if disc_res['p_val'] >= MAX_PVAL:
        results['verdict'] = f'KILLED_DISCOVERY (p={disc_res["p_val"]:.3f} >= {MAX_PVAL})'
        return results

    # Holdout (out-of-sample)
    hold_res = backtest_signal(holdout, signal_col)
    results['hold_trades'] = hold_res['trades']
    results['hold_pf'] = hold_res.get('pf', np.nan)
    results['hold_sharpe'] = hold_res.get('sharpe', np.nan)
    results['hold_pval'] = hold_res.get('p_val', np.nan)
    results['hold_ret'] = hold_res.get('total_ret', np.nan)

    if not hold_res.get('valid', False):
        results['verdict'] = 'INSUFFICIENT_HOLDOUT_DATA'
        return results

    if hold_res['pf'] < 1.0:
        results['verdict'] = f'KILLED_HOLDOUT (PF={hold_res["pf"]:.2f} < 1.0)'
        return results

    # Falsification: Reversed signal
    rev_res = backtest_signal(discovery, signal_col, reverse=True)
    results['rev_pf'] = rev_res.get('pf', np.nan)

    if rev_res.get('pf', 0) >= 1.0:
        results['verdict'] = f'KILLED_REVERSAL (rev_PF={rev_res["pf"]:.2f} >= 1.0)'
        return results

    # Falsification: Shuffled returns (run multiple times)
    shuffle_pfs = []
    for _ in range(100):
        shuf_res = backtest_signal(discovery, signal_col, shuffle=True)
        if shuf_res.get('valid', False):
            shuffle_pfs.append(shuf_res['pf'])

    if shuffle_pfs:
        shuffle_pf_mean = np.mean(shuffle_pfs)
        shuffle_pf_p95 = np.percentile(shuffle_pfs, 95)
        results['shuf_pf_mean'] = shuffle_pf_mean
        results['shuf_pf_p95'] = shuffle_pf_p95

        # Original should beat 95% of shuffled
        if disc_res['pf'] < shuffle_pf_p95:
            results['verdict'] = f'KILLED_SHUFFLE (PF={disc_res["pf"]:.2f} < shuffle_p95={shuffle_pf_p95:.2f})'
            return results

    # Falsification: With costs
    cost_res = backtest_signal(discovery, signal_col, cost_bps=COST_BPS)
    results['cost_pf'] = cost_res.get('pf', np.nan)
    results['cost_ret'] = cost_res.get('total_ret', np.nan)

    if cost_res.get('pf', 0) < 1.0:
        results['verdict'] = f'KILLED_COSTS (cost_PF={cost_res["pf"]:.2f} < 1.0)'
        return results

    # If all tests pass
    results['verdict'] = f'PASSED (disc_PF={disc_res["pf"]:.2f}, hold_PF={hold_res["pf"]:.2f})'

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("FUTOI POSITIONING RESEARCH PIPELINE")
    log("=" * 70)

    # Load FUTOI
    futoi = load_futoi()

    # Get unique dates for price fetching
    all_dates = sorted(futoi['date'].unique())
    log(f"Total unique dates in FUTOI: {len(all_dates)}")

    # Build/load price dataset
    prices = build_price_dataset(all_dates, TICKERS)
    if prices.empty:
        log("ERROR: No price data available")
        return

    log(f"Price data: {len(prices):,} records for {prices['ticker'].nunique()} tickers")

    # Build positioning features
    features = build_positioning_features(futoi)

    # Generate signals
    signals = generate_signals(features, prices)

    # Save intermediate data for inspection
    signals.to_csv(DATA_DIR / "futoi_signals.csv", index=False)
    log(f"Saved signals to {DATA_DIR / 'futoi_signals.csv'}")

    # Summary stats
    log("\n" + "=" * 70)
    log("SIGNAL SUMMARY")
    log("=" * 70)

    for sig in ['sig_h1_reversal', 'sig_h2_continuation', 'sig_h3_divergence', 'sig_h4_long_ratio']:
        counts = signals[sig].value_counts()
        log(f"{sig}: {dict(counts)}")

    # Run hypothesis tests
    log("\n" + "=" * 70)
    log("HYPOTHESIS TESTING")
    log("=" * 70)

    hypotheses = [
        ('sig_h1_reversal', 'H1: Extreme YUR position → reversal'),
        ('sig_h2_continuation', 'H2: YUR flow delta → continuation'),
        ('sig_h3_divergence', 'H3: YUR/FIZ divergence → follow YUR'),
        ('sig_h4_long_ratio', 'H4: Extreme YUR long ratio → reversal'),
    ]

    all_results = []

    for ticker in TICKERS:
        log(f"\n--- {ticker} ---")

        for sig_col, hyp_name in hypotheses:
            results = evaluate_hypothesis(signals, sig_col, ticker, hyp_name)
            all_results.append(results)

            verdict = results['verdict']
            disc_pf = results.get('disc_pf', np.nan)
            hold_pf = results.get('hold_pf', np.nan)

            status = "PASS" if "PASSED" in verdict else "FAIL"
            log(f"  {hyp_name}: {status}")
            log(f"    Discovery: PF={disc_pf:.2f}, trades={results.get('disc_trades', 0)}")
            if not np.isnan(hold_pf):
                log(f"    Holdout:   PF={hold_pf:.2f}, trades={results.get('hold_trades', 0)}")
            log(f"    Verdict: {verdict}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(DATA_DIR / "futoi_research_results.csv", index=False)
    log(f"\nSaved results to {DATA_DIR / 'futoi_research_results.csv'}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    passed = results_df[results_df['verdict'].str.contains('PASSED', na=False)]
    killed = results_df[~results_df['verdict'].str.contains('PASSED', na=False)]

    log(f"Total tests: {len(results_df)}")
    log(f"PASSED: {len(passed)}")
    log(f"KILLED: {len(killed)}")

    if len(passed) > 0:
        log("\nPASSED HYPOTHESES:")
        for _, row in passed.iterrows():
            log(f"  {row['ticker']} - {row['hypothesis']}: {row['verdict']}")

    # Final verdict
    log("\n" + "=" * 70)
    if len(passed) == 0:
        log("FINAL VERDICT: NO EDGE FOUND")
        log("All hypotheses killed during discovery, holdout, or falsification.")
    else:
        log("FINAL VERDICT: POTENTIAL EDGE DETECTED")
        log("WARNING: Requires further validation before live trading.")
    log("=" * 70)

    return results_df


if __name__ == "__main__":
    main()
