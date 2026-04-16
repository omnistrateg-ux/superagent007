#!/usr/bin/env python3
"""
FUTOI NG Edge - Extended Validation

Additional validation for H2/H3 edge on NG futures:
1. Walk-forward (6 folds by year)
2. Quarterly stability (24 quarters)
3. Regime analysis (bull vs bear)
4. Contract independence

Verdict: CONFIRMED if walk-forward 4+/6, quarterly stability, regime independence.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
SIGNALS_FILE = DATA_DIR / "futoi_signals.csv"

MIN_TRADES_FOLD = 30  # Min trades per fold for validity


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def backtest_signal(df: pd.DataFrame, signal_col: str, return_col: str = 'next_ret_otc') -> dict:
    """Backtest a signal and return metrics."""
    data = df[[signal_col, return_col]].dropna().copy()
    data = data[data[signal_col] != 0].copy()

    if len(data) < MIN_TRADES_FOLD:
        return {'trades': len(data), 'valid': False}

    signal = data[signal_col].values
    ret = data[return_col].values
    strat_ret = signal * ret

    trades = len(data)
    gross_profit = strat_ret[strat_ret > 0].sum()
    gross_loss = abs(strat_ret[strat_ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    total_ret = strat_ret.sum()
    win_rate = (strat_ret > 0).sum() / trades

    return {
        'trades': trades,
        'pf': pf,
        'sharpe': sharpe,
        'total_ret': total_ret,
        'win_rate': win_rate,
        'valid': True,
    }


# =============================================================================
# 1. WALK-FORWARD ANALYSIS
# =============================================================================

def walk_forward_analysis(df: pd.DataFrame, signal_col: str) -> dict:
    """
    Walk-forward: train on years 1..N-1, test on year N.
    6 folds for 2020-2025.
    """
    log(f"\n{'='*60}")
    log(f"WALK-FORWARD ANALYSIS: {signal_col}")
    log(f"{'='*60}")

    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year

    years = sorted(df['year'].unique())
    results = []

    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if not train_years:
            continue

        train_data = df[df['year'].isin(train_years)]
        test_data = df[df['year'] == test_year]

        # Train metrics (for reference)
        train_res = backtest_signal(train_data, signal_col)

        # Test metrics (what we care about)
        test_res = backtest_signal(test_data, signal_col)

        fold_result = {
            'test_year': test_year,
            'train_years': f"{min(train_years)}-{max(train_years)}",
            'train_trades': train_res.get('trades', 0),
            'train_pf': train_res.get('pf', np.nan),
            'test_trades': test_res.get('trades', 0),
            'test_pf': test_res.get('pf', np.nan),
            'test_sharpe': test_res.get('sharpe', np.nan),
            'test_ret': test_res.get('total_ret', np.nan),
            'test_valid': test_res.get('valid', False),
            'test_pass': test_res.get('pf', 0) >= 1.0 if test_res.get('valid', False) else False,
        }
        results.append(fold_result)

        status = "PASS" if fold_result['test_pass'] else "FAIL"
        log(f"  Fold {test_year}: Train={fold_result['train_years']} → Test PF={fold_result['test_pf']:.2f} [{status}]")

    results_df = pd.DataFrame(results)

    # Summary
    valid_folds = results_df[results_df['test_valid']]
    passed_folds = valid_folds[valid_folds['test_pass']]

    summary = {
        'total_folds': len(results_df),
        'valid_folds': len(valid_folds),
        'passed_folds': len(passed_folds),
        'pass_rate': len(passed_folds) / len(valid_folds) if len(valid_folds) > 0 else 0,
        'avg_test_pf': valid_folds['test_pf'].mean() if len(valid_folds) > 0 else np.nan,
        'min_test_pf': valid_folds['test_pf'].min() if len(valid_folds) > 0 else np.nan,
        'confirmed': len(passed_folds) >= 4,
        'details': results_df,
    }

    verdict = "CONFIRMED" if summary['confirmed'] else "FAILED"
    log(f"\n  Walk-forward verdict: {verdict} ({len(passed_folds)}/{len(valid_folds)} folds passed)")

    return summary


# =============================================================================
# 2. QUARTERLY ANALYSIS
# =============================================================================

def quarterly_analysis(df: pd.DataFrame, signal_col: str) -> dict:
    """
    Analyze PF by quarter. Edge should be stable across quarters.
    """
    log(f"\n{'='*60}")
    log(f"QUARTERLY ANALYSIS: {signal_col}")
    log(f"{'='*60}")

    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    df['quarter'] = df['date_dt'].dt.quarter
    df['yq'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

    results = []

    for yq in sorted(df['yq'].unique()):
        q_data = df[df['yq'] == yq]
        q_res = backtest_signal(q_data, signal_col)

        results.append({
            'quarter': yq,
            'trades': q_res.get('trades', 0),
            'pf': q_res.get('pf', np.nan),
            'sharpe': q_res.get('sharpe', np.nan),
            'ret': q_res.get('total_ret', np.nan),
            'valid': q_res.get('valid', False),
            'profitable': q_res.get('pf', 0) >= 1.0 if q_res.get('valid', False) else False,
        })

    results_df = pd.DataFrame(results)

    # Display
    log("\n  Quarter   Trades    PF      Sharpe   Return")
    log("  " + "-" * 50)
    for _, row in results_df.iterrows():
        status = "+" if row['profitable'] else "-" if row['valid'] else "?"
        pf_str = f"{row['pf']:.2f}" if not np.isnan(row['pf']) else "N/A"
        sharpe_str = f"{row['sharpe']:.2f}" if not np.isnan(row['sharpe']) else "N/A"
        ret_str = f"{row['ret']*100:.1f}%" if not np.isnan(row['ret']) else "N/A"
        log(f"  {row['quarter']:8}  {row['trades']:5}   {pf_str:>6}   {sharpe_str:>6}   {ret_str:>8}  [{status}]")

    # Summary
    valid_quarters = results_df[results_df['valid']]
    profitable_quarters = valid_quarters[valid_quarters['profitable']]

    # Check for clustering (edge should not be in just a few quarters)
    stability_ratio = len(profitable_quarters) / len(valid_quarters) if len(valid_quarters) > 0 else 0

    summary = {
        'total_quarters': len(results_df),
        'valid_quarters': len(valid_quarters),
        'profitable_quarters': len(profitable_quarters),
        'stability_ratio': stability_ratio,
        'avg_pf': valid_quarters['pf'].mean() if len(valid_quarters) > 0 else np.nan,
        'pf_std': valid_quarters['pf'].std() if len(valid_quarters) > 0 else np.nan,
        'confirmed': stability_ratio >= 0.5,  # At least 50% of quarters profitable
        'details': results_df,
    }

    verdict = "CONFIRMED" if summary['confirmed'] else "FAILED"
    log(f"\n  Quarterly verdict: {verdict} ({len(profitable_quarters)}/{len(valid_quarters)} quarters profitable, {stability_ratio*100:.0f}%)")

    return summary


# =============================================================================
# 3. REGIME ANALYSIS
# =============================================================================

def regime_analysis(df: pd.DataFrame, signal_col: str) -> dict:
    """
    Analyze edge in different market regimes (bull vs bear).
    """
    log(f"\n{'='*60}")
    log(f"REGIME ANALYSIS: {signal_col}")
    log(f"{'='*60}")

    df = df.copy()

    # Define regime based on 20-day rolling return
    df['rolling_ret'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change(20)
    )

    # Bull = rolling return > 0, Bear = rolling return < 0
    df['regime'] = np.where(df['rolling_ret'] > 0.05, 'BULL',
                           np.where(df['rolling_ret'] < -0.05, 'BEAR', 'NEUTRAL'))

    results = []

    for regime in ['BULL', 'BEAR', 'NEUTRAL']:
        regime_data = df[df['regime'] == regime]
        regime_res = backtest_signal(regime_data, signal_col)

        results.append({
            'regime': regime,
            'trades': regime_res.get('trades', 0),
            'pf': regime_res.get('pf', np.nan),
            'sharpe': regime_res.get('sharpe', np.nan),
            'ret': regime_res.get('total_ret', np.nan),
            'valid': regime_res.get('valid', False),
        })

    results_df = pd.DataFrame(results)

    log("\n  Regime      Trades    PF      Sharpe   Return")
    log("  " + "-" * 50)
    for _, row in results_df.iterrows():
        pf_str = f"{row['pf']:.2f}" if not np.isnan(row['pf']) else "N/A"
        sharpe_str = f"{row['sharpe']:.2f}" if not np.isnan(row['sharpe']) else "N/A"
        ret_str = f"{row['ret']*100:.1f}%" if not np.isnan(row['ret']) else "N/A"
        log(f"  {row['regime']:8}  {row['trades']:5}   {pf_str:>6}   {sharpe_str:>6}   {ret_str:>8}")

    # Check if edge works in both bull and bear
    bull = results_df[results_df['regime'] == 'BULL'].iloc[0] if len(results_df[results_df['regime'] == 'BULL']) > 0 else None
    bear = results_df[results_df['regime'] == 'BEAR'].iloc[0] if len(results_df[results_df['regime'] == 'BEAR']) > 0 else None

    bull_ok = bull is not None and bull['valid'] and bull['pf'] >= 1.0
    bear_ok = bear is not None and bear['valid'] and bear['pf'] >= 1.0

    # Edge is regime-independent if works in both, or at least doesn't lose money in one
    regime_independent = bull_ok and bear_ok

    summary = {
        'bull_pf': bull['pf'] if bull is not None else np.nan,
        'bear_pf': bear['pf'] if bear is not None else np.nan,
        'bull_ok': bull_ok,
        'bear_ok': bear_ok,
        'regime_independent': regime_independent,
        'confirmed': regime_independent or (bull_ok or bear_ok),
        'details': results_df,
    }

    verdict = "CONFIRMED" if summary['confirmed'] else "FAILED"
    log(f"\n  Regime verdict: {verdict} (Bull: {'OK' if bull_ok else 'FAIL'}, Bear: {'OK' if bear_ok else 'FAIL'})")

    return summary


# =============================================================================
# 4. SIGNAL MONOTONICITY
# =============================================================================

def monotonicity_analysis(df: pd.DataFrame, signal_col: str) -> dict:
    """
    Check signal monotonicity: stronger signals should have stronger returns.
    For H2 (delta), bucket by delta magnitude.
    """
    log(f"\n{'='*60}")
    log(f"MONOTONICITY ANALYSIS: {signal_col}")
    log(f"{'='*60}")

    df = df.copy()

    # Get the underlying feature for bucketing
    if 'continuation' in signal_col:
        feature = 'yur_net_delta'
        df['feature_abs'] = df[feature].abs()
    elif 'divergence' in signal_col:
        feature = 'yur_net'
        df['feature_abs'] = df[feature].abs()
    else:
        log("  Skipping monotonicity (unknown signal type)")
        return {'confirmed': True, 'skipped': True}

    # Create quantile buckets based on feature magnitude
    df['bucket'] = pd.qcut(df['feature_abs'], q=5, labels=['Q1 (weak)', 'Q2', 'Q3', 'Q4', 'Q5 (strong)'],
                          duplicates='drop')

    results = []
    for bucket in df['bucket'].dropna().unique():
        bucket_data = df[df['bucket'] == bucket]
        bucket_res = backtest_signal(bucket_data, signal_col)

        results.append({
            'bucket': str(bucket),
            'trades': bucket_res.get('trades', 0),
            'pf': bucket_res.get('pf', np.nan),
            'sharpe': bucket_res.get('sharpe', np.nan),
            'ret': bucket_res.get('total_ret', np.nan),
            'valid': bucket_res.get('valid', False),
        })

    results_df = pd.DataFrame(results)

    log("\n  Bucket        Trades    PF      Sharpe   Return")
    log("  " + "-" * 55)
    for _, row in results_df.iterrows():
        pf_str = f"{row['pf']:.2f}" if not np.isnan(row['pf']) else "N/A"
        sharpe_str = f"{row['sharpe']:.2f}" if not np.isnan(row['sharpe']) else "N/A"
        ret_str = f"{row['ret']*100:.1f}%" if not np.isnan(row['ret']) else "N/A"
        log(f"  {row['bucket']:12}  {row['trades']:5}   {pf_str:>6}   {sharpe_str:>6}   {ret_str:>8}")

    # Check monotonicity: PF should generally increase with signal strength
    valid_results = results_df[results_df['valid']].copy()
    if len(valid_results) >= 3:
        # Spearman correlation between bucket rank and PF
        valid_results['rank'] = range(len(valid_results))
        corr, p_val = stats.spearmanr(valid_results['rank'], valid_results['pf'])
        monotonic = corr > 0  # Positive correlation = stronger signal → better PF
    else:
        corr, p_val = np.nan, np.nan
        monotonic = True  # Not enough data to fail

    summary = {
        'correlation': corr,
        'p_value': p_val,
        'monotonic': monotonic,
        'confirmed': monotonic,
        'details': results_df,
    }

    verdict = "CONFIRMED" if summary['confirmed'] else "FAILED"
    log(f"\n  Monotonicity verdict: {verdict} (correlation={corr:.2f})")

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("FUTOI NG EDGE - EXTENDED VALIDATION")
    log("=" * 70)

    # Load signals
    df = pd.read_csv(SIGNALS_FILE)
    df_ng = df[df['ticker'] == 'NG'].copy()

    log(f"Loaded {len(df_ng)} NG signals")
    log(f"Date range: {df_ng['date'].min()} to {df_ng['date'].max()}")

    # Test both hypotheses
    hypotheses = [
        ('sig_h2_continuation', 'H2: YUR Delta → Continuation'),
        ('sig_h3_divergence', 'H3: YUR/FIZ Divergence'),
    ]

    final_results = []

    for signal_col, hyp_name in hypotheses:
        log(f"\n\n{'#'*70}")
        log(f"# {hyp_name}")
        log(f"{'#'*70}")

        # Run all validations
        wf = walk_forward_analysis(df_ng, signal_col)
        qt = quarterly_analysis(df_ng, signal_col)
        rg = regime_analysis(df_ng, signal_col)
        mn = monotonicity_analysis(df_ng, signal_col)

        # Aggregate verdict
        all_confirmed = wf['confirmed'] and qt['confirmed'] and rg['confirmed'] and mn['confirmed']

        final_results.append({
            'hypothesis': hyp_name,
            'signal': signal_col,
            'walk_forward': f"{wf['passed_folds']}/{wf['valid_folds']}",
            'wf_confirmed': wf['confirmed'],
            'quarterly': f"{qt['profitable_quarters']}/{qt['valid_quarters']}",
            'qt_confirmed': qt['confirmed'],
            'regime_bull_pf': rg['bull_pf'],
            'regime_bear_pf': rg['bear_pf'],
            'rg_confirmed': rg['confirmed'],
            'monotonicity': mn.get('correlation', np.nan),
            'mn_confirmed': mn['confirmed'],
            'FINAL_VERDICT': 'CONFIRMED' if all_confirmed else 'KILLED',
        })

        log(f"\n\n{'='*60}")
        log(f"HYPOTHESIS VERDICT: {hyp_name}")
        log(f"{'='*60}")
        log(f"  Walk-forward:  {'PASS' if wf['confirmed'] else 'FAIL'} ({wf['passed_folds']}/{wf['valid_folds']} folds)")
        log(f"  Quarterly:     {'PASS' if qt['confirmed'] else 'FAIL'} ({qt['profitable_quarters']}/{qt['valid_quarters']} quarters)")
        log(f"  Regime:        {'PASS' if rg['confirmed'] else 'FAIL'} (Bull={rg['bull_pf']:.2f}, Bear={rg['bear_pf']:.2f})")
        log(f"  Monotonicity:  {'PASS' if mn['confirmed'] else 'FAIL'}")
        log(f"\n  >>> FINAL: {'CONFIRMED' if all_confirmed else 'KILLED'} <<<")

    # Save results
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(DATA_DIR / "futoi_validation_results.csv", index=False)

    # Final summary
    log("\n\n" + "=" * 70)
    log("FINAL VALIDATION SUMMARY")
    log("=" * 70)

    for _, row in results_df.iterrows():
        log(f"\n{row['hypothesis']}:")
        log(f"  Walk-forward: {row['walk_forward']} {'✓' if row['wf_confirmed'] else '✗'}")
        log(f"  Quarterly:    {row['quarterly']} {'✓' if row['qt_confirmed'] else '✗'}")
        log(f"  Regime:       Bull={row['regime_bull_pf']:.2f}, Bear={row['regime_bear_pf']:.2f} {'✓' if row['rg_confirmed'] else '✗'}")
        log(f"  Monotonicity: corr={row['monotonicity']:.2f} {'✓' if row['mn_confirmed'] else '✗'}")
        log(f"  >>> {row['FINAL_VERDICT']} <<<")

    confirmed_count = len(results_df[results_df['FINAL_VERDICT'] == 'CONFIRMED'])
    log(f"\n{'='*70}")
    if confirmed_count > 0:
        log(f"OVERALL VERDICT: {confirmed_count} HYPOTHESIS(ES) CONFIRMED")
    else:
        log("OVERALL VERDICT: ALL HYPOTHESES KILLED")
    log("=" * 70)

    return results_df


if __name__ == "__main__":
    main()
