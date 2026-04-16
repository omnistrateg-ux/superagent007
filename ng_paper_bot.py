#!/usr/bin/env python3
"""
NG Paper Trading Bot

FROZEN SPEC - DO NOT MODIFY:
- Instrument: NG (Natural Gas futures)
- Signal: FUTOI YUR pos_long delta (increase → LONG, decrease → SHORT)
- Entry: 10:00 MSK
- Exit: 23:50 MSK
- Size: 1 contract
- Commission: 10 bps round-trip
- Data: FUTOI via moexalgo, Prices via MOEX ISS

Cron modes:
- 19:00 MSK: python ng_paper_bot.py --cron signal   # Generate tomorrow's signal
- 10:05 MSK: python ng_paper_bot.py --cron entry    # Record entry price
- 23:50 MSK: python ng_paper_bot.py --cron exit     # Record exit price, calc PnL
- Fri 23:55: python ng_paper_bot.py --cron report   # Weekly summary
"""

import argparse
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

# MOEX AlgoPack
try:
    from moexalgo import Market
    import moexalgo.session as moex_session
except ImportError:
    print("ERROR: moexalgo not installed. Run: pip install moexalgo")
    sys.exit(1)

# Config
MSK = ZoneInfo("Europe/Moscow")
DATA_DIR = Path("data")
JOURNAL_FILE = DATA_DIR / "paper_journal.csv"
SIGNAL_FILE = DATA_DIR / "pending_signal.txt"

TICKER = "NG"
COMMISSION_BPS = 10  # 10 bps round-trip

# ISS API
ISS_BASE = "https://iss.moex.com/iss"


def log(msg: str):
    """Print with timestamp."""
    ts = datetime.now(MSK).strftime("%Y-%m-%d %H:%M:%S MSK")
    print(f"[{ts}] {msg}")


def alert(msg: str):
    """Critical alert - printed prominently."""
    print()
    print("=" * 60)
    print(f"ALERT: {msg}")
    print("=" * 60)
    print()


# =============================================================================
# FUTOI DATA
# =============================================================================

def authenticate_algopack():
    """Authenticate with MOEX AlgoPack."""
    api_key = os.environ.get("MOEX_ALGOPACK_KEY", "")
    if not api_key:
        raise ValueError("MOEX_ALGOPACK_KEY environment variable not set")
    moex_session.TOKEN = api_key
    log("Authenticated with MOEX AlgoPack")


def fetch_futoi_snapshot(target_date: date) -> Optional[dict]:
    """
    Fetch FUTOI snapshot for NG on given date.
    Returns dict with YUR and FIZ data or None if unavailable.
    """
    market = Market('FORTS', 'FUT')

    try:
        df = market.futoi(date=target_date)
        if df is None or df.empty:
            return None

        # Filter to NG only
        ng_data = df[df['ticker'] == TICKER]
        if ng_data.empty:
            return None

        result = {'date': target_date}

        for _, row in ng_data.iterrows():
            clgroup = row['clgroup']
            result[f'{clgroup.lower()}_pos_long'] = row['pos_long']
            result[f'{clgroup.lower()}_pos_short'] = row['pos_short']
            result[f'{clgroup.lower()}_net'] = row['pos']

        return result

    except Exception as e:
        log(f"Error fetching FUTOI: {e}")
        return None


# =============================================================================
# PRICE DATA
# =============================================================================

def get_front_month_contract() -> str:
    """Get current front-month NG contract code."""
    # Contract months: F(Jan), G(Feb), H(Mar), J(Apr), K(May), M(Jun),
    #                  N(Jul), Q(Aug), U(Sep), V(Oct), X(Nov), Z(Dec)
    month_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

    today = date.today()
    # Use next month's contract (front-month logic)
    month_idx = today.month % 12  # next month index (0-11)
    year = today.year if today.month < 12 else today.year + 1

    month_code = month_codes[month_idx]
    year_digit = year % 10

    return f"NG{month_code}{year_digit}"


def fetch_current_price() -> Optional[float]:
    """Fetch current NG price from MOEX ISS."""
    contract = get_front_month_contract()
    url = f"{ISS_BASE}/engines/futures/markets/forts/securities/{contract}.json"

    params = {
        "iss.meta": "off",
        "iss.only": "marketdata",
        "marketdata.columns": "SECID,LAST,SETTLEPRICE",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        if not rows:
            log(f"No market data for {contract}")
            return None

        idx = {c: i for i, c in enumerate(cols)}
        row = rows[0]

        last = row[idx["LAST"]] if row[idx["LAST"]] else row[idx["SETTLEPRICE"]]

        if last:
            log(f"Price for {contract}: {last}")
            return float(last)

        return None

    except Exception as e:
        log(f"Error fetching price: {e}")
        return None


# =============================================================================
# JOURNAL
# =============================================================================

def ensure_journal():
    """Create journal file with headers if doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'date', 'signal', 'entry_price', 'exit_price',
                'gross_pnl', 'commission', 'net_pnl', 'cumulative_pnl'
            ])
        log(f"Created journal: {JOURNAL_FILE}")


def load_journal() -> list[dict]:
    """Load journal entries."""
    ensure_journal()

    entries = []
    with open(JOURNAL_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    return entries


def append_journal(entry: dict):
    """Append entry to journal."""
    ensure_journal()

    with open(JOURNAL_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            entry.get('date', ''),
            entry.get('signal', ''),
            entry.get('entry_price', ''),
            entry.get('exit_price', ''),
            entry.get('gross_pnl', ''),
            entry.get('commission', ''),
            entry.get('net_pnl', ''),
            entry.get('cumulative_pnl', ''),
        ])


def update_last_journal_entry(updates: dict):
    """Update the last entry in journal."""
    entries = load_journal()
    if not entries:
        log("No journal entries to update")
        return

    # Update last entry
    last = entries[-1]
    last.update(updates)

    # Rewrite journal
    with open(JOURNAL_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'signal', 'entry_price', 'exit_price',
            'gross_pnl', 'commission', 'net_pnl', 'cumulative_pnl'
        ])
        for e in entries:
            writer.writerow([
                e.get('date', ''),
                e.get('signal', ''),
                e.get('entry_price', ''),
                e.get('exit_price', ''),
                e.get('gross_pnl', ''),
                e.get('commission', ''),
                e.get('net_pnl', ''),
                e.get('cumulative_pnl', ''),
            ])


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signal():
    """
    Generate signal for tomorrow based on today's FUTOI.
    Called at 19:00 MSK.
    """
    log("=== SIGNAL GENERATION ===")

    authenticate_algopack()

    today = date.today()
    yesterday = today - timedelta(days=1)

    # Skip weekends for yesterday
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)

    log(f"Fetching FUTOI: today={today}, prev={yesterday}")

    today_data = fetch_futoi_snapshot(today)
    yesterday_data = fetch_futoi_snapshot(yesterday)

    if not today_data or not yesterday_data:
        alert("FUTOI data unavailable - NO SIGNAL")
        return

    # Calculate YUR pos_long delta
    yur_long_today = today_data.get('yur_pos_long', 0)
    yur_long_yesterday = yesterday_data.get('yur_pos_long', 0)
    delta = yur_long_today - yur_long_yesterday

    log(f"YUR pos_long: today={yur_long_today}, yesterday={yur_long_yesterday}, delta={delta}")

    # Generate signal
    if delta > 0:
        signal = "LONG"
    elif delta < 0:
        signal = "SHORT"
    else:
        signal = "FLAT"

    # Save signal for tomorrow
    tomorrow = today + timedelta(days=1)
    # Skip to Monday if tomorrow is weekend
    while tomorrow.weekday() >= 5:
        tomorrow += timedelta(days=1)

    with open(SIGNAL_FILE, 'w') as f:
        f.write(f"{tomorrow},{signal},{delta},{yur_long_today}")

    alert(f"SIGNAL FOR {tomorrow}: {signal} (delta={delta})")

    # Create journal entry placeholder
    ensure_journal()
    append_journal({
        'date': tomorrow.isoformat(),
        'signal': signal,
        'entry_price': '',
        'exit_price': '',
        'gross_pnl': '',
        'commission': '',
        'net_pnl': '',
        'cumulative_pnl': '',
    })

    log(f"Created journal entry for {tomorrow}")


# =============================================================================
# ENTRY RECORDING
# =============================================================================

def record_entry():
    """
    Record entry price at 10:05 MSK.
    """
    log("=== RECORD ENTRY ===")

    # Load pending signal
    if not SIGNAL_FILE.exists():
        alert("No pending signal file - skipping entry")
        return

    with open(SIGNAL_FILE, 'r') as f:
        content = f.read().strip()

    parts = content.split(',')
    signal_date = parts[0]
    signal = parts[1]

    today = date.today().isoformat()

    if signal_date != today:
        alert(f"Signal date mismatch: expected {today}, got {signal_date}")
        return

    if signal == "FLAT":
        alert("Signal is FLAT - no entry")
        return

    # Fetch current price
    price = fetch_current_price()

    if price is None:
        alert("Could not fetch entry price")
        return

    # Update journal
    update_last_journal_entry({'entry_price': price})

    alert(f"ENTRY RECORDED: {signal} @ {price}")


# =============================================================================
# EXIT RECORDING
# =============================================================================

def record_exit():
    """
    Record exit price at 23:50 MSK and calculate PnL.
    """
    log("=== RECORD EXIT ===")

    entries = load_journal()
    if not entries:
        alert("No journal entries")
        return

    last = entries[-1]

    # Check if we have an entry price
    if not last.get('entry_price'):
        alert("No entry price - skipping exit")
        return

    signal = last.get('signal', '')
    if signal == "FLAT":
        alert("Signal was FLAT - no exit needed")
        return

    # Fetch exit price
    price = fetch_current_price()

    if price is None:
        alert("Could not fetch exit price")
        return

    entry_price = float(last['entry_price'])
    exit_price = price

    # Calculate PnL
    if signal == "LONG":
        gross_pnl = exit_price - entry_price
    else:  # SHORT
        gross_pnl = entry_price - exit_price

    # Commission: 10 bps round-trip on notional
    avg_price = (entry_price + exit_price) / 2
    commission = avg_price * COMMISSION_BPS / 10000

    net_pnl = gross_pnl - commission

    # Calculate cumulative PnL
    cumulative = 0.0
    for e in entries[:-1]:
        if e.get('net_pnl'):
            cumulative += float(e['net_pnl'])
    cumulative += net_pnl

    # Update journal
    update_last_journal_entry({
        'exit_price': exit_price,
        'gross_pnl': f"{gross_pnl:.4f}",
        'commission': f"{commission:.4f}",
        'net_pnl': f"{net_pnl:.4f}",
        'cumulative_pnl': f"{cumulative:.4f}",
    })

    # Clear pending signal
    if SIGNAL_FILE.exists():
        SIGNAL_FILE.unlink()

    result = "WIN" if net_pnl > 0 else "LOSS"
    alert(f"EXIT: {signal} {entry_price:.2f} → {exit_price:.2f} | Net PnL: {net_pnl:.4f} ({result})")
    log(f"Cumulative PnL: {cumulative:.4f}")


# =============================================================================
# WEEKLY REPORT
# =============================================================================

def weekly_report():
    """
    Generate weekly summary report on Friday 23:55.
    """
    log("=== WEEKLY REPORT ===")

    entries = load_journal()
    if not entries:
        alert("No journal entries for report")
        return

    # Filter to this week
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday

    week_entries = []
    for e in entries:
        try:
            entry_date = date.fromisoformat(e.get('date', ''))
            if entry_date >= week_start:
                week_entries.append(e)
        except:
            pass

    print()
    print("=" * 60)
    print(f"WEEKLY REPORT: {week_start} to {today}")
    print("=" * 60)

    if not week_entries:
        print("No trades this week")
        print("=" * 60)
        return

    # Stats
    total_trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0

    for e in week_entries:
        if e.get('net_pnl'):
            total_trades += 1
            pnl = float(e['net_pnl'])
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Week PnL: {total_pnl:.4f}")

    # All-time cumulative
    all_pnl = 0.0
    all_trades = 0
    for e in entries:
        if e.get('net_pnl'):
            all_trades += 1
            all_pnl += float(e['net_pnl'])

    print()
    print(f"All-time trades: {all_trades}")
    print(f"All-time PnL: {all_pnl:.4f}")
    print("=" * 60)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NG Paper Trading Bot")
    parser.add_argument(
        '--cron',
        choices=['signal', 'entry', 'exit', 'report'],
        required=True,
        help='Cron mode: signal (19:00), entry (10:05), exit (23:50), report (Fri 23:55)'
    )

    args = parser.parse_args()

    if args.cron == 'signal':
        generate_signal()
    elif args.cron == 'entry':
        record_entry()
    elif args.cron == 'exit':
        record_exit()
    elif args.cron == 'report':
        weekly_report()


if __name__ == "__main__":
    main()
