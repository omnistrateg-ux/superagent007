#!/usr/bin/env python3
"""
Download FUTOI (Futures Open Interest) data from MOEX AlgoPack.
Uses Market.futoi() to get all futures data for each trading day.

Usage: MOEX_ALGOPACK_KEY='...' python download_futoi.py
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from moexalgo import Market
import moexalgo.session as moex_session

# API Key for AlgoPack (JWT token)
API_KEY = os.environ.get("MOEX_ALGOPACK_KEY", "")

# Output directory
OUTPUT_DIR = Path("data/futoi_futures")


def authenticate():
    """Authenticate with MOEX AlgoPack using JWT token."""
    if not API_KEY:
        raise ValueError(
            "MOEX_ALGOPACK_KEY environment variable not set. "
            "Set it with: export MOEX_ALGOPACK_KEY='your_key_here'"
        )
    # Set JWT token directly in session module
    moex_session.TOKEN = API_KEY
    print("Authenticated with MOEX AlgoPack (JWT token set)")


def get_trading_dates(start: date, end: date) -> list[date]:
    """Generate list of potential trading dates (weekdays only)."""
    dates = []
    current = start
    while current <= end:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def download_futoi_by_year(year: int, market: Market) -> pd.DataFrame | None:
    """Download FUTOI data for an entire year."""
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today())

    if start > date.today():
        print(f"  Year {year} is in the future, skipping")
        return None

    dates = get_trading_dates(start, end)
    all_data = []

    print(f"  Processing {len(dates)} trading days...")

    for i, d in enumerate(dates):
        try:
            df = market.futoi(date=d)
            if df is not None and not df.empty:
                all_data.append(df)
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(dates)} days processed ({len(all_data)} with data)")
        except Exception as e:
            # Skip dates with no data or errors (holidays, etc.)
            if "403" in str(e) or "404" in str(e):
                continue
            print(f"    Warning: {d}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  Year {year}: {len(combined)} total records from {len(all_data)} days")
        return combined

    return None


def main():
    print("MOEX FUTOI Data Downloader")
    print("=" * 40)

    authenticate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize futures market
    market = Market('FORTS', 'FUT')
    print(f"Market: {market.engine}/{market.market}")

    # Years to download
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_years_data = []

    for year in years:
        print(f"\nDownloading {year}...")
        df = download_futoi_by_year(year, market)

        if df is not None and not df.empty:
            # Save year file
            year_file = OUTPUT_DIR / f"futoi_{year}.parquet"
            df.to_parquet(year_file, index=False)
            print(f"  Saved to {year_file}")
            all_years_data.append(df)

    # Save combined file
    if all_years_data:
        all_combined = pd.concat(all_years_data, ignore_index=True)
        combined_file = OUTPUT_DIR / "futoi_all.parquet"
        all_combined.to_parquet(combined_file, index=False)
        print(f"\nTotal: {len(all_combined):,} records saved to {combined_file}")

        # Print sample
        print("\nSample data:")
        print(all_combined.head(10))
        print("\nColumns:", list(all_combined.columns))
    else:
        print("\nNo data downloaded!")

    print("\nDone!")


if __name__ == "__main__":
    main()
