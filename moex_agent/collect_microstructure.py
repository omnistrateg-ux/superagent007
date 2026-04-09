"""
MOEX Agent v2.5 Microstructure Data Collector

Phase 2: Collect order book and trade tape data for 30 days.

Usage:
    python -m moex_agent.collect_microstructure --tickers SBER,GAZP,BR

Data collection:
    - Order book snapshots every 500ms
    - All trades from tape
    - Open Interest updates (futures)

Storage:
    - SQLite: data/microstructure.db
    - ~50-100 MB/day for 4 futures + 10 stocks

Requirements:
    - QUIK terminal running
    - BCS account connected
    - QUIK Lua scripts or QuikPy library
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .broker import BCSClient, BCSConfig, MicrostructureStorage
from .microstructure import OrderBookSnapshot, Trade

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


class MicrostructureCollector:
    """
    Collects microstructure data from BCS QUIK API.

    Runs continuously, saving data to SQLite.
    """

    def __init__(
        self,
        tickers: List[str],
        db_path: Path,
        snapshot_interval_ms: int = 500,
    ):
        self.tickers = tickers
        self.snapshot_interval_ms = snapshot_interval_ms
        self.running = False

        # Initialize storage
        self.storage = MicrostructureStorage(db_path)

        # Initialize BCS client
        self.client = BCSClient(BCSConfig())

        # Set up callbacks
        self.client.on_book_update = self._on_book_update
        self.client.on_trade = self._on_trade
        self.client.on_oi_update = self._on_oi_update

        # Stats
        self.book_count = 0
        self.trade_count = 0
        self.oi_count = 0

    def _on_book_update(self, ticker: str, snapshot: OrderBookSnapshot) -> None:
        """Handle order book update."""
        self.storage.save_orderbook(snapshot)
        self.book_count += 1

        if self.book_count % 1000 == 0:
            logger.info(f"Saved {self.book_count} order book snapshots")

    def _on_trade(self, ticker: str, trade: Trade) -> None:
        """Handle trade update."""
        self.storage.save_trade(trade)
        self.trade_count += 1

        if self.trade_count % 1000 == 0:
            logger.info(f"Saved {self.trade_count} trades")

    def _on_oi_update(self, ticker: str, oi: float) -> None:
        """Handle OI update."""
        import pandas as pd
        self.storage.save_oi(ticker, pd.Timestamp.now(tz='UTC'), oi)
        self.oi_count += 1

    def start(self) -> None:
        """Start data collection."""
        logger.info(f"Starting microstructure collection for {len(self.tickers)} tickers")
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        logger.info(f"Snapshot interval: {self.snapshot_interval_ms}ms")

        # Connect to QUIK
        if not self.client.connect():
            logger.error("Failed to connect to QUIK")
            return

        # Subscribe to tickers
        if not self.client.subscribe(self.tickers):
            logger.error("Failed to subscribe to tickers")
            return

        self.running = True
        logger.info("Collection started. Press Ctrl+C to stop.")

        # Main loop
        try:
            while self.running:
                # In real implementation, QUIK would push updates via callbacks
                # Here we just sleep and wait for callbacks
                time.sleep(self.snapshot_interval_ms / 1000.0)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop data collection."""
        self.running = False
        self.client.disconnect()

        logger.info("Collection stopped")
        logger.info(f"Total saved: {self.book_count} orderbooks, "
                   f"{self.trade_count} trades, {self.oi_count} OI updates")

        # Show storage stats
        stats = self.storage.get_data_stats()
        logger.info(f"Database stats: {stats}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect microstructure data from BCS QUIK"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="SBER,GAZP,LKOH,ROSN,GMKN,BR,RI,MX",
        help="Comma-separated list of tickers"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/microstructure.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Order book snapshot interval in milliseconds"
    )

    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    db_path = Path(args.db_path)

    # Create data directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)

    collector = MicrostructureCollector(
        tickers=tickers,
        db_path=db_path,
        snapshot_interval_ms=args.interval_ms,
    )

    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    collector.start()


if __name__ == "__main__":
    main()
