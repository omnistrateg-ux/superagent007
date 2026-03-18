"""
MOEX Agent v2 CLI

Unified command-line interface for all operations.

Usage:
    python -m moex_agent init-db              # Initialize database
    python -m moex_agent bootstrap --days 180 # Load historical data
    python -m moex_agent train                # Train ML models
    python -m moex_agent live                 # Run live signal loop
    python -m moex_agent paper                # Paper trading mode
    python -m moex_agent margin               # Margin trading mode
    python -m moex_agent backtest             # Run backtest
    python -m moex_agent web --port 8000      # Start web dashboard
    python -m moex_agent status               # Show system status
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moex_agent")


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize database schema."""
    from .config import load_config
    from .storage import connect, init_db

    config = load_config(args.config)
    conn = connect(config.sqlite_path)

    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return 1

    init_db(conn, schema_path)
    conn.close()

    logger.info(f"Database initialized: {config.sqlite_path}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Load historical candle data."""
    from .config import load_config
    from .iss import fetch_candles
    from .storage import connect, upsert_many

    config = load_config(args.config)
    conn = connect(config.sqlite_path)

    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=args.days)).isoformat()
    till_date = today.isoformat()

    logger.info(f"Loading {args.days} days for {len(config.tickers)} tickers")

    total_candles = 0
    for i, secid in enumerate(config.tickers, 1):
        try:
            candles = fetch_candles(
                config.engine,
                config.market,
                config.board,
                secid,
                interval=1,
                from_date=from_date,
                till_date=till_date,
            )
            rows = [
                (secid, config.board, 1, c.ts, c.open, c.high, c.low, c.close, c.value, c.volume)
                for c in candles
            ]
            upsert_many(
                conn,
                table="candles",
                columns=("secid", "board", "interval", "ts", "open", "high", "low", "close", "value", "volume"),
                rows=rows,
            )
            total_candles += len(candles)
            logger.info(f"[{i}/{len(config.tickers)}] {secid}: {len(candles)} candles")
        except Exception as e:
            logger.warning(f"[{i}/{len(config.tickers)}] {secid}: ERROR - {e}")

    conn.close()
    logger.info(f"Bootstrap complete: {total_candles:,} candles")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train ML models."""
    from .train import main as train_main
    logger.info("Starting model training...")
    train_main()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run live signal generation loop."""
    from .config import load_config
    from .engine import PipelineEngine
    from .iss import close_session
    from .storage import connect, save_alert

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutdown signal received...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = load_config(args.config)
    conn = connect(config.sqlite_path)
    engine = PipelineEngine(config)
    engine.load_models()

    cooldown_map = defaultdict(lambda: datetime(1970, 1, 1, tzinfo=timezone.utc))
    cycle_count = 0
    alerts_count = 0

    logger.info("Live loop started")
    logger.info(f"Tickers: {len(config.tickers)} | Poll: {config.poll_seconds}s | P threshold: {config.p_threshold}")

    try:
        while not shutdown_requested:
            cycle_count += 1

            try:
                result = engine.run_cycle(conn, cooldown_map=cooldown_map)

                if result.errors:
                    for err in result.errors:
                        logger.warning(f"Error: {err}")

                if not result.signals:
                    if cycle_count % 12 == 0:
                        logger.info(f"Cycle {cycle_count}: no signals (anomalies: {result.anomalies_count})")

                    if args.once:
                        break

                    time.sleep(config.poll_seconds)
                    continue

                for sig in result.signals:
                    alert_id = save_alert(
                        conn,
                        secid=sig.secid,
                        direction=sig.direction.value,
                        horizon=sig.horizon,
                        p=sig.probability,
                        signal_type=sig.signal_type,
                        entry=sig.entry,
                        take=sig.take,
                        stop=sig.stop,
                        ttl_minutes=sig.ttl_minutes,
                        anomaly_score=sig.anomaly_score,
                        payload_json=str(sig.to_dict()),
                    )
                    alerts_count += 1
                    cooldown_map[sig.secid] = datetime.now(timezone.utc)

                    logger.info(
                        f"SIGNAL: {sig.secid} {sig.direction.value} {sig.horizon} "
                        f"p={sig.probability:.0%} score={sig.anomaly_score:.1f}"
                    )

                if args.once:
                    break

                time.sleep(config.poll_seconds)

            except Exception as e:
                logger.error(f"Cycle error: {repr(e)}")
                time.sleep(max(5, config.poll_seconds))

    finally:
        conn.close()
        close_session()
        logger.info(f"Shutdown. Cycles: {cycle_count}, Alerts: {alerts_count}")

    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    """Run paper trading mode using Trader class."""
    from .config import load_config
    from .trader import Trader

    config = load_config(args.config)

    trader = Trader(
        config=config,
        initial_equity=args.equity,
        leverage=1.0,
        state_path=Path("data/paper_trader_state.json"),
    )

    logger.info(f"Paper trading started with equity: {args.equity:,.0f}")
    trader.run()

    # Print final statistics
    stats = trader.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Trades: {stats['trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Total PnL: {stats['total_pnl']:+,.0f}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Final Equity: {stats['equity']:,.0f}")

    return 0


def cmd_margin(args: argparse.Namespace) -> int:
    """Run margin trading mode using Trader class with leverage."""
    from .config import load_config
    from .trader import Trader

    config = load_config(args.config)

    trader = Trader(
        config=config,
        initial_equity=args.equity,
        leverage=args.leverage,
        state_path=Path("data/margin_trader_state.json"),
    )

    logger.info(f"Margin trading started: equity={args.equity:,.0f}, leverage={args.leverage}x")
    trader.run()

    stats = trader.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Trades: {stats['trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Total PnL: {stats['total_pnl']:+,.0f}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Final Equity: {stats['equity']:,.0f}")
    print(f"  Max Drawdown: {stats['drawdown_pct']:.1f}%")

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtest using Backtester class."""
    from .backtest import run_backtest

    # Default to top 6 liquid tickers
    default_tickers = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "VTBR"]
    tickers = args.tickers.split(",") if args.tickers else default_tickers

    ticker_count = len(tickers) if tickers else "all"
    logger.info(f"Starting backtest: {args.days} days, {ticker_count} tickers")
    metrics = run_backtest(
        config_path=args.config,
        export_csv=not args.no_export,
        days=args.days,
        tickers=tickers,
    )

    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Start web dashboard."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn fastapi")
        return 1

    logger.info(f"Starting web server on http://0.0.0.0:{args.port}")
    uvicorn.run(
        "moex_agent.web:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status."""
    from .config import load_config
    from .storage import connect

    config = load_config(args.config)

    print(f"\n{'=' * 50}")
    print("MOEX Agent v2 Status")
    print(f"{'=' * 50}")

    print(f"\nConfiguration:")
    print(f"  Tickers: {len(config.tickers)}")
    print(f"  Poll: {config.poll_seconds}s")
    print(f"  P threshold: {config.p_threshold}")
    print(f"  Fee: {config.fee_bps} bps")

    print(f"\nDatabase:")
    print(f"  Path: {config.sqlite_path}")
    if config.sqlite_path.exists():
        conn = connect(config.sqlite_path)
        cur = conn.execute("SELECT COUNT(*) as cnt FROM candles")
        candles_count = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM alerts")
        alerts_count = cur.fetchone()["cnt"]
        conn.close()
        print(f"  Candles: {candles_count:,}")
        print(f"  Alerts: {alerts_count:,}")
    else:
        print("  Status: NOT INITIALIZED")

    print(f"\nModels:")
    models_dir = Path("./models")
    meta_path = models_dir / "meta.json"
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        print(f"  Loaded: {list(meta.keys())}")
        for h, info in meta.items():
            m = info.get("metrics", {})
            print(f"    {h}: WR={m.get('win_rate', 0):.1f}%, MaxP={m.get('max_probability', 0):.2f}")
    else:
        print("  Status: NOT TRAINED")

    print(f"\nRisk:")
    print(f"  Max spread: {config.risk.max_spread_bps} bps")
    print(f"  Min turnover: {config.risk.min_turnover_rub_5m:,.0f} RUB")
    print(f"  Max daily loss: {config.risk.max_daily_loss_pct}%")
    print(f"  Max consecutive losses: {config.risk.max_consecutive_losses}")
    print(f"  Max drawdown: {config.risk.max_drawdown_pct}%")

    print(f"\n{'=' * 50}\n")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="moex_agent",
        description="MOEX Trading Signal Agent v2",
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init-db
    subparsers.add_parser("init-db", help="Initialize database")

    # bootstrap
    sub = subparsers.add_parser("bootstrap", help="Load historical data")
    sub.add_argument("--days", type=int, default=180, help="Days of history")

    # train
    subparsers.add_parser("train", help="Train ML models")

    # live
    sub = subparsers.add_parser("live", help="Run live signal loop")
    sub.add_argument("--once", action="store_true", help="Run one cycle")

    # paper
    sub = subparsers.add_parser("paper", help="Paper trading mode")
    sub.add_argument("--equity", type=float, default=1_000_000, help="Starting equity")

    # margin
    sub = subparsers.add_parser("margin", help="Margin trading mode")
    sub.add_argument("--equity", type=float, default=1_000_000, help="Starting equity")
    sub.add_argument("--leverage", type=float, default=3.0, help="Leverage multiplier")

    # backtest
    sub = subparsers.add_parser("backtest", help="Run backtest")
    sub.add_argument("--no-export", action="store_true", help="Don't export trades to CSV")
    sub.add_argument("--days", type=int, default=30, help="Days of history for backtest")
    sub.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: Tier1)")

    # web
    sub = subparsers.add_parser("web", help="Start web dashboard")
    sub.add_argument("--port", type=int, default=8000, help="Port")
    sub.add_argument("--reload", action="store_true", help="Auto-reload")

    # status
    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    commands = {
        "init-db": cmd_init_db,
        "bootstrap": cmd_bootstrap,
        "train": cmd_train,
        "live": cmd_live,
        "paper": cmd_paper,
        "margin": cmd_margin,
        "backtest": cmd_backtest,
        "web": cmd_web,
        "status": cmd_status,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
