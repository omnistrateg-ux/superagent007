"""
Microstructure Data Validation Script

Validates collected data quality:
- Coverage by symbol/day
- Missing intervals
- Quote/trade counts
- Session completeness
- Gap detection

Usage:
    python -m moex_agent.microstructure_validate --db-path data/microstructure.db
    python -m moex_agent.microstructure_validate --date 2026-04-15
    python -m moex_agent.microstructure_validate --ticker SBER --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .microstructure_storage import MicrostructureStorage, MOEX_SESSIONS, MSK

logger = logging.getLogger(__name__)


@dataclass
class SessionCoverage:
    """Coverage stats for a single session."""
    session: str
    expected_minutes: int
    covered_minutes: int
    coverage_pct: float
    trades_count: int
    quotes_count: int
    has_data: bool


class QualityVerdict:
    """
    Quality verdict thresholds.

    GREEN:  score >= 80, unknown_side_pct < 5%, max_gap < 60s
            PASS for 30-day campaign
    YELLOW: score >= 60, some issues but usable
            WARN - investigate before continuing
    RED:    score < 60 or unknown_side_pct > 50%
            FAIL - do not use for research
    """
    GREEN = "GREEN"   # score >= 80, all thresholds pass
    YELLOW = "YELLOW" # score >= 60, some warnings
    RED = "RED"       # score < 60 or critical blockers


@dataclass
class DailyReport:
    """Daily validation report for a ticker."""
    date: str
    ticker: str
    trades_count: int
    quotes_count: int
    depth_count: int
    oi_count: int
    first_ts: Optional[str]
    last_ts: Optional[str]
    gaps_count: int
    max_gap_seconds: float
    unknown_side_count: int
    unknown_side_pct: float
    coverage_pct: float
    sessions: List[SessionCoverage]
    quality_score: float  # 0-100
    quality_verdict: str  # GREEN/YELLOW/RED
    issues: List[str]


@dataclass
class ValidationSummary:
    """Overall validation summary."""
    db_path: str
    validation_time: str
    total_days: int
    total_tickers: int
    total_trades: int
    total_quotes: int
    avg_quality_score: float
    reports: List[DailyReport]
    blockers: List[str]
    warnings: List[str]


class DataValidator:
    """Validates microstructure data quality."""

    # Expected data rates (per minute during trading hours)
    EXPECTED_RATES = {
        "quotes_per_min": 120,   # ~2 per second
        "trades_per_min": 10,    # Varies by liquidity
        "depth_per_min": 120,    # ~2 per second
    }

    # Quality thresholds for BURN-IN
    THRESHOLDS = {
        "min_coverage_pct": 80.0,
        "max_unknown_side_pct": 5.0,  # CRITICAL: > 50% = RED always
        "max_gaps_per_day": 10,
        "max_gap_seconds": 300.0,     # 5 min max gap
        "min_trades_per_day": 100,
    }

    def __init__(self, storage: MicrostructureStorage):
        self.storage = storage

    def validate_day(
        self,
        target_date: date,
        ticker: str,
        verbose: bool = False,
    ) -> DailyReport:
        """Validate data for a single day and ticker."""
        date_str = target_date.isoformat()

        # Get counts from storage
        df_trades = self.storage.get_trades(
            ticker,
            start=datetime.combine(target_date, datetime.min.time()).replace(tzinfo=MSK),
            end=datetime.combine(target_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=MSK),
        )

        conn = self.storage._connect()

        # Quotes count
        quotes_count = conn.execute(
            "SELECT COUNT(*) as c FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'",
            (ticker, date_str)
        ).fetchone()["c"]

        # Depth count
        depth_count = conn.execute(
            "SELECT COUNT(*) as c FROM raw_depth WHERE ticker=? AND ts LIKE ? || '%'",
            (ticker, date_str)
        ).fetchone()["c"]

        # OI count
        oi_count = conn.execute(
            "SELECT COUNT(*) as c FROM raw_oi WHERE ticker=? AND ts LIKE ? || '%'",
            (ticker, date_str)
        ).fetchone()["c"]

        # First/last timestamps
        first_ts = conn.execute(
            """SELECT MIN(ts) as ts FROM (
                SELECT ts FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%'
                UNION ALL SELECT ts FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'
            )""",
            (ticker, date_str, ticker, date_str)
        ).fetchone()["ts"]

        last_ts = conn.execute(
            """SELECT MAX(ts) as ts FROM (
                SELECT ts FROM raw_trades WHERE ticker=? AND ts LIKE ? || '%'
                UNION ALL SELECT ts FROM raw_quotes WHERE ticker=? AND ts LIKE ? || '%'
            )""",
            (ticker, date_str, ticker, date_str)
        ).fetchone()["ts"]

        # Gaps count and max gap
        gaps_count = conn.execute(
            "SELECT COUNT(*) as c FROM collection_gaps WHERE ticker=? AND ts LIKE ? || '%'",
            (ticker, date_str)
        ).fetchone()["c"]

        max_gap_result = conn.execute(
            "SELECT MAX(gap_seconds) as max_gap FROM collection_gaps WHERE ticker=? AND ts LIKE ? || '%'",
            (ticker, date_str)
        ).fetchone()
        max_gap_seconds = max_gap_result["max_gap"] if max_gap_result["max_gap"] else 0.0

        # Unknown side analysis
        trades_count = len(df_trades)
        unknown_side_count = len(df_trades[df_trades["side"] == "UNKNOWN"]) if not df_trades.empty else 0
        unknown_side_pct = (unknown_side_count / trades_count * 100) if trades_count > 0 else 0

        # Session coverage analysis
        sessions = self._analyze_sessions(df_trades, target_date, ticker)

        # Calculate overall coverage
        total_expected = sum(s.expected_minutes for s in sessions if s.expected_minutes > 0)
        total_covered = sum(s.covered_minutes for s in sessions)
        coverage_pct = (total_covered / total_expected * 100) if total_expected > 0 else 0

        # Identify issues
        issues = []

        if trades_count < self.THRESHOLDS["min_trades_per_day"]:
            issues.append(f"Low trade count: {trades_count} (min: {self.THRESHOLDS['min_trades_per_day']})")

        if unknown_side_pct > self.THRESHOLDS["max_unknown_side_pct"]:
            issues.append(f"High unknown side: {unknown_side_pct:.1f}% (max: {self.THRESHOLDS['max_unknown_side_pct']}%)")

        if coverage_pct < self.THRESHOLDS["min_coverage_pct"]:
            issues.append(f"Low coverage: {coverage_pct:.1f}% (min: {self.THRESHOLDS['min_coverage_pct']}%)")

        if gaps_count > self.THRESHOLDS["max_gaps_per_day"]:
            issues.append(f"Many gaps: {gaps_count} (max: {self.THRESHOLDS['max_gaps_per_day']})")

        if max_gap_seconds > self.THRESHOLDS["max_gap_seconds"]:
            issues.append(f"Large gap: {max_gap_seconds:.0f}s (max: {self.THRESHOLDS['max_gap_seconds']:.0f}s)")

        # Calculate quality score (0-100) and verdict
        quality_score, quality_verdict = self._calculate_quality_score(
            coverage_pct=coverage_pct,
            unknown_side_pct=unknown_side_pct,
            gaps_count=gaps_count,
            trades_count=trades_count,
            max_gap_seconds=max_gap_seconds,
        )

        report = DailyReport(
            date=date_str,
            ticker=ticker,
            trades_count=trades_count,
            quotes_count=quotes_count,
            depth_count=depth_count,
            oi_count=oi_count,
            first_ts=first_ts,
            last_ts=last_ts,
            gaps_count=gaps_count,
            max_gap_seconds=max_gap_seconds,
            unknown_side_count=unknown_side_count,
            unknown_side_pct=unknown_side_pct,
            coverage_pct=coverage_pct,
            sessions=sessions,
            quality_score=quality_score,
            quality_verdict=quality_verdict,
            issues=issues,
        )

        if verbose:
            self._print_report(report)

        return report

    def _analyze_sessions(
        self,
        df_trades: pd.DataFrame,
        target_date: date,
        ticker: str,
    ) -> List[SessionCoverage]:
        """Analyze coverage by session."""
        results = []

        for session in MOEX_SESSIONS:
            if not session.trading:
                continue

            # Calculate session duration
            start = datetime.combine(
                target_date,
                datetime.strptime(f"{session.start_hour}:{session.start_minute}", "%H:%M").time()
            ).replace(tzinfo=MSK)

            end = datetime.combine(
                target_date,
                datetime.strptime(f"{session.end_hour}:{session.end_minute}", "%H:%M").time()
            ).replace(tzinfo=MSK)

            expected_minutes = int((end - start).total_seconds() / 60)

            # Count trades in session
            if not df_trades.empty:
                session_trades = df_trades[df_trades["session"] == session.name]
                trades_count = len(session_trades)

                if trades_count > 0:
                    # Estimate covered minutes from trade timestamps
                    unique_minutes = session_trades["ts"].dt.floor("min").nunique()
                    covered_minutes = unique_minutes
                else:
                    covered_minutes = 0
            else:
                trades_count = 0
                covered_minutes = 0

            coverage_pct = (covered_minutes / expected_minutes * 100) if expected_minutes > 0 else 0

            results.append(SessionCoverage(
                session=session.name,
                expected_minutes=expected_minutes,
                covered_minutes=covered_minutes,
                coverage_pct=coverage_pct,
                trades_count=trades_count,
                quotes_count=0,  # Would need separate query
                has_data=trades_count > 0,
            ))

        return results

    def _calculate_quality_score(
        self,
        coverage_pct: float,
        unknown_side_pct: float,
        gaps_count: int,
        trades_count: int,
        max_gap_seconds: float = 0.0,
    ) -> tuple:
        """
        Calculate overall quality score (0-100) and verdict.

        Returns:
            (score, verdict) tuple
        """
        score = 100.0
        has_blocker = False

        # Coverage penalty
        if coverage_pct < 100:
            score -= (100 - coverage_pct) * 0.5

        # Unknown side penalty (critical for orderflow research)
        score -= unknown_side_pct * 5  # -5 points per 1% unknown
        if unknown_side_pct > 50:
            has_blocker = True  # Cannot test orderflow hypotheses

        # Gaps penalty
        score -= gaps_count * 2  # -2 points per gap

        # Max gap penalty (> 5 min gap is severe)
        if max_gap_seconds > 300:
            score -= 10
        elif max_gap_seconds > 60:
            score -= 5

        # Low trades penalty
        if trades_count < self.THRESHOLDS["min_trades_per_day"]:
            score -= 20

        score = max(0, min(100, score))

        # Determine verdict
        if has_blocker or score < 60:
            verdict = QualityVerdict.RED
        elif score < 80:
            verdict = QualityVerdict.YELLOW
        else:
            verdict = QualityVerdict.GREEN

        return score, verdict

    def _print_report(self, report: DailyReport) -> None:
        """Print report to console."""
        # Big verdict banner
        verdict_colors = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "❌"}
        banner = verdict_colors.get(report.quality_verdict, "?")

        print(f"\n{'=' * 60}")
        print(f"{banner} VALIDATION: {report.ticker} | {report.date} | [{report.quality_verdict}]")
        print(f"{'=' * 60}")

        print(f"\n📊 COUNTS:")
        print(f"  Trades:  {report.trades_count:,}")
        print(f"  Quotes:  {report.quotes_count:,}")
        print(f"  Depth:   {report.depth_count:,}")
        print(f"  OI:      {report.oi_count:,}")

        print(f"\n⏱️ TIME RANGE:")
        print(f"  First:   {report.first_ts}")
        print(f"  Last:    {report.last_ts}")

        print(f"\n📈 QUALITY METRICS:")
        print(f"  Coverage:       {report.coverage_pct:.1f}%")
        print(f"  Unknown Side:   {report.unknown_side_pct:.1f}% ({report.unknown_side_count:,} trades)")
        print(f"  Gaps:           {report.gaps_count} (max {report.max_gap_seconds:.0f}s)")
        print(f"  Quality Score:  {report.quality_score:.1f}/100")

        print(f"\n📅 SESSION COVERAGE:")
        for s in report.sessions:
            if s.expected_minutes > 0:
                status = "✅" if s.coverage_pct >= 80 else "⚠️" if s.has_data else "❌"
                print(f"  {status} {s.session:20s} | {s.coverage_pct:5.1f}% | {s.trades_count:5,} trades")

        if report.issues:
            print(f"\n⚠️ ISSUES:")
            for issue in report.issues:
                print(f"  - {issue}")

        # Burn-in verdict
        print(f"\n{'=' * 60}")
        if report.quality_verdict == "GREEN":
            print("✅ BURN-IN: PASS - Day acceptable for 30-day campaign")
        elif report.quality_verdict == "YELLOW":
            print("⚠️ BURN-IN: WARN - Investigate issues before continuing")
        else:
            print("❌ BURN-IN: FAIL - Day not usable for research")
        print(f"{'=' * 60}")

    def validate_range(
        self,
        start_date: date,
        end_date: date,
        tickers: Optional[List[str]] = None,
    ) -> ValidationSummary:
        """Validate data for a date range."""
        conn = self.storage._connect()

        # Get tickers with data
        if tickers is None:
            cur = conn.execute(
                "SELECT DISTINCT ticker FROM raw_trades WHERE ts >= ? AND ts < ?",
                (start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
            )
            tickers = [row["ticker"] for row in cur.fetchall()]

        reports = []
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                for ticker in tickers:
                    report = self.validate_day(current, ticker)
                    if report.trades_count > 0:  # Only include days with data
                        reports.append(report)

            current += timedelta(days=1)

        # Calculate summary
        total_trades = sum(r.trades_count for r in reports)
        total_quotes = sum(r.quotes_count for r in reports)
        avg_quality = sum(r.quality_score for r in reports) / len(reports) if reports else 0

        # Identify blockers and warnings
        blockers = []
        warnings = []

        # Check for missing aggressor side
        unknown_trades = sum(r.unknown_side_count for r in reports)
        total_all_trades = sum(r.trades_count for r in reports)
        if total_all_trades > 0:
            unknown_pct = unknown_trades / total_all_trades * 100
            if unknown_pct > 50:
                blockers.append(f"BLOCKER: {unknown_pct:.1f}% trades have UNKNOWN side - cannot test orderflow hypotheses")
            elif unknown_pct > 10:
                warnings.append(f"WARNING: {unknown_pct:.1f}% trades have UNKNOWN side")

        # Check coverage
        low_coverage = [r for r in reports if r.coverage_pct < 50]
        if len(low_coverage) > len(reports) * 0.3:
            warnings.append(f"WARNING: {len(low_coverage)} days have <50% coverage")

        return ValidationSummary(
            db_path=str(self.storage.db_path),
            validation_time=datetime.now().isoformat(),
            total_days=len(set(r.date for r in reports)),
            total_tickers=len(tickers),
            total_trades=total_trades,
            total_quotes=total_quotes,
            avg_quality_score=avg_quality,
            reports=reports,
            blockers=blockers,
            warnings=warnings,
        )

    def print_summary(self, summary: ValidationSummary) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nDatabase: {summary.db_path}")
        print(f"Validated: {summary.validation_time}")

        print(f"\nOVERALL:")
        print(f"  Days:         {summary.total_days}")
        print(f"  Tickers:      {summary.total_tickers}")
        print(f"  Total Trades: {summary.total_trades:,}")
        print(f"  Total Quotes: {summary.total_quotes:,}")
        print(f"  Avg Quality:  {summary.avg_quality_score:.1f}/100")

        if summary.blockers:
            print(f"\nBLOCKERS:")
            for b in summary.blockers:
                print(f"  {b}")

        if summary.warnings:
            print(f"\nWARNINGS:")
            for w in summary.warnings:
                print(f"  {w}")

        # Per-ticker summary
        print(f"\nPER TICKER:")
        ticker_stats: Dict[str, Dict[str, Any]] = {}
        for r in summary.reports:
            if r.ticker not in ticker_stats:
                ticker_stats[r.ticker] = {
                    "days": 0,
                    "trades": 0,
                    "quality_sum": 0,
                }
            ticker_stats[r.ticker]["days"] += 1
            ticker_stats[r.ticker]["trades"] += r.trades_count
            ticker_stats[r.ticker]["quality_sum"] += r.quality_score

        for ticker, stats in sorted(ticker_stats.items()):
            avg_q = stats["quality_sum"] / stats["days"] if stats["days"] > 0 else 0
            print(f"  {ticker:8s} | {stats['days']:3d} days | {stats['trades']:8,} trades | Q={avg_q:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate microstructure data quality"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/microstructure.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date to validate (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date for range validation"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for range validation"
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Specific ticker to validate"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed reports"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Open storage
    storage = MicrostructureStorage(Path(args.db_path))
    validator = DataValidator(storage)

    # Determine date range
    if args.date:
        start_date = date.fromisoformat(args.date)
        end_date = start_date
    elif args.start_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
    else:
        # Default: last 7 days
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

    # Validate
    tickers = [args.ticker] if args.ticker else None

    if args.ticker and args.date:
        # Single day/ticker validation
        report = validator.validate_day(
            date.fromisoformat(args.date),
            args.ticker,
            verbose=args.verbose,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
    else:
        # Range validation
        summary = validator.validate_range(start_date, end_date, tickers)
        validator.print_summary(summary)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(summary), f, indent=2, default=str)


if __name__ == "__main__":
    main()
