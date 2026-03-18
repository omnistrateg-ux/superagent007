"""
MOEX Agent v2 Futures (FORTS)

Trading futures on Moscow Exchange FORTS market:
- Si (USD/RUB), BR (Brent), RI (RTS Index)
- NG (Natural Gas), GD (Gold), MX (MOEX Index)
- Live quotes via MOEX ISS API
- Open Interest tracking
- Risk management for leveraged positions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("moex_agent.futures")

ISS_BASE = "https://iss.moex.com/iss"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MOEX-Agent/2.0"})


class FuturesContract(Enum):
    """Main FORTS futures contracts."""
    SI = "Si"      # USD/RUB
    BR = "BR"      # Brent Oil
    RI = "RI"      # RTS Index
    NG = "NG"      # Natural Gas
    GD = "GD"      # Gold
    MX = "MX"      # MOEX Index


# Contract specifications
CONTRACT_SPECS = {
    FuturesContract.SI: {
        "name": "USD/RUB",
        "lot_size": 1000,       # $1000 per contract
        "tick_size": 1.0,       # 1 RUB
        "tick_value": 1.0,      # 1 RUB per tick
        "margin_pct": 10.0,     # ~10% initial margin
        "settlement": "cash",
    },
    FuturesContract.BR: {
        "name": "Brent Oil",
        "lot_size": 10,         # 10 barrels
        "tick_size": 0.01,      # $0.01
        "tick_value": 0.1,      # $0.10 per tick
        "margin_pct": 15.0,
        "settlement": "cash",
    },
    FuturesContract.RI: {
        "name": "RTS Index",
        "lot_size": 1,
        "tick_size": 10.0,      # 10 points
        "tick_value": 10.0,     # ~$0.10 per point
        "margin_pct": 12.0,
        "settlement": "cash",
    },
    FuturesContract.NG: {
        "name": "Natural Gas",
        "lot_size": 100,        # 100 MMBtu
        "tick_size": 0.001,
        "tick_value": 0.1,
        "margin_pct": 20.0,
        "settlement": "cash",
    },
    FuturesContract.GD: {
        "name": "Gold",
        "lot_size": 1,          # 1 troy oz
        "tick_size": 0.1,
        "tick_value": 0.1,
        "margin_pct": 10.0,
        "settlement": "cash",
    },
    FuturesContract.MX: {
        "name": "MOEX Index",
        "lot_size": 1,
        "tick_size": 1.0,
        "tick_value": 1.0,
        "margin_pct": 12.0,
        "settlement": "cash",
    },
}


@dataclass
class FuturesQuote:
    """Live futures quote."""
    secid: str
    contract: FuturesContract
    last: float
    bid: float
    ask: float
    spread_pct: float
    volume: int
    value: float              # Turnover in RUB
    open_interest: int
    oi_change: int            # OI change from previous session
    high: float
    low: float
    open: float
    prev_close: float
    change_pct: float
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread_pct * 100

    def __repr__(self) -> str:
        return (
            f"FuturesQuote({self.secid}: {self.last:,.2f} "
            f"[{self.change_pct:+.2f}%] OI={self.open_interest:,})"
        )


@dataclass
class FuturesPosition:
    """Open futures position."""
    secid: str
    contract: FuturesContract
    direction: str            # "LONG" or "SHORT"
    quantity: int             # Number of contracts
    entry_price: float
    current_price: float
    margin_used: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pnl_pct(self) -> float:
        """Unrealized PnL as percentage of margin."""
        if self.margin_used <= 0:
            return 0.0
        return (self.unrealized_pnl / self.margin_used) * 100


@dataclass
class FuturesRiskLimits:
    """Risk management limits for futures trading."""
    max_contracts_per_position: int = 10
    max_total_contracts: int = 50
    max_margin_usage_pct: float = 50.0     # Max 50% of equity in margin
    max_loss_per_trade_pct: float = 2.0    # Max 2% loss per trade
    max_daily_loss_pct: float = 5.0        # Max 5% daily loss
    max_drawdown_pct: float = 15.0         # Max 15% drawdown
    min_liquidity_contracts: int = 100     # Min OI for entry


def _find_nearest_contract(base: str) -> Optional[str]:
    """
    Find nearest (most liquid) contract for a given base symbol.

    Args:
        base: Base symbol (e.g., "Si", "BR")

    Returns:
        Full contract code (e.g., "SiM4") or None
    """
    try:
        url = f"{ISS_BASE}/engines/futures/markets/forts/securities.json"
        resp = SESSION.get(url, params={"iss.meta": "off"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        if not rows or "SECID" not in cols:
            return None

        secid_idx = cols.index("SECID")
        oi_idx = cols.index("OPENPOSITION") if "OPENPOSITION" in cols else None
        vol_idx = cols.index("VOLTODAY") if "VOLTODAY" in cols else None

        # Find contracts matching base symbol
        candidates = []
        for row in rows:
            secid = row[secid_idx]
            if secid and secid.startswith(base):
                oi = row[oi_idx] if oi_idx else 0
                vol = row[vol_idx] if vol_idx else 0
                candidates.append((secid, oi or 0, vol or 0))

        if not candidates:
            return None

        # Sort by OI (most liquid first)
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]

    except Exception as e:
        logger.warning(f"Failed to find contract for {base}: {e}")
        return None


def fetch_futures_quote(contract: FuturesContract) -> Optional[FuturesQuote]:
    """
    Fetch live quote for a futures contract.

    Args:
        contract: FuturesContract enum value

    Returns:
        FuturesQuote or None if fetch failed
    """
    secid = _find_nearest_contract(contract.value)
    if not secid:
        logger.warning(f"No active contract found for {contract.value}")
        return None

    try:
        url = f"{ISS_BASE}/engines/futures/markets/forts/securities/{secid}.json"
        resp = SESSION.get(url, params={"iss.meta": "off"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])

        if not rows:
            return None

        row = dict(zip(cols, rows[0]))

        last = row.get("LAST") or row.get("WAPRICE") or 0
        bid = row.get("BID") or last * 0.999
        ask = row.get("OFFER") or last * 1.001
        open_price = row.get("OPEN") or last
        prev_close = row.get("PREVSETTLEPRICE") or open_price

        spread_pct = ((ask - bid) / last * 100) if last > 0 else 0
        change_pct = ((last - prev_close) / prev_close * 100) if prev_close > 0 else 0

        return FuturesQuote(
            secid=secid,
            contract=contract,
            last=float(last),
            bid=float(bid),
            ask=float(ask),
            spread_pct=float(spread_pct),
            volume=int(row.get("VOLTODAY") or 0),
            value=float(row.get("VALTODAY") or 0),
            open_interest=int(row.get("OPENPOSITION") or 0),
            oi_change=int(row.get("OPENPOSITIONVALUE") or 0),
            high=float(row.get("HIGH") or last),
            low=float(row.get("LOW") or last),
            open=float(open_price),
            prev_close=float(prev_close),
            change_pct=float(change_pct),
        )

    except Exception as e:
        logger.error(f"Failed to fetch quote for {secid}: {e}")
        return None


def fetch_all_futures_quotes() -> Dict[FuturesContract, FuturesQuote]:
    """Fetch quotes for all tracked futures contracts."""
    quotes = {}
    for contract in FuturesContract:
        quote = fetch_futures_quote(contract)
        if quote:
            quotes[contract] = quote
    return quotes


def calculate_margin(
    contract: FuturesContract,
    price: float,
    quantity: int,
) -> float:
    """
    Calculate required margin for a position.

    Args:
        contract: Futures contract
        price: Current price
        quantity: Number of contracts

    Returns:
        Required margin in RUB
    """
    spec = CONTRACT_SPECS.get(contract)
    if not spec:
        return 0.0

    contract_value = price * spec["lot_size"]
    margin_pct = spec["margin_pct"] / 100
    return contract_value * quantity * margin_pct


def calculate_position_pnl(
    contract: FuturesContract,
    direction: str,
    entry_price: float,
    current_price: float,
    quantity: int,
) -> float:
    """
    Calculate unrealized PnL for a position.

    Args:
        contract: Futures contract
        direction: "LONG" or "SHORT"
        entry_price: Entry price
        current_price: Current price
        quantity: Number of contracts

    Returns:
        Unrealized PnL in RUB
    """
    spec = CONTRACT_SPECS.get(contract)
    if not spec:
        return 0.0

    price_diff = current_price - entry_price
    if direction.upper() == "SHORT":
        price_diff = -price_diff

    ticks = price_diff / spec["tick_size"]
    pnl_per_contract = ticks * spec["tick_value"]
    return pnl_per_contract * quantity


def check_risk_limits(
    position: FuturesPosition,
    equity: float,
    limits: FuturesRiskLimits,
) -> Tuple[bool, List[str]]:
    """
    Check if position complies with risk limits.

    Args:
        position: Current position
        equity: Account equity
        limits: Risk limits configuration

    Returns:
        (is_compliant, list of violations)
    """
    violations = []

    # Check position size
    if position.quantity > limits.max_contracts_per_position:
        violations.append(
            f"Position size {position.quantity} exceeds limit {limits.max_contracts_per_position}"
        )

    # Check margin usage
    margin_usage_pct = (position.margin_used / equity * 100) if equity > 0 else 100
    if margin_usage_pct > limits.max_margin_usage_pct:
        violations.append(
            f"Margin usage {margin_usage_pct:.1f}% exceeds limit {limits.max_margin_usage_pct}%"
        )

    # Check loss limit
    loss_pct = (-position.unrealized_pnl / equity * 100) if equity > 0 else 0
    if loss_pct > limits.max_loss_per_trade_pct:
        violations.append(
            f"Position loss {loss_pct:.1f}% exceeds limit {limits.max_loss_per_trade_pct}%"
        )

    return len(violations) == 0, violations


def compute_stop_loss(
    contract: FuturesContract,
    direction: str,
    entry_price: float,
    risk_pct: float = 2.0,
) -> float:
    """
    Compute stop loss price based on risk percentage.

    Args:
        contract: Futures contract
        direction: "LONG" or "SHORT"
        entry_price: Entry price
        risk_pct: Max risk as percentage of entry

    Returns:
        Stop loss price
    """
    risk_amount = entry_price * (risk_pct / 100)

    if direction.upper() == "LONG":
        return entry_price - risk_amount
    else:
        return entry_price + risk_amount


def compute_take_profit(
    contract: FuturesContract,
    direction: str,
    entry_price: float,
    risk_reward: float = 2.0,
    risk_pct: float = 2.0,
) -> float:
    """
    Compute take profit price based on risk/reward ratio.

    Args:
        contract: Futures contract
        direction: "LONG" or "SHORT"
        entry_price: Entry price
        risk_reward: Risk/reward ratio
        risk_pct: Risk percentage used for stop loss

    Returns:
        Take profit price
    """
    risk_amount = entry_price * (risk_pct / 100)
    reward_amount = risk_amount * risk_reward

    if direction.upper() == "LONG":
        return entry_price + reward_amount
    else:
        return entry_price - reward_amount
