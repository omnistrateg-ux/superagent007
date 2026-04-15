"""
QUIK Data Source Implementation via QuikPy

Production-grade QUIK connection for microstructure data collection.
Requires: pip install quikpy (or manual QUIK Lua bridge)

Features:
- Trade tape with aggressor side (OnAllTrade callback)
- L2 orderbook depth (getQuoteLevel2)
- Best bid/ask quotes
- Open Interest updates (futures)
- Automatic reconnection
- Quality metrics

QUIK Setup:
1. Install QUIK terminal (requires broker account)
2. Enable "Trans2QUIK" or "QuikLua" in settings
3. Configure firewall for port 34130
4. Run this collector while QUIK is open

Alternative: LuaSockets bridge if QuikPy unavailable.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .microstructure_storage import msk_now, MSK

logger = logging.getLogger(__name__)


# =============================================================================
# QUIK PROTOCOL CONSTANTS
# =============================================================================

class QUIKMessageType(Enum):
    """QUIK Lua bridge message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    GET_QUOTE_LEVEL2 = "getQuoteLevel2"
    GET_PARAM_EX = "getParamEx"
    TRADE = "trade"
    QUOTE = "quote"
    DEPTH = "depth"
    OI = "oi"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


# QUIK class codes
CLASS_CODES = {
    # Futures (FORTS)
    "BR": "SPBFUT",
    "RI": "SPBFUT",
    "MX": "SPBFUT",
    "Si": "SPBFUT",
    "NG": "SPBFUT",
    "GD": "SPBFUT",
    # Stocks (TQBR)
    "SBER": "TQBR",
    "GAZP": "TQBR",
    "LKOH": "TQBR",
    "ROSN": "TQBR",
    "GMKN": "TQBR",
    "NVTK": "TQBR",
    "VTBR": "TQBR",
    "MGNT": "TQBR",
    "TATN": "TQBR",
    "PLZL": "TQBR",
}


@dataclass
class QUIKConfig:
    """QUIK connection configuration."""
    host: str = "127.0.0.1"
    port: int = 34130
    timeout_sec: float = 5.0
    reconnect_delay_sec: float = 10.0
    max_reconnect_attempts: int = 10
    depth_levels: int = 20
    snapshot_interval_ms: int = 500
    heartbeat_interval_sec: float = 30.0
    buffer_size: int = 65536


@dataclass
class QUIKStats:
    """QUIK connection statistics."""
    connected: bool = False
    connect_time: Optional[datetime] = None
    last_message_time: Optional[datetime] = None
    trades_received: int = 0
    quotes_received: int = 0
    depth_received: int = 0
    oi_received: int = 0
    errors_count: int = 0
    reconnect_count: int = 0
    unknown_side_count: int = 0

    def unknown_side_pct(self) -> float:
        """Percentage of trades with unknown side."""
        if self.trades_received == 0:
            return 0.0
        return (self.unknown_side_count / self.trades_received) * 100


# =============================================================================
# RAW DATA TYPES (matching collector types)
# =============================================================================

@dataclass
class RawTrade:
    ts: datetime
    ticker: str
    price: float
    qty: int
    side: str  # 'BUY', 'SELL', 'UNKNOWN'
    trade_id: Optional[str] = None


@dataclass
class RawQuote:
    ts: datetime
    ticker: str
    bid_price: Optional[float]
    bid_size: Optional[int]
    ask_price: Optional[float]
    ask_size: Optional[int]


@dataclass
class RawDepth:
    ts: datetime
    ticker: str
    bid_prices: List[float]
    bid_sizes: List[int]
    ask_prices: List[float]
    ask_sizes: List[int]


@dataclass
class RawOI:
    ts: datetime
    ticker: str
    open_interest: int


# =============================================================================
# QUIKPY WRAPPER
# =============================================================================

class QuikPyWrapper:
    """
    Wrapper for QuikPy library.

    QuikPy provides:
    - OnAllTrade callback for trade tape
    - getQuoteLevel2 for L2 orderbook
    - getParamEx for security parameters
    - subscribe/unsubscribe for real-time updates

    Install: pip install quikpy
    Docs: https://github.com/cia76/QuikPy
    """

    def __init__(self, config: QUIKConfig):
        self.config = config
        self._quik = None
        self._connected = False
        self._subscriptions: Dict[str, bool] = {}

    def connect(self) -> bool:
        """Connect to QUIK via QuikPy."""
        try:
            from QuikPy import QuikPy
            self._quik = QuikPy(
                host=self.config.host,
                port=self.config.port,
            )
            self._connected = True
            logger.info(f"QuikPy connected to {self.config.host}:{self.config.port}")
            return True

        except ImportError:
            logger.warning("QuikPy not installed. Run: pip install quikpy")
            return False

        except Exception as e:
            logger.error(f"QuikPy connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from QUIK."""
        if self._quik:
            try:
                # Unsubscribe all
                for ticker in list(self._subscriptions.keys()):
                    self.unsubscribe(ticker)
                self._quik.CloseConnectionAndThread()
            except Exception as e:
                logger.debug(f"Disconnect error: {e}")
            self._quik = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check connection status."""
        if not self._quik or not self._connected:
            return False
        try:
            # Simple check via getInfoParam
            result = self._quik.GetInfoParam("SERVERTIME")
            return result is not None
        except Exception:
            return False

    def subscribe_trades(self, ticker: str, callback: Callable[[RawTrade], None]) -> bool:
        """Subscribe to trade tape via OnAllTrade."""
        if not self._quik:
            return False

        class_code = CLASS_CODES.get(ticker, "TQBR")

        try:
            def on_trade(data: dict):
                """Convert QUIK trade to RawTrade."""
                try:
                    # QUIK OnAllTrade fields:
                    # tradeno, datetime, flags, price, qty, value
                    # flags bit 0: 0=sell, 1=buy (aggressor)

                    flags = int(data.get("flags", 0))
                    side = "BUY" if flags & 1 else "SELL"

                    # Parse datetime
                    trade_dt = data.get("datetime", {})
                    if isinstance(trade_dt, dict):
                        ts = datetime(
                            year=trade_dt.get("year", 2024),
                            month=trade_dt.get("month", 1),
                            day=trade_dt.get("day", 1),
                            hour=trade_dt.get("hour", 0),
                            minute=trade_dt.get("min", 0),
                            second=trade_dt.get("sec", 0),
                            microsecond=trade_dt.get("mcs", 0),
                            tzinfo=MSK,
                        )
                    else:
                        ts = msk_now()

                    trade = RawTrade(
                        ts=ts,
                        ticker=data.get("sec_code", ticker),
                        price=float(data.get("price", 0)),
                        qty=int(data.get("qty", 0)),
                        side=side,
                        trade_id=str(data.get("trade_num") or data.get("tradeno")),
                    )
                    callback(trade)

                except Exception as e:
                    logger.error(f"Trade parse error: {e}")

            # Subscribe via OnAllTrade
            self._quik.OnAllTrade = on_trade
            self._subscriptions[ticker] = True
            logger.info(f"Subscribed to trades: {ticker}")
            return True

        except Exception as e:
            logger.error(f"Trade subscription failed for {ticker}: {e}")
            return False

    def get_quote_level2(self, ticker: str) -> Optional[RawDepth]:
        """Get L2 orderbook snapshot."""
        if not self._quik:
            return None

        class_code = CLASS_CODES.get(ticker, "TQBR")

        try:
            result = self._quik.GetQuoteLevel2(class_code, ticker)

            if not result or "bid" not in result or "offer" not in result:
                return None

            bids = result.get("bid", [])
            asks = result.get("offer", [])

            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []

            for bid in bids[:self.config.depth_levels]:
                bid_prices.append(float(bid.get("price", 0)))
                bid_sizes.append(int(bid.get("quantity", 0)))

            for ask in asks[:self.config.depth_levels]:
                ask_prices.append(float(ask.get("price", 0)))
                ask_sizes.append(int(ask.get("quantity", 0)))

            return RawDepth(
                ts=msk_now(),
                ticker=ticker,
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )

        except Exception as e:
            logger.error(f"GetQuoteLevel2 failed for {ticker}: {e}")
            return None

    def get_param(self, ticker: str, param: str) -> Optional[Any]:
        """Get security parameter via getParamEx."""
        if not self._quik:
            return None

        class_code = CLASS_CODES.get(ticker, "TQBR")

        try:
            result = self._quik.GetParamEx(class_code, ticker, param)
            if result and "param_value" in result:
                return result["param_value"]
            return None

        except Exception as e:
            logger.debug(f"GetParamEx failed for {ticker}/{param}: {e}")
            return None

    def get_open_interest(self, ticker: str) -> Optional[int]:
        """Get Open Interest for futures."""
        value = self.get_param(ticker, "NUMCONTRACTS")
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
        return None

    def unsubscribe(self, ticker: str) -> None:
        """Unsubscribe from ticker."""
        self._subscriptions.pop(ticker, None)


# =============================================================================
# LUA SOCKET BRIDGE (Alternative)
# =============================================================================

class LuaSocketBridge:
    """
    Alternative QUIK connection via custom Lua socket server.

    Requires running Lua script in QUIK that:
    1. Listens on TCP port
    2. Sends JSON messages for trades/quotes
    3. Responds to depth/param requests

    Use this if QuikPy doesn't work with your QUIK version.
    """

    def __init__(self, config: QUIKConfig):
        self.config = config
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = b""

        # Callbacks
        self.on_trade: Optional[Callable[[RawTrade], None]] = None
        self.on_quote: Optional[Callable[[RawQuote], None]] = None
        self.on_depth: Optional[Callable[[RawDepth], None]] = None
        self.on_oi: Optional[Callable[[RawOI], None]] = None

    def connect(self) -> bool:
        """Connect to Lua socket server."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.config.timeout_sec)
            self._socket.connect((self.config.host, self.config.port))
            self._socket.setblocking(False)
            self._connected = True

            # Start receiver thread
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

            logger.info(f"Lua bridge connected to {self.config.host}:{self.config.port}")
            return True

        except socket.timeout:
            logger.error(f"Connection timeout to {self.config.host}:{self.config.port}")
            return False

        except ConnectionRefusedError:
            logger.error(f"Connection refused - is QUIK Lua server running?")
            return False

        except Exception as e:
            logger.error(f"Lua bridge connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Lua server."""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._socket is not None

    def _recv_loop(self) -> None:
        """Receive messages from Lua server."""
        while self._running and self._socket:
            try:
                data = self._socket.recv(self.config.buffer_size)
                if not data:
                    logger.warning("Lua server disconnected")
                    self._connected = False
                    break

                self._buffer += data
                self._process_buffer()

            except socket.error as e:
                if e.errno not in (10035, 11):  # WSAEWOULDBLOCK / EAGAIN
                    logger.error(f"Socket error: {e}")
                    self._connected = False
                    break
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Recv error: {e}")
                time.sleep(0.1)

    def _process_buffer(self) -> None:
        """Process received data buffer."""
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                msg = json.loads(line.decode("utf-8"))
                self._handle_message(msg)
            except json.JSONDecodeError:
                logger.debug(f"Invalid JSON: {line[:100]}")

    def _handle_message(self, msg: dict) -> None:
        """Handle incoming message from Lua."""
        msg_type = msg.get("type", "")

        if msg_type == "trade" and self.on_trade:
            trade = RawTrade(
                ts=datetime.fromisoformat(msg.get("ts", msk_now().isoformat())),
                ticker=msg.get("ticker", ""),
                price=float(msg.get("price", 0)),
                qty=int(msg.get("qty", 0)),
                side=msg.get("side", "UNKNOWN"),
                trade_id=msg.get("trade_id"),
            )
            self.on_trade(trade)

        elif msg_type == "quote" and self.on_quote:
            quote = RawQuote(
                ts=datetime.fromisoformat(msg.get("ts", msk_now().isoformat())),
                ticker=msg.get("ticker", ""),
                bid_price=msg.get("bid_price"),
                bid_size=msg.get("bid_size"),
                ask_price=msg.get("ask_price"),
                ask_size=msg.get("ask_size"),
            )
            self.on_quote(quote)

        elif msg_type == "depth" and self.on_depth:
            depth = RawDepth(
                ts=datetime.fromisoformat(msg.get("ts", msk_now().isoformat())),
                ticker=msg.get("ticker", ""),
                bid_prices=msg.get("bid_prices", []),
                bid_sizes=msg.get("bid_sizes", []),
                ask_prices=msg.get("ask_prices", []),
                ask_sizes=msg.get("ask_sizes", []),
            )
            self.on_depth(depth)

        elif msg_type == "oi" and self.on_oi:
            oi = RawOI(
                ts=datetime.fromisoformat(msg.get("ts", msk_now().isoformat())),
                ticker=msg.get("ticker", ""),
                open_interest=int(msg.get("oi", 0)),
            )
            self.on_oi(oi)

    def send_command(self, cmd: dict) -> bool:
        """Send command to Lua server."""
        if not self._socket or not self._connected:
            return False

        try:
            data = json.dumps(cmd).encode("utf-8") + b"\n"
            self._socket.sendall(data)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    def subscribe(self, ticker: str) -> bool:
        """Subscribe to ticker updates."""
        return self.send_command({
            "type": "subscribe",
            "ticker": ticker,
            "class_code": CLASS_CODES.get(ticker, "TQBR"),
        })

    def request_depth(self, ticker: str) -> bool:
        """Request L2 depth snapshot."""
        return self.send_command({
            "type": "get_depth",
            "ticker": ticker,
            "class_code": CLASS_CODES.get(ticker, "TQBR"),
            "levels": self.config.depth_levels,
        })


# =============================================================================
# UNIFIED QUIK DATA SOURCE
# =============================================================================

class QUIKDataSource:
    """
    Unified QUIK data source.

    Tries QuikPy first, falls back to Lua bridge.
    Provides reconnection logic and quality metrics.
    """

    def __init__(self, tickers: List[str], config: Optional[QUIKConfig] = None):
        self.tickers = tickers
        self.config = config or QUIKConfig()
        self.stats = QUIKStats()
        self.running = False

        # Backend
        self._quikpy: Optional[QuikPyWrapper] = None
        self._lua_bridge: Optional[LuaSocketBridge] = None
        self._backend: str = "none"

        # Threads
        self._snapshot_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_trade: Optional[Callable[[RawTrade], None]] = None
        self.on_quote: Optional[Callable[[RawQuote], None]] = None
        self.on_depth: Optional[Callable[[RawDepth], None]] = None
        self.on_oi: Optional[Callable[[RawOI], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None
        self.on_reconnect: Optional[Callable[[], None]] = None

        # OI tracking
        self._last_oi: Dict[str, int] = {}

    def connect(self) -> bool:
        """Connect to QUIK using available backend."""
        # Try QuikPy first
        logger.info("Attempting QuikPy connection...")
        self._quikpy = QuikPyWrapper(self.config)
        if self._quikpy.connect():
            self._backend = "quikpy"
            self.stats.connected = True
            self.stats.connect_time = msk_now()
            logger.info("Connected via QuikPy")
            return True

        # Fall back to Lua bridge
        logger.info("QuikPy failed, trying Lua bridge...")
        self._lua_bridge = LuaSocketBridge(self.config)
        if self._lua_bridge.connect():
            self._backend = "lua"
            self.stats.connected = True
            self.stats.connect_time = msk_now()
            logger.info("Connected via Lua bridge")
            return True

        logger.error("All QUIK connection methods failed")
        return False

    def disconnect(self) -> None:
        """Disconnect from QUIK."""
        self.running = False

        if self._snapshot_thread:
            self._snapshot_thread.join(timeout=2.0)

        if self._quikpy:
            self._quikpy.disconnect()
        if self._lua_bridge:
            self._lua_bridge.disconnect()

        self.stats.connected = False
        logger.info("Disconnected from QUIK")

    def start(self) -> None:
        """Start data collection."""
        if not self.stats.connected:
            logger.error("Not connected to QUIK")
            return

        self.running = True

        # Set up callbacks
        if self._backend == "quikpy" and self._quikpy:
            for ticker in self.tickers:
                self._quikpy.subscribe_trades(ticker, self._handle_trade)

        elif self._backend == "lua" and self._lua_bridge:
            self._lua_bridge.on_trade = self._handle_trade
            self._lua_bridge.on_quote = self._handle_quote
            self._lua_bridge.on_depth = self._handle_depth
            self._lua_bridge.on_oi = self._handle_oi

            for ticker in self.tickers:
                self._lua_bridge.subscribe(ticker)

        # Start snapshot thread for depth/quotes/OI
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_loop,
            daemon=True
        )
        self._snapshot_thread.start()

        # Start reconnect monitor
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True
        )
        self._reconnect_thread.start()

        logger.info(f"QUIK data collection started: {self.tickers}")

    def stop(self) -> None:
        """Stop data collection."""
        self.running = False
        self.disconnect()

    def _handle_trade(self, trade: RawTrade) -> None:
        """Handle incoming trade."""
        self.stats.trades_received += 1
        self.stats.last_message_time = msk_now()

        if trade.side == "UNKNOWN":
            self.stats.unknown_side_count += 1

        if self.on_trade:
            self.on_trade(trade)

    def _handle_quote(self, quote: RawQuote) -> None:
        """Handle incoming quote."""
        self.stats.quotes_received += 1
        self.stats.last_message_time = msk_now()

        if self.on_quote:
            self.on_quote(quote)

    def _handle_depth(self, depth: RawDepth) -> None:
        """Handle incoming depth."""
        self.stats.depth_received += 1
        self.stats.last_message_time = msk_now()

        if self.on_depth:
            self.on_depth(depth)

    def _handle_oi(self, oi: RawOI) -> None:
        """Handle incoming OI."""
        self.stats.oi_received += 1
        self.stats.last_message_time = msk_now()

        if self.on_oi:
            self.on_oi(oi)

    def _snapshot_loop(self) -> None:
        """Periodic snapshot collection."""
        interval = self.config.snapshot_interval_ms / 1000.0

        while self.running:
            try:
                ts = msk_now()

                for ticker in self.tickers:
                    # Get L2 depth
                    if self._backend == "quikpy" and self._quikpy:
                        depth = self._quikpy.get_quote_level2(ticker)
                        if depth:
                            self._handle_depth(depth)

                            # Extract BBO from depth
                            if depth.bid_prices and depth.ask_prices:
                                quote = RawQuote(
                                    ts=ts,
                                    ticker=ticker,
                                    bid_price=depth.bid_prices[0] if depth.bid_prices else None,
                                    bid_size=depth.bid_sizes[0] if depth.bid_sizes else None,
                                    ask_price=depth.ask_prices[0] if depth.ask_prices else None,
                                    ask_size=depth.ask_sizes[0] if depth.ask_sizes else None,
                                )
                                self._handle_quote(quote)

                        # Get OI for futures
                        if ticker in ("BR", "RI", "MX", "Si", "NG", "GD"):
                            oi_value = self._quikpy.get_open_interest(ticker)
                            if oi_value and oi_value != self._last_oi.get(ticker):
                                self._last_oi[ticker] = oi_value
                                oi = RawOI(ts=ts, ticker=ticker, open_interest=oi_value)
                                self._handle_oi(oi)

                    elif self._backend == "lua" and self._lua_bridge:
                        self._lua_bridge.request_depth(ticker)

            except Exception as e:
                logger.error(f"Snapshot error: {e}")
                self.stats.errors_count += 1
                if self.on_error:
                    self.on_error("snapshot", e)

            time.sleep(interval)

    def _reconnect_loop(self) -> None:
        """Monitor connection and reconnect if needed."""
        while self.running:
            time.sleep(self.config.heartbeat_interval_sec)

            if not self.running:
                break

            # Check connection
            is_alive = False
            if self._backend == "quikpy" and self._quikpy:
                is_alive = self._quikpy.is_connected()
            elif self._backend == "lua" and self._lua_bridge:
                is_alive = self._lua_bridge.is_connected()

            if not is_alive and self.stats.connected:
                logger.warning("QUIK connection lost, attempting reconnect...")
                self.stats.connected = False
                self._attempt_reconnect()

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to QUIK."""
        for attempt in range(self.config.max_reconnect_attempts):
            if not self.running:
                break

            logger.info(f"Reconnect attempt {attempt + 1}/{self.config.max_reconnect_attempts}")

            if self.connect():
                self.stats.reconnect_count += 1
                if self.on_reconnect:
                    self.on_reconnect()
                # Resubscribe
                self.start()
                return

            time.sleep(self.config.reconnect_delay_sec)

        logger.error("Max reconnect attempts reached")
        self.running = False

    @property
    def provides_aggressor_side(self) -> bool:
        """QUIK provides aggressor side via trade flags."""
        return True

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics."""
        return {
            "connected": self.stats.connected,
            "backend": self._backend,
            "connect_time": self.stats.connect_time.isoformat() if self.stats.connect_time else None,
            "last_message": self.stats.last_message_time.isoformat() if self.stats.last_message_time else None,
            "trades_received": self.stats.trades_received,
            "quotes_received": self.stats.quotes_received,
            "depth_received": self.stats.depth_received,
            "oi_received": self.stats.oi_received,
            "unknown_side_pct": self.stats.unknown_side_pct(),
            "errors_count": self.stats.errors_count,
            "reconnect_count": self.stats.reconnect_count,
        }


# =============================================================================
# QUIK LUA SCRIPT (for reference)
# =============================================================================

QUIK_LUA_SERVER_SCRIPT = '''
-- QUIK Lua Socket Server for Python data collection
-- Place this in QUIK scripts folder and run from QUIK menu

local socket = require("socket")
local json = require("json")  -- or cjson

local HOST = "127.0.0.1"
local PORT = 34130

local server = nil
local client = nil
local subscriptions = {}

function main()
    server = socket.bind(HOST, PORT)
    if not server then
        message("Failed to bind to " .. HOST .. ":" .. PORT)
        return
    end

    server:settimeout(0.1)
    message("Lua server listening on " .. HOST .. ":" .. PORT)

    while true do
        -- Accept new connections
        local new_client = server:accept()
        if new_client then
            client = new_client
            client:settimeout(0)
            message("Python client connected")
        end

        -- Process commands from client
        if client then
            local line, err = client:receive("*l")
            if line then
                processCommand(json.decode(line))
            elseif err == "closed" then
                client = nil
                message("Python client disconnected")
            end
        end

        sleep(10)
    end
end

function OnAllTrade(trade)
    if not client then return end

    local ticker = trade.sec_code
    if not subscriptions[ticker] then return end

    -- Determine aggressor side from flags
    local side = "UNKNOWN"
    if trade.flags then
        if bit.band(trade.flags, 1) == 1 then
            side = "BUY"
        else
            side = "SELL"
        end
    end

    local msg = {
        type = "trade",
        ts = os.date("!%Y-%m-%dT%H:%M:%S") .. "+03:00",
        ticker = ticker,
        price = trade.price,
        qty = trade.qty,
        side = side,
        trade_id = tostring(trade.trade_num),
    }

    sendToClient(msg)
end

function processCommand(cmd)
    if cmd.type == "subscribe" then
        subscriptions[cmd.ticker] = true
        message("Subscribed: " .. cmd.ticker)

    elseif cmd.type == "unsubscribe" then
        subscriptions[cmd.ticker] = nil

    elseif cmd.type == "get_depth" then
        local depth = getQuoteLevel2(cmd.class_code, cmd.ticker)
        if depth then
            local msg = {
                type = "depth",
                ts = os.date("!%Y-%m-%dT%H:%M:%S") .. "+03:00",
                ticker = cmd.ticker,
                bid_prices = {},
                bid_sizes = {},
                ask_prices = {},
                ask_sizes = {},
            }

            for i, bid in ipairs(depth.bid or {}) do
                if i <= (cmd.levels or 20) then
                    table.insert(msg.bid_prices, bid.price)
                    table.insert(msg.bid_sizes, bid.quantity)
                end
            end

            for i, ask in ipairs(depth.offer or {}) do
                if i <= (cmd.levels or 20) then
                    table.insert(msg.ask_prices, ask.price)
                    table.insert(msg.ask_sizes, ask.quantity)
                end
            end

            sendToClient(msg)
        end
    end
end

function sendToClient(msg)
    if client then
        local data = json.encode(msg) .. "\\n"
        client:send(data)
    end
end
'''


def get_lua_server_script() -> str:
    """Get Lua server script for QUIK."""
    return QUIK_LUA_SERVER_SCRIPT
