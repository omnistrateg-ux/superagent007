#!/usr/bin/env python3
"""
Paper Trading — Фьючерсы FORTS (MOEX)
Mean reversion + новостной фильтр.
Контракты: Si, BR, RI, NG, GD, MX
"""
import json, os, time, logging, math, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("futures_paper")

# ── BKS Live bridge (safe import) ──
try:
    from moex_agent.bks_live import is_live_enabled, submit_futures_market_order, build_client_order_id
    _HAS_LIVE_BRIDGE = True
except ImportError:
    _HAS_LIVE_BRIDGE = False

# ── SmartFilter + RegimeDetector (Phase 3) ──
try:
    from moex_agent.smart_filters import SmartFilter, get_smart_filter
    from moex_agent.regime import RegimeDetector, filter_signal_by_regime
    from moex_agent.calendar_features import CalendarFeatures, get_calendar_features
    _HAS_SMART_FILTER = True
except ImportError:
    _HAS_SMART_FILTER = False

_live_open_futures = {}  # secid -> live quantity actually sent on open


def _try_live_futures_order(*, secid: str, side: str, quantity: int, trade_id, action: str = "open"):
    """Mirror order to BKS через Position Manager. Never raises."""
    if not _HAS_LIVE_BRIDGE:
        return
    if not secid or not secid.strip():
        log.warning(f"LIVE BKS {action}: empty secid, skipping")
        return
    try:
        if not is_live_enabled():
            return

        from moex_agent.bks_position_manager import get_bks_pm
        pm = get_bks_pm()

        if action == "open":
            from moex_agent.live_config import calc_futures_quantity
            live_qty = calc_futures_quantity(quantity, secid)
            if live_qty <= 0:
                return
            
            # safe_open через Position Manager
            # Robust side conversion: handle various formats (BUY/SELL, 1/2, LONG/SHORT)
            side_upper = str(side).upper()
            if side_upper in ("1", "BUY", "LONG"):
                side_int = 1
            elif side_upper in ("2", "SELL", "SHORT"):
                side_int = 2
            else:
                log.warning(f"Unknown side '{side}', defaulting to SELL (2)")
                side_int = 2
            result = pm.safe_open(secid, side_int, live_qty)
            if result.get('ok'):
                log.info(f"LIVE BKS OPEN {side} {secid} x{live_qty} (paper={quantity}) ✅")
            else:
                log.warning(f"LIVE BKS OPEN {side} {secid} x{live_qty} FAILED: {result.get('reason')}")
            return

        elif action == "close":
            # safe_close через Position Manager
            result = pm.safe_close(secid)
            if result.get('ok'):
                log.info(f"LIVE BKS CLOSE {secid} ✅ (closed {result.get('closed_qty', '?')})")
            else:
                log.warning(f"LIVE BKS CLOSE {secid} FAILED: {result.get('reason')}")
            return

        else:
            live_qty = min(quantity, 1)

        coid = build_client_order_id("futures", action, trade_id, secid)
        # Получаем текущую цену для лимитного ордера
        try:
            import urllib.request as _ur
            _mdata = json.loads(_ur.urlopen(f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}.json?iss.meta=off&iss.only=marketdata&marketdata.columns=LAST", timeout=5).read())
            _rows = _mdata.get("marketdata", {}).get("data", [])
            _current_price = float(_rows[0][0]) if _rows and _rows[0][0] else 0
        except Exception:
            _current_price = 0
        result = submit_futures_market_order(ticker=secid, side=side, quantity=live_qty, client_order_id=coid, price=_current_price)
        if result.get("skipped"):
            return
        log.info(f"LIVE BKS {action.upper()} {side} {secid} x{live_qty} (paper={quantity}) → {result.get('broker', {}).get('orderId', '?')}")
        send_tg(f"🔴 <b>LIVE BKS</b> {action.upper()} {side} {secid} x{live_qty}")
    except Exception as exc:
        log.warning(f"LIVE BKS {action} {secid} FAILED: {exc}")
        send_tg(f"⚠️ <b>LIVE BKS ОШИБКА</b> {action} {secid}: {exc}")

MSK = timezone(timedelta(hours=3))

# ── Config ───────────────────────────────────────────────
INITIAL_BALANCE = 10_000_000
MAX_MARGIN_PCT = float(os.environ.get("MAX_MARGIN_PCT", "50"))   # Max 50% equity in margin
MAX_CONTRACTS = int(os.environ.get("MAX_CONTRACTS", "20"))       # Per position
DAILY_LOSS_LIMIT = float(os.environ.get("FUTURES_DAILY_LOSS_LIMIT", "30000"))  # Stop trading if daily loss > 30k
MAX_LOSS_PER_TRADE = float(os.environ.get("FUTURES_MAX_LOSS_PER_TRADE", "15000"))  # Emergency close if single trade loss > 15k
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "6"))
STOP_PCT = float(os.environ.get("STOP_PCT", "2.0"))             # Default stop % (widened to reduce stop-outs)
TARGET_RR = float(os.environ.get("TARGET_RR", "2.0"))           # Risk:Reward
MIN_RR = float(os.environ.get("MIN_RR", "1.0"))                 # Minimum risk:reward to allow entry
TIME_STOP_MINUTES = int(os.environ.get("TIME_STOP_MINUTES", "120"))
SIDE_MODE = os.environ.get("SIDE_MODE", "both")  # both, long_only, short_only
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "30"))      # seconds

# Trailing 2.0 — "Let Winners Run"
# Tiered trailing: earlier activation, adaptive cushion
TRAIL_TIERS = {
    # progress_pct: (cushion_pct, name)
    15: (50, "ЗАЩИТА"),       # 15% к цели → стоп держит 50% прибыли (breakeven+)
    30: (35, "ТРЕЙЛ_1"),      # 30% к цели → держим 65% прибыли
    50: (25, "ТРЕЙЛ_2"),      # 50% к цели → держим 75% прибыли
    70: (15, "ТРЕЙЛ_3"),      # 70% к цели → держим 85% прибыли
    90: (8,  "ЦЕЛЬ_БЛИЗКО"),  # 90% к цели → держим 92% прибыли
}
TRAIL_ACTIVATE_PCT = 15   # Активация трейлинга раньше (было 30)
TRAIL_STEP_PCT = 30       # Fallback cushion

# Partial Take Profit — фиксация части прибыли (v2.1: менее агрессивно)
PARTIAL_TAKE_ENABLED = True
PARTIAL_TAKE_LEVELS = [
    (60, 30),   # При 60% к цели → закрыть 30% позиции
    (80, 30),   # При 80% к цели → ещё 30% (осталось 40%)
]

# Kill Losers Fast — ступенчатое закрытие убыточных (v2.1: без микро-тира)
LOSER_TIERS = {
    5000:  50,   # Убыток 5k₽ → закрыть 50% позиции
    10000: 75,   # Убыток 10k₽ → закрыть ещё 25% (осталось 25%)
    # MAX_LOSS_PER_TRADE закрывает остаток
}

# Smart Time Stop (v2.1: DISABLED — держит лосеры слишком долго)
SMART_TIME_STOP = False

# Stop Multiplier — расширенные стопы для волатильных контрактов (v2)
STOP_MULTIPLIER = {
    "BR": 1.3,   # Нефть: +30% к стопу (было -134k на стопах)
    "NG": 1.2,   # Газ: +20% к стопу
    "RI": 1.0,   # РТС: стандарт
    "MX": 1.0,   # MOEX: стандарт
}

# Hourly Tactics — адаптивный min_dev по времени суток (v2)
# Формат: (hour_start, hour_end): min_dev_multiplier
HOURLY_TACTICS = {
    (7, 9):   0.7,    # Утро: высокая волатильность, снижаем порог
    (9, 10):  0.85,   # Открытие: умеренно
    (10, 14): 1.0,    # День: стандарт
    (14, 16): 1.3,    # Обед: низкая ликвидность, выше порог
    (16, 19): 0.9,    # Вечер: хорошо для mean reversion
    (19, 20): 1.0,    # Переход к вечерней сессии
    (20, 23): 1.5,    # Поздний вечер: осторожнее (или None для блокировки)
}

def get_hourly_min_dev_mult() -> float:
    """Get min_dev multiplier based on current hour."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=3)))  # MSK
    hour = now.hour
    for (h_start, h_end), mult in HOURLY_TACTICS.items():
        if h_start <= hour < h_end:
            return mult
    return 1.0  # Default

# News
NEWS_ENABLED = True
NEWS_ALERT_MAX_AGE_MIN = int(os.environ.get("NEWS_ALERT_MAX_AGE_MIN", "30"))

# Telegram
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "120171956")

# Paths
DATA_DIR = Path("/tmp/superagent007/data")
STATE_FILE = DATA_DIR / "futures_state.json"
EQUITY_FILE = DATA_DIR / "futures_equity.json"

# ── Contract specs ───────────────────────────────────────
# v2: оптимизация по бэктесту 2026-04-09
# Изменения: min_dev снижен (0.5 → 0.3), MX both directions, stop_mult в STOP_MULTIPLIER
CONTRACTS = {
    # Si: PF 0.99 — ОТКЛЮЧЁН (edge нет, Sharpe -0.05)
    # "Si": {"name": "💵 Доллар",  "base": "Si", "lot": 1000, "tick": 1.0,   "tick_val": 1.0,  "margin_pct": 10,
    #         "min_dev": 0.3, "side_mode": "both", "time_stop_bars": 4},

    # BR: min_dev 0.5 → 0.3, stop_mult 1.3 в STOP_MULTIPLIER (исправляет -134k на стопах)
    "BR": {"name": "🛢 Нефть Brent",   "base": "BR", "lot": 10,   "tick": 0.01,  "tick_val": 6.55, "margin_pct": 15,
            "min_dev": 0.3, "side_mode": "both", "time_stop_bars": 3, "min_rr": 0.3, "max_dev": 3.0},

    # RI: min_dev 0.5 → 0.3, Sharpe 2.54 — лучшее качество
    "RI": {"name": "📈 Индекс РТС",     "base": "RI", "lot": 1,    "tick": 10.0,  "tick_val": 10.0, "margin_pct": 12,
            "min_dev": 0.3, "side_mode": "both", "time_stop_bars": 3, "min_rr": 0.3, "max_dev": 3.0},

    # NG: min_dev 0.5 → 0.35, stop_mult 1.2 в STOP_MULTIPLIER
    "NG": {"name": "🔥 Природный газ",     "base": "NG", "lot": 100,  "tick": 0.001, "tick_val": 6.55, "margin_pct": 20,
            "min_dev": 0.35, "side_mode": "both", "time_stop_bars": 4, "min_rr": 0.3, "max_dev": 3.0},

    # GD: ОТКЛЮЧЁН
    # "GD": {"name": "🥇 Золото",  "base": "GD", "lot": 1,    "tick": 0.1,   "tick_val": 6.55, "margin_pct": 10,
    #         "min_dev": 0.8, "side_mode": "long_only", "time_stop_bars": 2},

    # MX: BEST performer — WR 80%. Теперь BOTH: short min_dev 0.15, long min_dev_long 0.25
    "MX": {"name": "📊 Индекс Мосбиржи",    "base": "MX", "lot": 1,    "tick": 1.0,   "tick_val": 1.0,  "margin_pct": 12,
            "min_dev": 0.15, "min_dev_long": 0.25, "side_mode": "both", "time_stop_bars": 2, "min_rr": 0.3, "max_dev": 2.0},
}

# FORTS extended session window (morning + main + evening)
FUTURES_SESSION_START = (6, 50)
FUTURES_SESSION_END = (23, 50)
FORCE_FLAT_AFTER = (23, 45)
NO_NEW_AFTER = (23, 40)

# ── Telegram ─────────────────────────────────────────────

def send_tg(text):
    if not BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = json.dumps({"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.error(f"TG error: {e}")


# ── TG Commands ──────────────────────────────────────────

_tg_last_update_id = 0

def _tg_init_offset():
    global _tg_last_update_id
    if not BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?limit=1&offset=-1"
        resp = json.loads(urllib.request.urlopen(url, timeout=5).read())
        if resp.get("result"):
            _tg_last_update_id = resp["result"][-1]["update_id"]
    except Exception:
        pass

def tg_get_updates():
    global _tg_last_update_id
    if not BOT_TOKEN:
        return []
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={_tg_last_update_id+1}&timeout=1&limit=10"
        resp = json.loads(urllib.request.urlopen(url, timeout=5).read())
        updates = resp.get("result", [])
        if updates:
            _tg_last_update_id = updates[-1]["update_id"]
        return updates
    except Exception:
        return []


# ── MOEX ISS API ─────────────────────────────────────────

_contract_cache = {}  # base -> (secid, timestamp)

def find_contract(base):
    """Find nearest (most liquid) contract for base symbol."""
    now = time.time()
    if base in _contract_cache and now - _contract_cache[base][1] < 3600:
        return _contract_cache[base][0]
    try:
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
            f"?iss.meta=off&iss.only=marketdata&marketdata.columns=SECID,OPENPOSITION,VOLTODAY"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        rows = data.get("marketdata", {}).get("data", [])
        candidates = [(r[0], r[1] or 0, r[2] or 0) for r in rows if r[0] and r[0].startswith(base)]
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if candidates:
            secid = candidates[0][0]
            _contract_cache[base] = (secid, now)
            return secid
    except Exception as e:
        log.debug(f"Contract lookup error {base}: {e}")
    return None


def fetch_futures_market():
    """Fetch all futures quotes. Returns {base: {secid, last, open, high, low, prev, volume, oi, change_pct}}"""
    market = {}
    try:
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
            f"?iss.meta=off&iss.only=marketdata"
            f"&marketdata.columns=SECID,LAST,OPENPOSITION,VOLTODAY,OPEN,HIGH,LOW,WAPRICE"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        cols = data.get("marketdata", {}).get("columns", [])
        rows = data.get("marketdata", {}).get("data", [])

        # Build column index map
        ci = {c: i for i, c in enumerate(cols)}

        for base, spec in CONTRACTS.items():
            # Find most liquid contract by OI
            candidates = []
            for r in rows:
                secid = r[ci.get("SECID", 0)]
                if secid and secid.startswith(base):
                    oi = r[ci.get("OPENPOSITION", 2)] or 0
                    vol = r[ci.get("VOLTODAY", 3)] or 0
                    candidates.append((secid, oi, vol, r))

            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            if not candidates:
                continue

            secid, oi, vol, row = candidates[0]
            last = row[ci.get("LAST", 1)] or 0
            if not last:
                continue

            open_price = row[ci.get("OPEN", 4)] or last
            high = row[ci.get("HIGH", 5)] or last
            low = row[ci.get("LOW", 6)] or last

            # WAP = VWAP approximation: (H+L+C)/3
            wap = round((high + low + last) / 3, 4)
            change_pct = ((last - open_price) / open_price * 100) if open_price else 0

            market[base] = {
                "secid": secid,
                "last": float(last),
                "wap": float(wap),
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "prev": float(open_price),
                "volume": int(vol),
                "oi": int(oi),
                "change_pct": round(change_pct, 2),
            }
    except Exception as e:
        log.error(f"Market fetch error: {e}")
    return market


def fetch_futures_candles(secid, interval=60, count=20):
    """Fetch recent candles for ATR calculation.

    MOEX ISS candles are paginated from older history by default, so request
    a recent date window explicitly; otherwise EMA init can accidentally use
    stale historical candles for active contracts.
    """
    try:
        now = datetime.now(MSK)
        lookback_days = max(7, math.ceil(count / 24) + 3) if interval >= 60 else 3
        date_from = (now - timedelta(days=lookback_days)).date().isoformat()
        date_till = now.date().isoformat()
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}/candles.json"
            f"?interval={interval}&from={date_from}&till={date_till}"
            f"&iss.meta=off&candles.columns=open,close,high,low,begin"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        candles = data.get("candles", {}).get("data", [])
        return candles[-count:] if candles else []
    except Exception:
        return []


def calc_atr(candles):
    """Calculate ATR from candles."""
    if len(candles) < 2:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i][2], candles[i][3], candles[i-1][1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


# ── Macro Context (correlations: USD/RUB, IMOEX) ─────────

_macro_cache = {"ts": 0.0}
MACRO_REFRESH_INTERVAL = 600  # 10 min


def fetch_macro_context() -> dict:
    """Fetch IMOEX and USD/RUB for cross-asset correlation filter.
    
    Returns:
        {"imoex_trend": "UP"/"DOWN"/"FLAT", "imoex_change_pct": float,
         "usdrub_trend": "UP"/"DOWN"/"FLAT", "usdrub_change_pct": float,
         "ts": str}
    """
    import time as _time

    if (_time.time() - _macro_cache.get("ts", 0)) < MACRO_REFRESH_INTERVAL and "imoex_trend" in _macro_cache:
        return _macro_cache

    result = {"ts": datetime.now(MSK).isoformat()}

    # IMOEX: today's candle
    try:
        today = datetime.now(MSK).date()
        yesterday = (today - timedelta(days=3)).isoformat()
        url = (
            f"https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json"
            f"?from={yesterday}&till={today.isoformat()}&interval=60&iss.meta=off"
            f"&candles.columns=open,close,begin"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=8).read())
        candles = data.get("candles", {}).get("data", [])
        if candles and len(candles) >= 2:
            first_open = candles[-min(6, len(candles))][0]  # Open ~6h ago
            last_close = candles[-1][1]
            pct = ((last_close - first_open) / first_open) * 100 if first_open else 0
            result["imoex_change_pct"] = round(pct, 2)
            result["imoex_trend"] = "UP" if pct > 0.3 else ("DOWN" if pct < -0.3 else "FLAT")
        else:
            result["imoex_trend"] = "FLAT"
            result["imoex_change_pct"] = 0
    except Exception:
        result["imoex_trend"] = "FLAT"
        result["imoex_change_pct"] = 0

    # USD/RUB (Si futures or CETS)
    try:
        today = datetime.now(MSK).date()
        yesterday = (today - timedelta(days=3)).isoformat()
        url = (
            f"https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/securities/USD000UTSTOM/candles.json"
            f"?from={yesterday}&till={today.isoformat()}&interval=60&iss.meta=off"
            f"&candles.columns=open,close,begin"
        )
        data = json.loads(urllib.request.urlopen(url, timeout=8).read())
        candles = data.get("candles", {}).get("data", [])
        if candles and len(candles) >= 2:
            first_open = candles[-min(6, len(candles))][0]
            last_close = candles[-1][1]
            pct = ((last_close - first_open) / first_open) * 100 if first_open else 0
            result["usdrub_change_pct"] = round(pct, 2)
            result["usdrub_trend"] = "UP" if pct > 0.2 else ("DOWN" if pct < -0.2 else "FLAT")
        else:
            result["usdrub_trend"] = "FLAT"
            result["usdrub_change_pct"] = 0
    except Exception:
        result["usdrub_trend"] = "FLAT"
        result["usdrub_change_pct"] = 0

    _macro_cache.update(result)
    _macro_cache["ts"] = _time.time()
    log.info(f"MACRO: IMOEX {result['imoex_trend']} ({result['imoex_change_pct']:+.2f}%), USD/RUB {result['usdrub_trend']} ({result.get('usdrub_change_pct', 0):+.2f}%)")
    return result


# Correlation rules:
# - BR correlates positively with oil, negatively with RUB strengthening
# - RI/MX correlate positively with IMOEX
# - When USD/RUB goes UP (ruble weakens), oil-exporters benefit
MACRO_CORRELATION = {
    "BR": {"contra_imoex": False, "contra_usdrub": True},   # Oil: if USD weakens (DOWN), oil often drops too
    "NG": {"contra_imoex": False, "contra_usdrub": False},   # Gas: less correlated
    "RI": {"contra_imoex": True, "contra_usdrub": False},    # RTS index: follows IMOEX
    "MX": {"contra_imoex": True, "contra_usdrub": False},    # MOEX index: follows IMOEX
}


def macro_filter(base: str, direction: str, macro: dict) -> tuple[bool, str]:
    """Check if macro context supports the trade direction.
    
    Returns: (allowed: bool, reason: str)
    """
    rules = MACRO_CORRELATION.get(base)
    if not rules:
        return True, ""

    imoex = macro.get("imoex_trend", "FLAT")
    usdrub = macro.get("usdrub_trend", "FLAT")

    # IMOEX correlation
    if rules.get("contra_imoex"):
        # RI/MX follow IMOEX: don't go SHORT if market is UP, don't go LONG if DOWN
        if imoex == "UP" and direction == "SHORT":
            return False, f"macro_block: IMOEX UP, skip {base} SHORT"
        if imoex == "DOWN" and direction == "LONG":
            return False, f"macro_block: IMOEX DOWN, skip {base} LONG"

    # USD/RUB correlation for BR
    if rules.get("contra_usdrub"):
        # When USD/RUB UP (ruble weakens), Brent in RUB goes up → don't SHORT
        if usdrub == "UP" and direction == "SHORT":
            return False, f"macro_block: USD/RUB UP (рубль слабеет), skip {base} SHORT"
        if usdrub == "DOWN" and direction == "LONG":
            return False, f"macro_block: USD/RUB DOWN (рубль крепнет), skip {base} LONG"

    return True, ""


# ── News ─────────────────────────────────────────────────

# ── EMA Cache (hourly candles only, no running drift) ─────────

_ema_cache = {}  # base -> {"ema": float, "count": int, "last_candle_ts": str, "refreshed_at": float}
EMA_PERIOD = 20
EMA_REFRESH_INTERVAL = 300  # Refresh from candles every 5 min (not every tick)


def _compute_ema_from_closes(closes: list, period: int = EMA_PERIOD) -> float:
    """Pure EMA calculation from a list of close prices."""
    if not closes:
        return 0.0
    ema = closes[0]
    k = 2 / (period + 1)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def refresh_ema_from_candles(base, secid, period=EMA_PERIOD):
    """Refresh EMA exclusively from hourly candles. No running updates from ticks.
    
    This is the ONLY way EMA gets updated. Called periodically (every 5 min).
    Returns: EMA value or None if not enough data.
    """
    import time as _time

    # Rate-limit: don't re-fetch if we just refreshed
    cache = _ema_cache.get(base)
    if cache and (_time.time() - cache.get("refreshed_at", 0)) < EMA_REFRESH_INTERVAL:
        return cache["ema"]

    candles = fetch_futures_candles(secid, interval=60, count=period * 3)
    if len(candles) < period:
        log.debug(f"EMA {base}/{secid}: only {len(candles)} candles, need {period}")
        return cache["ema"] if cache else None

    # MOEX can return stale candles for expired/rolled contracts. Don't use them.
    try:
        last_begin = candles[-1][4]
        last_dt = datetime.fromisoformat(str(last_begin)).replace(tzinfo=MSK)
        if (datetime.now(MSK) - last_dt).total_seconds() > 72 * 3600:
            log.warning(f"STALE candles for {base}/{secid}: last={last_begin}, skip EMA refresh")
            return cache["ema"] if cache else None
    except Exception:
        pass

    closes = [c[1] for c in candles if c[1]]
    if len(closes) < period:
        return cache["ema"] if cache else None

    ema = _compute_ema_from_closes(closes, period)
    last_candle_ts = str(candles[-1][4]) if candles else ""

    _ema_cache[base] = {
        "ema": ema,
        "count": len(closes),
        "last_candle_ts": last_candle_ts,
        "refreshed_at": _time.time(),
    }
    log.info(f"EMA {base}: {ema:.4f} (from {len(closes)} hourly candles, last={last_candle_ts})")
    return ema


def get_ema(base):
    """Get current EMA for a contract (from cache, hourly-candle-based)."""
    if base in _ema_cache:
        return _ema_cache[base]["ema"]
    return None


def check_news():
    if not NEWS_ENABLED:
        return 1.0, None, {}
    try:
        from moex_agent.news import get_systemic_news_impact, get_futures_sentiment
        mult, kw = get_systemic_news_impact()  # only CRITICAL stops all futures
        sentiment = get_futures_sentiment()    # contract-specific news handled per contract
        return mult, kw, sentiment
    except Exception:
        return 1.0, None, {}


# ── Engine ───────────────────────────────────────────────

class FuturesEngine:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.positions = []
        self.closed = []
        self.trade_count = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_equity = INITIAL_BALANCE
        self.equity_curve = []
        self.last_regime = "NEUTRAL"
        self.news_mult = 1.0
        self.news_sentiment = {}
        self._last_futures_news_alerts = {}
        self._last_global_news_alert = None
        self._disabled_sides = {}
        # Load persisted side_size_mult
        try:
            _ssm_file = Path(DATA_DIR) / "side_size_mult.json"
            if _ssm_file.exists():
                import json as _json
                _ssm = _json.loads(_ssm_file.read_text())
                self._side_size_mult = {tuple(k.split('_')): v for k, v in _ssm.items()}
                log.info(f"Loaded side_size_mult: {self._side_size_mult}")
            else:
                self._side_size_mult = {}
        except Exception:
            self._side_size_mult = {}
        self._last_self_optimize = 0.0  # timestamp
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                s = json.loads(STATE_FILE.read_text())
                self.balance = s.get("balance", INITIAL_BALANCE)
                self.positions = s.get("positions", [])
                self.closed = s.get("closed", [])
                self.trade_count = s.get("trade_count", len(self.closed))
                self.daily_trades = s.get("daily_trades", 0)
                self.daily_pnl = s.get("daily_pnl", 0.0)
                self.peak_equity = s.get("peak_equity", INITIAL_BALANCE)
                log.info(f"Loaded: balance={self.balance:,.0f}, positions={len(self.positions)}, closed={len(self.closed)}")
            except Exception as e:
                log.error(f"State load error: {e}")
        if EQUITY_FILE.exists():
            try:
                self.equity_curve = json.loads(EQUITY_FILE.read_text())
            except Exception:
                self.equity_curve = []

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "balance": self.balance,
            "positions": self.positions,
            "closed": self.closed,
            "trade_count": self.trade_count,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "peak_equity": self.peak_equity,
        }, indent=2, ensure_ascii=False))

    def _save_equity(self):
        EQUITY_FILE.write_text(json.dumps(self.equity_curve[-2000:], ensure_ascii=False))

    def self_optimize(self):
        """Adjust position size per contract based on performance.
        
        НЕ ОТКЛЮЧАЕТ — только уменьшает/увеличивает размер.
        WR < 30% → size × 0.3 (минимум, не ноль)
        WR 30-50% → size × 0.5
        WR 50-65% → size × 1.0 (нормальный)
        WR > 65% → size × 1.3 (увеличить)
        
        Учитывает PnL: если PnL положительный при низком WR → не уменьшает.
        Runs every 30 min.
        """
        import time as _time
        now = _time.time()
        if now - self._last_self_optimize < 1800:  # 30 min
            return
        self._last_self_optimize = now

        MIN_TRADES = 5
        RECENT_WINDOW = 20  # Look at last N trades

        # Compute WR + PnL per (base, direction)
        stats = {}  # (base, direction) -> {"wins": int, "total": int, "pnl": float}
        recent_stats = {}

        for trade in self.closed:
            base = trade.get("base", "")
            direction = trade.get("direction", "")
            if not base or not direction:
                continue
            key = (base, direction)
            s = stats.setdefault(key, {"wins": 0, "total": 0, "pnl": 0})
            s["total"] += 1
            pnl = float(trade.get("pnl_rub", trade.get("pnl", 0)) or 0)
            s["pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1

        # Recent window
        for trade in self.closed[-200:]:
            base = trade.get("base", "")
            direction = trade.get("direction", "")
            if not base or not direction:
                continue
            key = (base, direction)
            r = recent_stats.setdefault(key, {"wins": 0, "total": 0, "trades": []})
            r["trades"].append(trade)

        for key, r in recent_stats.items():
            # Keep only last RECENT_WINDOW
            last_n = r["trades"][-RECENT_WINDOW:]
            r["total"] = len(last_n)
            r["wins"] = sum(1 for t in last_n if float(t.get("pnl_rub", t.get("pnl", 0)) or 0) > 0)

        changed = False
        for (base, direction), s in stats.items():
            if s["total"] < MIN_TRADES:
                continue
            wr = s["wins"] / s["total"] * 100
            pnl = s["pnl"]

            # Determine size multiplier based on WR + PnL
            if pnl > 0:
                # Profitable — don't reduce even if WR is low (e.g. BR LONG WR 27% but +22k)
                if wr >= 65:
                    mult = 1.3
                else:
                    mult = 1.0
            else:
                # Losing money
                if wr < 30:
                    mult = 0.3
                elif wr < 50:
                    mult = 0.5
                else:
                    mult = 0.7

            old_mult = self._side_size_mult.get((base, direction), 1.0)
            if abs(old_mult - mult) > 0.05:
                self._side_size_mult[(base, direction)] = mult
                log.info(f"SELF-OPTIMIZE: {base} {direction} size_mult {old_mult:.1f} → {mult:.1f} (WR={wr:.0f}%, PnL={pnl:+,.0f}₽)")
                changed = True

        if changed:
            lines = ["🧠 <b>Self-optimize</b>"]
            for (b, d), m in sorted(self._side_size_mult.items()):
                s = stats.get((b, d), {})
                lines.append(f"  {b} {d}: ×{m:.1f} (WR={s.get('wins',0)/max(s.get('total',1),1)*100:.0f}%, PnL={s.get('pnl',0):+,.0f}₽)")
            send_tg("\n".join(lines))
            log.info(f"Self-optimize state: {self._side_size_mult}")
            # Persist to disk
            try:
                import json as _json
                _ssm_file = Path(DATA_DIR) / "side_size_mult.json"
                _ssm_data = {f"{k[0]}_{k[1]}": v for k, v in self._side_size_mult.items()}
                _ssm_file.write_text(_json.dumps(_ssm_data, indent=2))
            except Exception:
                pass

    def is_side_disabled(self, base: str, direction: str) -> bool:
        """Never disable — always trade, just adjust size."""
        return False  # НЕ ОТКЛЮЧАЕМ — используем size_mult
    
    def get_side_size_mult(self, base: str, direction: str) -> float:
        """Get size multiplier from self-optimize."""
        return self._side_size_mult.get((base, direction), 1.0)

    def _margin_used(self):
        total = 0
        for p in self.positions:
            spec = CONTRACTS.get(p["base"], {})
            margin_pct = spec.get("margin_pct", 15) / 100
            total += p["entry"] * spec.get("lot", 1) * p["qty"] * margin_pct
        return total

    def _calc_pnl(self, pos, price):
        spec = CONTRACTS[pos["base"]]
        ticks = (price - pos["entry"]) / spec["tick"]
        if pos["direction"] == "SHORT":
            ticks = -ticks
        return ticks * spec["tick_val"] * pos["qty"]

    def _calc_equity(self, market):
        eq = self.balance
        for p in self.positions:
            base = p["base"]
            if base in market:
                eq += self._calc_pnl(p, market[base]["last"])
        return eq

    def _update_trailing(self, pos, price):
        """Update trailing stop for position — Tiered Trailing 2.0.

        "Let Winners Run" logic:
        1. best_price ALWAYS tracks the most favorable price
        2. Trail activates EARLY (15% progress) with wide cushion
        3. Cushion tightens progressively as profit grows (TRAIL_TIERS)
        4. Stop only moves in profitable direction (ratchet)
        """
        direction = pos["direction"]
        entry = pos["entry"]
        target = pos["target"]

        # Initialize best_price if missing (handles old state)
        if "best_price" not in pos:
            pos["best_price"] = price
        best = pos["best_price"]

        # Update best_price (always tracks most favorable)
        if direction == "LONG":
            if price > best:
                pos["best_price"] = price
                best = price
        else:
            if price < best:
                pos["best_price"] = price
                best = price

        # Calculate progress toward target (0-100%)
        if direction == "LONG":
            progress_pct = ((price - entry) / (target - entry) * 100) if target != entry else 0
        else:
            progress_pct = ((entry - price) / (entry - target) * 100) if target != entry else 0

        progress_pct = max(0, progress_pct)

        # Calculate best progress (from best_price, not current)
        if direction == "LONG":
            best_progress_pct = ((best - entry) / (target - entry) * 100) if target != entry else 0
        else:
            best_progress_pct = ((entry - best) / (entry - target) * 100) if target != entry else 0

        best_progress_pct = max(0, best_progress_pct)

        # Find applicable tier based on best progress
        applicable_tier = None
        tier_name = None
        for tier_progress, (cushion, name) in sorted(TRAIL_TIERS.items(), reverse=True):
            if best_progress_pct >= tier_progress:
                applicable_tier = cushion
                tier_name = name
                break

        # Activate trailing if we hit any tier
        if applicable_tier is not None and not pos.get("trail_active"):
            pos["trail_active"] = True
            pos["trail_tier"] = tier_name
            # Initial stop at breakeven
            pos["stop"] = round(entry, 2)
            log.info(f"🔄 {tier_name} {pos['base']} прогресс={best_progress_pct:.0f}% → стоп на входе")

        # Update trailing stop using tiered cushion
        if pos.get("trail_active") and applicable_tier is not None:
            distance = abs(best - entry)
            cushion = distance * applicable_tier / 100

            old_tier = pos.get("trail_tier", "")
            if tier_name and tier_name != old_tier:
                pos["trail_tier"] = tier_name
                log.info(f"📈 {pos['base']} апгрейд {old_tier} → {tier_name} (cushion {applicable_tier}%)")

            if direction == "LONG":
                new_stop = round(best - cushion, 2)
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
            else:
                new_stop = round(best + cushion, 2)
                if new_stop < pos["stop"]:
                    pos["stop"] = new_stop

        return pos

    def _check_partial_take(self, pos, price):
        """Check and execute partial take profit.

        Returns: (took_partial, remaining_qty)
        """
        if not PARTIAL_TAKE_ENABLED:
            return False, pos["qty"]

        direction = pos["direction"]
        entry = pos["entry"]
        target = pos["target"]

        # Calculate progress
        if direction == "LONG":
            progress_pct = ((price - entry) / (target - entry) * 100) if target != entry else 0
        else:
            progress_pct = ((entry - price) / (entry - target) * 100) if target != entry else 0

        if progress_pct <= 0:
            return False, pos["qty"]

        # Check which partial levels we've hit
        taken_levels = pos.get("partial_taken", [])

        for level_progress, close_pct in PARTIAL_TAKE_LEVELS:
            if level_progress in taken_levels:
                continue
            if progress_pct >= level_progress:
                # Calculate qty to close
                original_qty = pos.get("original_qty", pos["qty"])
                close_qty = max(1, int(original_qty * close_pct / 100))

                if close_qty >= pos["qty"]:
                    continue  # Don't close everything here

                # Record partial take
                taken_levels.append(level_progress)
                pos["partial_taken"] = taken_levels

                # Calculate partial PnL
                partial_pnl = self._calc_pnl_for_qty(pos, price, close_qty)

                # Update position
                pos["qty"] -= close_qty
                self.balance += partial_pnl
                self.daily_pnl += partial_pnl

                log.info(f"💰 PARTIAL {pos['base']} {close_pct}% @ {progress_pct:.0f}% прогресс | -{close_qty} контр | PnL {partial_pnl:+,.0f}₽")
                send_tg(f"💰 <b>ЧАСТИЧНАЯ ФИКСАЦИЯ</b> {pos['base']}\n{close_pct}% позиции при {progress_pct:.0f}% к цели\nPnL: {partial_pnl:+,.0f}₽ | Осталось: {pos['qty']} контр")

                return True, pos["qty"]

        return False, pos["qty"]

    def _calc_pnl_for_qty(self, pos, price, qty):
        """Calculate PnL for specific quantity."""
        spec = CONTRACTS.get(pos["base"], {})
        ticks = (price - pos["entry"]) / spec.get("tick", 1)
        if pos["direction"] == "SHORT":
            ticks = -ticks
        pnl = ticks * spec.get("tick_val", 1) * qty
        # Fee
        fee = qty * 2.2 * 2
        return pnl - fee

    def _check_kill_loser(self, pos, price, current_pnl):
        """Check and execute partial close for losing positions.

        Returns: (killed_partial, remaining_qty)
        """
        if current_pnl >= 0:
            return False, pos["qty"]

        loss = abs(current_pnl)
        killed_levels = pos.get("loser_killed", [])

        for loss_threshold, total_close_pct in sorted(LOSER_TIERS.items()):
            if loss_threshold in killed_levels:
                continue
            if loss >= loss_threshold:
                # Calculate how much to close at this level
                original_qty = pos.get("original_qty", pos["qty"])
                prev_closed_pct = sum(LOSER_TIERS.get(l, 0) for l in killed_levels)
                this_close_pct = total_close_pct - prev_closed_pct

                close_qty = max(1, int(original_qty * this_close_pct / 100))

                if close_qty >= pos["qty"]:
                    continue  # Emergency close will handle full exit

                killed_levels.append(loss_threshold)
                pos["loser_killed"] = killed_levels

                # Calculate partial loss
                partial_pnl = self._calc_pnl_for_qty(pos, price, close_qty)

                # Update position
                pos["qty"] -= close_qty
                self.balance += partial_pnl
                self.daily_pnl += partial_pnl

                log.info(f"✂️ KILL LOSER {pos['base']} {this_close_pct}% @ убыток {loss:,.0f}₽ | -{close_qty} контр")
                send_tg(f"✂️ <b>СОКРАЩЕНИЕ УБЫТКА</b> {pos['base']}\nУбыток достиг {loss_threshold:,}₽\nЗакрыто {close_qty} контр | Осталось: {pos['qty']}")

                return True, pos["qty"]

        return False, pos["qty"]

    def _should_time_exit(self, pos, held_minutes, current_pnl, atr):
        """Smart time stop — don't cut winners.

        Returns: (should_exit, reason)
        """
        base_time = CONTRACTS.get(pos["base"], {}).get("time_stop_bars", 4) * 60

        if not SMART_TIME_STOP:
            # Old behavior
            return held_minutes >= base_time, "ВРЕМЯ"

        # In good profit → DON'T exit on time, let trailing work
        if atr and current_pnl > atr * 0.5:
            return False, None

        # In small profit → extend time
        if current_pnl > 0:
            return held_minutes >= base_time * 1.5, "ВРЕМЯ_ПРОФИТ"

        # Breaking even → standard time
        if current_pnl > -500:
            return held_minutes >= base_time, "ВРЕМЯ"

        # Losing → faster exit
        return held_minutes >= base_time * 0.75, "ВРЕМЯ_ЛОСС"

    def _check_momentum_reversal(self, pos, candles):
        """Check if momentum is reversing against position.

        Returns: (should_exit, reason)
        """
        if len(candles) < 5:
            return False, None

        direction = pos["direction"]
        entry = pos["entry"]
        best = pos.get("best_price", entry)

        last_3 = candles[-3:]

        # Check if we had meaningful profit first
        if direction == "LONG":
            had_profit = best > entry * 1.003  # Had > 0.3% profit
            # 3 consecutive red candles
            reversing = all(
                (c[1] if isinstance(c, (list, tuple)) else c.get("close", 0)) <
                (c[0] if isinstance(c, (list, tuple)) else c.get("open", 0))
                for c in last_3
            )
        else:
            had_profit = best < entry * 0.997  # Had > 0.3% profit (SHORT)
            # 3 consecutive green candles
            reversing = all(
                (c[1] if isinstance(c, (list, tuple)) else c.get("close", 0)) >
                (c[0] if isinstance(c, (list, tuple)) else c.get("open", 0))
                for c in last_3
            )

        if had_profit and reversing:
            return True, "РАЗВОРОТ"

        return False, None

    def _is_strong_contra_news(self, base, direction):
        sentiment = self.news_sentiment.get(base, {}) if self.news_sentiment else {}
        preferred_direction = sentiment.get("preferred_direction", "NEUTRAL")
        confidence = float(sentiment.get("confidence", 0.0) or 0.0)
        score = float(sentiment.get("score", 1.0) or 1.0)
        if preferred_direction not in ("LONG", "SHORT"):
            return False
        if direction == preferred_direction:
            return False
        if confidence < 0.75:
            return False
        return score <= 0.7

    def _send_futures_news_alerts(self):
        changed = []
        active = set()

        for base, sentiment in (self.news_sentiment or {}).items():
            score = float(sentiment.get("score", 1.0) or 1.0)
            alerts = tuple((sentiment.get("alerts") or [])[:2])
            if score >= 1.0 or not alerts:
                continue

            latest_published = sentiment.get("latest_published")
            if latest_published:
                try:
                    latest_dt = datetime.fromisoformat(latest_published)
                    if latest_dt.tzinfo is None:
                        latest_dt = latest_dt.replace(tzinfo=MSK)
                    age_min = (datetime.now(MSK) - latest_dt.astimezone(MSK)).total_seconds() / 60
                    if age_min > NEWS_ALERT_MAX_AGE_MIN:
                        continue
                except Exception:
                    pass

            active.add(base)
            event_ids = tuple((sentiment.get("event_ids") or [])[:2])
            signature = (sentiment.get("impact", "HIGH"), round(score, 3), event_ids or alerts)
            last_sig, last_ts = self._last_futures_news_alerts.get(base, (None, None))
            # Жёсткий cooldown: не слать чаще 15 мин на контракт, даже если новая новость
            if last_ts and (datetime.now(MSK) - last_ts).total_seconds() < 900:
                continue
            if last_sig == signature:
                continue

            self._last_futures_news_alerts[base] = (signature, datetime.now(MSK))
            changed.append((base, score, sentiment))

        # Clear resolved alerts so a future reappearance can notify again
        for base in list(self._last_futures_news_alerts.keys()):
            if base not in active:
                self._last_futures_news_alerts.pop(base, None)
            else:
                # Обновим timestamp даже если signature совпала, чтобы cooldown работал
                pass

        if not changed:
            return

        lines = ["📰 <b>ФЬЮЧЕРСЫ: NEWS ALERT</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        for base, score, sentiment in changed:
            name = CONTRACTS.get(base, {}).get("name", base)
            level = "⛔ CRITICAL" if score == 0 else "⚠️ HIGH"
            age_min = sentiment.get("latest_age_minutes")
            age_text = f" • ⏱ {age_min} мин назад" if age_min is not None else ""
            if base in ("BR", "NG"):
                action = "Осторожнее с SHORT по сырью; фактор учтён во futures sentiment"
            else:
                action = "Фактор учтён в фильтре и размере позиции"
            lines.append(f"{name} ({base}) — {level} ×{score:g}{age_text}")
            for title in (sentiment.get("alerts") or [])[:2]:
                lines.append(f"• {title[:110]}")
            lines.append(f"💡 {action}")
            lines.append("")

        send_tg("\n".join(lines).rstrip())

    def scan_and_trade(self, market):
        now = datetime.now(MSK)

        # Daily loss limit check
        _skip_new_trades = False
        _circuit_breaker_mult = 1.0
        if self.daily_pnl < -DAILY_LOSS_LIMIT:
            log.warning(f"🚨 DAILY LOSS LIMIT: {self.daily_pnl:+,.0f}₽ < -{DAILY_LOSS_LIMIT:,.0f}₽ — no new trades")
            _skip_new_trades = True

        # Circuit breaker: progressive risk reduction
        try:
            from moex_agent.circuit_breaker import check_circuit_breaker
            # Count consecutive losses
            consec = 0
            for t in reversed(self.closed):
                if float(t.get("pnl_rub", 0) or 0) < 0:
                    consec += 1
                else:
                    break
            equity = self._calc_equity(market) if market else self.balance
            cb = check_circuit_breaker(equity, self.peak_equity, self.daily_pnl, consec)
            _circuit_breaker_mult = cb["size_mult"]
            if not cb["can_trade"]:
                _skip_new_trades = True
                if cb["reason"]:
                    log.warning(f"CIRCUIT BREAKER: {cb['reason']}")
        except Exception:
            pass

        # Check news
        self.news_mult, kw, self.news_sentiment = check_news()
        if self.news_mult < 1 and kw:
            if not self._last_global_news_alert or (now - self._last_global_news_alert).total_seconds() > 900:
                self._last_global_news_alert = now
                # Не шлём дубль — per-contract алерты ниже покрывают это

        self._send_futures_news_alerts()

        # Макро-календарь
        try:
            from moex_agent.macro_calendar import get_macro_news_mult
            macro_mult, macro_reason = get_macro_news_mult()
            if macro_mult < 1.0:
                self.news_mult = min(self.news_mult, macro_mult)
                if not hasattr(self, '_last_macro_alert') or not self._last_macro_alert or (now - self._last_macro_alert).total_seconds() > 3600:
                    send_tg(f"📅 <b>МАКРО-СОБЫТИЕ</b>\n{macro_reason}\n💡 Размер ×{macro_mult}")
                    self._last_macro_alert = now
        except Exception:
            pass

        if self.news_mult == 0:
            log.info(f"📰 СТОП НОВОСТИ: {kw}")
            return

        # ══ NEWS MOMENTUM — торговля по тренду на сильных новостях ══
        if not _skip_new_trades and len(self.positions) == 0:
            try:
                from moex_agent.news_momentum import detect_news_momentum
                from moex_agent.news import fetch_news
                digest = fetch_news(max_age_minutes=120, force=True)  # свежие новости для momentum
                news_items = digest.items if digest else []
                # Собираем данные фьючерсов из market + вычисляем dev
                futures_data = {}
                for ticker, data in market.items():
                    for base in CONTRACTS:
                        if base in ticker:
                            price = data.get("last", data.get("price", data.get("close", 0)))
                            # Вычисляем dev через EMA
                            ema = get_ema(base)
                            if ema and ema > 0 and price > 0:
                                dev_pct = ((price - ema) / ema * 100)
                            else:
                                dev_pct = 3.0  # fallback: assume strong move if no EMA
                            futures_data[base] = {"dev": round(dev_pct, 2), "price": price, "ticker": ticker, "secid": data.get("secid", ticker)}
                            break
                
                log.info(f"NEWS_MOM: {len(news_items)} news, devs={[(k,round(v.get('dev',0),2)) for k,v in futures_data.items()]}")
                momentum = detect_news_momentum(
                    [{"title": n.title, "impact": n.impact, "score": n.score} for n in news_items],
                    futures_data
                )
                log.info(f"NEWS_MOM result: {momentum}")
                if momentum:
                    base, direction, reason, dev = momentum
                    spec = CONTRACTS[base]
                    ticker = futures_data[base].get("secid", futures_data[base].get("ticker", ""))
                    price = futures_data[base].get("price", 0)
                    log.info(f"🚀 MOMENTUM: {base} {direction} ticker={ticker} price={price} dev={dev}")
                    if price > 0:
                        # Агрессивный вход: больше контрактов
                        qty = max(3, int(abs(dev)))  # 3-5 контрактов по dev
                        atr = abs(dev) * price / 100 * 0.5  # примерный ATR
                        stop_dist = atr * 2.0
                        if direction == "SHORT":
                            stop = price + stop_dist
                            target = price - stop_dist * 1.5
                        else:
                            stop = price - stop_dist
                            target = price + stop_dist * 1.5
                        
                        log.info(f"🚀 NEWS MOMENTUM ENTRY: {base} {direction} @ {price} qty={qty} | {reason}")
                        send_tg(f"🚀 <b>NEWS MOMENTUM</b>\n{base} {direction} x{qty} @ {price}\n📰 {reason}\nStop: {stop:.0f} Target: {target:.0f}")
                        
                        spec = CONTRACTS[base]
                        margin_per = spec.get("margin_pct", 15) / 100 * price * spec.get("lot", 1)
                        margin = margin_per * qty
                        self.balance -= margin
                        self.trade_count += 1
                        self.daily_trades += 1
                        pos = {
                            "id": self.trade_count, "base": base, "secid": ticker,
                            "direction": direction, "entry": price, "stop": stop,
                            "target": target, "qty": qty, "original_qty": qty, "margin": round(margin, 2),
                            "entry_time": now.isoformat(), "ema": price, "dev": round(dev, 2),
                            "rr": 1.5, "atr": stop_dist, "oi": 0, "volume": 0,
                            "trail_active": False, "best_price": price, "news_score": 0.5,
                        }
                        self.positions.append(pos)
                        self._save()
                        
                        # Live BKS
                        side_str = "2" if direction == "SHORT" else "1"
                        _try_live_futures_order(secid=ticker, side=side_str, quantity=qty, trade_id=self.trade_count, action="open")
            except Exception as exc:
                log.warning(f"News momentum check error: {exc}", exc_info=True)

        # News health check every 10 min
        try:
            from moex_agent.news import news_health_check
            last_hc = getattr(self, '_last_health_check', None)
            if not last_hc or (now - last_hc).total_seconds() > 600:
                self._last_health_check = now
                healthy, diag = news_health_check()
                if not healthy:
                    last_sa = getattr(self, '_last_stale_alert', None)
                    if not last_sa or (now - last_sa).total_seconds() > 900:
                        send_tg(f"🔧 <b>ДИАГНОСТИКА НОВОСТЕЙ (ФЬЮЧЕРСЫ)</b>\n{diag}")
                        self._last_stale_alert = now
        except Exception as e:
            log.debug(f"Health check error: {e}")

        if (now.hour, now.minute) >= NO_NEW_AFTER:
            return

        # Manage existing positions
        for pos in list(self.positions):
            base = pos["base"]
            if base not in market:
                continue
            price = market[base]["last"]
            entry_time = datetime.fromisoformat(pos["entry_time"])
            held = (now - entry_time).total_seconds() / 60
            current_pnl = self._calc_pnl(pos, price)
            atr = pos.get("atr", price * 0.01)  # Fallback: 1% of price

            # Store original_qty for partial calculations (first time only)
            if "original_qty" not in pos:
                pos["original_qty"] = pos["qty"]

            # ══ 1. Update trailing stop (Tiered 2.0) ══
            self._update_trailing(pos, price)

            # ══ 2. Check partial take profit (Let Winners Run) ══
            took_partial, _ = self._check_partial_take(pos, price)
            if took_partial:
                self._save()

            # ══ 3. Kill losers fast (partial close on loss tiers) ══
            killed_partial, _ = self._check_kill_loser(pos, price, current_pnl)
            if killed_partial:
                self._save()

            # ══ 4. Check stop/target ══
            hit_stop = (pos["direction"] == "LONG" and price <= pos["stop"]) or \
                       (pos["direction"] == "SHORT" and price >= pos["stop"])
            hit_target = (pos["direction"] == "LONG" and price >= pos["target"]) or \
                         (pos["direction"] == "SHORT" and price <= pos["target"])

            # ══ 5. Smart time stop (don't cut winners) ══
            hit_time, time_reason = self._should_time_exit(pos, held, current_pnl, atr)

            # ══ 6. Momentum reversal check ══
            candles = market[base].get("candles", [])
            hit_reversal, reversal_reason = self._check_momentum_reversal(pos, candles)

            # ══ 7. Stale position detector ══
            STALE_MINUTES = 60
            STALE_MFE_PCT = 0.1
            hit_stale = False
            if held >= STALE_MINUTES:
                entry = pos["entry"]
                best = pos.get("best_price", price)
                if pos["direction"] == "LONG":
                    mfe_pct = (best - entry) / entry * 100 if entry else 0
                else:
                    mfe_pct = (entry - best) / entry * 100 if entry else 0
                if mfe_pct < STALE_MFE_PCT:
                    hit_stale = True

            # ══ 8. Emergency close: max loss per trade ══
            if current_pnl < -MAX_LOSS_PER_TRADE:
                log.warning(f"🚨 EMERGENCY CLOSE {base} {pos['direction']}: loss {current_pnl:+,.0f}₽ > limit {MAX_LOSS_PER_TRADE:,.0f}₽")
                self._close(pos, price, "МАКС_УБЫТОК", now)
                continue

            # ══ 9. Contra-news exit (commodities) ══
            contra_news = base in ("BR", "NG") and self._is_strong_contra_news(base, pos["direction"])
            if contra_news and current_pnl < 0:
                log.info(f"📰 CONTRA NEWS EXIT {base}: strong news against position")
                self._close(pos, price, "КОНТРА_НОВОСТЬ", now)
                continue

            # ══ Execute exits by priority ══
            if hit_stop:
                self._close(pos, price, "СТОП", now)
            elif hit_target:
                self._close(pos, price, "ЦЕЛЬ", now)
            elif hit_reversal:
                log.info(f"🔄 MOMENTUM REVERSAL {base}: {reversal_reason}")
                self._close(pos, price, reversal_reason, now)
            elif hit_stale:
                log.info(f"⏳ STALE {base} {pos['direction']}: {held:.0f}m, MFE < {STALE_MFE_PCT}%")
                self._close(pos, price, "STALE", now)
            elif hit_time:
                self._close(pos, price, time_reason or "ВРЕМЯ", now)

        # New entries
        if _skip_new_trades or len(self.positions) >= MAX_POSITIONS:
            return

        for base, data in sorted(market.items(), key=lambda x: abs(x[1]["change_pct"]), reverse=True):
            if len(self.positions) >= MAX_POSITIONS:
                break
            if any(p["base"] == base for p in self.positions):
                continue

            # Expiration check: skip contracts near expiration
            try:
                from moex_agent.events import should_skip_expiring
                if should_skip_expiring(data["secid"]):
                    log.info(f"SKIP {base}/{data['secid']}: near expiration")
                    continue
            except Exception:
                pass

            # Cooldown: don't re-enter same contract within 30 min after close
            last_close = None
            for t in reversed(self.closed):
                if t["base"] == base:
                    last_close = t
                    break
            if last_close:
                close_time = datetime.fromisoformat(last_close["exit_time"])
                if (now - close_time).total_seconds() < 900:  # 15 min cooldown
                    continue

            price = data["last"]
            spec = CONTRACTS[base]

            if not price:
                continue

            # Calculate EMA-based deviation (from hourly candles only — no tick drift)
            ema = get_ema(base)
            if not ema:
                ema = refresh_ema_from_candles(base, data["secid"])
            else:
                # Periodically refresh from candles (every 5 min)
                refresh_ema_from_candles(base, data["secid"])
                ema = get_ema(base)
            if not ema:
                log.info(f"No EMA data for {base}/{data['secid']}, waiting for candles...")
                continue

            dev = ((price - ema) / ema) * 100
            abs_dev = abs(dev)

            # Direction based on deviation from EMA
            if dev > 0:
                direction = "SHORT"
            else:
                direction = "LONG"

            # v2: min_dev with hourly tactics multiplier and direction-specific thresholds
            base_min_dev = spec["min_dev"]
            # MX has separate min_dev_long for LONG direction
            if direction == "LONG" and "min_dev_long" in spec:
                base_min_dev = spec["min_dev_long"]
            # Apply hourly tactics multiplier
            hourly_mult = get_hourly_min_dev_mult()
            effective_min_dev = base_min_dev * hourly_mult

            if abs_dev < effective_min_dev:
                continue

            # Max deviation filter: dev > 3% = trend, not mean reversion (data: -25,632₽)
            max_dev = spec.get("max_dev", 3.0)
            if abs_dev > max_dev:
                log.info(f"SKIP {base}: dev {abs_dev:.1f}% > max {max_dev}% (trend, not MR)")
                continue

            # Per-contract side mode (from backtest optimization)
            contract_mode = spec.get("side_mode", SIDE_MODE)
            if contract_mode == "short_only" and direction != "SHORT":
                continue
            if contract_mode == "long_only" and direction != "LONG":
                continue

            # Self-optimization: skip auto-disabled sides
            if self.is_side_disabled(base, direction):
                log.debug(f"Self-optimize: {base} {direction} disabled, skip")
                continue

            # Macro correlation filter (IMOEX, USD/RUB)
            macro = fetch_macro_context()
            macro_ok, macro_reason = macro_filter(base, direction, macro)
            # Macro filter: НЕ блокируем, уменьшаем размер
            macro_size_mult = 1.0
            if not macro_ok:
                macro_size_mult = 0.3  # 30% размер против макро
                log.info(f"MACRO WARNING: {macro_reason} → size×0.3")

            # OI filter — skip if very low liquidity
            if data.get("oi", 0) < 10000:
                continue

            # Hourly tactics: already applied via get_hourly_min_dev_mult() above
            # Log current hourly multiplier for debugging
            entry_hour = now.hour
            hourly_mult_debug = get_hourly_min_dev_mult()
            if hourly_mult_debug != 1.0:
                log.debug(f"HOURLY {entry_hour}:00 → min_dev×{hourly_mult_debug:.1f}")

            # Volume confirmation: skip if volume is abnormally low
            vol = data.get("volume", 0)
            if vol < 500:
                log.debug(f"Low volume {base}: {vol} → skip")
                continue

            # Correlation filter: MX and RI correlate ~85%, don't open both same direction
            CORRELATED = {"MX": "RI", "RI": "MX"}
            corr_pair = CORRELATED.get(base)
            if corr_pair:
                for pos in self.positions:
                    if pos["base"] == corr_pair and pos["direction"] == direction:
                        log.info(f"SKIP {base} {direction}: correlated with open {corr_pair}")
                        break
                else:
                    pass  # No conflict
                if any(pos["base"] == corr_pair and pos["direction"] == direction for pos in self.positions):
                    continue

            # News sentiment check for this contract
            fut_sentiment = self.news_sentiment.get(base, {})
            if fut_sentiment.get("score", 1.0) == 0:
                log.info(f"📰 Пропуск {base}: негативный новостной фон")
                continue
            if base in ("BR", "NG") and self._is_strong_contra_news(base, direction):
                log.info(f"📰 Пропуск {base} {direction}: strong contra-news bias")
                continue

            # ATR-based stop + trend filter
            secid = data["secid"]
            candles = fetch_futures_candles(secid, interval=60, count=14)
            atr = calc_atr(candles)

            # ATR volatility filter (data-driven):
            # ATR 1.0-1.5% = 6 trades, WR 0%, -40,865₽ → BLOCK
            # ATR <0.3% = 27 trades, +55,708₽ → BEST
            # ATR 0.6-1.0% = 16 trades, +20,017₽ → OK
            if atr and price > 0:
                atr_pct = atr / price * 100
                if 1.0 <= atr_pct <= 1.5 and base == "BR":
                    log.info(f"VOL BLOCK {base}: ATR={atr_pct:.2f}% in toxic range 1.0-1.5% (WR 0%, -40k)")
                    continue

            # Short-term momentum: if last 3 candles all against us → don't catch knife
            # + Entry confirmation: last candle should show reversal in our direction
            if candles and len(candles) >= 3:
                last3 = candles[-3:]
                try:
                    moves = [(c[1] - c[0]) for c in last3 if c[0] and c[1]]  # close - open
                    if len(moves) == 3:
                        if direction == "SHORT" and all(m > 0 for m in moves):
                            log.debug(f"KNIFE {base}: 3 consecutive UP candles, skip SHORT")
                            continue
                        if direction == "LONG" and all(m < 0 for m in moves):
                            log.debug(f"KNIFE {base}: 3 consecutive DOWN candles, skip LONG")
                            continue

                    # Entry confirmation: last candle must show reversal
                    # For SHORT: last candle should close DOWN (bearish)
                    # For LONG: last candle should close UP (bullish)
                    if len(moves) >= 1:
                        last_move = moves[-1]
                        if direction == "SHORT" and last_move > 0:
                            log.debug(f"NO CONFIRM {base}: last candle UP, waiting for bearish confirm for SHORT")
                            continue
                        if direction == "LONG" and last_move < 0:
                            log.debug(f"NO CONFIRM {base}: last candle DOWN, waiting for bullish confirm for LONG")
                            continue
                except Exception:
                    pass

            # Trend filter: skip if price moved >2% in 6 hours against our direction
            try:
                if candles and len(candles) >= 6:
                    c0 = candles[0]
                    cN = candles[-1]
                    first_close = c0[1] if isinstance(c0, (list, tuple)) else float(c0) if not isinstance(c0, dict) else c0.get("close", price)
                    last_close = cN[1] if isinstance(cN, (list, tuple)) else float(cN) if not isinstance(cN, dict) else cN.get("close", price)
                    trend_pct = (last_close - first_close) / first_close * 100 if first_close else 0
                    # Block contra-trend entries (strong trend > 2%)
                    # Trend filter: НЕ блокируем, уменьшаем размер
                    if direction == "SHORT" and trend_pct > 2.0:
                        macro_size_mult *= 0.3
                        log.info(f"TREND WARNING {base} SHORT: восходящий +{trend_pct:.1f}% → size×0.3")
                    if direction == "LONG" and trend_pct < -2.0:
                        macro_size_mult *= 0.3
                        log.info(f"TREND WARNING {base} LONG: нисходящий {trend_pct:.1f}% → size×0.3")
                    # Multi-timeframe confirmation: mean-reversion INTO weak trend is best
                    # (Short when 1h trend is slightly down = trend + mean-rev aligned)
                    trend_aligned = (
                        (direction == "SHORT" and trend_pct < -0.3)
                        or (direction == "LONG" and trend_pct > 0.3)
                    )
                    # Will use trend_aligned later for position sizing boost
            except Exception as e:
                log.debug(f"Trend filter error {base}: {e}")

            if atr:
                # v2: Apply STOP_MULTIPLIER for volatile contracts (BR +30%, NG +20%)
                stop_mult = STOP_MULTIPLIER.get(base, 1.0)
                stop_dist = atr * 3.0 * stop_mult  # Base 3.0 ATR, adjusted per contract
                stop_pct_calc = (stop_dist / price) * 100
                stop_pct_calc = max(0.8, min(4.0, stop_pct_calc))  # Min 0.8% (was 0.3: way too tight for futures)
            else:
                stop_pct_calc = STOP_PCT

            # Calculate stop & target (mean reversion to EMA)
            if direction == "LONG":
                stop = round(price * (1 - stop_pct_calc / 100), 2)
                target = round(ema, 2)
            else:
                stop = round(price * (1 + stop_pct_calc / 100), 2)
                target = round(ema, 2)

            rr = abs(price - target) / abs(price - stop) if abs(price - stop) > 0 else 0
            min_rr_required = float(spec.get("min_rr", MIN_RR) or MIN_RR)
            if rr < min_rr_required:
                log.info(f"SKIP {base} {direction}: RR {rr:.2f} < min {min_rr_required:.2f}")
                continue

            # Position sizing — Kelly criterion (adaptive)
            margin_per_contract = price * spec["lot"] * (spec["margin_pct"] / 100)
            try:
                from moex_agent.kelly import kelly_position_size
                kelly = kelly_position_size(
                    balance=self.balance,
                    asset_type="futures",
                    signal_verdict=sig.get("verdict", "OK") if 'sig' in dir() else "OK",
                    base=base,
                )
                kelly_margin = kelly["size_rub"]
                max_qty_kelly = int(kelly_margin / margin_per_contract) if margin_per_contract > 0 else 0
            except Exception:
                max_qty_kelly = MAX_CONTRACTS

            available_margin = self.balance * (MAX_MARGIN_PCT / 100) - self._margin_used()
            max_qty = int(available_margin / margin_per_contract) if margin_per_contract > 0 else 0
            qty = min(max_qty, max_qty_kelly, MAX_CONTRACTS)
            # Apply circuit breaker reduction
            qty = max(1, int(qty * _circuit_breaker_mult))
            # Apply macro/trend size reduction
            try:
                qty = max(1, int(qty * macro_size_mult))
            except NameError:
                pass
            # Apply hourly size multiplier
            try:
                hourly_mult = hourly.get("size_mult", 1.0) if 'hourly' in dir() else 1.0
                qty = max(1, int(qty * hourly_mult))
            except Exception:
                pass
            # Apply per-contract size multiplier
            try:
                qty = max(1, int(qty * contract_size_mult))
            except NameError:
                pass

            # News multiplier — now direction-aware per contract
            base_news_score = float(fut_sentiment.get("score", 1.0) or 1.0)
            preferred_direction = fut_sentiment.get("preferred_direction", "NEUTRAL")
            news_confidence = float(fut_sentiment.get("confidence", 0.0) or 0.0)
            news_factor = self.news_mult

            if preferred_direction == "LONG":
                news_factor *= 1.2 if direction == "LONG" else 0.7
            elif preferred_direction == "SHORT":
                news_factor *= 1.2 if direction == "SHORT" else 0.7
            else:
                news_factor *= base_news_score
                if base_news_score >= 1.0 and direction == "LONG":
                    news_factor *= 1.1

            if news_confidence >= 0.85:
                news_factor *= 1.1
            elif 0.0 < news_confidence < 0.65:
                news_factor *= 0.85

            news_factor = max(0.5, min(1.5, news_factor))

            # Multi-timeframe boost: if 1h trend aligns with mean-reversion direction
            try:
                if trend_aligned:
                    news_factor *= 1.15  # 15% more when trend confirms
                    log.debug(f"MTF boost: {base} {direction} aligned with 1h trend ({trend_pct:+.1f}%)")
            except NameError:
                pass  # trend_aligned not computed (no candles)

            # ATR-based sizing: per-contract (data-driven)
            # MX med ATR = +16.7k → fine. NG med = +2.4k → fine. BR med = blocked above.
            # RI med ATR = -14.5k → reduce
            if atr and price > 0:
                atr_pct = atr / price * 100
                if base == "RI" and atr_pct >= 0.5:
                    news_factor *= 0.5  # RI medium vol = WR 75% but PnL -14.5k
                    log.debug(f"VOL adjust RI: ATR={atr_pct:.2f}% → size ×0.5")

            qty = max(1, int(qty * news_factor))

            if qty < 1 or margin_per_contract * qty > available_margin:
                continue

            # ── Per-contract strategy (data-driven tiers) ──
            try:
                from moex_agent.futures_strategy import check_futures_entry
                fut_check = check_futures_entry(
                    base=base, direction=direction, dev=dev, rr=rr,
                    entry_hour=now.hour, signal_score=sig.get("score", 65),
                )
                if not fut_check["allowed"]:
                    log.info(f"📋 CONTRACT GATE {base} {direction}: {fut_check['tier']} blocked by {fut_check['blocks']}")
                    continue
                # Apply contract-level size adjustment
                contract_size_mult = fut_check["size_pct"] / 100
                log.debug(f"📋 {base} {direction}: tier={fut_check['tier']} size={fut_check['size_pct']}%")
            except Exception as exc:
                contract_size_mult = 1.0
                log.debug(f"Contract strategy error: {exc}")

            # ── Self-analysis lessons check ──
            try:
                from moex_agent.self_analysis import should_skip_by_lessons
                skip, skip_reason = should_skip_by_lessons(
                    base=base, direction=direction, dev=dev, rr=rr, entry_hour=now.hour,
                )
                if skip:
                    log.info(f"🧠 LESSONS SKIP {base} {direction}: {skip_reason}")
                    continue
            except Exception:
                pass

            # ── SmartFilter + RegimeDetector (Phase 3) ──
            if _HAS_SMART_FILTER:
                try:
                    smart_filter = get_smart_filter()
                    calendar = get_calendar_features()
                    calendar_state = calendar.get_features(now, ticker=base)

                    # Build minimal regime state from available candle data
                    regime_state = None
                    if candles and len(candles) >= 5:
                        try:
                            import pandas as pd
                            # Calculate basic regime features from candles
                            closes = [c[1] if isinstance(c, (list, tuple)) else c.get("close", 0) for c in candles[-14:]]
                            if closes and all(c > 0 for c in closes):
                                volatility = pd.Series(closes).pct_change().std() * 100
                                momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0
                                features = pd.Series({
                                    "adx": 20.0,  # Default, no ADX calc
                                    "volatility_30": volatility,
                                    "r_60m": momentum,
                                    "bb_width": volatility * 2,
                                    "sma20_sma50_ratio": 1.0,
                                })
                                regime_detector = RegimeDetector()
                                regime_state = regime_detector.detect(features)
                        except Exception:
                            pass

                    filter_decision = smart_filter.should_trade(
                        ticker=base,
                        direction=direction,
                        regime_state=regime_state,
                        calendar_state=calendar_state,
                    )

                    if not filter_decision.allow:
                        log.info(f"🚫 SMART_FILTER {base} {direction}: {filter_decision.reason}")
                        continue

                    # Apply position size multiplier from SmartFilter
                    if filter_decision.position_mult != 1.0:
                        qty = max(1, int(qty * filter_decision.position_mult))
                        log.debug(f"SMART_FILTER {base}: position_mult={filter_decision.position_mult:.2f}")
                except Exception as e:
                    log.debug(f"SmartFilter error: {e}")

            # ── Signal Quality Score + Strategy Ladder ──
            sig = {"score": 65, "verdict": "OK"}
            try:
                from moex_agent.signal_scorer import score_futures_signal
                sig = score_futures_signal(
                    base=base, direction=direction, dev=dev, rr=rr,
                    atr=atr or 0, price=price, entry_hour=now.hour,
                    news_direction=fut_sentiment.get(base, {}).get("preferred_direction", "NEUTRAL"),
                    trend_pct=trend_pct if 'trend_pct' in dir() else 0,
                    last_3_candles_same=False,
                )
            except Exception:
                pass

            # Strategy Ladder: every signal gets a level, even weak ones
            try:
                from moex_agent.strategy_ladder import get_level, apply_level_params, log_observation
                level_name, level_config = get_level(sig["score"])

                if level_config["size_pct"] == 0:
                    # OBSERVE: don't trade but LOG the signal for learning
                    log_observation(
                        base=base, secid=data["secid"], direction=direction,
                        score=sig["score"], level=level_name,
                        dev=dev, rr=rr, price=price,
                    )
                    log.info(f"👁 OBSERVE {base} {direction}: score={sig['score']} level={level_name} (logged for learning)")
                    continue
                else:
                    # Apply level-specific parameters
                    level_params = apply_level_params(
                        score=sig["score"],
                        base_size=qty,
                        base_stop=stop_pct_calc if 'stop_pct_calc' in dir() else 1.0,
                        base_target=abs(ema - price) if ema else price * 0.01,
                        base_trail_activate=30,
                        base_time_stop=CONTRACTS.get(base, {}).get("time_stop_bars", 4) * 60,
                    )
                    # Adjust qty by level
                    qty = max(1, int(level_params["size"] if level_params["size"] > 0 else qty * level_config["size_pct"] / 100))
                    log.info(f"📊 LADDER {base} {direction}: score={sig['score']} level={level_name} qty={qty}")
            except Exception as exc:
                log.debug(f"Ladder error: {exc}")

            # BCS fee check: round-trip fee < expected profit at target
            FEE_PER_CONTRACT = 2.2  # broker 1.2₽ + exchange ~1₽
            round_trip_fee = qty * FEE_PER_CONTRACT * 2  # open + close
            ticks_to_target = abs(price - target) / spec["tick"]
            expected_profit = ticks_to_target * spec["tick_val"] * qty
            if expected_profit <= round_trip_fee * 1.5:
                log.debug(f"SKIP {base}: profit {expected_profit:.0f}₽ < fee×1.5 {round_trip_fee*1.5:.0f}₽")
                continue

            margin = margin_per_contract * qty
            self.balance -= margin  # Block margin
            self.trade_count += 1
            self.daily_trades += 1

            pos = {
                "id": self.trade_count,
                "base": base,
                "secid": secid,
                "direction": direction,
                "entry": price,
                "stop": stop,
                "target": target,
                "qty": qty,
                "original_qty": qty,  # For partial take/kill calculations
                "margin": round(margin, 2),
                "entry_time": now.isoformat(),
                "ema": round(ema, 2),
                "dev": round(dev, 2),
                "rr": round(rr, 2),
                "atr": round(atr, 2) if atr else None,
                "oi": data.get("oi", 0),
                "volume": data.get("volume", 0),
                "trail_active": False,
                "best_price": price,
                "news_score": round(news_factor, 2),
            }
            self.positions.append(pos)
            self._save()

            atr_tag = f"ATR×2.0" if atr else f"Фикс {stop_pct_calc}%"
            emoji = "🟢" if direction == "LONG" else "🔴"
            dir_ru = "ПОКУПКА" if direction == "LONG" else "ПРОДАЖА"
            # Calculate potential profit/loss in RUB
            ticks_profit = abs(price - target) / spec["tick"]
            ticks_loss = abs(price - stop) / spec["tick"]
            pot_profit = ticks_profit * spec["tick_val"] * qty
            pot_loss = ticks_loss * spec["tick_val"] * qty
            send_tg(
                f"{emoji} <b>ФЬЮЧЕРС {dir_ru} {spec['name']} ({secid})</b>\n"
                f"📅 {now.strftime('%Y-%m-%d %H:%M')}\n"
                f"📊 Цена: {price} | EMA20: {ema:.2f} | Откл: {dev:+.1f}%\n"
                f"🎯 Цель: {target} (+{pot_profit:,.0f}₽) | 🔴 Стоп: {stop} (-{pot_loss:,.0f}₽)\n"
                f"📦 {qty} контр. | Маржа: {margin:,.0f}₽\n"
                f"📰 Новостной фон: ×{news_factor:.1f}"
            )
            log.info(f"OPEN {direction} {base} ({secid}) @ {price} dev={dev:+.1f}% qty={qty}")
            # Mirror open to BKS live
            open_side = "BUY" if direction == "LONG" else "SELL"
            _try_live_futures_order(secid=secid, side=open_side, quantity=qty, trade_id=len(self.closed) + len(self.positions), action="open")

    def _close(self, pos, price, reason, now):
        pnl = self._calc_pnl(pos, price)
        held = (now - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 60

        # BCS Trader fee: 1.2₽/contract broker + ~1₽ exchange ≈ 2.2₽/contract (open+close = x2)
        fee = pos["qty"] * 2.2 * 2  # open + close
        pnl -= fee

        # CRITICAL: Mirror close to BKS live BEFORE updating paper state
        # This ensures live position is closed even if paper state update fails
        secid = pos.get("secid")
        if secid:  # Only call bridge if secid exists
            close_side = "SELL" if pos["direction"] == "LONG" else "BUY"
            _try_live_futures_order(secid=secid, side=close_side, quantity=pos["qty"], trade_id=len(self.closed), action="close")

        self.balance += pos["margin"] + pnl  # Return margin + PnL
        self.daily_pnl += pnl

        trade = {**pos, "exit_price": price, "exit_time": now.isoformat(),
                 "pnl_rub": round(pnl, 2), "reason": reason, "held_minutes": round(held, 1)}
        self.closed.append(trade)
        self.positions.remove(pos)
        self._save()

        spec = CONTRACTS[pos["base"]]
        emoji = "✅" if pnl > 0 else "❌"
        result = "ПРИБЫЛЬ" if pnl > 0 else "УБЫТОК"
        entry_dt = pos.get("entry_time", "")[:16].replace("T", " ")
        exit_dt = now.strftime("%Y-%m-%d %H:%M")
        send_tg(
            f"{emoji} <b>{spec['name']} {pos['direction']}</b>\n"
            f"📅 {entry_dt} → {exit_dt}\n"
            f"Вход: {pos['entry']} → Выход: {price}\n"
            f"💰 <b>{result}: {pnl:+,.0f}₽</b> | {held:.0f} мин | {reason}\n"
            f"Баланс: {self.balance:,.0f}₽"
        )
        log.info(f"CLOSED {pos['base']} {reason} pnl={pnl:+,.0f}₽")

        # Self-analysis: learn from this trade
        try:
            from moex_agent.self_analysis import analyze_and_learn
            analysis = analyze_and_learn(trade)
            if analysis.get("lessons"):
                lesson_text = " | ".join(l["rule"] for l in analysis["lessons"][:2])
                log.info(f"🧠 LESSON: {lesson_text}")
        except Exception as exc:
            log.debug(f"Self-analysis error: {exc}")

        # Strategy Ladder: update level stats
        try:
            from moex_agent.strategy_ladder import update_level_stats, get_level
            # Estimate what level this trade was (from its score if available)
            trade_score = int(trade.get("signal_score", 65) or 65)
            level_name, _ = get_level(trade_score)
            update_level_stats(level_name, pnl)
        except Exception:
            pass

        # Push notification
        try:
            from moex_agent.push import send_push
            spec = CONTRACTS.get(pos["base"], {})
            emoji = "✅" if pnl > 0 else "❌"
            send_push(f"{emoji} {spec.get('name', pos['base'])} {reason}", f"PnL: {pnl:+,.0f}₽", tag="futures_close")
        except Exception:
            pass
        # NOTE: Live bridge call moved to beginning of _close() for atomic close behavior

    def force_flat_all(self, market, reason="EOD"):
        for pos in list(self.positions):
            base = pos["base"]
            if base in market:
                self._close(pos, market[base]["last"], reason, datetime.now(MSK))

    def daily_report(self, market):
        equity = self._calc_equity(market)
        self.equity_curve.append({"ts": datetime.now(MSK).isoformat(), "equity": round(equity, 2)})
        self._save_equity()

        today_trades = [t for t in self.closed
                       if t.get("exit_time", "")[:10] == datetime.now(MSK).date().isoformat()]
        wins = sum(1 for t in today_trades if float(t.get("pnl_rub", 0)) > 0)
        total_pnl = sum(float(t.get("pnl_rub", 0)) for t in today_trades)
        all_pnl = sum(float(t.get("pnl_rub", 0)) for t in self.closed)

        # Enhanced daily report
        losses = len(today_trades) - wins
        wr = wins / len(today_trades) * 100 if today_trades else 0
        gross_p = sum(float(t.get("pnl_rub", 0)) for t in today_trades if float(t.get("pnl_rub", 0)) > 0)
        gross_l = abs(sum(float(t.get("pnl_rub", 0)) for t in today_trades if float(t.get("pnl_rub", 0)) < 0))
        pf_today = round(gross_p / gross_l, 2) if gross_l else "∞"

        all_wins = sum(1 for t in self.closed if float(t.get("pnl_rub", 0)) > 0)
        all_wr = all_wins / len(self.closed) * 100 if self.closed else 0
        all_gp = sum(float(t.get("pnl_rub", 0)) for t in self.closed if float(t.get("pnl_rub", 0)) > 0)
        all_gl = abs(sum(float(t.get("pnl_rub", 0)) for t in self.closed if float(t.get("pnl_rub", 0)) < 0))
        pf_all = round(all_gp / all_gl, 2) if all_gl else "∞"

        # By contract today
        by_base = {}
        for t in today_trades:
            b = t.get("base", "?")
            if b not in by_base:
                by_base[b] = {"pnl": 0, "trades": 0}
            by_base[b]["pnl"] += float(t.get("pnl_rub", 0))
            by_base[b]["trades"] += 1

        base_lines = " · ".join(f"{b} {s['pnl']:+,.0f}₽({s['trades']})" for b, s in sorted(by_base.items(), key=lambda x: x[1]["pnl"]))

        # By exit reason
        by_reason = {}
        for t in today_trades:
            r = t.get("reason", "?")
            if r not in by_reason:
                by_reason[r] = 0
            by_reason[r] += 1
        reason_line = " · ".join(f"{r}:{n}" for r, n in by_reason.items())

        # Disabled sides info
        disabled = []
        for b, dirs in self._disabled_sides.items():
            for d in dirs:
                disabled.append(f"{b} {d}")
        disabled_line = ", ".join(disabled) if disabled else "нет"

        dd = self.peak_equity - equity
        dd_pct = dd / self.peak_equity * 100 if self.peak_equity else 0

        send_tg(
            f"📊 <b>ФЬЮЧЕРСЫ — ДНЕВНОЙ ОТЧЁТ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Эквити: {equity:,.0f}₽\n"
            f"📈 День: {total_pnl:+,.0f}₽ | WR {wr:.0f}% ({wins}✅/{losses}❌) | PF {pf_today}\n"
            f"📊 All-time: {all_pnl:+,.0f}₽ | WR {all_wr:.0f}% | PF {pf_all}\n"
            f"📉 DD: {dd:,.0f}₽ ({dd_pct:.1f}%)\n"
            f"📋 Контракты: {base_lines or 'нет сделок'}\n"
            f"🔄 Выходы: {reason_line or '—'}\n"
            f"🧠 Отключены: {disabled_line}\n"
            f"📂 Открыто: {len(self.positions)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _handle_manual_news_message(self, msg):
        try:
            from moex_agent.manual_news import (
                build_source_name,
                extract_candidate_text,
                ingest_manual_news,
                is_forwarded_message,
            )
            text = extract_candidate_text(msg)
            if not text or text.startswith("/"):
                return False

            forwarded = is_forwarded_message(msg)
            ts = msg.get("forward_date") or msg.get("date")
            published_at = datetime.fromtimestamp(ts, tz=MSK) if ts else datetime.now(MSK)
            result = ingest_manual_news(
                text,
                source_name=build_source_name(msg),
                forwarded=forwarded,
                published_at=published_at,
                chat_message_id=msg.get("message_id"),
            )
            if result.get("stored"):
                assets = ", ".join(result.get("futures_affected") or []) or "общий контекст"
                send_tg(
                    f"📝 <b>MANUAL NEWS SIGNAL</b>\n"
                    f"⏱ 0 мин назад • Источник: {result.get('source', 'manual_forward')}\n"
                    f"Impact: {result.get('impact')} ×{result.get('score')}\n"
                    f"Активы: {assets}\n"
                    f"{result.get('title', '')[:180]}"
                )
                return True
        except Exception as e:
            log.debug(f"Manual news ingest error: {e}")
        return False

    def handle_commands(self, market):
        updates = tg_get_updates()
        for update in updates:
            msg = update.get("message", {})
            text = (msg.get("text") or msg.get("caption") or "").strip().lower()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if chat_id != CHAT_ID:
                continue

            if text and not text.startswith("/"):
                self._handle_manual_news_message(msg)
                continue

            cmd = text.split()[0] if text else ""
            if cmd in ("/fstatus", "/fs"):
                equity = self._calc_equity(market)
                margin = self._margin_used()
                send_tg(
                    f"📊 <b>ФЬЮЧЕРСЫ СТАТУС</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 Баланс: {self.balance:,.0f}₽\n"
                    f"📈 Эквити: {equity:,.0f}₽\n"
                    f"🔒 Маржа: {margin:,.0f}₽ ({margin/equity*100:.1f}%)\n"
                    f"📂 Позиции: {len(self.positions)}\n"
                    f"📰 Новости: ×{self.news_mult}\n"
                    f"🔄 Сделок: {self.daily_trades} сегодня, {len(self.closed)} всего"
                )
            elif cmd in ("/fpos", "/fp"):
                if not self.positions:
                    send_tg("📂 Фьючерсы: нет открытых позиций")
                    continue
                lines = ["📂 <b>ФЬЮЧЕРСЫ — ПОЗИЦИИ</b>\n"]
                for pos in self.positions:
                    base = pos["base"]
                    price = market.get(base, {}).get("last", pos["entry"])
                    pnl = self._calc_pnl(pos, price)
                    spec = CONTRACTS[base]
                    trail = "🔄" if pos.get("trail_active") else ""
                    emoji = "✅" if pnl > 0 else "❌"
                    lines.append(
                        f"{emoji} <b>{pos['direction']} {spec['name']}</b> {trail}\n"
                        f"   {pos['entry']}→{price} | {pnl:+,.0f}₽ | {pos['qty']} контр."
                    )
                send_tg("\n".join(lines))
            elif cmd in ("/fflat", "/ff"):
                if self.positions:
                    count = len(self.positions)
                    self.force_flat_all(market, reason="MANUAL")
                    send_tg(f"🏳️ Фьючерсы: закрыто {count} позиций")
                else:
                    send_tg("📂 Нечего закрывать")
            elif cmd in ("/fnews", "/fn"):
                from moex_agent.news import fetch_news, format_news_digest, get_futures_sentiment
                digest = fetch_news(60, force=True)
                send_tg(format_news_digest(digest))
                sentiment = get_futures_sentiment()
                lines = ["\n⚡ <b>Фьючерсный фон:</b>"]
                for base, s in sentiment.items():
                    if s["count"]:
                        score_e = "🟢" if s["score"] >= 1.0 else "🟡" if s["score"] >= 0.5 else "🔴"
                        name = CONTRACTS.get(base, {}).get("name", base)
                        lines.append(f"  {score_e} {name}: {s['impact']} (×{s['score']})")
                send_tg("\n".join(lines))


# ── Main ─────────────────────────────────────────────────

def main():
    global BOT_TOKEN
    env_file = Path("/tmp/superagent007/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                BOT_TOKEN = line.split("=", 1)[1].strip().strip('"')

    engine = FuturesEngine()
    _tg_init_offset()

    send_tg(
        f"🚀 <b>ФЬЮЧЕРСЫ Paper Trading ЗАПУЩЕН</b>\n"
        f"💰 Депозит: {INITIAL_BALANCE:,}₽\n"
        f"📊 Контракты: {', '.join(CONTRACTS.keys())}\n"
        f"🛡 Макс маржа: {MAX_MARGIN_PCT}% | Стоп: {STOP_PCT}%\n"
        f"⏱ Time-stop: {TIME_STOP_MINUTES} мин\n"
        f"📰 Новости: {'ВКЛ' if NEWS_ENABLED else 'ВЫКЛ'}\n"
        f"↔️ Режим: {SIDE_MODE}"
    )

    last_report = None
    last_equity_save = 0

    while True:
        try:
            now = datetime.now(MSK)

            if now.weekday() >= 5:
                log.info("Weekend")
                time.sleep(300)
                continue

            market = {}
            in_active_session = (
                (now.hour > FUTURES_SESSION_START[0] or (now.hour == FUTURES_SESSION_START[0] and now.minute >= FUTURES_SESSION_START[1]))
                and (now.hour < FUTURES_SESSION_END[0] or (now.hour == FUTURES_SESSION_END[0] and now.minute <= FUTURES_SESSION_END[1]))
            )
            if in_active_session:
                market = fetch_futures_market()
                if market:
                    # Self-optimization: periodically review WR by side
                    engine.self_optimize()

                    if (now.hour, now.minute) >= FORCE_FLAT_AFTER:
                        engine.force_flat_all(market, reason="EOD")
                    else:
                        engine.scan_and_trade(market)

                    equity = engine._calc_equity(market)
                    if equity > engine.peak_equity:
                        engine.peak_equity = equity

                    log.info(f"Scan: {len(market)} фьючерсов, {len(engine.positions)} open, eq={equity:,.0f}₽")

                    # Save equity every 10 min
                    if time.time() - last_equity_save > 600:
                        engine.equity_curve.append({"ts": now.isoformat(), "equity": round(equity, 2)})
                        engine._save_equity()
                        last_equity_save = time.time()
            else:
                log.info("Outside futures session, skip trading scan")

            # TG commands always
            try:
                engine.handle_commands(market)
            except Exception as e:
                log.debug(f"TG cmd error: {e}")

            # ── SAFETY: orphan killer — проверяем БКС каждые 5 мин ──
            try:
                from moex_agent.bks_live import is_live_enabled
                if is_live_enabled():
                    import time as _t
                    _last_orphan_check = getattr(engine, '_last_orphan_check', 0)
                    if _t.time() - _last_orphan_check > 300:  # 5 min
                        engine._last_orphan_check = _t.time()
                        from moex_agent.bks_position_manager import get_bks_pm
                        pm = get_bks_pm()
                        paper_tickers = {p.get('secid', p.get('base', '')): p.get('direction') for p in engine.positions}
                        orphans = pm.close_all_orphans(paper_tickers)
                        for o in orphans:
                            send_tg(f"🚨 <b>ORPHAN KILLER</b>: {o['ticker']} {o['direction']} x{abs(o['qty'])}")
            except Exception as exc:
                log.debug(f"Orphan check error: {exc}")

            # ── Position Sync: paper ↔ БКС (every 5 min) ──
            try:
                from moex_agent.bks_live import is_live_enabled
                if is_live_enabled():
                    import time as _t2
                    _last_sync = getattr(engine, '_last_position_sync', 0)
                    if _t2.time() - _last_sync > 300:
                        engine._last_position_sync = _t2.time()
                        from moex_agent.bks_position_manager import get_bks_pm
                        pm = get_bks_pm()
                        bks_positions = pm.get_all_positions()
                        paper_map = {p.get('secid', p.get('base', '')): p.get('direction') for p in engine.positions}
                        
                        # Проверка: БКС qty ≠ paper qty
                        for ticker, bks_pos in bks_positions.items():
                            if ticker in paper_map:
                                if bks_pos['direction'] != paper_map[ticker]:
                                    log.warning(f"⚠️ SYNC MISMATCH: {ticker} paper={paper_map[ticker]} bks={bks_pos['direction']}")
                                    send_tg(f"⚠️ <b>SYNC</b>: {ticker} paper={paper_map[ticker]} bks={bks_pos['direction']}")
                            else:
                                log.warning(f"⚠️ SYNC ORPHAN: {ticker} на БКС но не в paper")
                        
                        if bks_positions:
                            log.info(f"SYNC: paper={len(engine.positions)} bks={len(bks_positions)} tickers")
            except Exception as exc:
                log.debug(f"Position sync error: {exc}")

            # ── Reconciliation check (every 5 min when live is on) ──
            try:
                from moex_agent.reconciliation import maybe_reconcile
                maybe_reconcile()
            except Exception as exc:
                log.debug(f"Reconciliation check error: {exc}")

            # ── Performance alerting ──
            try:
                from moex_agent.alerting import check_performance_alerts
                check_performance_alerts(
                    closed_trades=engine.closed,
                    daily_pnl=engine.daily_pnl,
                    current_drawdown=engine.peak_equity - engine._calc_equity(market) if market else 0,
                    agent_name="Фьючерсы",
                )
            except Exception as exc:
                log.debug(f"Alert check error: {exc}")

            # Daily report at 23:50
            if now.hour == 23 and now.minute >= 50:
                today = now.date()
                if last_report != today and market:
                    engine.daily_report(market)
                    last_report = today
                    engine.daily_pnl = 0
                    engine.daily_trades = 0

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            send_tg("🛑 <b>Фьючерсы Paper Trading Остановлен</b>")
            engine._save()
            engine._save_equity()
            break
        except Exception as e:
            import traceback
            log.error(f"Error: {e}\n{traceback.format_exc()}")
            time.sleep(30)


if __name__ == "__main__":
    main()
