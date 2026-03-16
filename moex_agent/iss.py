"""
MOEX ISS API Client

Fetches candles and quotes from Moscow Exchange Information & Statistical Server.
Includes retry logic with exponential backoff.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("moex_agent.iss")

ISS_BASE = "https://iss.moex.com/iss"

_DEFAULT_HEADERS = {
    "User-Agent": "moex-agent/2.0",
    "Accept": "application/json",
}

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create a requests session with retry logic."""
    global _session
    if _session is None:
        _session = requests.Session()
        retry = Retry(
            total=6,
            connect=6,
            read=6,
            status=6,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


@dataclass
class Candle:
    """OHLCV candle data."""
    ts: str
    open: float
    high: float
    low: float
    close: float
    value: float
    volume: float


def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make GET request with timeout and error handling."""
    session = _get_session()
    try:
        r = session.get(
            url,
            params=params,
            timeout=(5, 30),
            headers=_DEFAULT_HEADERS,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {url}")
        raise
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {url}: {e}")
        raise


def fetch_candles(
    engine: str,
    market: str,
    board: str,
    secid: str,
    interval: int,
    from_date: str,
    till_date: str,
) -> List[Candle]:
    """
    Fetch candles from MOEX ISS with automatic pagination.

    Args:
        engine: Engine name (e.g., 'stock')
        market: Market name (e.g., 'shares')
        board: Board name (e.g., 'TQBR')
        secid: Security ID (e.g., 'SBER')
        interval: Candle interval (1=1min, 10=10min, 60=1h, 24=1d)
        from_date: Start date (YYYY-MM-DD)
        till_date: End date (YYYY-MM-DD)

    Returns:
        List of Candle objects
    """
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"
    base_params = {
        "from": from_date,
        "till": till_date,
        "interval": interval,
        "iss.meta": "off",
        "candles.columns": "begin,open,high,low,close,value,volume",
    }

    limit = 500
    start = 0
    out: List[Candle] = []
    idx: Optional[Dict[str, int]] = None

    while True:
        params = dict(base_params)
        params["start"] = start
        params["limit"] = limit

        try:
            data = _get_json(url, params=params)
        except Exception:
            break

        candles_data = data.get("candles", {})
        cols = candles_data.get("columns", [])
        rows = candles_data.get("data", [])

        if idx is None and cols:
            idx = {c: i for i, c in enumerate(cols)}

        if not rows or idx is None:
            break

        for r in rows:
            out.append(
                Candle(
                    ts=str(r[idx["begin"]]),
                    open=float(r[idx["open"]]) if r[idx["open"]] is not None else 0.0,
                    high=float(r[idx["high"]]) if r[idx["high"]] is not None else 0.0,
                    low=float(r[idx["low"]]) if r[idx["low"]] is not None else 0.0,
                    close=float(r[idx["close"]]) if r[idx["close"]] is not None else 0.0,
                    value=float(r[idx["value"]]) if r[idx["value"]] is not None else 0.0,
                    volume=float(r[idx["volume"]]) if r[idx["volume"]] is not None else 0.0,
                )
            )

        start += len(rows)
        if len(rows) < limit:
            break

    logger.debug(f"Fetched {len(out)} candles for {secid}")
    return out


def fetch_quote(engine: str, market: str, board: str, secid: str) -> Dict[str, Any]:
    """
    Fetch current quote (last/bid/ask) from MOEX ISS.

    Args:
        engine: Engine name
        market: Market name
        board: Board name
        secid: Security ID

    Returns:
        Dict with keys: secid, last, bid, ask, numtrades, voltoday, valtoday
    """
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}.json"
    params = {
        "iss.meta": "off",
        "iss.only": "marketdata,marketdata_yields",
        "marketdata.columns": "SECID,LAST,BID,OFFER,NUMTRADES,VOLTODAY,VALTODAY",
    }

    try:
        data = _get_json(url, params=params)
    except Exception:
        return {"secid": secid}

    md = data.get("marketdata")
    if not md or not md.get("data"):
        return {"secid": secid}

    cols = md["columns"]
    row = md["data"][0]
    idx = {c: i for i, c in enumerate(cols)}

    def getf(name: str) -> Optional[float]:
        if name not in idx:
            return None
        v = row[idx[name]]
        return None if v is None else float(v)

    return {
        "secid": secid,
        "last": getf("LAST"),
        "bid": getf("BID"),
        "ask": getf("OFFER"),
        "numtrades": getf("NUMTRADES"),
        "voltoday": getf("VOLTODAY"),
        "valtoday": getf("VALTODAY"),
    }


def close_session() -> None:
    """Close the HTTP session."""
    global _session
    if _session is not None:
        _session.close()
        _session = None
        logger.debug("ISS session closed")
