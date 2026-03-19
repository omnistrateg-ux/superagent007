"""
MOEX Agent v2 Telegram Integration

Send alerts to Telegram with retry logic and rate limit handling.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger("moex_agent.telegram")


def send_telegram_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to Telegram using env credentials.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.

    Args:
        text: Message text
        parse_mode: Parse mode (Markdown, HTML)

    Returns:
        True if sent successfully
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    return send_telegram(bot_token, chat_id, text, parse_mode=parse_mode)

# Rate limit settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0
RATE_LIMIT_DELAY = 30.0


def send_telegram(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: Optional[str] = None,
    disable_notification: bool = False,
) -> bool:
    """
    Send message to Telegram with retry logic.

    Handles:
    - 429 Too Many Requests (rate limit) - waits and retries
    - Network errors - retries with backoff
    - API errors - logs and returns False

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID to send to
        text: Message text (plain text by default, no Markdown)
        parse_mode: Optional parse mode ('HTML', 'Markdown', 'MarkdownV2')
        disable_notification: Send silently

    Returns:
        True if message sent successfully
    """
    if not bot_token or not chat_id:
        logger.error("Telegram bot_token or chat_id not configured")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_notification": disable_notification,
    }

    if parse_mode:
        payload["parse_mode"] = parse_mode

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.debug(f"Telegram message sent to {chat_id}")
                return True

            if response.status_code == 429:
                # Rate limited
                try:
                    data = response.json()
                    retry_after = data.get("parameters", {}).get("retry_after", RATE_LIMIT_DELAY)
                except Exception:
                    retry_after = RATE_LIMIT_DELAY

                logger.warning(f"Telegram rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if response.status_code >= 500:
                # Server error, retry
                logger.warning(f"Telegram server error {response.status_code}, retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            # Client error (4xx except 429)
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False

        except requests.exceptions.Timeout:
            logger.warning(f"Telegram timeout, attempt {attempt + 1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (attempt + 1))

        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram request error: {e}, attempt {attempt + 1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    logger.error(f"Telegram send failed after {MAX_RETRIES} attempts")
    return False


def send_signal_alert(
    bot_token: str,
    chat_id: str,
    ticker: str,
    direction: str,
    horizon: str,
    p: float,
    score: float,
    entry: Optional[float] = None,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    volume_spike: float = 1.0,
) -> bool:
    """
    Send formatted signal alert to Telegram.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID
        ticker: Ticker symbol
        direction: LONG or SHORT
        horizon: Time horizon
        p: Probability
        score: Anomaly score
        entry: Entry price
        take: Take profit price
        stop: Stop loss price
        volume_spike: Volume spike ratio

    Returns:
        True if sent successfully
    """
    # Plain text format (no Markdown)
    lines = [
        f"SIGNAL: {ticker} {direction}",
        f"Horizon: {horizon}",
        f"Probability: {p:.1%}",
        f"Score: {score:.2f}",
    ]

    if entry:
        lines.append(f"Entry: {entry:.2f}")
    if take:
        lines.append(f"Take: {take:.2f}")
    if stop:
        lines.append(f"Stop: {stop:.2f}")

    if volume_spike > 1.5:
        lines.append(f"Volume: {volume_spike:.1f}x")

    text = "\n".join(lines)

    return send_telegram(bot_token, chat_id, text)


def send_trade_result(
    bot_token: str,
    chat_id: str,
    ticker: str,
    direction: str,
    pnl: float,
    pnl_pct: float,
    equity: float,
) -> bool:
    """
    Send trade result notification.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID
        ticker: Ticker symbol
        direction: LONG or SHORT
        pnl: Absolute PnL
        pnl_pct: PnL percentage
        equity: Current equity

    Returns:
        True if sent successfully
    """
    emoji = "+" if pnl > 0 else ""
    result = "WIN" if pnl > 0 else "LOSS"

    text = (
        f"TRADE {result}: {ticker} {direction}\n"
        f"PnL: {emoji}{pnl:,.0f} ({emoji}{pnl_pct:.2%})\n"
        f"Equity: {equity:,.0f}"
    )

    return send_telegram(bot_token, chat_id, text)


def send_kill_switch_alert(
    bot_token: str,
    chat_id: str,
    reason: str,
    equity: float,
    drawdown_pct: float,
) -> bool:
    """
    Send kill-switch activation alert.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID
        reason: Kill-switch reason
        equity: Current equity
        drawdown_pct: Current drawdown percentage

    Returns:
        True if sent successfully
    """
    text = (
        f"KILL-SWITCH ACTIVATED\n"
        f"Reason: {reason}\n"
        f"Equity: {equity:,.0f}\n"
        f"Drawdown: {drawdown_pct:.1f}%"
    )

    return send_telegram(bot_token, chat_id, text)


def send_daily_summary(
    bot_token: str,
    chat_id: str,
    trades_count: int,
    wins: int,
    losses: int,
    daily_pnl: float,
    equity: float,
) -> bool:
    """
    Send daily trading summary.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID
        trades_count: Number of trades
        wins: Number of winning trades
        losses: Number of losing trades
        daily_pnl: Daily PnL
        equity: Current equity

    Returns:
        True if sent successfully
    """
    win_rate = wins / trades_count * 100 if trades_count > 0 else 0
    emoji = "+" if daily_pnl > 0 else ""

    text = (
        f"DAILY SUMMARY\n"
        f"Trades: {trades_count} ({wins}W / {losses}L)\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"PnL: {emoji}{daily_pnl:,.0f}\n"
        f"Equity: {equity:,.0f}"
    )

    return send_telegram(bot_token, chat_id, text)
