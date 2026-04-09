"""
MOEX Agent v2 — Интеграция с Telegram

Отправка уведомлений в Telegram с повторными попытками и обработкой лимитов.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger("moex_agent.telegram")


def _get_credentials() -> tuple[str, str]:
    """Получить токен и chat_id из переменных окружения."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    return bot_token, chat_id


def send_telegram_message(text: str, parse_mode: Optional[str] = None) -> bool:
    """
    Отправка сообщения в Telegram используя переменные окружения.

    Args:
        text: Текст сообщения
        parse_mode: Режим парсинга (Markdown, HTML)

    Returns:
        True если отправлено успешно
    """
    bot_token, chat_id = _get_credentials()

    if not bot_token or not chat_id:
        logger.warning("Telegram не настроен (нет TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID)")
        return False

    return send_telegram(bot_token, chat_id, text, parse_mode=parse_mode)


def notify_signal(
    ticker: str,
    direction: str,
    entry: float,
    vwap: float,
    deviation_pct: float,
    target: float,
    stop: float,
    volume_rub: float,
    volume_status: str = "норма",
) -> bool:
    """
    Отправка сигнала используя переменные окружения.
    """
    bot_token, chat_id = _get_credentials()
    if not bot_token or not chat_id:
        logger.warning("Telegram не настроен")
        return False

    return send_signal_alert(
        bot_token, chat_id, ticker, direction, entry, vwap,
        deviation_pct, target, stop, volume_rub, volume_status
    )


def notify_trade_result(
    ticker: str,
    direction: str = "",
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    equity: float = 0.0,
    is_stop: bool = False,
) -> bool:
    """Отправка результата сделки используя переменные окружения."""
    bot_token, chat_id = _get_credentials()
    if not bot_token or not chat_id:
        return False
    return send_trade_result(bot_token, chat_id, ticker, direction, pnl, pnl_pct, equity, is_stop)


def notify_daily_summary(
    date_str: str,
    trades_count: int,
    wins: int,
    losses: int,
    daily_pnl: float,
    best_ticker: str = "",
    best_pct: float = 0.0,
    worst_ticker: str = "",
    worst_pct: float = 0.0,
) -> bool:
    """Отправка итогов дня используя переменные окружения."""
    bot_token, chat_id = _get_credentials()
    if not bot_token or not chat_id:
        return False
    return send_daily_summary(
        bot_token, chat_id, date_str, trades_count, wins, losses,
        daily_pnl, best_ticker, best_pct, worst_ticker, worst_pct
    )

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
    entry: float,
    vwap: float,
    deviation_pct: float,
    target: float,
    stop: float,
    volume_rub: float,
    volume_status: str = "норма",
) -> bool:
    """
    Отправка сигнала в Telegram.

    Args:
        bot_token: Токен бота Telegram
        chat_id: ID чата
        ticker: Тикер
        direction: LONG или SHORT
        entry: Цена входа
        vwap: Цена VWAP
        deviation_pct: Девиация от VWAP в %
        target: Цена цели
        stop: Цена стопа
        volume_rub: Объём в рублях
        volume_status: Статус объёма (норма, высокий, низкий)

    Returns:
        True если отправлено успешно
    """
    # Эмодзи направления
    dir_emoji = "🟢 ПОКУПКА" if direction == "LONG" else "🔴 ПРОДАЖА"

    # Расчёт R:R
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk if risk > 0 else 0

    # Стоп в процентах
    stop_pct = (stop - entry) / entry * 100 if entry > 0 else 0

    # Форматирование объёма
    if volume_rub >= 1_000_000_000:
        vol_str = f"{volume_rub / 1_000_000_000:,.1f}B₽"
    elif volume_rub >= 1_000_000:
        vol_str = f"{volume_rub / 1_000_000:,.0f}M₽"
    else:
        vol_str = f"{volume_rub / 1_000:,.0f}K₽"

    lines = [
        f"{dir_emoji} {ticker} по {entry:.2f}",
        f"📊 VWAP: {vwap:.2f} | Девиация: {deviation_pct:+.1f}%",
        f"🎯 Цель: {target:.2f} (возврат к VWAP)",
        f"🛑 Стоп: {stop:.2f} ({stop_pct:+.1f}%)",
        f"💰 R:R: {rr_ratio:.1f}:1",
        f"📈 Объём: {vol_str} ({volume_status})",
    ]

    text = "\n".join(lines)
    return send_telegram(bot_token, chat_id, text)


def send_trade_result(
    bot_token: str,
    chat_id: str,
    ticker: str,
    direction: str = "",
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    equity: float = 0.0,
    is_stop: bool = False,
) -> bool:
    """
    Отправка результата сделки в Telegram.

    Args:
        bot_token: Токен бота Telegram
        chat_id: ID чата
        ticker: Тикер
        direction: Направление (LONG/SHORT)
        pnl: Абсолютный PnL в рублях
        pnl_pct: PnL в процентах
        equity: Текущий капитал
        is_stop: True если закрыта по стопу

    Returns:
        True если отправлено успешно
    """
    dir_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else ""

    if pnl >= 0:
        text = f"✅ {dir_emoji}{ticker} закрыта {pnl_pct:+.1f}% | PnL: {pnl:+,.0f}₽"
    else:
        if is_stop:
            text = f"❌ {dir_emoji}{ticker} стоп {pnl_pct:.1f}% | PnL: {pnl:,.0f}₽"
        else:
            text = f"❌ {dir_emoji}{ticker} закрыта {pnl_pct:.1f}% | PnL: {pnl:,.0f}₽"

    if equity > 0:
        text += f"\n💰 Капитал: {equity:,.0f}₽"

    return send_telegram(bot_token, chat_id, text)


def send_kill_switch_alert(
    bot_token: str,
    chat_id: str,
    reason: str,
    equity: float,
    drawdown_pct: float,
) -> bool:
    """
    Отправка уведомления об активации аварийной остановки.

    Args:
        bot_token: Токен бота Telegram
        chat_id: ID чата
        reason: Причина остановки
        equity: Текущий капитал
        drawdown_pct: Просадка в процентах

    Returns:
        True если отправлено успешно
    """
    text = (
        f"🚨 АВАРИЙНАЯ ОСТАНОВКА\n"
        f"Причина: {reason}\n"
        f"Капитал: {equity:,.0f}₽\n"
        f"Просадка: {drawdown_pct:.1f}%"
    )

    return send_telegram(bot_token, chat_id, text)


def send_daily_summary(
    bot_token: str,
    chat_id: str,
    date_str: str,
    trades_count: int,
    wins: int,
    losses: int,
    daily_pnl: float,
    best_ticker: str = "",
    best_pct: float = 0.0,
    worst_ticker: str = "",
    worst_pct: float = 0.0,
) -> bool:
    """
    Отправка итогов дня в Telegram.

    Args:
        bot_token: Токен бота Telegram
        chat_id: ID чата
        date_str: Дата в формате ДД.ММ.ГГГГ
        trades_count: Количество сделок
        wins: Выигрышных сделок
        losses: Проигрышных сделок
        daily_pnl: Дневной PnL
        best_ticker: Лучший тикер
        best_pct: Лучший результат в %
        worst_ticker: Худший тикер
        worst_pct: Худший результат в %

    Returns:
        True если отправлено успешно
    """
    win_rate = wins / trades_count * 100 if trades_count > 0 else 0

    lines = [
        f"📊 Итоги {date_str}:",
        f"Сделок: {trades_count} | Выигрыш: {wins} | Проигрыш: {losses}",
        f"WR: {win_rate:.0f}% | PnL: {daily_pnl:+,.0f}₽",
    ]

    if best_ticker:
        lines.append(f"Лучшая: {best_ticker} {best_pct:+.1f}%")
    if worst_ticker:
        lines.append(f"Худшая: {worst_ticker} {worst_pct:.1f}%")

    text = "\n".join(lines)
    return send_telegram(bot_token, chat_id, text)
