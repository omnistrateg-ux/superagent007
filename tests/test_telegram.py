"""Tests for telegram module."""
import pytest
from unittest.mock import patch, MagicMock

from moex_agent.telegram import send_telegram, send_signal_alert


def test_send_telegram_no_credentials():
    """Should return False without credentials."""
    result = send_telegram("", "123", "test")
    assert result is False

    result = send_telegram("token", "", "test")
    assert result is False


@patch("moex_agent.telegram.requests.post")
def test_send_telegram_success(mock_post):
    """Should return True on successful send."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    result = send_telegram("token123", "chat123", "Hello")

    assert result is True
    mock_post.assert_called_once()


@patch("moex_agent.telegram.requests.post")
def test_send_telegram_rate_limit(mock_post):
    """Should retry on rate limit."""
    # First call: rate limited, second call: success
    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.json.return_value = {"parameters": {"retry_after": 0.1}}

    success_response = MagicMock()
    success_response.status_code = 200

    mock_post.side_effect = [rate_limit_response, success_response]

    result = send_telegram("token123", "chat123", "Hello")

    assert result is True
    assert mock_post.call_count == 2


@patch("moex_agent.telegram.requests.post")
def test_send_signal_alert(mock_post):
    """Test formatted signal alert."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    result = send_signal_alert(
        bot_token="token",
        chat_id="chat",
        ticker="SBER",
        direction="LONG",
        horizon="5m",
        p=0.58,
        score=2.5,
        entry=250.0,
        take=252.0,
        stop=248.0,
        volume_spike=2.0,
    )

    assert result is True

    # Check message content
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    text = payload["text"]

    assert "SBER" in text
    assert "LONG" in text
    assert "5m" in text
