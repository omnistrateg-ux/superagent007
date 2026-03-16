"""Tests for risk management."""
import pytest

from moex_agent.risk import (
    RiskEngine,
    KillSwitchConfig,
    RiskParams,
    pass_gatekeeper,
    DayMode,
)


def test_kill_switch_consecutive_losses():
    """Kill-switch should activate after 2 consecutive losses."""
    engine = RiskEngine(initial_equity=1_000_000)

    # First loss
    engine.record_trade_result(-5000, is_win=False)
    active, reason = engine.check_kill_switch()
    assert not active

    # Second loss - should trigger
    engine.record_trade_result(-5000, is_win=False)
    active, reason = engine.check_kill_switch()
    assert active
    assert "consecutive" in reason.lower()


def test_kill_switch_daily_loss():
    """Kill-switch should activate on 2% daily loss."""
    engine = RiskEngine(initial_equity=1_000_000)

    # 2% loss = 20,000
    engine.record_trade_result(-20000, is_win=False)

    active, reason = engine.check_kill_switch()
    assert active
    assert "daily" in reason.lower()


def test_kill_switch_drawdown():
    """Kill-switch should activate on 10% drawdown."""
    engine = RiskEngine(initial_equity=1_000_000)

    # Record large loss to trigger 10% DD
    engine.record_trade_result(-100000, is_win=False)

    active, reason = engine.check_kill_switch()
    assert active
    assert "drawdown" in reason.lower()


def test_risk_engine_reset():
    """Manual reset should work with confirmation."""
    engine = RiskEngine(initial_equity=1_000_000)

    # Trigger kill-switch
    engine.record_trade_result(-5000, is_win=False)
    engine.record_trade_result(-5000, is_win=False)

    # Reset without confirmation - should fail
    assert not engine.reset_kill_switch(confirm=False)

    # Reset with confirmation - should work
    assert engine.reset_kill_switch(confirm=True)

    active, _ = engine.check_kill_switch()
    assert not active


def test_pass_gatekeeper():
    """Test gatekeeper logic."""
    risk = RiskParams(max_spread_bps=200, min_turnover_rub_5m=1_000_000)

    # Should pass
    assert pass_gatekeeper(p=0.55, p_threshold=0.54, turnover_5m=2_000_000, spread=100, risk=risk)

    # Low probability - should fail
    assert not pass_gatekeeper(p=0.50, p_threshold=0.54, turnover_5m=2_000_000, spread=100, risk=risk)

    # Low turnover - should fail
    assert not pass_gatekeeper(p=0.55, p_threshold=0.54, turnover_5m=500_000, spread=100, risk=risk)

    # High spread - should fail
    assert not pass_gatekeeper(p=0.55, p_threshold=0.54, turnover_5m=2_000_000, spread=300, risk=risk)


def test_win_resets_loss_streak():
    """A win should reset the loss streak."""
    engine = RiskEngine(initial_equity=1_000_000)

    # One loss
    engine.record_trade_result(-5000, is_win=False)
    assert engine.state.consecutive_losses == 1

    # Win resets streak
    engine.record_trade_result(10000, is_win=True)
    assert engine.state.consecutive_losses == 0
    assert engine.state.consecutive_wins == 1
