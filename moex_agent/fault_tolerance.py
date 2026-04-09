"""
MOEX Agent v2.5 Fault Tolerance

Handles:
1. Network reconnection with exponential backoff
2. Crash recovery from saved state
3. Graceful degradation when services unavailable
4. Circuit breaker pattern for external APIs
5. Health monitoring and self-healing

Usage:
    from moex_agent.fault_tolerance import (
        with_retry,
        CircuitBreaker,
        HealthMonitor,
        StateRecovery,
    )

    # Retry decorator
    @with_retry(max_attempts=3, backoff_factor=2)
    def fetch_data():
        return requests.get(url)

    # Circuit breaker
    breaker = CircuitBreaker("moex_api")
    if breaker.is_closed():
        try:
            result = api_call()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
"""
from __future__ import annotations

import functools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# === Retry Decorator ===

class RetryError(Exception):
    """Raised when all retry attempts fail."""
    pass


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for delay between attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between attempts
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback function on each retry (exception, attempt_number)

    Example:
        @with_retry(max_attempts=3, backoff_factor=2)
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise RetryError(f"All {max_attempts} attempts failed") from e

                    if on_retry:
                        on_retry(e, attempt)

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise RetryError(f"All {max_attempts} attempts failed") from last_exception

        return wrapper
    return decorator


# === Circuit Breaker ===

class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for external services.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    name: str
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 60.0   # Seconds before trying again
    half_open_requests: int = 3      # Requests allowed in half-open

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_count: int = 0

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        self._check_state_transition()
        return self.state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        self._check_state_transition()
        return self.state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        self._check_state_transition()

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_count < self.half_open_requests
        else:
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_requests:
                self._close()
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._half_open()

    def _open(self) -> None:
        """Open the circuit."""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now()

    def _half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
        self.state = CircuitState.HALF_OPEN
        self.half_open_count = 0
        self.success_count = 0

    def _close(self) -> None:
        """Close the circuit (recovery complete)."""
        logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_count = 0

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


# === Health Monitor ===

@dataclass
class HealthCheck:
    """Single health check result."""
    name: str
    healthy: bool
    message: str
    last_check: datetime
    response_time_ms: float = 0.0


class HealthMonitor:
    """
    Monitor health of system components.
    """

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker for monitoring."""
        self.circuit_breakers[breaker.name] = breaker

    def check_database(self, db_path: Path) -> HealthCheck:
        """Check database health."""
        start = time.perf_counter()
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.execute("SELECT 1")
            conn.close()
            response_time = (time.perf_counter() - start) * 1000

            check = HealthCheck(
                name="database",
                healthy=True,
                message="OK",
                last_check=datetime.now(),
                response_time_ms=response_time,
            )
        except Exception as e:
            check = HealthCheck(
                name="database",
                healthy=False,
                message=str(e),
                last_check=datetime.now(),
            )

        self.checks["database"] = check
        return check

    def check_moex_api(self) -> HealthCheck:
        """Check MOEX ISS API health."""
        start = time.perf_counter()
        try:
            import requests
            resp = requests.get(
                "https://iss.moex.com/iss/index.json",
                timeout=10,
            )
            resp.raise_for_status()
            response_time = (time.perf_counter() - start) * 1000

            check = HealthCheck(
                name="moex_api",
                healthy=True,
                message=f"OK ({resp.status_code})",
                last_check=datetime.now(),
                response_time_ms=response_time,
            )
        except Exception as e:
            check = HealthCheck(
                name="moex_api",
                healthy=False,
                message=str(e),
                last_check=datetime.now(),
            )

        self.checks["moex_api"] = check
        return check

    def check_models(self, models_dir: Path) -> HealthCheck:
        """Check ML models health."""
        try:
            meta_path = models_dir / "meta.json"
            if not meta_path.exists():
                check = HealthCheck(
                    name="models",
                    healthy=False,
                    message="meta.json not found",
                    last_check=datetime.now(),
                )
            else:
                meta = json.loads(meta_path.read_text())
                model_count = len(meta)
                check = HealthCheck(
                    name="models",
                    healthy=True,
                    message=f"{model_count} models loaded",
                    last_check=datetime.now(),
                )
        except Exception as e:
            check = HealthCheck(
                name="models",
                healthy=False,
                message=str(e),
                last_check=datetime.now(),
            )

        self.checks["models"] = check
        return check

    def get_overall_health(self) -> Dict:
        """Get overall system health."""
        all_healthy = all(c.healthy for c in self.checks.values())

        # Check circuit breakers
        breakers_ok = all(
            not b.is_open() for b in self.circuit_breakers.values()
        )

        return {
            "healthy": all_healthy and breakers_ok,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "healthy": c.healthy,
                    "message": c.message,
                    "response_time_ms": c.response_time_ms,
                }
                for name, c in self.checks.items()
            },
            "circuit_breakers": {
                name: b.get_status()
                for name, b in self.circuit_breakers.items()
            },
        }


# === State Recovery ===

class StateRecovery:
    """
    Handles crash recovery from saved state.
    """

    def __init__(
        self,
        state_dir: Path = Path("data"),
        max_state_age_hours: int = 24,
    ):
        self.state_dir = state_dir
        self.max_state_age_hours = max_state_age_hours

    def save_checkpoint(
        self,
        name: str,
        state: Dict[str, Any],
    ) -> Path:
        """
        Save a state checkpoint.

        Args:
            name: Checkpoint name
            state: State dictionary to save

        Returns:
            Path to saved checkpoint
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "state": state,
        }

        path = self.state_dir / f"{name}_checkpoint.json"
        path.write_text(json.dumps(checkpoint, indent=2, default=str))

        logger.debug(f"Saved checkpoint: {path}")
        return path

    def load_checkpoint(
        self,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a state checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            State dictionary or None if not found/expired
        """
        path = self.state_dir / f"{name}_checkpoint.json"

        if not path.exists():
            return None

        try:
            checkpoint = json.loads(path.read_text())

            # Check age
            timestamp = datetime.fromisoformat(checkpoint["timestamp"])
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600

            if age_hours > self.max_state_age_hours:
                logger.warning(f"Checkpoint {name} too old ({age_hours:.1f}h), ignoring")
                return None

            logger.info(f"Loaded checkpoint: {name} (age: {age_hours:.1f}h)")
            return checkpoint["state"]

        except Exception as e:
            logger.error(f"Failed to load checkpoint {name}: {e}")
            return None

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        checkpoints = []

        for path in self.state_dir.glob("*_checkpoint.json"):
            try:
                data = json.loads(path.read_text())
                timestamp = datetime.fromisoformat(data["timestamp"])
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600

                checkpoints.append({
                    "name": data["name"],
                    "path": str(path),
                    "timestamp": data["timestamp"],
                    "age_hours": age_hours,
                    "valid": age_hours <= self.max_state_age_hours,
                })
            except Exception:
                continue

        return checkpoints

    def cleanup_old_checkpoints(self) -> int:
        """Remove checkpoints older than max_state_age_hours."""
        removed = 0

        for path in self.state_dir.glob("*_checkpoint.json"):
            try:
                data = json.loads(path.read_text())
                timestamp = datetime.fromisoformat(data["timestamp"])
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600

                if age_hours > self.max_state_age_hours * 2:  # 2x margin
                    path.unlink()
                    removed += 1
                    logger.debug(f"Removed old checkpoint: {path}")
            except Exception:
                continue

        return removed


# === Graceful Degradation ===

class GracefulDegradation:
    """
    Handle service degradation gracefully.
    """

    def __init__(self):
        self.degraded_services: Dict[str, str] = {}

    def mark_degraded(self, service: str, reason: str) -> None:
        """Mark a service as degraded."""
        self.degraded_services[service] = reason
        logger.warning(f"Service '{service}' degraded: {reason}")

    def mark_healthy(self, service: str) -> None:
        """Mark a service as healthy."""
        if service in self.degraded_services:
            del self.degraded_services[service]
            logger.info(f"Service '{service}' recovered")

    def is_degraded(self, service: str) -> bool:
        """Check if a service is degraded."""
        return service in self.degraded_services

    def get_fallback_value(
        self,
        service: str,
        default: Any,
    ) -> Any:
        """Get fallback value for degraded service."""
        if self.is_degraded(service):
            logger.debug(f"Using fallback for degraded service '{service}'")
            return default
        return None  # Signal to use real value

    def get_status(self) -> Dict:
        """Get degradation status."""
        return {
            "degraded_count": len(self.degraded_services),
            "services": dict(self.degraded_services),
        }


# Singletons
_health_monitor: Optional[HealthMonitor] = None
_state_recovery: Optional[StateRecovery] = None
_degradation: Optional[GracefulDegradation] = None


def get_health_monitor() -> HealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def get_state_recovery() -> StateRecovery:
    global _state_recovery
    if _state_recovery is None:
        _state_recovery = StateRecovery()
    return _state_recovery


def get_degradation() -> GracefulDegradation:
    global _degradation
    if _degradation is None:
        _degradation = GracefulDegradation()
    return _degradation
