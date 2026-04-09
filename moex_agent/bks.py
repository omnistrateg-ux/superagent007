"""BKS Trade API client.

Safe-by-default integration layer for BKS broker connectivity.

Current scope:
- refresh token -> access token exchange
- portfolio fetch
- orders search

Order creation/execution is intentionally not implemented yet.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterable, Optional

import requests


class BksApiError(RuntimeError):
    """Raised when BKS API returns an invalid or unsuccessful response."""


@dataclass(slots=True)
class BksConfig:
    """Runtime configuration for BKS API."""

    base_url: str = field(default_factory=lambda: os.environ.get("BKS_BASE_URL", "https://be.broker.ru").rstrip("/"))
    client_id: str = field(default_factory=lambda: os.environ.get("BKS_CLIENT_ID", "trade-api-write"))
    refresh_token: Optional[str] = field(default_factory=lambda: os.environ.get("BKS_REFRESH_TOKEN"))
    timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("BKS_TIMEOUT_SECONDS", "15")))
    execution_enabled: bool = field(default_factory=lambda: os.environ.get("BKS_EXECUTION_ENABLED", "false").lower() in {"1", "true", "yes", "on"})
    create_order_path: str = field(default_factory=lambda: os.environ.get("BKS_CREATE_ORDER_PATH", "/trade-api-bff-operations/api/v1/orders"))
    cancel_order_path: str = field(default_factory=lambda: os.environ.get("BKS_CANCEL_ORDER_PATH", "/trade-api-bff-operations/api/v1/orders/{order_id}/cancel"))
    get_order_path: str = field(default_factory=lambda: os.environ.get("BKS_GET_ORDER_PATH", "/trade-api-bff-operations/api/v1/orders/{order_id}"))
    execution_log_path: str = field(default_factory=lambda: os.environ.get("BKS_EXECUTION_LOG_PATH", "data/bks_execution_log.jsonl"))
    order_registry_path: str = field(default_factory=lambda: os.environ.get("BKS_ORDER_REGISTRY_PATH", "data/bks_order_registry.json"))

    def validate(self) -> None:
        if not self.refresh_token:
            raise BksApiError("Missing BKS refresh token. Set BKS_REFRESH_TOKEN in .env")
        if self.client_id not in {"trade-api-read", "trade-api-write"}:
            raise BksApiError("BKS_CLIENT_ID must be trade-api-read or trade-api-write")


class BksClient:
    """Thin client over BKS Trade API."""

    TOKEN_PATH = "/trade-api-keycloak/realms/tradeapi/protocol/openid-connect/token"
    PORTFOLIO_PATH = "/trade-api-bff-portfolio/api/v1/portfolio"
    ORDERS_SEARCH_PATH = "/trade-api-bff-order-details/api/v1/orders/search"

    def __init__(
        self,
        config: Optional[BksConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or BksConfig()
        self.session = session or requests.Session()
        self._access_token: Optional[str] = None
        self._access_token_expires_at: Optional[datetime] = None

    def _request_with_retries(self, method: str, url: str, attempts: int = 3, **kwargs: Any) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                headers = dict(kwargs.pop("headers", {}) or {})
                headers.setdefault("Connection", "close")
                return self.session.request(method=method, url=url, headers=headers, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                sleep(0.5 * attempt)
        raise BksApiError(f"BKS request failed after {attempts} attempts: {last_exc}")

    def exchange_refresh_token(self, force: bool = False) -> str:
        """Exchange refresh token for access token."""
        self.config.validate()

        if not force and self._access_token and self._access_token_expires_at:
            now = datetime.now(timezone.utc)
            if now < self._access_token_expires_at - timedelta(seconds=30):
                return self._access_token

        response = self._request_with_retries(
            "POST",
            f"{self.config.base_url}{self.TOKEN_PATH}",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "client_id": self.config.client_id,
                "refresh_token": self.config.refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=self.config.timeout_seconds,
        )
        payload = self._parse_json(response, "token exchange")

        access_token = payload.get("access_token")
        if not access_token:
            raise BksApiError("BKS token exchange succeeded without access_token in response")

        expires_in = int(payload.get("expires_in", 300))
        self._access_token = access_token
        self._access_token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return access_token

    def get_portfolio(self) -> Dict[str, Any]:
        """Fetch portfolio state from BKS."""
        response = self._request_with_retries(
            "GET",
            f"{self.config.base_url}{self.PORTFOLIO_PATH}",
            headers=self._auth_headers(),
            timeout=self.config.timeout_seconds,
        )
        return self._parse_json(response, "portfolio fetch")

    def search_orders(
        self,
        start_datetime: str,
        end_datetime: str,
        side: Optional[int] = None,
        order_status: Optional[Iterable[int]] = None,
        order_types: Optional[Iterable[int]] = None,
        tickers: Optional[Iterable[str]] = None,
        class_codes: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Search orders using BKS order-details endpoint."""
        payload: Dict[str, Any] = {
            "startDateTime": start_datetime,
            "endDateTime": end_datetime,
        }
        if side is not None:
            payload["side"] = side
        if order_status is not None:
            payload["orderStatus"] = list(order_status)
        if order_types is not None:
            payload["orderTypes"] = list(order_types)
        if tickers is not None:
            payload["tickers"] = list(tickers)
        if class_codes is not None:
            payload["classCodes"] = list(class_codes)

        response = self._request_with_retries(
            "POST",
            f"{self.config.base_url}{self.ORDERS_SEARCH_PATH}",
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        return self._parse_json(response, "orders search")

    def _execution_log_append(self, event: Dict[str, Any]) -> None:
        path = Path(self.config.execution_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _registry_read(self) -> Dict[str, Any]:
        path = Path(self.config.order_registry_path)
        if not path.exists():
            return {"orders": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("orders"), dict):
                return data
        except Exception:
            pass
        return {"orders": {}}

    def _registry_write(self, data: Dict[str, Any]) -> None:
        path = Path(self.config.order_registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _registry_upsert_order(self, client_order_id: str, patch: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not client_order_id:
            return patch
        data = self._registry_read()
        orders = data.setdefault("orders", {})
        current = orders.get(client_order_id, {"clientOrderId": client_order_id, "events": []})
        current.update({k: v for k, v in patch.items() if v is not None})
        current["updatedAt"] = datetime.now(timezone.utc).isoformat()
        if "createdAt" not in current:
            current["createdAt"] = current["updatedAt"]
        if event:
            current.setdefault("events", []).append({
                "ts": current["updatedAt"],
                **event,
            })
            current["events"] = current["events"][-50:]
        orders[client_order_id] = current
        self._registry_write(data)
        return current

    def list_tracked_orders(self, limit: int = 50) -> Dict[str, Any]:
        data = self._registry_read()
        orders = list(data.get("orders", {}).values())
        orders.sort(key=lambda x: x.get("updatedAt", ""), reverse=True)
        return {
            "items": orders[:limit],
            "count": len(orders),
        }

    @staticmethod
    def _normalize_order_response(payload: Dict[str, Any], raw: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        order_id = payload.get("orderId") or payload.get("order_id") or payload.get("clientOrderId") or payload.get("id") or payload.get("orderNum")
        status = payload.get("status") or payload.get("orderStatus") or payload.get("state") or ("dry_run" if dry_run else "submitted")
        return {
            "ok": True,
            "dry_run": dry_run,
            "order_id": order_id,
            "status": status,
            "payload": payload,
            "raw": raw,
        }

    def create_order(
        self,
        ticker: str,
        side: str | int,
        quantity: float,
        order_type: str | int = "market",
        price: Optional[float] = None,
        class_code: Optional[str] = None,
        board: Optional[str] = None,
        dry_run: Optional[bool] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        side_value = side
        if isinstance(side, str):
            side_upper = str(side or "").upper()
            if side_upper not in {"BUY", "SELL"}:
                raise BksApiError("side must be BUY/SELL or 1/2")
            side_value = 1 if side_upper == "BUY" else 2
        elif side not in {1, 2}:
            raise BksApiError("side must be BUY/SELL or 1/2")

        if float(quantity or 0) <= 0:
            raise BksApiError("quantity must be > 0")

        order_type_value = order_type
        if isinstance(order_type, str):
            order_type_upper = str(order_type or "market").upper()
            order_type_map = {"MARKET": 1, "LIMIT": 2}
            if order_type_upper not in order_type_map:
                raise BksApiError("order_type must be market/limit or 1/2")
            order_type_value = order_type_map[order_type_upper]
        elif order_type not in {1, 2}:
            raise BksApiError("order_type must be market/limit or 1/2")

        if int(order_type_value) == 2 and price is None:
            raise BksApiError("price is required for LIMIT order")
        if not (class_code or board):
            raise BksApiError("class_code is required")

        # BKS API требует side и orderType как строки "1"/"2"
        payload: Dict[str, Any] = {
            "clientOrderId": client_order_id or extra_fields.get("clientOrderId") if extra_fields else client_order_id,
            "side": str(int(side_value)),
            "orderType": str(int(order_type_value)),
            "orderQuantity": int(quantity),
            "ticker": ticker,
            "classCode": class_code or board,
        }
        if not payload.get("clientOrderId"):
            import uuid
            payload["clientOrderId"] = str(uuid.uuid4())
        if price is not None:
            payload["price"] = float(price)
        if extra_fields:
            payload.update(extra_fields)

        effective_dry_run = (not self.config.execution_enabled) if dry_run is None else bool(dry_run)
        if effective_dry_run:
            result = self._normalize_order_response(payload, {"dry_run": True, "would_send": payload}, dry_run=True)
            self._registry_upsert_order(
                payload["clientOrderId"],
                {
                    "ticker": ticker,
                    "classCode": payload.get("classCode"),
                    "side": int(side_value),
                    "orderType": int(order_type_value),
                    "orderQuantity": int(quantity),
                    "price": payload.get("price"),
                    "status": "dry_run",
                    "mode": "dry_run",
                },
                event={"action": "create_order", "payload": payload},
            )
            self._execution_log_append({"action": "create_order", "mode": "dry_run", "ticker": ticker, "side": int(side_value), "quantity": quantity, "payload": payload})
            return result

        response = self._request_with_retries(
            "POST",
            f"{self.config.base_url}{self.config.create_order_path}",
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        raw = self._parse_json(response, "order create")
        result = self._normalize_order_response(raw, raw, dry_run=False)
        self._registry_upsert_order(
            payload["clientOrderId"],
            {
                "ticker": ticker,
                "classCode": payload.get("classCode"),
                "side": int(side_value),
                "orderType": int(order_type_value),
                "orderQuantity": int(quantity),
                "price": payload.get("price"),
                "status": result.get("status") or "submitted",
                "mode": "live",
                "brokerOrderId": raw.get("orderId") or raw.get("id"),
            },
            event={"action": "create_order", "payload": payload, "result": result},
        )
        self._execution_log_append({"action": "create_order", "mode": "live", "ticker": ticker, "side": int(side_value), "quantity": quantity, "payload": payload, "result": result})
        return result

    def cancel_order(self, order_id: str, dry_run: Optional[bool] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        if not order_id:
            raise BksApiError("order_id is required")

        effective_dry_run = (not self.config.execution_enabled) if dry_run is None else bool(dry_run)
        cancel_path = self.config.cancel_order_path.format(order_id=order_id)
        payload = {"clientOrderId": client_order_id or order_id}

        if effective_dry_run:
            result = self._normalize_order_response({"orderId": order_id, **payload}, {"dry_run": True, "would_cancel": order_id}, dry_run=True)
            self._registry_upsert_order(
                order_id,
                {
                    "lastCancelRequestId": payload["clientOrderId"],
                    "status": "cancel_requested_dry_run",
                    "mode": "dry_run",
                },
                event={"action": "cancel_order", "payload": payload, "targetOrderId": order_id},
            )
            self._execution_log_append({"action": "cancel_order", "mode": "dry_run", "order_id": order_id, "payload": payload})
            return result

        response = self._request_with_retries(
            "POST",
            f"{self.config.base_url}{cancel_path}",
            headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        raw = self._parse_json(response, "order cancel")
        result = self._normalize_order_response(raw, raw, dry_run=False)
        self._registry_upsert_order(
            order_id,
            {
                "lastCancelRequestId": payload["clientOrderId"],
                "status": result.get("status") or "cancel_requested",
                "mode": "live",
            },
            event={"action": "cancel_order", "payload": payload, "result": result, "targetOrderId": order_id},
        )
        self._execution_log_append({"action": "cancel_order", "mode": "live", "order_id": order_id, "payload": payload, "result": result})
        return result

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if not order_id:
            raise BksApiError("order_id is required")
        response = self._request_with_retries(
            "GET",
            f"{self.config.base_url}{self.config.get_order_path.format(order_id=order_id)}",
            headers=self._auth_headers(),
            timeout=self.config.timeout_seconds,
        )
        raw = self._parse_json(response, "order status")
        result = self._normalize_order_response(raw, raw, dry_run=False)
        self._registry_upsert_order(
            order_id,
            {
                "status": result.get("status"),
                "mode": "live",
                "brokerOrderId": raw.get("orderId") or raw.get("id"),
            },
            event={"action": "order_status", "result": result},
        )
        return result

    def recent_orders_summary(self, days: int = 30, limit: int = 12) -> Dict[str, Any]:
        """Return recent orders in a mobile/UI-friendly shape."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        def fmt(dt: datetime) -> str:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        raw = self.search_orders(fmt(start), fmt(end))
        records = raw.get("records") or []
        if not isinstance(records, list):
            records = []

        def infer_status(order: Dict[str, Any]) -> str:
            if order.get("rejectReason"):
                return "REJECTED"
            executed = float(order.get("executedQuantity") or 0.0)
            remained = float(order.get("remainedQuantity") or 0.0)
            total = float(order.get("orderQuantity") or 0.0)
            status_code = int(order.get("orderStatus") or 0)
            if executed > 0 and remained <= 0:
                return "FILLED"
            if executed > 0 and remained > 0:
                return "PARTIAL"
            if status_code == 1 and remained >= total:
                return "ACTIVE"
            if status_code == 3:
                return "CANCELED"
            if remained <= 0:
                return "CLOSED"
            return f"STATUS_{status_code}"

        def side_label(side: Any) -> str:
            if int(side or 0) == 1:
                return "BUY"
            if int(side or 0) == 2:
                return "SELL"
            return f"SIDE_{side}"

        normalized = []
        for record in records:
            if not isinstance(record, dict):
                continue
            normalized.append({
                "ticker": record.get("ticker"),
                "class_code": record.get("classCode"),
                "side": record.get("side"),
                "side_label": side_label(record.get("side")),
                "status": infer_status(record),
                "status_code": record.get("orderStatus"),
                "price": record.get("price"),
                "quantity": record.get("orderQuantity"),
                "executed_quantity": record.get("executedQuantity"),
                "remained_quantity": record.get("remainedQuantity"),
                "order_datetime": record.get("orderDateTime"),
                "reject_reason": record.get("rejectReason"),
                "order_id": record.get("orderId"),
            })

        normalized.sort(key=lambda item: str(item.get("order_datetime") or ""), reverse=True)
        counts = {
            "total": len(normalized),
            "active": sum(1 for item in normalized if item.get("status") == "ACTIVE"),
            "filled": sum(1 for item in normalized if item.get("status") == "FILLED"),
            "partial": sum(1 for item in normalized if item.get("status") == "PARTIAL"),
            "canceled": sum(1 for item in normalized if item.get("status") == "CANCELED"),
            "rejected": sum(1 for item in normalized if item.get("status") == "REJECTED"),
        }
        return {
            "counts": counts,
            "items": normalized[:limit],
        }

    def portfolio_summary(self) -> Dict[str, Any]:
        """Return a human-friendly normalized portfolio summary."""
        raw = self.get_portfolio()
        items = raw.get("content") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return {"raw": raw, "counts": {}}

        try:
            recent_orders = self.recent_orders_summary(days=2, limit=200).get("items", [])
        except Exception:
            recent_orders = []
        recent_fills_by_key: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
        for order in recent_orders:
            if not isinstance(order, dict) or order.get("status") != "FILLED":
                continue
            key = (str(order.get("ticker") or ""), str(order.get("class_code") or ""))
            recent_fills_by_key.setdefault(key, []).append(order)

        grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue

            key = (str(item.get("type") or "unknown"), str(item.get("ticker") or ""))
            chosen = grouped.get(key)
            if chosen is None:
                chosen = dict(item)
                chosen["terms_seen"] = [item.get("term")]
                chosen["max_locked"] = float(item.get("locked") or 0.0)
                chosen["max_locked_for_futures"] = float(item.get("lockedForFutures") or 0.0)
                grouped[key] = chosen
            else:
                if item.get("term") not in chosen["terms_seen"]:
                    chosen["terms_seen"].append(item.get("term"))
                chosen["max_locked"] = max(chosen["max_locked"], float(item.get("locked") or 0.0))
                chosen["max_locked_for_futures"] = max(chosen["max_locked_for_futures"], float(item.get("lockedForFutures") or 0.0))
                if chosen.get("term") != "T0" and item.get("term") == "T0":
                    preserved_terms = chosen["terms_seen"]
                    max_locked = chosen["max_locked"]
                    max_locked_for_futures = chosen["max_locked_for_futures"]
                    chosen.clear()
                    chosen.update(item)
                    chosen["terms_seen"] = preserved_terms
                    chosen["max_locked"] = max_locked
                    chosen["max_locked_for_futures"] = max_locked_for_futures

        def compact(entry: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "ticker": entry.get("ticker"),
                "name": entry.get("displayName"),
                "quantity": entry.get("quantity"),
                "current_value_rub": entry.get("currentValueRub"),
                "balance_value_rub": entry.get("balanceValueRub"),
                "currency": entry.get("currency"),
                "instrument_type": entry.get("instrumentType"),
                "board": entry.get("board"),
                "balance_price": entry.get("balancePrice"),
                "current_price": entry.get("currentPrice"),
                "unrealized_pl": entry.get("unrealizedPL"),
                "unrealized_pct": entry.get("unrealizedPercentPL"),
                "daily_pl": entry.get("dailyPL"),
                "daily_pct": entry.get("dailyPercentPL"),
                "logo_link": entry.get("logoLink"),
                "trade_blocked": bool(entry.get("isBlockedTradeAccount")),
                "is_blocked": bool(entry.get("isBlocked")),
                "max_locked": entry.get("max_locked", 0.0),
                "locked_for_futures": entry.get("max_locked_for_futures", 0.0),
                "terms_seen": entry.get("terms_seen", []),
                "term": entry.get("term"),
            }

        def settlement_tail_reason(entry: Dict[str, Any]) -> Optional[str]:
            quantity = float(entry.get("quantity") or 0.0)
            if abs(quantity) < 1e-9 or str(entry.get("term") or "") != "T0":
                return None
            key = (str(entry.get("ticker") or ""), str(entry.get("board") or ""))
            fills = recent_fills_by_key.get(key, [])
            if not fills:
                return None
            needed_side = 2 if quantity > 0 else 1
            for order in fills:
                executed_qty = float(order.get("executed_quantity") or 0.0)
                if int(order.get("side") or 0) == needed_side and executed_qty >= abs(quantity):
                    return f"today_fill_side_{needed_side}"
            return None

        cash = []
        longs = []
        shorts = []
        otc = []
        russian = []
        blocked_trade = []
        tradeable_positions = []
        top_positions = []
        closed_positions = []
        settlement_tail_positions = []

        for entry in grouped.values():
            entry_type = str(entry.get("type") or "")
            quantity = float(entry.get("quantity") or 0.0)
            current_value_rub = float(entry.get("currentValueRub") or 0.0)
            balance_value_rub = float(entry.get("balanceValueRub") or 0.0)
            upper_type = str(entry.get("upperType") or "")
            instrument_type = str(entry.get("instrumentType") or "")
            is_otc = upper_type == "OTC" or instrument_type.startswith("OTC_")
            is_russian = upper_type == "RUSSIA"
            is_trade_blocked = bool(entry.get("isBlockedTradeAccount"))

            if entry_type == "moneyLimit":
                cash.append(compact(entry))
                continue

            compact_entry = compact(entry)
            is_closed_zero = (
                abs(quantity) < 1e-9
                and abs(current_value_rub) < 1e-9
                and abs(balance_value_rub) < 1e-9
            )
            if is_closed_zero:
                closed_positions.append(compact_entry)
                continue

            settle_reason = settlement_tail_reason(entry)
            if settle_reason:
                compact_entry["reconcile_reason"] = settle_reason
                settlement_tail_positions.append(compact_entry)
                continue

            top_positions.append(compact_entry)

            if quantity < 0:
                shorts.append(compact_entry)
            else:
                longs.append(compact_entry)

            if is_otc:
                otc.append(compact_entry)
            if is_russian:
                russian.append(compact_entry)
            if is_trade_blocked:
                blocked_trade.append(compact_entry)
            else:
                tradeable_positions.append(compact_entry)

        top_positions.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        tradeable_positions.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        longs.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        shorts.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        otc.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        russian.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        cash.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)
        closed_positions.sort(key=lambda item: (item.get("name") or item.get("ticker") or ""))
        settlement_tail_positions.sort(key=lambda item: abs(float(item.get("current_value_rub") or 0.0)), reverse=True)

        cash_total = round(sum(float(item.get("current_value_rub") or 0.0) for item in cash), 2)
        longs_total = round(sum(float(item.get("current_value_rub") or 0.0) for item in longs if float(item.get("quantity") or 0.0) >= 0), 2)
        shorts_abs_total = round(sum(abs(float(item.get("current_value_rub") or 0.0)) for item in shorts), 2)
        tradeable_total = round(sum(float(item.get("current_value_rub") or 0.0) for item in tradeable_positions), 2)
        total_positions_value = round(sum(float(item.get("current_value_rub") or 0.0) for item in top_positions), 2)
        total_positions_balance = round(sum(float(item.get("balance_value_rub") or 0.0) for item in top_positions), 2)
        total_unrealized = round(sum(float(item.get("unrealized_pl") or 0.0) for item in top_positions), 2)
        total_unrealized_pct = round((total_unrealized / total_positions_balance * 100.0), 2) if total_positions_balance else 0.0

        russian_total = round(sum(float(item.get("current_value_rub") or 0.0) for item in russian), 2)
        russian_unrealized = round(sum(float(item.get("unrealized_pl") or 0.0) for item in russian), 2)
        russian_balance_total = round(sum(float(item.get("balance_value_rub") or 0.0) for item in russian), 2)
        russian_unrealized_pct = round((russian_unrealized / russian_balance_total * 100.0), 2) if russian_balance_total else 0.0
        blocked_total = round(sum(float(item.get("current_value_rub") or 0.0) for item in blocked_trade), 2)
        blocked_unrealized = round(sum(float(item.get("unrealized_pl") or 0.0) for item in blocked_trade), 2)
        locked_for_futures_total = round(sum(float(item.get("locked_for_futures") or 0.0) for item in tradeable_positions), 2)

        return {
            "account": next((item.get("account") for item in grouped.values() if item.get("account")), None),
            "agreement_id": next((item.get("agreementId") for item in grouped.values() if item.get("agreementId")), None),
            "counts": {
                "raw_items": len(items),
                "unique_items": len(grouped),
                "cash": len(cash),
                "longs": len(longs),
                "shorts": len(shorts),
                "otc": len(otc),
                "russian": len(russian),
                "blocked_trade": len(blocked_trade),
                "tradeable": len(tradeable_positions),
                "closed_positions": len(closed_positions),
                "settlement_tail_positions": len(settlement_tail_positions),
            },
            "totals_rub": {
                "cash": cash_total,
                "longs": longs_total,
                "shorts_abs": shorts_abs_total,
                "otc": round(sum(float(item.get("current_value_rub") or 0.0) for item in otc), 2),
                "russian": russian_total,
                "russian_unrealized": russian_unrealized,
                "russian_unrealized_pct": russian_unrealized_pct,
                "blocked_total": blocked_total,
                "blocked_unrealized": blocked_unrealized,
                "locked_for_futures": locked_for_futures_total,
                "tradeable": tradeable_total,
                "net_liquid": round(cash_total + tradeable_total, 2),
                "working_portfolio": round(cash_total + russian_total, 2),
                "total_portfolio": round(cash_total + total_positions_value, 2),
                "total_balance": total_positions_balance,
                "total_unrealized": total_unrealized,
                "total_unrealized_pct": total_unrealized_pct,
            },
            "cash": cash,
            "top_positions": top_positions[:10],
            "tradeable_positions": tradeable_positions,
            "shorts": shorts,
            "blocked_trade": blocked_trade,
            "closed_positions": closed_positions,
            "settlement_tail_positions": settlement_tail_positions,
        }

    def _auth_headers(self) -> Dict[str, str]:
        token = self.exchange_refresh_token()
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    @staticmethod
    def _parse_json(response: requests.Response, context: str) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            snippet = (response.text or "")[:300]
            raise BksApiError(
                f"BKS {context} returned non-JSON response: {response.status_code} {snippet!r}"
            ) from exc

        if response.status_code >= 400:
            raise BksApiError(f"BKS {context} failed: {response.status_code} {payload}")

        if not isinstance(payload, dict):
            return {"content": payload}
        return payload
