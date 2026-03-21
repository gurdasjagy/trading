"""Custom webhook receiver for external trading signals."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List

from loguru import logger

from .base_source import BaseSource, DataItem, DataSourceType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Type alias for async handler functions
WebhookHandler = Callable[[dict], Awaitable[None]]


class CustomWebhookReceiver(BaseSource):
    """Receives custom webhook data from external sources via a FastAPI endpoint."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        secret_token: str = "",
        path_prefix: str = "/webhook",
    ):
        super().__init__("custom_webhook", DataSourceType.REST_API)
        self._host = host
        self._port = port
        self._secret_token = secret_token
        self._path_prefix = path_prefix.rstrip("/")
        self._handlers: Dict[str, List[WebhookHandler]] = {}  # event_type -> handlers
        self._items: List[DataItem] = []
        self._app = None  # FastAPI application
        self._server = None  # uvicorn server

    async def start_monitoring(self) -> None:
        self._running = True
        await self.start_server()

    async def stop_monitoring(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.should_exit = True

    async def fetch_latest(self, limit: int = 50) -> List[DataItem]:
        return self._items[-limit:]

    async def start_server(self) -> None:
        """Build and launch the FastAPI webhook server."""
        try:
            import uvicorn  # type: ignore
            from fastapi import Depends, FastAPI, HTTPException, Request  # type: ignore

            app = FastAPI(title="CryptoBot Webhook Receiver", docs_url=None)
            self._app = app
            monitor = self  # closure

            async def _verify_token(request: Request) -> None:
                if monitor._secret_token:
                    token = request.headers.get("X-Webhook-Secret") or request.headers.get(
                        "Authorization", ""
                    ).replace("Bearer ", "")
                    if token != monitor._secret_token:
                        raise HTTPException(status_code=401, detail="Invalid webhook secret")

            @app.post(f"{self._path_prefix}/{{event_type}}")
            async def receive_webhook(
                event_type: str,
                request: Request,
                _: None = Depends(_verify_token),
            ) -> Dict[str, Any]:
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                data = {"event_type": event_type, "payload": body}
                await monitor._process_webhook(data)
                return {"status": "ok", "event_type": event_type}

            @app.get("/health")
            async def health() -> Dict[str, str]:
                return {"status": "running"}

            config = uvicorn.Config(
                app,
                host=self._host,
                port=self._port,
                log_level="warning",
            )
            self._server = uvicorn.Server(config)
            logger.info(
                f"Webhook Receiver starting on {self._host}:{self._port}{self._path_prefix}/<event_type>"
            )
            # Run uvicorn in background so it doesn't block the event loop
            asyncio.create_task(self._server.serve())
        except ImportError as exc:
            logger.warning(
                f"CustomWebhookReceiver: FastAPI/uvicorn not installed – {exc}. "
                "Install with: pip install fastapi uvicorn"
            )
        except Exception as exc:
            logger.error(f"Webhook server start error: {exc}")
            self._errors += 1

    def register_handler(self, event_type: str, handler: WebhookHandler) -> None:
        """Register an async handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Webhook handler registered for event_type='{event_type}'")

    async def _process_webhook(self, data: dict) -> None:
        """Process an incoming webhook payload and notify registered handlers."""
        event_type = data.get("event_type", "generic")
        payload = data.get("payload", {})

        content = self._build_content(event_type, payload)
        assets = self._extract_mentioned_assets(content)
        urgency = self._calculate_urgency(content, author_influence=0.5)

        item = DataItem(
            source_type=self.source_type,
            source_name=f"webhook/{event_type}",
            content=content,
            timestamp=_utcnow(),
            raw_data=payload,
            metadata={"event_type": event_type},
            relevance_score=0.8,
            urgency_score=urgency,
            mentioned_assets=assets,
        )
        self._items.append(item)
        self._items_collected += 1
        if len(self._items) > 500:
            self._items = self._items[-500:]

        # Invoke registered handlers
        handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                await handler(payload)
            except Exception as exc:
                logger.warning(f"Webhook handler error for event '{event_type}': {exc}")

        self._last_update = _utcnow()

    def _build_content(self, event_type: str, payload: dict) -> str:
        """Build a human-readable content string from the webhook payload."""
        try:
            return f"Webhook [{event_type}]: " + json.dumps(payload, default=str)[:500]
        except Exception:
            return f"Webhook [{event_type}]: (unparseable payload)"
