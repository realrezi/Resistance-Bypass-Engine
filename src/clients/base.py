import asyncio
import json
import logging
import os
import tempfile
from typing import Any

import diskcache
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "developer@example.com")
USER_AGENT = f"ResistanceBypassEngine/1.0 (mailto:{CONTACT_EMAIL})"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "mailto": CONTACT_EMAIL,
    "Accept": "application/json",
}

CACHE_DIR = os.getenv(
    "CACHE_DIR", os.path.join(tempfile.gettempdir(), "bypass_engine_cache")
)
CACHE_SCHEMA_VERSION = os.getenv("CACHE_SCHEMA_VERSION", "evidence-priority-0.3")
cache = diskcache.Cache(CACHE_DIR, size_limit=int(1e9))
_SEMAPHORES_BY_LOOP: dict[int, asyncio.Semaphore] = {}
_CLIENTS_BY_LOOP: dict[int, httpx.AsyncClient] = {}


def _get_semaphore() -> asyncio.Semaphore:
    loop_key = id(asyncio.get_running_loop())
    semaphore = _SEMAPHORES_BY_LOOP.get(loop_key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(5)
        _SEMAPHORES_BY_LOOP[loop_key] = semaphore
    return semaphore


def _get_connection_pool(timeout: float) -> httpx.AsyncClient:
    """Return one pooled client per event loop to reuse keep-alive connections."""
    loop = asyncio.get_running_loop()
    loop_key = id(loop)
    client = _CLIENTS_BY_LOOP.get(loop_key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        _CLIENTS_BY_LOOP[loop_key] = client
    return client


async def close_connection_pools() -> None:
    """Close pooled clients during graceful application shutdown."""
    clients = list(_CLIENTS_BY_LOOP.values())
    _CLIENTS_BY_LOOP.clear()
    _SEMAPHORES_BY_LOOP.clear()
    for client in clients:
        if not client.is_closed:
            await client.aclose()


def _stable_cache_key(method: str, url: str, payload: Any) -> str:
    """Build a deterministic cache key using sorted JSON serialization."""
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(payload)
    return f"{CACHE_SCHEMA_VERSION}:{method}:{url}:{serialized}"


class BaseHTTPClient:
    def __init__(self, headers: dict[str, str] | None = None, timeout: float = 30.0):
        req_headers = DEFAULT_HEADERS.copy()
        if headers:
            req_headers.update(headers)
        self.headers = req_headers
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        cache_key = _stable_cache_key(
            "GET",
            url,
            {"params": params, "headers": {**self.headers, **(headers or {})}},
        )
        try:
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Cache read failed for %s: %s", cache_key, exc)

        merged_headers = self.headers.copy()
        if headers:
            merged_headers.update(headers)

        # Acquire semaphore INSIDE retry block to avoid deadlocks
        async with _get_semaphore():
            client = _get_connection_pool(self.timeout)
            response = await client.get(url, params=params, headers=merged_headers)
            response.raise_for_status()
            data = response.json()

        # Cache only primitive JSON types (dict/list) with 7-day TTL
        try:
            cache.set(cache_key, data, expire=604800)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Cache write failed for %s: %s", cache_key, exc)
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def post_json(
        self,
        url: str,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        cache_key = _stable_cache_key(
            "POST",
            url,
            {"json": json_data, "headers": {**self.headers, **(headers or {})}},
        )
        try:
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Cache read failed for %s: %s", cache_key, exc)

        merged_headers = self.headers.copy()
        if headers:
            merged_headers.update(headers)

        # Acquire semaphore INSIDE retry block to avoid deadlocks
        async with _get_semaphore():
            client = _get_connection_pool(self.timeout)
            response = await client.post(url, json=json_data, headers=merged_headers)
            response.raise_for_status()
            data = response.json()

        try:
            cache.set(cache_key, data, expire=604800)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Cache write failed for %s: %s", cache_key, exc)
        return data
