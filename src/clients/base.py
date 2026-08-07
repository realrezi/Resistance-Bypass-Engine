from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import diskcache
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "realrezi@users.noreply.github.com")
USER_AGENT = f"ResistanceBypassEngine/0.2 (mailto:{CONTACT_EMAIL})"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "From": CONTACT_EMAIL,
    "Accept": "application/json",
}
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

CACHE_DIR = os.getenv(
    "CACHE_DIR", os.path.join(tempfile.gettempdir(), "bypass_engine_cache")
)
cache = diskcache.Cache(CACHE_DIR, size_limit=int(1e9))
SEMAPHORE = asyncio.Semaphore(5)

_shared_client: httpx.AsyncClient | None = None


def _stable_cache_key(method: str, url: str, payload: Any) -> str:
    """Build a deterministic cache key using sorted JSON serialization."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"v2:{method}:{url}:{serialized}"


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_is_json_value(item) for item in value)
    return False


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def get_http_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        timeout = httpx.Timeout(30.0, connect=10.0, pool=10.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        _shared_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    return _shared_client


async def close_http_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


class BaseHTTPClient:
    def __init__(self, headers: dict[str, str] | None = None, timeout: float = 30.0):
        request_headers = DEFAULT_HEADERS.copy()
        if headers:
            request_headers.update(headers)
        self.headers = request_headers
        self.timeout = timeout

    async def _cache_get(self, key: str) -> Any:
        try:
            return await asyncio.to_thread(cache.get, key)
        except (diskcache.Timeout, OSError, sqlite3.Error):
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        if not _is_json_value(value):
            return
        try:
            await asyncio.to_thread(cache.set, key, value, expire=CACHE_TTL_SECONDS)
        except (diskcache.Timeout, OSError, sqlite3.Error):
            return

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        cache_key = _stable_cache_key("GET", url, params)
        cached_value = await self._cache_get(cache_key)
        if cached_value is not None:
            return cached_value

        merged_headers = self.headers | (headers or {})
        async with SEMAPHORE:
            response = await get_http_client().get(
                url, params=params, headers=merged_headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

        await self._cache_set(cache_key, data)
        return data

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def post_json(
        self,
        url: str,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        cache_key = _stable_cache_key("POST", url, json_data)
        cached_value = await self._cache_get(cache_key)
        if cached_value is not None:
            return cached_value

        merged_headers = self.headers | (headers or {})
        async with SEMAPHORE:
            response = await get_http_client().post(
                url, json=json_data, headers=merged_headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

        await self._cache_set(cache_key, data)
        return data
