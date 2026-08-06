import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
import diskcache
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

USER_AGENT = "ResistanceBypassEngine/1.0 (mailto:developer@example.com)"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(tempfile.gettempdir(), "bypass_engine_cache"))
cache = diskcache.Cache(CACHE_DIR, size_limit=int(1e9))
SEMAPHORE = asyncio.Semaphore(5)


def _stable_cache_key(method: str, url: str, payload: Any) -> str:
    """Build a deterministic cache key using sorted JSON serialization."""
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(payload)
    return f"{method}:{url}:{serialized}"


class BaseHTTPClient:
    def __init__(self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
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
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        cache_key = _stable_cache_key("GET", url, params)
        try:
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        except Exception:
            pass

        merged_headers = self.headers.copy()
        if headers:
            merged_headers.update(headers)

        # Acquire semaphore INSIDE retry block to avoid deadlocks
        async with SEMAPHORE:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=merged_headers)
                response.raise_for_status()
                data = response.json()

        # Cache only primitive JSON types (dict/list) with 7-day TTL
        try:
            cache.set(cache_key, data, expire=604800)
        except Exception:
            pass
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
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        cache_key = _stable_cache_key("POST", url, json_data)
        try:
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        except Exception:
            pass

        merged_headers = self.headers.copy()
        if headers:
            merged_headers.update(headers)

        # Acquire semaphore INSIDE retry block to avoid deadlocks
        async with SEMAPHORE:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=json_data, headers=merged_headers)
                response.raise_for_status()
                data = response.json()

        try:
            cache.set(cache_key, data, expire=604800)
        except Exception:
            pass
        return data

