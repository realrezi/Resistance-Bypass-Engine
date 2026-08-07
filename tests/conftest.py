import pytest

from src.clients.base import cache


@pytest.fixture(scope="session", autouse=True)
def close_disk_cache_after_tests():
    """Release diskcache's SQLite handle before pytest checks resources."""
    yield
    cache.close()
