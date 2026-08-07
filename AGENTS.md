# Project: Targeted Oncology Resistance Bypass Engine
**Goal:** An open-source microservice that models acquired drug resistance pathways in cancer. Given a primary drug/target and a secondary resistance marker, the engine resolves canonical biological IDs, queries network APIs with automated subnetwork expansion, constructs an undirected signaling graph in NetworkX, calculates hub-penalized bottleneck nodes, and ranks active, non-withdrawn clinical dual-drug combination therapies.
**Core Technical Principles:**
1. Zero PDF/OCR Scraping: Operates 100% on clean, structured REST/GraphQL APIs.
2. Deterministic Graph Math: Graph analysis runs via pure Python (`NetworkX`, `SciPy`). Computations must guard against empty topologies, extract the Largest Connected Component (LCC), and strip self-loops.
3. Resilience & Concurrency: Async API calls with `asyncio.Semaphore(5)`, exponential backoff via `tenacity` (yielding semaphore), and exhaustive pagination loops.
4. Thread Boundaries: Heavy NetworkX CPU-bound math MUST be offloaded via `asyncio.to_thread()`. All `httpx` async I/O MUST complete on the main event loop prior to offloading.
5. Caching & Compliance: Cache *only* primitive types via `diskcache`. Enforce a 7-day TTL and 1GB cache size limit.
6. Network Compliance: All HTTP clients must inject a custom `User-Agent` and `mailto:` header.
**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, NetworkX, SciPy, NumPy, statistics, httpx, gql, tenacity, diskcache, Pydantic v2, pytest, pytest-asyncio.
