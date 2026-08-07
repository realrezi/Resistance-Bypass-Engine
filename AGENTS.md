# Project: Targeted Oncology Resistance Bypass Engine
**Goal:** An open-source, research-use-only microservice that explores acquired drug-resistance networks. Given a primary drug/target, resistance marker, cancer context, and optional alteration, it resolves canonical IDs, builds a STRING physical-association graph, discovers targetable nodes, and prioritizes active, non-withdrawn clinical target–drug records with explicit evidence and limitations. It must never describe its heuristic priority as measured synergy or imply that two drugs form a validated combination without pair-level evidence.
**Core Technical Principles:**
1. Zero PDF/OCR Scraping: Operates 100% on clean, structured REST/GraphQL APIs.
2. Deterministic Graph Math: Graph analysis runs via pure Python (`NetworkX`, `SciPy`). Computations must guard against empty topologies, extract the Largest Connected Component (LCC), and strip self-loops.
3. Resilience & Concurrency: Async API calls with `asyncio.Semaphore(5)`, exponential backoff via `tenacity` (yielding semaphore), and exhaustive pagination loops.
4. Thread Boundaries: Heavy NetworkX CPU-bound math MUST be offloaded via `asyncio.to_thread()`. All `httpx` async I/O MUST complete on the main event loop prior to offloading.
5. Caching & Compliance: Cache *only* primitive types via `diskcache`. Enforce a 7-day TTL and 1GB cache size limit.
6. Network Compliance: All HTTP clients must inject an identifying `User-Agent` and contact header.
7. Evidence Honesty: Missing values remain null/empty; never fabricate drugs, identifiers, phases, structures, clinical status, or biological claims.
8. Research Safety: Every report must state that the score is heuristic and the output is not clinical decision support.
**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, NetworkX, statistics, httpx, tenacity, diskcache, Pydantic v2, pytest, pytest-asyncio.
