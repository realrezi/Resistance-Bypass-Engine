# Workflow: Build Resistance Bypass Engine
- Phase 1: Initialize `pyproject.toml` dependencies and `src/schemas/models.py`.
- Phase 2: Build `src/services/id_mapper.py` and async API clients in `src/clients/` with resilience, caching, pagination, and `withdrawn_flag=false`.
- Phase 3: Build `src/engine/graph_builder.py` and `src/engine/scorer.py` with self-loop removal, LCC extraction, weighted centrality, and normalization guards.
- Phase 4: Build FastAPI REST endpoints in `src/main.py`. Offload `scorer.py` graph math using `await asyncio.to_thread()`.
- Phase 5: Build integration tests in `tests/test_pipeline.py`.
