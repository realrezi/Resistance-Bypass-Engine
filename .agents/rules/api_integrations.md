# Rule: Biological API Client Contracts & ID Resolution
- Global Requirements: Identifying `User-Agent` and contact header. Retry only timeouts/network errors, HTTP 429, and HTTP 5xx with jitter. Nest `async with semaphore:` inside retry. Cache JSON-compatible values only with a 7-day TTL and 1GB limit.
- ID Mapper (`src/services/id_mapper.py`): HGNC REST API (strip Ensembl version suffixes via `.split('.')[0]`), UniProt REST API, ChEMBL Target API (filter `target_type == 'SINGLE PROTEIN'`).
- STRING-DB (`src/clients/string_db.py`): Endpoint `/api/json/network`, human, score >=400, add_nodes=25, `network_type=physical`, and caller identity.
- Open Targets (`src/clients/open_targets.py`): Fetch current `drugAndClinicalCandidates` disease, trial-report, status, warning, and mechanism fields. Surface GraphQL errors.
- ChEMBL Client (`src/clients/chembl.py`): Follow `page_meta.next` with an explicit safety cap; use human binding assays, IC50/Ki/Kd, valid non-duplicate records, and return median plus measurement metadata.
