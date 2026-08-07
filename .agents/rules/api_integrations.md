# Rule: Biological API Client Contracts & ID Resolution
- Global Requirements: Header `User-Agent: "ResistanceBypassEngine/1.0 (mailto:developer@example.com)"`. Use `@retry` from `tenacity`. Nest `async with semaphore:` INSIDE retry block. Caching via `diskcache.Cache(size_limit=1e9)` with 7-day TTL (`expire=604800`).
- ID Mapper (`src/services/id_mapper.py`): HGNC REST API (strip Ensembl version suffixes via `.split('.')[0]`), UniProt REST API, ChEMBL Target API (filter `target_type == 'SINGLE PROTEIN'`).
- STRING-DB (`src/clients/string_db.py`): Endpoint `/api/json/network`, `identifiers="{T_primary}\r{T_resistance}"`, `species=9606`, `required_score=400`, `add_nodes=25`.
- Open Targets (`src/clients/open_targets.py`): GraphQL endpoint `/api/v4/graphql`, fetch `knownDrugs(size: 100)`.
- ChEMBL Client (`src/clients/chembl.py`): Endpoint `/api/data/molecule.json`, `max_phase>=2`, `withdrawn_flag=false`, `limit=100`. Iterate `page_meta.next` until all pages fetched. Group by `pref_name` using `median(pchembl_value)`.
