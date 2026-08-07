# Targeted Oncology Resistance Bypass Engine

An open-source, research-use-only service for exploring acquired resistance networks and prioritizing evidence-linked target–drug pairs for expert review.

[![Live Demo](https://img.shields.io/badge/Live_Workstation-resistance--bypass--engine.vercel.app-0080FF)](https://resistance-bypass-engine.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)

> **Important:** This software does not measure experimental drug synergy and must not be used to select treatment for a patient. Its output is hypothesis-generating and requires independent biological, pharmacological, and clinical review.

## What version 0.2 does

Given a primary drug/target, resistance marker, cancer type, and optional alteration, the engine:

1. Resolves canonical HGNC, Ensembl, UniProt, and ChEMBL target identifiers.
2. Retrieves a human **physical** interaction neighborhood from STRING.
3. Requires the primary and resistance targets to share a connected component.
4. Computes deterministic hub-penalized topology and weighted shortest paths.
5. Selects the resistance marker plus high-priority intermediary network targets.
6. Retrieves target-linked clinical candidates, disease labels, trial reports, status, and warnings from Open Targets.
7. Excludes the primary drug, withdrawn/stopped records, and phase-1-only candidates.
8. Follows ChEMBL pagination up to a documented 50,000-record safety cap and filters human binding activities (IC50, Ki, and Kd).
9. Returns a decomposed **Heuristic Combination Priority Score**, evidence links, limitations, warnings, and provenance.

The engine never invents a fallback drug, identifier, phase, PDB structure, or clinical claim when evidence is missing.

## What it does not do

- It does not calculate Bliss, Loewe, HSA, ZIP, or another experimental synergy measure.
- A target-linked drug is not automatically a validated combination with the primary drug.
- STRING physical associations are undirected and are not a signed, tissue-specific signaling model.
- Clinical stage can be global rather than approval for the requested indication.
- On-target results are not variant-sensitive unless a future curated sensitivity layer supplies that evidence.
- Safety, dosing, pharmacokinetic compatibility, blood–brain barrier exposure, and patient-specific factors are not modeled.

These limitations are also returned in API responses instead of being hidden by the UI.

## Priority model

For candidate drug \(d\) targeting node \(v\), the service reports four bounded components:

- **Topology:** a stable transform of hub-penalized composite centrality.
- **Proximity:** \(e^{-distance(primary,v)}\) over confidence-derived edge costs.
- **Pharmacology:** a logistic transform of median filtered pChEMBL, when available.
- **Clinical evidence:** phase, indication match, pair-level report mention, and report status.

The default priority is:

\[
P(d,v)=0.30T(v)+0.25D(v)+0.20A(d,v)+0.25E(d,v)
\]

If pharmacology is unavailable, remaining weights are renormalized. Components use fixed transforms: adding or removing another candidate does not change an existing candidate's score. A missing graph target is rejected rather than assigned a default distance.

This is deliberately named a priority score—not a synergy score. The legacy `synergy_score` JSON field is retained temporarily for API compatibility and contains the same heuristic value.

## API

### Request

```json
POST /api/v1/analyze-resistance
{
  "primary_drug": "Osimertinib",
  "primary_target": "EGFR",
  "primary_alteration": "L858R",
  "resistance_marker": "MET",
  "resistance_alteration": "amplification",
  "resistance_alteration_type": "amplification",
  "cancer_type": "Non-Small Cell Lung Cancer"
}
```

### Response highlights

- Canonical identifiers and alteration context
- Relevant physical-network nodes and edges
- Primary-to-resistance weighted distance
- Ranked phase-2+ target–drug research candidates
- Score components rather than only one opaque number
- Indication and pair-evidence flags
- Trial/ChEMBL record links
- Candidate-specific limitations
- Global warnings and source provenance

Interactive OpenAPI documentation is available at `/docs`.

## Architecture

```text
FastAPI request
  ├─ canonical ID mapping (HGNC → UniProt → ChEMBL)
  ├─ STRING physical-network retrieval
  ├─ NetworkX component validation and target discovery (worker thread)
  ├─ concurrent Open Targets + ChEMBL evidence retrieval
  ├─ NetworkX scoring and serialization (worker thread)
  └─ evidence-rich Pydantic response
```

Network calls use a shared pooled `httpx.AsyncClient`, a global concurrency limit of five, selective exponential backoff with jitter, a seven-day JSON-only disk cache, and identifying request headers. Graph computation is kept outside the event loop.

## Installation

```bash
git clone https://github.com/realrezi/Resistance-Bypass-Engine.git
cd Resistance-Bypass-Engine

uv venv
source .venv/bin/activate
uv sync --extra dev

uv run pytest --cov=src
uv run uvicorn src.main:app --reload --port 8000
```

Optional environment variables:

```bash
export CONTACT_EMAIL="maintainer@example.org"
export ALLOWED_ORIGINS="http://localhost:8000,https://your-domain.example"
export CACHE_DIR="/tmp/bypass_engine_cache"
```

## Verification strategy

The deterministic suite covers:

- canonicalization and alias handling;
- physical-network request parameters;
- self-loop, duplicate-edge, and technical-hub handling;
- connected-component requirements;
- deterministic target discovery;
- stable score transforms and missing-evidence behavior;
- ChEMBL bounded pagination and aggregation;
- Open Targets disease, trial, status, and warning parsing;
- API branching, error mapping, security headers, and absence of fabricated fallbacks.

Live contract smoke tests should be run separately because upstream schemas and availability can change.

## Data sources

- [HGNC REST API](https://www.genenames.org/help/rest/)
- [UniProt REST API](https://www.uniprot.org/help/api)
- [STRING API](https://string-db.org/help/api/)
- [Open Targets GraphQL API](https://platform-docs.opentargets.org/data-access/graphql-api)
- [ChEMBL Data Web Services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)

Users are responsible for reviewing source-specific licensing, attribution, and acceptable-use requirements before redistribution or commercial use.

## Roadmap toward scientific validation

- Add signed, directed, tissue-specific pathway evidence.
- Add structured HGVS/variant normalization and variant-specific sensitivity evidence.
- Add true pair-level trial-arm and combination-response datasets.
- Add toxicity, essentiality, exposure, and pharmacokinetic compatibility.
- Publish a versioned positive/negative benchmark and temporal validation protocol.
- Calibrate or replace the heuristic score using held-out experimental evidence.

## License

MIT. See [LICENSE](LICENSE).

Author: [Ahmadreza Shirdel](https://github.com/realrezi)
