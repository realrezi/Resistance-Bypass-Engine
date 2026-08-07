# Targeted Oncology Resistance Bypass Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2.1-orange.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** [Ahmadreza Shirdel](https://github.com/realrezi)  
**Live Platform:** [https://resistance-bypass-engine.vercel.app/](https://resistance-bypass-engine.vercel.app/)

---

## Executive Summary

The **Targeted Oncology Resistance Bypass Engine** is an open-source computational biology microservice designed to model acquired drug resistance mechanisms in human malignancies. When cancer patients experience disease progression on frontline targeted therapies (such as EGFR, ALK, BRAF, or KRAS inhibitors), tumor cells frequently acquire secondary resistance through **on-target gatekeeper mutations** (e.g., *EGFR* C797S, *ABL1* T315I) or **off-target receptor tyrosine kinase (RTK) bypass hyperactivation** (e.g., *MET* amplification, *HER2* overexpression, *PIK3CA* activation).

This engine resolves biological target identifiers, queries REST/GraphQL bioactivity endpoints with automated subnetwork expansion, constructs an undirected signaling network graph in `NetworkX`, evaluates **Hub-Penalized Bottleneck Centrality**, and ranks active, non-withdrawn clinical combination therapies to overcome therapeutic resistance.

---

## Key Features

- **Zero PDF/OCR Scraping:** Operates 100% on clean, structured REST and GraphQL biological APIs (Open Targets, ChEMBL, STRING-DB).
- **Deterministic Graph Mathematics:** Evaluates Dijkstra shortest path distances, isolates the Largest Connected Component ($G_{\text{LCC}}$), strips self-loops, and guards against empty topologies.
- **Hub-Penalized Bottleneck Centrality:** Penalizes non-specific promiscuous super-hubs while identifying critical signaling bottlenecks.
- **Interactive Biological Network Visualization:** Dynamic Cytoscape.js visualizer with interaction confidence edge-scaling and node inspection.
- **Deep Multi-Omics Node Inspector & 3D Structure Viewer:** Integrates Ensembl IDs, UniProt accession codes, COSMIC clinical resistance hotspot variants, and interactive 3D protein structure rendering via `3Dmol.js`.
- **Prevalent Clinical Resistance Matrix:** Interactive clinical scenario selector covering 9 tumor types (NSCLC, Breast Carcinoma, Colorectal, Melanoma, CML/AML, Prostate, Ovarian, Glioma, Thyroid).

---

## Mathematical Formulation & Scoring Engine

### 1. Network Topology Construction

Given a primary target gene $T_{\text{primary}}$ and a secondary resistance marker $M_{\text{resistance}}$, the engine queries the STRING-DB REST API to retrieve physical protein-protein interactions (PPI) with confidence scores $w \ge 0.400$. The resulting undirected graph $G = (V, E, W)$ is constructed with edge weights defined by confidence scores $w(u, v) \in [0.4, 1.0]$.

To prevent topology fragmentation, the engine extracts the Largest Connected Component ($G_{\text{LCC}}$):

$$G_{\text{LCC}} = \arg\max_{C \subseteq G} |V(C)| \quad \text{subject to } |V(G_{\text{LCC}})| \ge 2$$

---

### 2. Hub-Penalized Bottleneck Centrality

Standard betweenness centrality $C_B(v)$ identifies bottleneck nodes that control signal flow:

$$C_B(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

To prevent non-specific promiscuous proteins from dominating network topology, the engine applies a logarithmic degree penalty:

$$C_{\text{target}}(v) = \frac{C_B(v) + 0.5 \times \frac{\text{degree}(v)}{\text{max\_degree}(G_{\text{LCC}})}}{\log_2(\text{degree}(v) + 2)}$$

This formulation guarantees non-zero score attribution for peripheral secondary targets while penalizing topological super-hubs.

---

### 3. Multi-Objective Combination Synergy Scoring

Candidate combination therapies targeting secondary target $v$ are scored based on topological proximity, centrality bottlenecking, and bioactivity affinity ($p\text{ChEMBL} = -\log_{10}(\text{IC}_{50} \times 10^{-9})$):

#### **Case A: Bioactivity Affinity Available ($p\text{ChEMBL}$ Present)**

$$S_{\text{combination}} = 0.40 \cdot C_{\text{norm}}(v) + 0.30 \cdot (1.0 - d_{\text{norm}}(s, v)) + 0.30 \cdot \text{Aff}_{\text{norm}}(v)$$

#### **Case B: Bioactivity Affinity Missing**

$$S_{\text{combination}} = 0.55 \cdot C_{\text{norm}}(v) + 0.45 \cdot (1.0 - d_{\text{norm}}(s, v))$$

Where $d(s, v)$ is the Dijkstra shortest path length between the primary target $s$ and secondary target $v$.

---

## Data Architecture & External Services

| API Service | Data Type | Protocol | Compliance / Caching |
| :--- | :--- | :--- | :--- |
| **Open Targets Platform** | Disease-target associations & clinical drug trials | GraphQL | Async `httpx`, custom `User-Agent` |
| **ChEMBL Database** | Compound bioactivity ($\text{IC}_{50}$ / $K_i$) & mechanism of action | REST API v33 | Exhaustive pagination, non-withdrawn drug filter |
| **STRING-DB** | Physical protein-protein interaction network graph | REST API v12 | Confidence threshold $w \ge 0.400$ |
| **Ensembl & UniProt** | Gene loci, Ensembl IDs, UniProt accessions, 3D PDB models | REST / Static | `diskcache` 7-day TTL, 1GB cache limit |

---

## Repository Structure

```
resistance-bypass-engine/
├── api/
│   └── index.py             # Vercel serverless entrypoint
├── src/
│   ├── clients/             # Async HTTP API clients (ChEMBL, Open Targets, STRING-DB)
│   │   ├── base.py
│   │   ├── chembl.py
│   │   ├── open_targets.py
│   │   └── string_db.py
│   ├── engine/              # Graph math & synergy scoring engine
│   │   ├── graph_builder.py
│   │   └── scorer.py
│   ├── schemas/             # Pydantic v2 data contracts
│   │   └── models.py
│   ├── services/            # Bio-data services & multi-omics annotation lookup
│   │   ├── gene_annotation.py
│   │   └── id_mapper.py
│   └── main.py              # FastAPI application & clinical workstation UI
├── tests/                   # Pytest automated test suite (25 tests)
├── pyproject.toml           # Python dependencies & project config
├── requirements.txt         # Vercel deployment requirements
└── README.md                # Project documentation
```

---

## Getting Started

### Prerequisites
- **Python:** Version 3.11 or higher
- **Package Manager:** `uv` or `pip`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/realrezi/Resistance-Bypass-Engine.git
   cd Resistance-Bypass-Engine
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. Run the automated test suite:
   ```bash
   uv run pytest -v
   ```

4. Start the development server:
   ```bash
   uv run uvicorn src.main:app --reload --port 8000
   ```
   Navigate to `http://localhost:8000` to access the clinical workstation.

---

## REST API Specification

### `POST /api/v1/bypass-candidates`

Computes active combination therapies for a given drug resistance scenario.

#### **Request Body Example:**
```json
{
  "primary_target_symbol": "EGFR",
  "primary_drug_name": "Osimertinib",
  "secondary_resistance_marker": "MET",
  "cancer_indication": "Non-Small Cell Lung Cancer"
}
```

#### **Response Body Example:**
```json
{
  "primary_target_resolved": "EGFR",
  "secondary_marker_resolved": "MET",
  "mechanistic_branch": "Off-Target Bypass Hyperactivation",
  "network_nodes": [
    { "id": "EGFR", "degree": 14, "role": "primary" },
    { "id": "MET", "degree": 10, "role": "resistance" },
    { "id": "GRB2", "degree": 16, "role": "secondary" }
  ],
  "ranked_combinations": [
    {
      "secondary_drug": "Capmatinib",
      "secondary_target": "MET",
      "clinical_phase": 4,
      "synergy_score": 0.8420,
      "hub_penalized_centrality": 0.6250,
      "shortest_path_distance": 2.0,
      "biological_rationale": "Dual EGFR (Osimertinib) + MET (Capmatinib) inhibition neutralizes parallel ERBB3/PI3K reactivation."
    }
  ]
}
```

---

## Author & Contact

**Ahmadreza Shirdel**  
- **GitHub:** [https://github.com/realrezi](https://github.com/realrezi)  
- **Project Repo:** [https://github.com/realrezi/Resistance-Bypass-Engine](https://github.com/realrezi/Resistance-Bypass-Engine)  
- **Live Platform:** [https://resistance-bypass-engine.vercel.app/](https://resistance-bypass-engine.vercel.app/)

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
