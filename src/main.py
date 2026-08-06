import asyncio
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.clients.base import cache
from src.clients.chembl import ChEMBLClient
from src.clients.open_targets import OpenTargetsClient
from src.clients.string_db import StringDBClient
from src.engine.graph_builder import build_signaling_graph
from src.engine.scorer import PathwayScorer
from src.schemas.models import (
    CombinationCandidate,
    ResistanceBypassReport,
    ResistanceRequest,
)
from src.services.id_mapper import IDMapper

app = FastAPI(
    title="Targeted Oncology Resistance Bypass Engine",
    description="Microservice modeling acquired drug resistance pathways in cancer",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Targeted Oncology Resistance Bypass Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070a12;
            --card-bg: rgba(15, 23, 42, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --accent-purple: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            --accent-emerald: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.06) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.06) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            line-height: 1.5;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            background: rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.25);
            color: #38bdf8;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 680px;
            margin: 0 auto 1.5rem auto;
        }

        .nav-links {
            display: flex;
            justify-content: center;
            gap: 1rem;
        }

        .nav-link {
            color: #94a3b8;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            border: 1px solid var(--card-border);
            transition: all 0.2s ease;
        }

        .nav-link:hover {
            color: #fff;
            border-color: rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 850px) {
            .grid-layout { grid-template-columns: 1fr; }
        }

        .glass-panel {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        .panel-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-group {
            margin-bottom: 1.1rem;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 0.4rem;
        }

        input {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            color: #fff;
            font-family: var(--font-main);
            font-size: 0.95rem;
            transition: border-color 0.2s ease;
        }

        input:focus {
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        .preset-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 1.25rem;
        }

        .btn-preset {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #94a3b8;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-preset:hover {
            color: #fff;
            border-color: #38bdf8;
            background: rgba(56, 189, 248, 0.08);
        }

        .btn-submit {
            width: 100%;
            padding: 0.85rem;
            background: var(--primary-gradient);
            border: none;
            border-radius: 8px;
            color: #070a12;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }

        .metric-val {
            font-size: 1.5rem;
            font-weight: 800;
            color: #38bdf8;
            font-family: var(--font-mono);
        }

        .metric-lbl {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }

        .therapy-card {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease;
        }

        .therapy-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
        }

        .therapy-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.6rem;
        }

        .drug-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f1f5f9;
        }

        .phase-badge {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .synergy-bar-bg {
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.6rem 0;
        }

        .synergy-bar-fill {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 999px;
            width: 0%;
            transition: width 0.6s ease;
        }

        .rationale {
            font-size: 0.85rem;
            color: #94a3b8;
            line-height: 1.4;
        }

        .loader {
            display: none;
            text-align: center;
            padding: 3rem 1rem;
            color: #94a3b8;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top-color: #38bdf8;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .placeholder-state {
            text-align: center;
            padding: 4rem 1rem;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">🧬 Targeted Oncology Microservice</div>
            <h1>Resistance Bypass Engine</h1>
            <p class="subtitle">Predict acquired drug resistance mechanisms, resolve canonical biological IDs, expand PPI signaling networks, and rank clinical combination therapies.</p>
            <div class="nav-links">
                <a href="/docs" target="_blank" class="nav-link">⚡ OpenAPI Docs (/docs)</a>
                <a href="/health" target="_blank" class="nav-link">💚 Health Diagnostics</a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" class="nav-link">📦 GitHub Repository</a>
            </div>
        </header>

        <div class="grid-layout">
            <!-- Sidebar Form -->
            <div class="glass-panel">
                <div class="panel-title">🎯 Analysis Inputs</div>
                
                <div class="form-group">
                    <label>Quick Demo Presets</label>
                    <div class="preset-buttons">
                        <button class="btn-preset" onclick="setPreset('EGFR', 'Osimertinib', 'MET')">EGFR + MET (Bypass)</button>
                        <button class="btn-preset" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR')">EGFR + EGFR (On-Target)</button>
                        <button class="btn-preset" onclick="setPreset('HER2', 'Trastuzumab', 'MET')">HER2 + MET (Alias)</button>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label for="primary_target">Primary Target Symbol</label>
                        <input type="text" id="primary_target" value="EGFR" required placeholder="e.g. EGFR, ERBB2">
                    </div>
                    <div class="form-group">
                        <label for="primary_drug">Primary Drug Name</label>
                        <input type="text" id="primary_drug" value="Osimertinib" required placeholder="e.g. Osimertinib">
                    </div>
                    <div class="form-group">
                        <label for="resistance_marker">Secondary Resistance Marker</label>
                        <input type="text" id="resistance_marker" value="MET" required placeholder="e.g. MET, KRAS">
                    </div>
                    <div class="form-group">
                        <label for="cancer_type">Cancer Indication</label>
                        <input type="text" id="cancer_type" value="Non-Small Cell Lung Cancer">
                    </div>
                    <button type="submit" id="submitBtn" class="btn-submit">🚀 Run Resistance Pipeline</button>
                </form>
            </div>

            <!-- Results Panel -->
            <div class="glass-panel">
                <div class="panel-title">📊 Resistance Analysis & Synergy Ranking</div>

                <div id="loader" class="loader">
                    <div class="spinner"></div>
                    <p>Resolving HGNC/UniProt IDs & Querying STRING-DB / Open Targets...</p>
                </div>

                <div id="placeholder" class="placeholder-state">
                    <p>Enter biological targets on the left or select a preset to analyze signaling graph topologies.</p>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div class="metrics-summary">
                        <div class="metric-card">
                            <div class="metric-val" id="resTypeVal">-</div>
                            <div class="metric-lbl">Resistance Type</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="nodesCountVal">0</div>
                            <div class="metric-lbl">Network Nodes</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="distVal">0.0</div>
                            <div class="metric-lbl">Shortest Distance</div>
                        </div>
                    </div>

                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: #e2e8f0;">Ranked Dual-Drug Combinations</h3>
                    <div id="therapiesList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setPreset(target, drug, marker) {
            document.getElementById('primary_target').value = target;
            document.getElementById('primary_drug').value = drug;
            document.getElementById('resistance_marker').value = marker;
            runAnalysis(new Event('submit'));
        }

        async function runAnalysis(e) {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            const loader = document.getElementById('loader');
            const placeholder = document.getElementById('placeholder');
            const resultsContent = document.getElementById('resultsContent');
            const therapiesList = document.getElementById('therapiesList');

            submitBtn.disabled = true;
            placeholder.style.display = 'none';
            resultsContent.style.display = 'none';
            loader.style.display = 'block';

            const payload = {
                primary_target: document.getElementById('primary_target').value,
                primary_drug: document.getElementById('primary_drug').value,
                resistance_marker: document.getElementById('resistance_marker').value,
                cancer_type: document.getElementById('cancer_type').value
            };

            try {
                const response = await fetch('/api/v1/analyze-resistance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Analysis failed');

                document.getElementById('resTypeVal').innerText = data.resistance_type;
                document.getElementById('nodesCountVal').innerText = data.pathway_nodes_count;
                document.getElementById('distVal').innerText = data.shortest_path_distance.toFixed(2);

                therapiesList.innerHTML = '';
                data.ranked_combinations.forEach((c, idx) => {
                    const pct = Math.round(c.synergy_score * 100);
                    const card = document.createElement('div');
                    card.className = 'therapy-card';
                    card.innerHTML = `
                        <div class="therapy-header">
                            <span class="drug-title">#${idx+1} ${c.secondary_drug} + ${payload.primary_drug}</span>
                            <span class="phase-badge">Phase ${c.clinical_phase}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #38bdf8; margin-bottom: 0.2rem;">
                            Target: <strong>${c.secondary_target}</strong> | Synergy Score: <strong>${c.synergy_score}</strong> | Hub Penalty Centrality: ${c.hub_penalized_centrality}
                        </div>
                        <div class="synergy-bar-bg">
                            <div class="synergy-bar-fill" style="width: ${pct}%"></div>
                        </div>
                        <div class="rationale">${c.biological_rationale}</div>
                    `;
                    therapiesList.appendChild(card);
                });

                resultsContent.style.display = 'block';
            } catch (err) {
                alert('Error: ' + err.message);
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

def _sync_build_and_score(
    interactions: List[Dict[str, Any]],
    primary_target: str,
    raw_candidates: List[Dict[str, Any]],
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """CPU-bound worker function offloaded via asyncio.to_thread."""
    G = build_signaling_graph(interactions)
    if len(G.nodes) < 2:
        raise ValueError("NoPathwayFound: Insufficient biological interactions.")

    scored = PathwayScorer.score_candidates(G, primary_target, raw_candidates)
    pathway_nodes_count = len(G.nodes)

    if scored:
        shortest_path_distance = float(scored[0].get("shortest_path_distance", 2.0))
    else:
        shortest_path_distance = 2.0

    return pathway_nodes_count, shortest_path_distance, scored


@app.get("/", response_class=HTMLResponse)
async def root_dashboard() -> str:

    """Serve the interactive web UI dashboard for the Resistance Bypass Engine."""
    return INDEX_HTML


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check diagnostic endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "cache_size_bytes": cache.volume(),
        "cache_limit_bytes": cache.size_limit,
    }


def _is_drug_withdrawn(drug_row: Dict[str, Any]) -> bool:
    """Check if an Open Targets drug row indicates a withdrawn status."""
    status = (drug_row.get("status") or "").strip().lower()
    return status == "withdrawn"


@app.post("/api/v1/analyze-resistance", response_model=ResistanceBypassReport)
async def analyze_resistance(req: ResistanceRequest) -> ResistanceBypassReport:
    """Analyze drug resistance pathways and rank dual-drug combination candidates."""
    id_mapper = IDMapper()

    try:
        mapped_primary = await id_mapper.map_identifier(req.primary_target)
        mapped_resistance = await id_mapper.map_identifier(req.resistance_marker)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"ID Resolution failed: {str(e)}")

    primary_target_canonical = mapped_primary.canonical_symbol
    resistance_marker_canonical = mapped_resistance.canonical_symbol

    chembl_client = ChEMBLClient()

    # Branching Evaluation
    if primary_target_canonical == resistance_marker_canonical:
        # On-Target Mutation Branching
        resistance_type = "On-Target Mutation"
        pathway_nodes_count = 1
        shortest_path_distance = 0.0

        molecules = await chembl_client.get_clinical_molecules(
            target_chembl_id=mapped_primary.chembl_target_id,
            max_phase_gte=2,
            withdrawn_flag=False,
        )

        ranked_combinations: List[CombinationCandidate] = []
        for mol in molecules[:10]:
            drug_name = mol.get("pref_name") or mol.get("molecule_chembl_id", "Unknown Drug")
            max_phase = mol.get("max_phase", 2) or 2

            ranked_combinations.append(
                CombinationCandidate(
                    secondary_drug=drug_name.upper(),
                    secondary_target=primary_target_canonical,
                    mechanism_of_action="Next-Generation Inhibitor",
                    clinical_phase=int(max_phase),
                    is_withdrawn=False,
                    synergy_score=1.0,
                    hub_penalized_centrality=1.0,
                    chembl_ic50_nm=None,
                    biological_rationale=f"On-target mutation in {primary_target_canonical}. Next-generation inhibitor overrides resistance.",
                )
            )

        if not ranked_combinations:
            # Fallback candidate if no specific clinical molecule returned
            ranked_combinations.append(
                CombinationCandidate(
                    secondary_drug=f"Next-Gen {primary_target_canonical} Inhibitor",
                    secondary_target=primary_target_canonical,
                    mechanism_of_action="Next-Generation Inhibitor",
                    clinical_phase=3,
                    is_withdrawn=False,
                    synergy_score=1.0,
                    hub_penalized_centrality=1.0,
                    chembl_ic50_nm=None,
                    biological_rationale=f"On-target mutation in {primary_target_canonical}. Next-generation inhibitor overrides resistance.",
                )
            )

        return ResistanceBypassReport(
            primary_target_canonical=primary_target_canonical,
            resistance_marker_canonical=resistance_marker_canonical,
            resistance_type=resistance_type,
            pathway_nodes_count=pathway_nodes_count,
            shortest_path_distance=shortest_path_distance,
            ranked_combinations=ranked_combinations,
        )

    else:
        # Off-Target Bypass Branching
        resistance_type = "Off-Target Bypass"

        string_client = StringDBClient()
        ot_client = OpenTargetsClient()

        # Fetch STRING-DB network
        interactions = await string_client.get_network(
            primary_target_canonical, resistance_marker_canonical
        )

        # Fetch Open Targets known drugs for resistance marker
        ot_drugs = await ot_client.get_known_drugs(mapped_resistance.ensembl_id)

        # Filter out withdrawn drugs — AGENTS.md: "ranks active, non-withdrawn clinical dual-drug combination therapies"
        ot_drugs = [d for d in ot_drugs if not _is_drug_withdrawn(d)]

        # Fetch ChEMBL activities for resistance target if available
        activity_map: Dict[str, float] = {}
        if mapped_resistance.chembl_target_id:
            activity_map = await chembl_client.get_target_activities(
                mapped_resistance.chembl_target_id
            )

        raw_candidates: List[Dict[str, Any]] = []

        if ot_drugs:
            for drug in ot_drugs:
                drug_name = drug.get("prefName") or drug.get("drugId") or "Unknown"
                moa = drug.get("mechanismOfAction") or "Bypass Pathway Inhibitor"
                phase = drug.get("phase", 2) or 2
                target_sym = drug.get("targetSymbol") or resistance_marker_canonical

                # Try matching activity by both prefName and drugId
                pchembl_val = activity_map.get(drug_name.upper())
                if pchembl_val is None:
                    drug_id = drug.get("drugId", "")
                    if drug_id:
                        pchembl_val = activity_map.get(drug_id.upper())

                raw_candidates.append(
                    {
                        "secondary_drug": drug_name.upper(),
                        "secondary_target": target_sym.upper(),
                        "mechanism_of_action": moa,
                        "clinical_phase": phase,
                        "is_withdrawn": False,
                        "pchembl_value": pchembl_val,
                        "biological_rationale": f"Inhibits bypass target {target_sym.upper()} to restore sensitivity to primary drug targeting {primary_target_canonical}.",
                    }
                )
        else:
            # Fallback candidate for resistance marker
            raw_candidates.append(
                {
                    "secondary_drug": f"{resistance_marker_canonical} Inhibitor",
                    "secondary_target": resistance_marker_canonical,
                    "mechanism_of_action": "Bypass Pathway Inhibitor",
                    "clinical_phase": 2,
                    "is_withdrawn": False,
                    "pchembl_value": None,
                    "biological_rationale": f"Inhibits bypass node {resistance_marker_canonical} to bypass resistance.",
                }
            )

        # THREAD SAFETY: Offload heavy NetworkX CPU-bound math via asyncio.to_thread
        try:
            pathway_nodes_count, shortest_path_distance, scored_raw = (
                await asyncio.to_thread(
                    _sync_build_and_score,
                    interactions,
                    primary_target_canonical,
                    raw_candidates,
                )
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        ranked_combinations: List[CombinationCandidate] = [
            CombinationCandidate(
                secondary_drug=c.get("secondary_drug", "Unknown"),
                secondary_target=c.get("secondary_target", resistance_marker_canonical),
                mechanism_of_action=c.get("mechanism_of_action", "Combination Therapy"),
                clinical_phase=int(c.get("clinical_phase", 2)),
                is_withdrawn=bool(c.get("is_withdrawn", False)),
                synergy_score=float(c.get("synergy_score", 0.0)),
                hub_penalized_centrality=float(c.get("hub_penalized_centrality", 0.0)),
                chembl_ic50_nm=c.get("chembl_ic50_nm"),
                biological_rationale=c.get(
                    "biological_rationale",
                    f"Targets bypass node {c.get('secondary_target')} to overcome {resistance_marker_canonical} resistance.",
                ),
            )
            for c in scored_raw
        ]

        return ResistanceBypassReport(
            primary_target_canonical=primary_target_canonical,
            resistance_marker_canonical=resistance_marker_canonical,
            resistance_type=resistance_type,
            pathway_nodes_count=pathway_nodes_count,
            shortest_path_distance=shortest_path_distance,
            ranked_combinations=ranked_combinations,
        )
