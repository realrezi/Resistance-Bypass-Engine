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
            --card-bg: rgba(15, 23, 42, 0.75);
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
            max-width: 1150px;
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
            max-width: 720px;
            margin: 0 auto 1.5rem auto;
        }

        .nav-links {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .nav-link {
            color: #94a3b8;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            border: 1px solid var(--card-border);
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .nav-link:hover {
            color: #fff;
            border-color: rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
        }

        .nav-link.highlight {
            background: rgba(168, 85, 247, 0.15);
            border-color: rgba(168, 85, 247, 0.3);
            color: #c084fc;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 900px) {
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
            justify-content: space-between;
        }

        .form-group {
            margin-bottom: 1.1rem;
        }

        label {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 0.4rem;
        }

        .info-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            font-size: 0.7rem;
            cursor: help;
            position: relative;
        }

        .tooltip-box {
            position: relative;
        }

        .tooltip-box .tooltip-text {
            visibility: hidden;
            width: 220px;
            background-color: #0f172a;
            color: #e2e8f0;
            text-align: left;
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 0.78rem;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            font-weight: 400;
            line-height: 1.35;
        }

        .tooltip-box:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
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

        .preset-section-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0.8rem 0 0.4rem 0;
        }

        .preset-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.5rem;
        }

        .btn-preset {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #94a3b8;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            font-size: 0.76rem;
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
            margin-top: 0.5rem;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .error-banner {
            display: none;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            padding: 1rem;
            color: #fca5a5;
            font-size: 0.88rem;
            margin-bottom: 1.25rem;
        }

        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
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
            font-size: clamp(1rem, 2vw, 1.4rem);
            font-weight: 800;
            color: #38bdf8;
            font-family: var(--font-mono);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .metric-lbl {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.25rem;
        }

        .canonical-tags {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .tag-pill {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            color: #7dd3fc;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-family: var(--font-mono);
        }

        .filter-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .filter-tabs {
            display: flex;
            gap: 0.3rem;
        }

        .btn-tab {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: #94a3b8;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-tab.active {
            background: rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.4);
            color: #38bdf8;
            font-weight: 600;
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
            font-size: 1.05rem;
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

        .btn-action {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-action:hover {
            color: #fff;
            border-color: #38bdf8;
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

        /* Clinician Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(7, 10, 18, 0.85);
            backdrop-filter: blur(12px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal-content {
            background: #0f172a;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            max-width: 720px;
            width: 100%;
            max-height: 85vh;
            overflow-y: auto;
            padding: 2rem;
            color: #e2e8f0;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .modal-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .modal-close {
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size: 1.5rem;
            cursor: pointer;
        }

        .glossary-item {
            margin-bottom: 1.25rem;
        }

        .glossary-term {
            font-weight: 700;
            color: #f8fafc;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .glossary-desc {
            font-size: 0.88rem;
            color: #94a3b8;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">🧬 Targeted Oncology Precision Engine</div>
            <h1>Resistance Bypass Engine</h1>
            <p class="subtitle">Models acquired drug resistance pathways in cancer, resolves canonical biological IDs, constructs NetworkX PPI graphs, and ranks active clinical dual-drug combination therapies.</p>
            <div class="nav-links">
                <a onclick="toggleModal(true)" class="nav-link highlight">📖 Clinician & Researcher Guide</a>
                <a href="/docs" target="_blank" class="nav-link">⚡ OpenAPI Docs (/docs)</a>
                <a href="/health" target="_blank" class="nav-link">💚 Diagnostics</a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" class="nav-link">📦 GitHub Repository</a>
            </div>
        </header>

        <div class="grid-layout">
            <!-- Sidebar Form -->
            <div class="glass-panel">
                <div class="panel-title">
                    <span>🎯 Analysis Inputs</span>
                </div>
                
                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label for="primary_target">
                            Primary Target Symbol
                            <span class="tooltip-box info-icon">i
                                <span class="tooltip-text">Oncogenic driver gene targeted by frontline therapy (e.g. EGFR, ERBB2, ALK, BRAF).</span>
                            </span>
                        </label>
                        <input type="text" id="primary_target" value="EGFR" required placeholder="e.g. EGFR, ERBB2">
                    </div>

                    <div class="form-group">
                        <label for="primary_drug">
                            Primary Drug Name
                            <span class="tooltip-box info-icon">i
                                <span class="tooltip-text">Frontline targeted agent administered (e.g. Osimertinib, Trastuzumab, Sotorasib).</span>
                            </span>
                        </label>
                        <input type="text" id="primary_drug" value="Osimertinib" required placeholder="e.g. Osimertinib">
                    </div>

                    <div class="form-group">
                        <label for="resistance_marker">
                            Secondary Resistance Marker
                            <span class="tooltip-box info-icon">i
                                <span class="tooltip-text">Bypass marker or secondary mutation driving acquired resistance (e.g. MET, KRAS, BRAF).</span>
                            </span>
                        </label>
                        <input type="text" id="resistance_marker" value="MET" required placeholder="e.g. MET, KRAS">
                    </div>

                    <div class="form-group">
                        <label for="cancer_type">Cancer Indication</label>
                        <input type="text" id="cancer_type" value="Non-Small Cell Lung Cancer">
                    </div>

                    <button type="submit" id="submitBtn" class="btn-submit">🚀 Run Resistance Pipeline</button>
                </form>

                <div style="margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem;">
                    <div class="preset-section-title">Lung Cancer (NSCLC) Presets</div>
                    <div class="preset-buttons">
                        <button class="btn-preset" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'NSCLC')">EGFR + MET (Bypass)</button>
                        <button class="btn-preset" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'NSCLC')">EGFR + EGFR (C797S)</button>
                        <button class="btn-preset" onclick="setPreset('ALK', 'Alectinib', 'MET', 'NSCLC')">ALK + MET (Bypass)</button>
                        <button class="btn-preset" onclick="setPreset('MET', 'Capmatinib', 'EGFR', 'NSCLC')">MET + EGFR (Reciprocal)</button>
                    </div>

                    <div class="preset-section-title">Breast & Gynecologic Presets</div>
                    <div class="preset-buttons">
                        <button class="btn-preset" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'Breast Cancer')">HER2 + MET (Bypass)</button>
                        <button class="btn-preset" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'Breast Cancer')">ESR1 + CDK4 (Cyclin)</button>
                    </div>

                    <div class="preset-section-title">Melanoma & GI Presets</div>
                    <div class="preset-buttons">
                        <button class="btn-preset" onclick="setPreset('BRAF', 'Dabrafenib', 'RAF1', 'Melanoma')">BRAF + RAF1 (MAPK)</button>
                        <button class="btn-preset" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer')">KRAS + EGFR (RTK)</button>
                        <button class="btn-preset" onclick="setPreset('AR', 'Enzalutamide', 'PIK3CA', 'Prostate Cancer')">AR + PIK3CA (PI3K)</button>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div class="glass-panel">
                <div class="panel-title">
                    <span>📊 Resistance Analysis & Synergy Ranking</span>
                    <button id="copyJsonBtn" class="btn-action" style="display: none;" onclick="copyResultJson()">📋 Copy JSON</button>
                </div>

                <div id="errorBanner" class="error-banner"></div>

                <div id="loader" class="loader">
                    <div class="spinner"></div>
                    <p>Resolving HGNC/UniProt IDs & Querying STRING-DB / Open Targets...</p>
                </div>

                <div id="placeholder" class="placeholder-state">
                    <p>Select a clinical preset or enter custom targets to model signaling graph topologies.</p>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div class="canonical-tags">
                        <span class="tag-pill" id="primaryTag">Target: -</span>
                        <span class="tag-pill" id="resistanceTag">Marker: -</span>
                    </div>

                    <div class="metrics-summary">
                        <div class="metric-card">
                            <div class="metric-val" id="resTypeVal">-</div>
                            <div class="metric-lbl">
                                Resistance Type
                                <span class="tooltip-box info-icon">i
                                    <span class="tooltip-text">Off-Target Bypass (parallel activation) vs On-Target Mutation (binding pocket alteration).</span>
                                </span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="nodesCountVal">0</div>
                            <div class="metric-lbl">
                                Network Nodes
                                <span class="tooltip-box info-icon">i
                                    <span class="tooltip-text">Number of interacting proteins in the STRING-DB signaling subnet.</span>
                                </span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="distVal">0.0</div>
                            <div class="metric-lbl">
                                Shortest Distance
                                <span class="tooltip-box info-icon">i
                                    <span class="tooltip-text">Topological distance in network between target and resistance marker.</span>
                                </span>
                            </div>
                        </div>
                    </div>

                    <div class="filter-controls">
                        <h3 style="font-size: 1rem; color: #e2e8f0;">Ranked Dual-Drug Combination Therapies</h3>
                        <div class="filter-tabs">
                            <button class="btn-tab active" onclick="filterPhase('ALL')">All Phases</button>
                            <button class="btn-tab" onclick="filterPhase(4)">Phase 4 (Approved)</button>
                            <button class="btn-tab" onclick="filterPhase(3)">Phase 3</button>
                            <button class="btn-tab" onclick="filterPhase(2)">Phase 2</button>
                        </div>
                    </div>

                    <div id="therapiesList"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Clinician Guide Modal -->
    <div id="clinicianModal" class="modal-overlay" onclick="if(event.target===this) toggleModal(false)">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">📖 Clinical & Methodological Guide</div>
                <button class="modal-close" onclick="toggleModal(false)">&times;</button>
            </div>
            
            <div class="glossary-item">
                <div class="glossary-term">🎯 Primary Target & Primary Drug</div>
                <div class="glossary-desc">The frontline oncogenic driver protein (e.g. EGFR in lung cancer) and the primary targeted inhibitor administered to the patient (e.g. Osimertinib).</div>
            </div>

            <div class="glossary-item">
                <div class="glossary-term">⚡ Secondary Resistance Marker</div>
                <div class="glossary-desc">The acquired gene or protein driving therapeutic resistance. Resistance occurs via two primary biological mechanisms:</div>
                <ul style="margin: 0.4rem 0 0 1.2rem; font-size: 0.85rem; color: #94a3b8;">
                    <li><strong>Off-Target Bypass:</strong> Activation of a parallel signaling pathway (e.g., MET amplification) that bypasses the blocked primary target to sustain downstream cell survival.</li>
                    <li><strong>On-Target Mutation:</strong> Secondary mutations directly inside the primary target gene (e.g., EGFR C797S) preventing drug binding.</li>
                </ul>
            </div>

            <div class="glossary-item">
                <div class="glossary-term">🧮 Hub-Penalized Bottleneck Centrality</div>
                <div class="glossary-desc">NetworkX centrality algorithm computed as: <code>Betweenness / log2(Degree + 2)</code>. This penalizes generic, highly-connected hub proteins (e.g., Ubiquitin) while isolating true bottleneck signaling nodes driving resistance.</div>
            </div>

            <div class="glossary-item">
                <div class="glossary-term">🧪 Synergy Score & Dual-Drug Ranking</div>
                <div class="glossary-desc">Calculated score (0.0 to 1.0) combining network shortest path distance, bottleneck centrality, and ChEMBL binding affinity (pChEMBL). Recommends active, non-withdrawn clinical combination therapies to override resistance.</div>
            </div>

            <div style="text-align: right; margin-top: 1.5rem;">
                <button class="btn-submit" style="width: auto; padding: 0.6rem 1.2rem;" onclick="toggleModal(false)">Got It</button>
            </div>
        </div>
    </div>

    <script>
        let latestAnalysisData = null;
        let currentFilterPhase = 'ALL';

        function toggleModal(show) {
            document.getElementById('clinicianModal').style.display = show ? 'flex' : 'none';
        }

        function setPreset(target, drug, marker, indication) {
            document.getElementById('primary_target').value = target;
            document.getElementById('primary_drug').value = drug;
            document.getElementById('resistance_marker').value = marker;
            if (indication) document.getElementById('cancer_type').value = indication;
            executePipeline();
        }

        function runAnalysis(e) {
            if (e && e.preventDefault) e.preventDefault();
            executePipeline();
        }

        function filterPhase(phase) {
            currentFilterPhase = phase;
            document.querySelectorAll('.btn-tab').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            renderTherapies();
        }

        async function executePipeline() {
            const submitBtn = document.getElementById('submitBtn');
            const loader = document.getElementById('loader');
            const placeholder = document.getElementById('placeholder');
            const resultsContent = document.getElementById('resultsContent');
            const errorBanner = document.getElementById('errorBanner');
            const copyJsonBtn = document.getElementById('copyJsonBtn');

            submitBtn.disabled = true;
            errorBanner.style.display = 'none';
            placeholder.style.display = 'none';
            resultsContent.style.display = 'none';
            copyJsonBtn.style.display = 'none';
            loader.style.display = 'block';

            const payload = {
                primary_target: document.getElementById('primary_target').value.trim(),
                primary_drug: document.getElementById('primary_drug').value.trim(),
                resistance_marker: document.getElementById('resistance_marker').value.trim(),
                cancer_type: document.getElementById('cancer_type').value.trim()
            };

            try {
                const response = await fetch('/api/v1/analyze-resistance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Analysis pipeline failed');

                latestAnalysisData = data;
                document.getElementById('primaryTag').innerText = 'Target: ' + data.primary_target_canonical;
                document.getElementById('resistanceTag').innerText = 'Marker: ' + data.resistance_marker_canonical;
                document.getElementById('resTypeVal').innerText = data.resistance_type;
                document.getElementById('nodesCountVal').innerText = data.pathway_nodes_count;
                
                const distNum = typeof data.shortest_path_distance === 'number' 
                    ? data.shortest_path_distance 
                    : Number(data.shortest_path_distance) || 0;
                document.getElementById('distVal').innerText = distNum.toFixed(3);

                renderTherapies();

                resultsContent.style.display = 'block';
                copyJsonBtn.style.display = 'inline-block';
            } catch (err) {
                errorBanner.innerText = '❌ ' + err.message;
                errorBanner.style.display = 'block';
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        }

        function renderTherapies() {
            if (!latestAnalysisData) return;
            const therapiesList = document.getElementById('therapiesList');
            therapiesList.innerHTML = '';

            let candidates = latestAnalysisData.ranked_combinations || [];
            if (currentFilterPhase !== 'ALL') {
                candidates = candidates.filter(c => c.clinical_phase === Number(currentFilterPhase));
            }

            if (candidates.length === 0) {
                therapiesList.innerHTML = '<div style="color: #94a3b8; font-size: 0.9rem; text-align: center; padding: 2rem;">No clinical combination therapies found for the selected filter.</div>';
                return;
            }

            const primaryDrug = document.getElementById('primary_drug').value.trim();
            candidates.forEach((c, idx) => {
                const pct = Math.round((c.synergy_score || 0) * 100);
                const card = document.createElement('div');
                card.className = 'therapy-card';
                card.innerHTML = `
                    <div class="therapy-header">
                        <span class="drug-title">#${idx+1} ${c.secondary_drug} + ${primaryDrug}</span>
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
        }

        function copyResultJson() {
            if (!latestAnalysisData) return;
            navigator.clipboard.writeText(JSON.stringify(latestAnalysisData, null, 2))
                .then(() => alert('JSON response copied to clipboard!'))
                .catch(() => alert('Failed to copy to clipboard'));
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
        mapped_primary, mapped_resistance = await asyncio.gather(
            id_mapper.map_identifier(req.primary_target),
            id_mapper.map_identifier(req.resistance_marker),
        )
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

        async def _fetch_activities() -> Dict[str, float]:
            if mapped_resistance.chembl_target_id:
                return await chembl_client.get_target_activities(
                    mapped_resistance.chembl_target_id
                )
            return {}

        # Fetch external APIs concurrently via asyncio.gather
        interactions, ot_drugs, activity_map = await asyncio.gather(
            string_client.get_network(
                primary_target_canonical, resistance_marker_canonical
            ),
            ot_client.get_known_drugs(mapped_resistance.ensembl_id),
            _fetch_activities(),
        )

        # Filter out withdrawn drugs — AGENTS.md: "ranks active, non-withdrawn clinical dual-drug combination therapies"
        ot_drugs = [d for d in ot_drugs if not _is_drug_withdrawn(d)]


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
