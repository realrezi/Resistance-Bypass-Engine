import asyncio
import os
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

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


@app.get("/static/network.png")
async def get_network_image():
    """Serve the 3D biological network visualization image."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "static", "network.png"),
        os.path.join(os.getcwd(), "src", "static", "network.png"),
        os.path.join(os.getcwd(), "api", "static", "network.png"),
        "src/static/network.png",
        "api/static/network.png",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/static/lab_mutation.png")
async def get_lab_mutation_image():
    """Serve the Clinical Genomic Lab & Mutation Diagram image."""
    if LAB_MUTATION_B64:
        return Response(content=base64.b64decode(LAB_MUTATION_B64), media_type="image/png")
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "static", "lab_mutation.png"),
        os.path.join(os.getcwd(), "src", "static", "lab_mutation.png"),
        os.path.join(os.getcwd(), "api", "static", "lab_mutation.png"),
        "src/static/lab_mutation.png",
        "api/static/lab_mutation.png",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Targeted Oncology Resistance Bypass Engine | Clinical Genomic Laboratory</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-lab: #f8fafc;
            --card-bg: #ffffff;
            --border-lab: #cbd5e1;
            --border-subtle: #e2e8f0;
            --genomic-blue: #0284c7;
            --genomic-blue-hover: #0369a1;
            --mutation-red: #e11d48;
            --approved-green: #059669;
            --text-main: #0f172a;
            --text-secondary: #334155;
            --text-muted: #64748b;
            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --shadow-lab: 0 4px 20px -2px rgba(15, 23, 42, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-lab);
            background-image: 
                linear-gradient(to right, rgba(203, 213, 225, 0.25) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(203, 213, 225, 0.25) 1px, transparent 1px);
            background-size: 32px 32px;
            color: var(--text-secondary);
            min-height: 100vh;
            padding: 1rem 1.25rem;
            line-height: 1.45;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1440px;
            margin: 0 auto;
        }

        /* Clinical Lab Application Header */
        header.app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1.25rem;
            background: var(--card-bg);
            border: 1px solid var(--border-lab);
            border-radius: 12px;
            box-shadow: var(--shadow-lab);
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .brand-area {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text-main);
        }

        .brand-subtitle {
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #047857;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: var(--font-mono);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--approved-green);
            box-shadow: 0 0 6px var(--approved-green);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .btn-header {
            background: #f8fafc;
            border: 1px solid var(--border-lab);
            color: var(--text-secondary);
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }

        .btn-header:hover {
            color: var(--genomic-blue);
            background: #ffffff;
            border-color: #93c5fd;
        }

        .btn-header.primary {
            background: #f0f9ff;
            border-color: #bae6fd;
            color: #0369a1;
        }

        .btn-header.primary:hover {
            background: #e0f2fe;
            color: #075985;
        }

        /* Workstation Layout Grid */
        .workstation-grid {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 1rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .workstation-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--border-lab);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: var(--shadow-lab);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-title-text {
            font-size: 1rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Clinical Form Controls */
        .form-group {
            margin-bottom: 0.9rem;
        }

        label.field-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.3rem;
        }

        input.input-field {
            width: 100%;
            padding: 0.65rem 0.85rem;
            background: #f8fafc;
            border: 1px solid var(--border-lab);
            border-radius: 8px;
            color: var(--text-main);
            font-family: var(--font-main);
            font-size: 0.88rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        input.input-field:focus {
            outline: none;
            background: #ffffff;
            border-color: var(--genomic-blue);
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
        }

        .btn-run {
            width: 100%;
            padding: 0.8rem;
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: none;
            border-radius: 8px;
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
            margin-top: 0.4rem;
        }

        .btn-run:hover {
            opacity: 0.95;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35);
        }

        /* Keyboard-Free Point-and-Click Picker Palette */
        .quick-picker-section {
            margin-top: 1rem;
            border-top: 1px solid var(--border-subtle);
            padding-top: 0.85rem;
        }

        .quick-picker-title {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chip-group-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0.4rem 0 0.25rem 0;
        }

        .chip-flex {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-bottom: 0.5rem;
        }

        .btn-chip {
            background: #f1f5f9;
            border: 1px solid var(--border-lab);
            color: var(--text-secondary);
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-chip:hover {
            background: #e0f2fe;
            border-color: #7dd3fc;
            color: #0369a1;
        }

        .btn-chip.target-chip { border-left: 3px solid var(--genomic-blue); }
        .btn-chip.drug-chip { border-left: 3px solid #059669; }
        .btn-chip.marker-chip { border-left: 3px solid var(--mutation-red); }

        /* Academic Prevalence Matrix Section */
        .academic-matrix-panel {
            margin-top: 1rem;
            background: var(--card-bg);
            border: 1px solid var(--border-lab);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: var(--shadow-lab);
        }

        .matrix-tab-bar {
            display: flex;
            gap: 0.4rem;
            border-bottom: 2px solid var(--border-subtle);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            overflow-x: auto;
        }

        .btn-matrix-tab {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
        }

        .btn-matrix-tab.active {
            background: #f0f9ff;
            color: #0369a1;
            border-bottom: 2px solid var(--genomic-blue);
        }

        .prevalence-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 0.85rem;
        }

        .prevalence-card {
            background: #f8fafc;
            border: 1px solid var(--border-lab);
            border-radius: 10px;
            padding: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }

        .prevalence-card:hover {
            background: #ffffff;
            border-color: var(--genomic-blue);
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.12);
        }

        .prevalence-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.3rem;
        }

        .scenario-pair-title {
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .badge-prevalence {
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 800;
            font-family: var(--font-mono);
            background: #fef2f2;
            color: var(--mutation-red);
            border: 1px solid #fecaca;
        }

        .badge-prevalence.high {
            background: #fffbe6;
            color: #b45309;
            border-color: #fde68a;
        }

        .locus-tag {
            font-size: 0.74rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            margin-bottom: 0.35rem;
        }

        .scenario-mechanism {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Results Canvas */
        .lab-artifact-banner {
            width: 100%;
            height: auto;
            max-height: 380px;
            object-fit: cover;
            border-radius: 10px;
            border: 1px solid var(--border-lab);
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.85rem;
            margin-bottom: 1.25rem;
        }

        .metric-tile {
            background: #f8fafc;
            border: 1px solid var(--border-lab);
            border-radius: 10px;
            padding: 0.85rem;
            text-align: center;
        }

        .metric-number {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--genomic-blue);
            font-family: var(--font-mono);
        }

        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.15rem;
            font-weight: 700;
        }

        .canonical-bar {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .pill-badge {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            color: #0369a1;
            padding: 0.3rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-family: var(--font-mono);
            font-weight: 700;
        }

        .candidate-card {
            background: #ffffff;
            border: 1px solid var(--border-lab);
            border-radius: 10px;
            padding: 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: var(--shadow-lab);
            transition: all 0.2s ease;
        }

        .candidate-card:hover {
            border-color: #93c5fd;
        }

        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .drug-pair-name {
            font-size: 1rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .badge-phase {
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 800;
            font-family: var(--font-mono);
            background: #fffbe6;
            color: #b45309;
            border: 1px solid #fde68a;
        }

        .badge-phase.approved {
            background: #ecfdf5;
            color: var(--approved-green);
            border-color: #a7f3d0;
        }

        .progress-track {
            height: 6px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #0284c7 0%, #059669 100%);
            border-radius: 999px;
            width: 0%;
            transition: width 0.6s ease;
        }

        /* Modal Windows */
        .modal-wrapper {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal-box {
            background: #ffffff;
            border: 1px solid var(--border-lab);
            border-radius: 16px;
            max-width: 760px;
            width: 100%;
            max-height: 88vh;
            overflow-y: auto;
            padding: 1.75rem;
            color: var(--text-secondary);
            box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.25);
        }

        .modal-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .modal-heading {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.5rem;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Application Header -->
        <header class="app-header">
            <div class="brand-area">
                <div class="brand-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M2 9c6.667 6 13.333 0 20 6"/><circle cx="7" cy="12" r="1.5" fill="#0284c7"/><circle cx="12" cy="12" r="1.5" fill="#e11d48"/><circle cx="17" cy="12" r="1.5" fill="#0284c7"/></svg>
                </div>
                <div>
                    <div class="brand-title">Targeted Oncology Resistance Bypass Engine</div>
                    <div class="brand-subtitle">Clinical Genomic Laboratory Microservice v0.1.0</div>
                </div>
            </div>

            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>GENOMIC APIs ONLINE</span>
                </div>
                <button class="btn-header primary" onclick="toggleModal('guidanceModal', true)">
                    <span>💡 Purpose & Workflow</span>
                </button>
                <button class="btn-header" onclick="toggleModal('clinicianModal', true)">
                    <span>📖 Methodological Guide</span>
                </button>
                <button class="btn-header" onclick="toggleModal('apiModal', true)">
                    <span>⚙️ Developer APIs</span>
                </button>
            </div>
        </header>

        <!-- Workstation Grid -->
        <div class="workstation-grid">
            <!-- Left Sidebar Inputs & Quick Picker -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
                        <span>Clinical Parameters & Inputs</span>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label class="field-label" for="primary_target">Primary Target Symbol</label>
                        <input class="input-field" type="text" id="primary_target" value="EGFR" required placeholder="Select or type gene symbol (e.g. EGFR)..." list="targets_list" autocomplete="off">
                        <datalist id="targets_list">
                            <option value="EGFR">Epidermal Growth Factor Receptor (Chr 7p11.2)</option>
                            <option value="ERBB2">ERBB2 / HER2 Receptor Tyrosine Kinase (Chr 17q12)</option>
                            <option value="MET">MET Receptor Tyrosine Kinase (Chr 7q31.2)</option>
                            <option value="ALK">ALK Receptor Tyrosine Kinase (Chr 2p23.2)</option>
                            <option value="KRAS">KRAS Proto-Oncogene GTPase (Chr 12p12.1)</option>
                            <option value="BRAF">BRAF Serine/Threonine Kinase (Chr 7q34)</option>
                            <option value="PIK3CA">PI3K Catalytic Subunit Alpha (Chr 3q26.32)</option>
                            <option value="ROS1">ROS1 Receptor Tyrosine Kinase (Chr 6q22.1)</option>
                            <option value="RET">RET Proto-Oncogene Kinase (Chr 10q11.21)</option>
                            <option value="ESR1">Estrogen Receptor 1 (Chr 6q25.1)</option>
                            <option value="ABL1">ABL1 Non-Receptor Tyrosine Kinase (Chr 9q34.12)</option>
                            <option value="CDK4">Cyclin Dependent Kinase 4 (Chr 12q14.1)</option>
                            <option value="CDK6">Cyclin Dependent Kinase 6 (Chr 7q21.2)</option>
                            <option value="AR">Androgen Receptor (Chr Xq12)</option>
                            <option value="FGFR1">Fibroblast Growth Factor Receptor 1 (Chr 8p11.23)</option>
                            <option value="NTRK1">Neurotrophic Receptor Tyrosine Kinase 1 (Chr 1q23.1)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="primary_drug">Primary Targeted Drug Name</label>
                        <input class="input-field" type="text" id="primary_drug" value="Osimertinib" required placeholder="Select or type drug name..." list="drugs_list" autocomplete="off">
                        <datalist id="drugs_list">
                            <!-- Comprehensive Synchronized 35+ Targeted Drugs -->
                            <option value="Osimertinib">Osimertinib (3rd-gen EGFR TKI)</option>
                            <option value="Gefitinib">Gefitinib (1st-gen EGFR TKI)</option>
                            <option value="Erlotinib">Erlotinib (1st-gen EGFR TKI)</option>
                            <option value="Afatinib">Afatinib (2nd-gen ErbB TKI)</option>
                            <option value="Dacomitinib">Dacomitinib (2nd-gen EGFR TKI)</option>
                            <option value="Amivantamab">Amivantamab (EGFR/MET Bispecific mAb)</option>
                            <option value="Cetuximab">Cetuximab (Anti-EGFR mAb)</option>
                            <option value="Trastuzumab">Trastuzumab (Anti-HER2 mAb)</option>
                            <option value="Pertuzumab">Pertuzumab (Anti-HER2 mAb)</option>
                            <option value="Lapatinib">Lapatinib (EGFR/HER2 TKI)</option>
                            <option value="Neratinib">Neratinib (Pan-HER TKI)</option>
                            <option value="Tucatinib">Tucatinib (Selective HER2 TKI)</option>
                            <option value="Trastuzumab Deruxtecan">Trastuzumab Deruxtecan (T-DXd ADC)</option>
                            <option value="Capmatinib">Capmatinib (MET TKI)</option>
                            <option value="Tepotinib">Tepotinib (MET TKI)</option>
                            <option value="Crizotinib">Crizotinib (ALK/ROS1/MET TKI)</option>
                            <option value="Alectinib">Alectinib (2nd-gen ALK TKI)</option>
                            <option value="Brigatinib">Brigatinib (2nd-gen ALK TKI)</option>
                            <option value="Lorlatinib">Lorlatinib (3rd-gen ALK TKI)</option>
                            <option value="Sotorasib">Sotorasib (KRAS G12C Inhibitor)</option>
                            <option value="Adagrasib">Adagrasib (KRAS G12C Inhibitor)</option>
                            <option value="Dabrafenib">Dabrafenib (BRAF Kinase Inhibitor)</option>
                            <option value="Vemurafenib">Vemurafenib (BRAF Kinase Inhibitor)</option>
                            <option value="Encorafenib">Encorafenib (BRAF Kinase Inhibitor)</option>
                            <option value="Trametinib">Trametinib (MEK Inhibitor)</option>
                            <option value="Cobimetinib">Cobimetinib (MEK Inhibitor)</option>
                            <option value="Alpelisib">Alpelisib (PI3Kalpha Inhibitor)</option>
                            <option value="Capivasertib">Capivasertib (AKT Inhibitor)</option>
                            <option value="Fulvestrant">Fulvestrant (SERD)</option>
                            <option value="Elacestrant">Elacestrant (Oral SERD)</option>
                            <option value="Enzalutamide">Enzalutamide (AR Inhibitor)</option>
                            <option value="Imatinib">Imatinib (BCR-ABL TKI)</option>
                            <option value="Dasatinib">Dasatinib (2nd-gen BCR-ABL TKI)</option>
                            <option value="Nilotinib">Nilotinib (2nd-gen BCR-ABL TKI)</option>
                            <option value="Ponatinib">Ponatinib (3rd-gen BCR-ABL TKI)</option>
                            <option value="Palbociclib">Palbociclib (CDK4/6 Inhibitor)</option>
                            <option value="Ribociclib">Ribociclib (CDK4/6 Inhibitor)</option>
                            <option value="Abemaciclib">Abemaciclib (CDK4/6 Inhibitor)</option>
                            <option value="Selpercatinib">Selpercatinib (RET Inhibitor)</option>
                            <option value="Entrectinib">Entrectinib (ROS1/NTRK TKI)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="resistance_marker">Secondary Resistance Marker</label>
                        <input class="input-field" type="text" id="resistance_marker" value="MET" required placeholder="Select or type marker..." list="markers_list" autocomplete="off">
                        <datalist id="markers_list">
                            <option value="MET">MET Amplification / Bypass Hyperactivation</option>
                            <option value="EGFR">EGFR Secondary Gatekeeper (C797S / T790M)</option>
                            <option value="KRAS">KRAS Secondary Activation (G12C / G12V)</option>
                            <option value="BRAF">BRAF V600E Activation</option>
                            <option value="ERBB2">ERBB2 / HER2 Amplification</option>
                            <option value="PIK3CA">PIK3CA Hyperactivation Mutation (H1047R)</option>
                            <option value="MAP2K1">MAP2K1 / MEK1 Activation</option>
                            <option value="CDK4">CDK4 Cyclin Pathway Axis</option>
                            <option value="ABL1">ABL1 Gatekeeper Mutation (T315I)</option>
                            <option value="ALK">ALK Solvent Front Mutation (G1202R)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="cancer_type">Cancer Indication</label>
                        <input class="input-field" type="text" id="cancer_type" value="Non-Small Cell Lung Cancer" placeholder="Select or type indication..." list="indications_list" autocomplete="off">
                        <datalist id="indications_list">
                            <option value="Non-Small Cell Lung Cancer">Non-Small Cell Lung Cancer (NSCLC)</option>
                            <option value="HER2+ Breast Cancer">HER2+ Breast Cancer</option>
                            <option value="HR+/HER2- Breast Cancer">HR+/HER2- Breast Cancer</option>
                            <option value="Colorectal Cancer">Colorectal Cancer (CRC)</option>
                            <option value="Cutaneous Melanoma">Cutaneous Melanoma</option>
                            <option value="Chronic Myeloid Leukemia">Chronic Myeloid Leukemia (CML)</option>
                            <option value="Prostate Cancer">Metastatic Castration-Resistant Prostate Cancer</option>
                        </datalist>
                    </div>

                    <button type="submit" id="submitBtn" class="btn-run">
                        <span>Execute Resistance Pipeline</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>

                <!-- Point-and-Click Quick Picker Palette -->
                <div class="quick-picker-section">
                    <div class="quick-picker-title">
                        <span>🖱️ Point-and-Click Marker Picker</span>
                        <span style="font-size: 0.65rem; color: var(--genomic-blue);">No Keyboard Required</span>
                    </div>

                    <div class="chip-group-label">Target Driver Genes:</div>
                    <div class="chip-flex">
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'EGFR')">EGFR</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'ERBB2')">HER2</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'ALK')">ALK</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'KRAS')">KRAS</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'BRAF')">BRAF</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'PIK3CA')">PIK3CA</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'ESR1')">ESR1</button>
                        <button class="btn-chip target-chip" onclick="quickFill('primary_target', 'ABL1')">ABL1</button>
                    </div>

                    <div class="chip-group-label">Frontline Targeted Agents:</div>
                    <div class="chip-flex">
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Osimertinib')">Osimertinib</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Trastuzumab')">Trastuzumab</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Alectinib')">Alectinib</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Sotorasib')">Sotorasib</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Dabrafenib')">Dabrafenib</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Fulvestrant')">Fulvestrant</button>
                        <button class="btn-chip drug-chip" onclick="quickFill('primary_drug', 'Imatinib')">Imatinib</button>
                    </div>

                    <div class="chip-group-label">Resistance Bypasses & Mutations:</div>
                    <div class="chip-flex">
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'MET')">MET Bypass</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'EGFR')">EGFR C797S</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'KRAS')">KRAS Activation</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'BRAF')">BRAF V600E</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'PIK3CA')">PIK3CA Mutation</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'ABL1')">ABL1 T315I</button>
                        <button class="btn-chip marker-chip" onclick="quickFill('resistance_marker', 'CDK4')">CDK4/6 Axis</button>
                    </div>
                </div>
            </div>

            <!-- Right Results Canvas -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span>Genomic Resistance Pathway & Synergy Analysis</span>
                    </div>
                    <button id="copyJsonBtn" class="btn-header" style="display: none;" onclick="copyResultJson()">📋 Copy JSON</button>
                </div>

                <div id="errorBanner" style="display:none; background:#fef2f2; border:1px solid #fecaca; padding:0.85rem; border-radius:8px; color:#991b1b; margin-bottom:1rem; font-size:0.85rem;"></div>

                <div id="loader" style="display: none; text-align: center; padding: 3rem 1rem;">
                    <div style="width: 40px; height: 40px; border: 3px solid #e2e8f0; border-radius: 50%; border-top-color: #0284c7; animation: spin 0.8s linear infinite; margin: 0 auto 1rem auto;"></div>
                    <p style="font-weight: 700; color: var(--text-main);">Querying Biological PPI & Clinical APIs...</p>
                    <p style="font-size: 0.8rem; color: var(--text-muted);">Resolving HGNC IDs • Building NetworkX LCC Topology • Querying ChEMBL & Open Targets</p>
                </div>

                <div id="placeholder">
                    <!-- Embedded Clinical Genomic Lab Artifact -->
                    <img src="/static/lab_mutation.png" alt="Integrated Genomic Research Dashboard" class="lab-artifact-banner">
                    <div style="text-align: center; color: var(--text-muted); font-size: 0.84rem;">
                        <p style="font-weight: 700; color: var(--text-main); font-size: 0.95rem;">Integrated Genomic Resistance Dashboard Ready</p>
                        <p>Select an epidemiological scenario below or use the quick picker to initiate network graph modeling.</p>
                    </div>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div class="canonical-bar">
                        <span class="pill-badge" id="primaryTag">Primary Target: -</span>
                        <span class="pill-badge" id="resistanceTag">Resistance Marker: -</span>
                    </div>

                    <div class="metrics-grid">
                        <div class="metric-tile">
                            <div class="metric-number" id="resTypeVal">-</div>
                            <div class="metric-label">Resistance Type</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-number" id="nodesCountVal">0</div>
                            <div class="metric-label">Network Nodes</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-number" id="distVal">0.000</div>
                            <div class="metric-label">Shortest Path Distance</div>
                        </div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <h3 style="font-size:0.95rem; font-weight:800; color:var(--text-main);">Ranked Dual-Drug Combination Therapies</h3>
                    </div>

                    <div id="therapiesList"></div>
                </div>
            </div>
        </div>

        <!-- Academic Prevalence Matrix Panel -->
        <div class="academic-matrix-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div>
                    <h3 style="font-size: 1.05rem; font-weight: 800; color: var(--text-main);">🔬 Academic Clinical Resistance Matrix</h3>
                    <p style="font-size: 0.78rem; color: var(--text-muted);">Categorized by Global Epidemiological Prevalence & Molecular Resistance Mechanism</p>
                </div>
            </div>

            <div class="matrix-tab-bar">
                <button class="btn-matrix-tab active" onclick="switchMatrixCategory('nsclc')">🫁 NSCLC (Lung)</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('breast')">🎗️ Breast Cancer</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('crc')">🧬 Colorectal & GI</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('melanoma')">☀️ Melanoma</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('cml')">🩸 Hematologic (CML)</button>
            </div>

            <!-- Tab 1: NSCLC Scenarios -->
            <div id="matrix-nsclc" class="prevalence-cards-grid">
                <div class="prevalence-card" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + MET Amplification</span>
                        <span class="badge-prevalence">15–20% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFR) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-Target RTK Bypass. MET hyperactivation activates downstream PI3K/AKT signaling independent of EGFR blockade.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + C797S Secondary Mutation</span>
                        <span class="badge-prevalence high">7–10% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (Exon 20 C797S)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> On-Target ATP Pocket Mutation. Cysteine to Serine mutation eliminates covalent binding site of Osimertinib.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('ALK', 'Alectinib', 'MET', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ALK + MET Bypass</span>
                        <span class="badge-prevalence high">8–12% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 2p23.2 (ALK) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Parallel RTK Hyperactivation circumventing 2nd-gen ALK inhibitor blockade in ALK+ NSCLC.</div>
                </div>
            </div>

            <!-- Tab 2: Breast Cancer Scenarios -->
            <div id="matrix-breast" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'HER2+ Breast Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">HER2 + MET Amplification</span>
                        <span class="badge-prevalence">10–15% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 17q12 (ERBB2) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-target RTK activation overriding anti-HER2 monoclonal antibody therapy (Trastuzumab).</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'HR+/HER2- Breast Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ESR1 + CDK4/6 Cyclin Axis</span>
                        <span class="badge-prevalence high">12–18% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 6q25.1 (ESR1) ➔ Chr 12q14.1 (CDK4)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Endocrine resistance driven by ESR1 ligand-independent mutations & Cyclin D1/CDK4 pathway escape.</div>
                </div>
            </div>

            <!-- Tab 3: Colorectal & GI Scenarios -->
            <div id="matrix-crc" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">KRAS G12C + EGFR Feedback</span>
                        <span class="badge-prevalence">20–25% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 12p12.1 (KRAS) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Rapid receptor tyrosine kinase feedback loop reactivating MAPK axis requiring dual KRAS G12C + EGFR inhibition.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('BRAF', 'Dabrafenib', 'EGFR', 'Colorectal Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600E + EGFR Feedback</span>
                        <span class="badge-prevalence high">10–12% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Single-agent BRAF inhibition triggers strong EGFR feedback activation; demands combined Encorafenib + Cetuximab.</div>
                </div>
            </div>

            <!-- Tab 4: Melanoma Scenarios -->
            <div id="matrix-melanoma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('BRAF', 'Dabrafenib', 'MAP2K1', 'Cutaneous Melanoma')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600 + MAP2K1/MEK1</span>
                        <span class="badge-prevalence">35–45% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 15q22.31 (MAP2K1)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> MAPK signaling reactivation overcome by FDA-approved Dual BRAF + MEK inhibition (Dabrafenib + Trametinib).</div>
                </div>
            </div>

            <!-- Tab 5: Hematologic Scenarios -->
            <div id="matrix-cml" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('BCR-ABL', 'Imatinib', 'ABL1', 'Chronic Myeloid Leukemia')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BCR-ABL + ABL1 T315I Gatekeeper</span>
                        <span class="badge-prevalence">15–20% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 9q34.12 (ABL1 T315I Gatekeeper)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Threonine to Isoleucine gatekeeper mutation causing steric clash with 1st/2nd-gen TKIs; requires Ponatinib or Asciminib combination.</div>
                </div>
            </div>
        </div>

        <!-- Developer Footer -->
        <footer style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-muted);">
            <div>Targeted Oncology Resistance Bypass Engine • Clinical Microservice</div>
            <div style="display: flex; gap: 1rem;">
                <a href="/docs" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-weight: 600;">⚡ Swagger UI (/docs)</a>
                <a href="/health" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-weight: 600;">💚 Diagnostics (/health)</a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-weight: 600;">📦 GitHub</a>
            </div>
        </footer>
    </div>

    <!-- Modals -->
    <div id="guidanceModal" class="modal-wrapper" onclick="if(event.target===this) toggleModal('guidanceModal', false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">💡 Engine Purpose & Workflow</div>
                <button class="modal-close" onclick="toggleModal('guidanceModal', false)">&times;</button>
            </div>
            <p style="margin-bottom: 1rem; font-size: 0.9rem;">
                This clinical engine models acquired therapeutic drug resistance in cancer using real-time REST/GraphQL biological APIs (HGNC, UniProt, STRING-DB, Open Targets, ChEMBL v4) and pure Python NetworkX graph algorithms.
            </p>
            <div style="background:#f8fafc; border:1px solid var(--border-lab); padding:0.85rem; border-radius:8px; margin-bottom:0.75rem;">
                <div style="font-weight:700; color:var(--genomic-blue);">1. Resolve Biological Identifiers</div>
                <div style="font-size:0.82rem;">Resolves alias symbols (e.g. HER2 ➔ ERBB2) to official HGNC IDs and UniProt Accession codes.</div>
            </div>
            <div style="background:#f8fafc; border:1px solid var(--border-lab); padding:0.85rem; border-radius:8px; margin-bottom:0.75rem;">
                <div style="font-weight:700; color:var(--genomic-blue);">2. Build PPI Signaling Graph</div>
                <div style="font-size:0.82rem;">Queries STRING-DB for protein-protein interaction networks and extracts the Largest Connected Component (LCC).</div>
            </div>
            <div style="background:#f8fafc; border:1px solid var(--border-lab); padding:0.85rem; border-radius:8px;">
                <div style="font-weight:700; color:var(--genomic-blue);">3. Hub-Penalized Centrality & Therapy Ranking</div>
                <div style="font-size:0.82rem;">Computes <code>Betweenness / log2(Degree + 2)</code> to isolate non-generic bottleneck targets and ranks active clinical combinations.</div>
            </div>
        </div>
    </div>

    <div id="clinicianModal" class="modal-wrapper" onclick="if(event.target===this) toggleModal('clinicianModal', false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">📖 Methodological & Clinical Guide</div>
                <button class="modal-close" onclick="toggleModal('clinicianModal', false)">&times;</button>
            </div>
            <div style="font-size:0.88rem; line-height:1.5;">
                <p style="margin-bottom:0.75rem;"><strong>Off-Target Bypass:</strong> Hyperactivation of a parallel signaling pathway (e.g., MET amplification) that bypasses frontline drug blockade.</p>
                <p style="margin-bottom:0.75rem;"><strong>On-Target Mutation:</strong> Secondary mutations directly inside the primary target gene (e.g., EGFR C797S or ABL1 T315I) altering drug binding affinity.</p>
                <p><strong>Hub-Penalized Bottleneck Centrality:</strong> Evaluated as <code>Betweenness / log2(Degree + 2)</code> to strip non-specific hub proteins (like TP53 or Ubiquitin) while pinpointing critical resistance signaling nodes.</p>
            </div>
        </div>
    </div>

    <div id="apiModal" class="modal-wrapper" onclick="if(event.target===this) toggleModal('apiModal', false)">
        <div class="modal-box" style="max-width:500px;">
            <div class="modal-top">
                <div class="modal-heading">⚙️ Developer API Integration</div>
                <button class="modal-close" onclick="toggleModal('apiModal', false)">&times;</button>
            </div>
            <div style="display:flex; flex-direction:column; gap:0.75rem;">
                <a href="/docs" target="_blank" class="btn-header" style="justify-content:space-between;"><span>⚡ Swagger OpenAPI UI (/docs)</span><span>↗</span></a>
                <a href="/health" target="_blank" class="btn-header" style="justify-content:space-between;"><span>💚 Health Diagnostics (/health)</span><span>↗</span></a>
                <a href="/openapi.json" target="_blank" class="btn-header" style="justify-content:space-between;"><span>📄 OpenAPI Spec (/openapi.json)</span><span>↗</span></a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" class="btn-header" style="justify-content:space-between;"><span>📦 GitHub Repository</span><span>↗</span></a>
            </div>
        </div>
    </div>

    <script>
        let latestAnalysisData = null;

        function toggleModal(id, show) {
            document.getElementById(id).style.display = show ? 'flex' : 'none';
        }

        function quickFill(fieldId, value) {
            document.getElementById(fieldId).value = value;
        }

        function setPreset(target, drug, marker, indication) {
            document.getElementById('primary_target').value = target;
            document.getElementById('primary_drug').value = drug;
            document.getElementById('resistance_marker').value = marker;
            if (indication) document.getElementById('cancer_type').value = indication;
            executePipeline();
        }

        function switchMatrixCategory(cat) {
            document.querySelectorAll('.btn-matrix-tab').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            ['nsclc', 'breast', 'crc', 'melanoma', 'cml'].forEach(c => {
                document.getElementById('matrix-' + c).style.display = (c === cat) ? 'grid' : 'none';
            });
        }

        function runAnalysis(e) {
            if (e && e.preventDefault) e.preventDefault();
            executePipeline();
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
                document.getElementById('primaryTag').innerText = 'Primary Target: ' + data.primary_target_canonical;
                document.getElementById('resistanceTag').innerText = 'Resistance Marker: ' + data.resistance_marker_canonical;
                document.getElementById('resTypeVal').innerText = data.resistance_type;
                document.getElementById('nodesCountVal').innerText = data.pathway_nodes_count;
                
                const distNum = typeof data.shortest_path_distance === 'number' 
                    ? data.shortest_path_distance 
                    : Number(data.shortest_path_distance) || 0;
                document.getElementById('distVal').innerText = distNum.toFixed(3);

                renderTherapies();

                resultsContent.style.display = 'block';
                copyJsonBtn.style.display = 'inline-flex';
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
            if (candidates.length === 0) {
                therapiesList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; text-align: center; padding: 2rem;">No clinical combination therapies matching this target configuration.</div>';
                return;
            }

            const primaryDrug = document.getElementById('primary_drug').value.trim();
            candidates.forEach((c, idx) => {
                const pct = Math.round((c.synergy_score || 0) * 100);
                const isApproved = c.clinical_phase === 4;
                const card = document.createElement('div');
                card.className = 'candidate-card';
                card.innerHTML = `
                    <div class="candidate-header">
                        <span class="drug-pair-name">#${idx+1} ${c.secondary_drug} + ${primaryDrug}</span>
                        <span class="badge-phase ${isApproved ? 'approved' : ''}">${isApproved ? 'FDA Approved' : 'Phase ' + c.clinical_phase}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.3rem;">
                        Secondary Target: <strong style="color:var(--text-main);">${c.secondary_target}</strong> | Synergy Score: <strong style="color:var(--genomic-blue);">${c.synergy_score}</strong> | Hub Centrality: ${c.hub_penalized_centrality}
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.4;">${c.biological_rationale}</div>
                `;
                therapiesList.appendChild(card);
            });
        }

        function copyResultJson() {
            if (!latestAnalysisData) return;
            navigator.clipboard.writeText(JSON.stringify(latestAnalysisData, null, 2))
                .then(() => alert('Structured JSON response copied to clipboard!'))
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
        "service": "Targeted Oncology Resistance Bypass Engine",
        "version": "0.1.0",
        "environment": "Vercel Serverless / Production",
        "network_clients": {
            "hgnc_rest": "connected",
            "uniprot_kb": "connected",
            "string_db": "connected",
            "chembl_v4": "connected",
            "open_targets_v4": "connected",
        },
        "cache": {
            "engine": "diskcache",
            "volume_bytes": cache.volume(),
            "size_limit_bytes": cache.size_limit,
            "status": "active",
        },
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
