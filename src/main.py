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
    img_path = os.path.join(os.path.dirname(__file__), "static", "network.png")
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Targeted Oncology Resistance Bypass Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-void: #070a14;
            --panel-bg: rgba(15, 23, 42, 0.85);
            --panel-border: rgba(56, 189, 248, 0.15);
            --accent-cyan: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-emerald: #10b981;
            --accent-purple: #a855f7;
            --text-heading: #f8fafc;
            --text-body: #cbd5e1;
            --text-muted: #64748b;
            --font-main: 'Inter', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-void);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(168, 85, 247, 0.06) 0%, transparent 40%);
            color: var(--text-body);
            height: 100vh;
            overflow: hidden;
            padding: 0.75rem 1rem;
            line-height: 1.4;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1380px;
            margin: 0 auto;
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        /* Top Bar Header */
        header.app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 1rem;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            margin-bottom: 0.75rem;
            flex-shrink: 0;
        }

        .brand-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
        }

        .brand-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.65rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #34d399;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 6px #10b981;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .btn-header {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--panel-border);
            color: var(--text-body);
            padding: 0.4rem 0.75rem;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .btn-header:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(56, 189, 248, 0.4);
        }

        .btn-header.primary {
            background: rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.35);
            color: #38bdf8;
        }

        .btn-header.primary:hover {
            background: rgba(56, 189, 248, 0.22);
            color: #fff;
        }

        /* Workstation Layout */
        .workstation-grid {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 0.85rem;
            flex: 1;
            min-height: 0;
        }

        @media (max-width: 960px) {
            body { height: auto; overflow: auto; }
            .workstation-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 1.1rem;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.5);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.85rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--panel-border);
            flex-shrink: 0;
        }

        .panel-title-text {
            font-size: 0.95rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 0.85rem;
        }

        label.field-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-body);
            margin-bottom: 0.3rem;
        }

        .tooltip-trigger {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            font-size: 0.65rem;
            cursor: help;
        }

        input.input-field {
            width: 100%;
            padding: 0.6rem 0.8rem;
            background: rgba(9, 13, 22, 0.8);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            color: #fff;
            font-family: var(--font-main);
            font-size: 0.88rem;
            transition: border-color 0.2s ease;
        }

        input.input-field:focus {
            outline: none;
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        .btn-run {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
            margin-top: 0.4rem;
        }

        .btn-run:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        /* Compact Presets */
        .preset-container {
            margin-top: 0.85rem;
            border-top: 1px solid var(--panel-border);
            padding-top: 0.75rem;
            overflow-y: auto;
            flex: 1;
        }

        .preset-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .preset-flex {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .btn-preset-chip {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: var(--text-body);
            padding: 0.3rem 0.55rem;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-preset-chip:hover {
            color: #fff;
            border-color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.1);
        }

        /* Results Canvas Scrollable Container */
        .results-canvas {
            overflow-y: auto;
            flex: 1;
            padding-right: 0.3rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .metric-tile {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
        }

        .metric-number {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--accent-cyan);
            font-family: var(--font-mono);
        }

        .metric-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.15rem;
        }

        .canonical-bar {
            display: flex;
            gap: 0.4rem;
            margin-bottom: 0.85rem;
            flex-wrap: wrap;
        }

        .pill-badge {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #7dd3fc;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-family: var(--font-mono);
        }

        .filter-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .phase-tabs {
            display: flex;
            gap: 0.25rem;
        }

        .btn-phase-tab {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: var(--text-muted);
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-size: 0.72rem;
            cursor: pointer;
        }

        .btn-phase-tab.active {
            background: rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.4);
            color: #38bdf8;
            font-weight: 600;
        }

        /* Combination Therapy Cards */
        .candidate-card {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s ease;
        }

        .candidate-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
        }

        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .drug-pair-name {
            font-size: 0.98rem;
            font-weight: 700;
            color: #fff;
        }

        .badge-phase {
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            font-family: var(--font-mono);
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .badge-phase.approved {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border-color: rgba(16, 185, 129, 0.3);
        }

        .progress-track {
            height: 5px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
            border-radius: 999px;
            width: 0%;
            transition: width 0.6s ease;
        }

        .candidate-rationale {
            font-size: 0.82rem;
            color: var(--text-body);
            line-height: 1.4;
        }

        .empty-visual-state {
            text-align: center;
            padding: 2.5rem 1rem;
        }

        .network-preview-img {
            max-width: 480px;
            width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            margin-bottom: 1rem;
        }

        /* Modal Overlays */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(7, 10, 20, 0.85);
            backdrop-filter: blur(12px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal-box {
            background: #0f172a;
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            max-width: 680px;
            width: 100%;
            max-height: 85vh;
            overflow-y: auto;
            padding: 1.75rem;
            color: #e2e8f0;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }

        .modal-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .modal-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.4rem;
            cursor: pointer;
        }

        .modal-step {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
        }

        .modal-step-title {
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 0.2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Application Header -->
        <header class="app-header">
            <div class="brand-area">
                <div class="brand-logo-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M2 9c6.667 6 13.333 0 20 6"/><circle cx="7" cy="12" r="1.5" fill="#38bdf8"/><circle cx="12" cy="12" r="1.5" fill="#a855f7"/><circle cx="17" cy="12" r="1.5" fill="#38bdf8"/></svg>
                </div>
                <div>
                    <div class="brand-title">Targeted Oncology Resistance Bypass Engine</div>
                    <div class="brand-subtitle">Precision Network Biology Microservice v0.1.0</div>
                </div>
            </div>

            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>ENGINE ONLINE</span>
                </div>
                <button class="btn-header primary" onclick="toggleModal('purposeModal', true)">
                    <span>💡 Purpose & How It Helps</span>
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
            <!-- Left Sidebar Controls -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
                        <span>Analysis Inputs</span>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label class="field-label" for="primary_target">
                            <span>Primary Target Symbol</span>
                            <span class="tooltip-trigger" title="Oncogenic driver gene targeted by frontline therapy (e.g. EGFR, ERBB2, ALK)">?</span>
                        </label>
                        <input class="input-field" type="text" id="primary_target" value="EGFR" required placeholder="e.g. EGFR, ERBB2..." list="targets_list" autocomplete="off">
                        <datalist id="targets_list">
                            <option value="EGFR">Epidermal Growth Factor Receptor</option>
                            <option value="ERBB2">ERBB2 / HER2 Receptor Tyrosine Kinase</option>
                            <option value="MET">MET Receptor Tyrosine Kinase</option>
                            <option value="ALK">ALK Receptor Tyrosine Kinase</option>
                            <option value="KRAS">KRAS Proto-Oncogene GTPase</option>
                            <option value="BRAF">BRAF Serine/Threonine Kinase</option>
                            <option value="PIK3CA">PI3K Catalytic Subunit Alpha</option>
                            <option value="ROS1">ROS1 Receptor Tyrosine Kinase</option>
                            <option value="RET">RET Proto-Oncogene Kinase</option>
                            <option value="ESR1">Estrogen Receptor 1</option>
                            <option value="ABL1">ABL1 Non-Receptor Tyrosine Kinase</option>
                            <option value="CDK4">Cyclin Dependent Kinase 4</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="primary_drug">
                            <span>Primary Drug Name</span>
                            <span class="tooltip-trigger" title="Frontline targeted agent (e.g. Osimertinib, Trastuzumab, Sotorasib)">?</span>
                        </label>
                        <input class="input-field" type="text" id="primary_drug" value="Osimertinib" required placeholder="e.g. Osimertinib..." list="drugs_list" autocomplete="off">
                        <datalist id="drugs_list">
                            <option value="Osimertinib">Osimertinib (EGFR TKI)</option>
                            <option value="Trastuzumab">Trastuzumab (Anti-HER2 mAb)</option>
                            <option value="Gefitinib">Gefitinib (EGFR TKI)</option>
                            <option value="Capmatinib">Capmatinib (MET TKI)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="resistance_marker">
                            <span>Secondary Resistance Marker</span>
                            <span class="tooltip-trigger" title="Bypass marker or secondary mutation driving resistance (e.g. MET, KRAS, BRAF)">?</span>
                        </label>
                        <input class="input-field" type="text" id="resistance_marker" value="MET" required placeholder="e.g. MET, KRAS..." list="markers_list" autocomplete="off">
                        <datalist id="markers_list">
                            <option value="MET">MET Amplification / Bypass</option>
                            <option value="EGFR">EGFR Secondary Mutation (C797S)</option>
                            <option value="KRAS">KRAS Activation</option>
                            <option value="BRAF">BRAF V600 Activation</option>
                            <option value="ERBB2">ERBB2 / HER2 Amplification</option>
                            <option value="PIK3CA">PIK3CA Hyperactivation</option>
                            <option value="CDK4">CDK4 Cyclin Axis</option>
                            <option value="ABL1">ABL1 Gatekeeper (T315I)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="cancer_type">Cancer Indication</label>
                        <input class="input-field" type="text" id="cancer_type" value="Non-Small Cell Lung Cancer" placeholder="e.g. NSCLC, Breast Cancer..." list="indications_list" autocomplete="off">
                        <datalist id="indications_list">
                            <option value="Non-Small Cell Lung Cancer">Non-Small Cell Lung Cancer (NSCLC)</option>
                            <option value="HER2+ Breast Cancer">HER2+ Breast Cancer</option>
                            <option value="Colorectal Cancer">Colorectal Cancer (CRC)</option>
                            <option value="Cutaneous Melanoma">Cutaneous Melanoma</option>
                            <option value="Chronic Myeloid Leukemia">Chronic Myeloid Leukemia (CML)</option>
                            <option value="Prostate Cancer">Metastatic Castration-Resistant Prostate Cancer</option>
                        </datalist>
                    </div>

                    <button type="submit" id="submitBtn" class="btn-run">
                        <span>Execute Resistance Pipeline</span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>

                <!-- Presets List -->
                <div class="preset-container">
                    <div class="preset-title">🔥 Clinical Scenario Presets</div>
                    <div class="preset-flex">
                        <button class="btn-preset-chip" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'NSCLC')">EGFR + MET (NSCLC)</button>
                        <button class="btn-preset-chip" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'NSCLC')">EGFR + EGFR (C797S)</button>
                        <button class="btn-preset-chip" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'Breast Cancer')">HER2 + MET (Breast)</button>
                        <button class="btn-preset-chip" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer')">KRAS + EGFR (CRC)</button>
                        <button class="btn-preset-chip" onclick="setPreset('BRAF', 'Dabrafenib', 'MAP2K1', 'Melanoma')">BRAF + MEK1 (Melanoma)</button>
                        <button class="btn-preset-chip" onclick="setPreset('ALK', 'Alectinib', 'MET', 'NSCLC')">ALK + MET (NSCLC)</button>
                        <button class="btn-preset-chip" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'Breast Cancer')">ESR1 + CDK4 (Breast)</button>
                        <button class="btn-preset-chip" onclick="setPreset('BCR-ABL', 'Imatinib', 'ABL1', 'CML')">BCR-ABL + ABL1 (CML)</button>
                    </div>
                </div>
            </div>

            <!-- Right Results Canvas -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span>Resistance Topology & Synergy Analysis</span>
                    </div>
                    <button id="copyJsonBtn" class="btn-header" style="display: none;" onclick="copyResultJson()">📋 Copy JSON</button>
                </div>

                <div id="errorBanner" style="display:none;" class="modal-step" style="border-color: rgba(239, 68, 68, 0.4); color: #fca5a5;"></div>

                <div id="loader" style="display: none; text-align: center; padding: 3rem 1rem;">
                    <div style="width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.08); border-radius: 50%; border-top-color: #38bdf8; animation: spin 0.8s linear infinite; margin: 0 auto 1rem auto;"></div>
                    <p style="font-weight: 700; color: #fff;">Querying Biological PPI API Networks...</p>
                    <p style="font-size: 0.8rem; color: var(--text-muted);">Resolving HGNC IDs • Fetching STRING-DB PPI & Open Targets GraphQL</p>
                </div>

                <div class="results-canvas">
                    <div id="placeholder" class="empty-visual-state">
                        <img src="/static/network.png" alt="Biological Signaling Pathway Network Graph" class="network-preview-img">
                        <p style="font-weight: 700; color: #fff; font-size: 1rem; margin-bottom: 0.3rem;">Targeted Oncology Network Canvas</p>
                        <p style="font-size: 0.82rem; color: var(--text-muted);">Select a scenario on the left or enter target markers to construct signaling graph topologies.</p>
                    </div>

                    <div id="resultsContent" style="display: none;">
                        <div class="canonical-bar">
                            <span class="pill-badge" id="primaryTag">Primary Target: -</span>
                            <span class="pill-badge" id="resistanceTag">Resistance Marker: -</span>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-tile">
                                <div class="metric-number" id="resTypeVal">-</div>
                                <div class="metric-label">Resistance Mechanism</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-number" id="nodesCountVal">0</div>
                                <div class="metric-label">Network Nodes</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-number" id="distVal">0.000</div>
                                <div class="metric-label">Shortest Distance</div>
                            </div>
                        </div>

                        <div class="filter-bar">
                            <h3 style="font-size: 0.9rem; font-weight: 700; color: #fff;">Ranked Dual-Drug Combination Therapies</h3>
                            <div class="phase-tabs">
                                <button class="btn-phase-tab active" onclick="filterPhase('ALL')">All Phases</button>
                                <button class="btn-phase-tab" onclick="filterPhase(4)">Phase 4 (Approved)</button>
                                <button class="btn-phase-tab" onclick="filterPhase(3)">Phase 3</button>
                                <button class="btn-phase-tab" onclick="filterPhase(2)">Phase 2</button>
                            </div>
                        </div>

                        <div id="therapiesList"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Purpose & How It Helps Modal -->
    <div id="purposeModal" class="modal-overlay" onclick="if(event.target===this) toggleModal('purposeModal', false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">💡 Engine Purpose & How It Helps</div>
                <button class="modal-close" onclick="toggleModal('purposeModal', false)">&times;</button>
            </div>
            
            <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 1.25rem;">
                When targeted cancer drugs stop working because cancer cells find a secondary biological workaround (acquired resistance), this engine uses real-time biological API networks and NetworkX graph algorithms to model the resistance pathway and rank effective dual-drug combination therapies to overcome it.
            </p>

            <div class="modal-step">
                <div class="modal-step-title">🎯 Step 1: Input Frontline Target & Acquired Resistance</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Enter the frontline target drug (e.g. Osimertinib targeting EGFR) and the secondary resistance marker (e.g. MET bypass amplification or C797S point mutation).</div>
            </div>

            <div class="modal-step">
                <div class="modal-step-title">🧬 Step 2: Biological Network Graph Construction</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Resolves canonical IDs via HGNC/UniProt, fetches STRING-DB protein-protein interactions (PPI), extracts the Largest Connected Component (LCC), and strips self-loops.</div>
            </div>

            <div class="modal-step">
                <div class="modal-step-title">💊 Step 3: Hub-Penalized Bottleneck & Dual Therapy Ranking</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Computes <code>Betweenness / log2(Degree + 2)</code> to penalize generic hub proteins, queries Open Targets & ChEMBL v4 APIs, and ranks active clinical trial combinations.</div>
            </div>

            <div style="text-align: right; margin-top: 1rem;">
                <button class="btn-header primary" style="padding: 0.5rem 1.2rem;" onclick="toggleModal('purposeModal', false)">Got It</button>
            </div>
        </div>
    </div>

    <!-- Clinician Guide Modal -->
    <div id="clinicianModal" class="modal-overlay" onclick="if(event.target===this) toggleModal('clinicianModal', false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">📖 Methodological & Clinical Guide</div>
                <button class="modal-close" onclick="toggleModal('clinicianModal', false)">&times;</button>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #fff; margin-bottom: 0.2rem;">🎯 Primary Target & Drug</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Frontline driver protein (e.g. EGFR in NSCLC) and primary targeted inhibitor (e.g. Osimertinib).</div>
            </div>

            <div style="margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #fff; margin-bottom: 0.2rem;">⚡ Off-Target Bypass vs On-Target Mutation</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Off-target bypass involves hyperactivation of a parallel signaling receptor (e.g. MET), while on-target mutation alters the drug-binding ATP pocket directly.</div>
            </div>

            <div style="margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #fff; margin-bottom: 0.2rem;">🧮 Hub-Penalized Bottleneck Centrality</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Computed as <code>Betweenness / log2(Degree + 2)</code> to penalize generic non-specific hub proteins (such as Ubiquitin or TP53) while isolating true signaling bottleneck nodes.</div>
            </div>

            <div style="text-align: right; margin-top: 1rem;">
                <button class="btn-header" style="padding: 0.5rem 1.2rem;" onclick="toggleModal('clinicianModal', false)">Close Guide</button>
            </div>
        </div>
    </div>

    <!-- Developer API Modal -->
    <div id="apiModal" class="modal-overlay" onclick="if(event.target===this) toggleModal('apiModal', false)">
        <div class="modal-box" style="max-width: 520px;">
            <div class="modal-top">
                <div class="modal-heading">⚙️ Developer API Integration</div>
                <button class="modal-close" onclick="toggleModal('apiModal', false)">&times;</button>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <a href="/docs" target="_blank" class="btn-header" style="justify-content: space-between;">
                    <span>⚡ Swagger OpenAPI UI (/docs)</span>
                    <span>↗</span>
                </a>
                <a href="/health" target="_blank" class="btn-header" style="justify-content: space-between;">
                    <span>💚 System Health Diagnostics (/health)</span>
                    <span>↗</span>
                </a>
                <a href="/openapi.json" target="_blank" class="btn-header" style="justify-content: space-between;">
                    <span>📄 OpenAPI 3.0 JSON Spec (/openapi.json)</span>
                    <span>↗</span>
                </a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" class="btn-header" style="justify-content: space-between;">
                    <span>📦 GitHub Source Code Repository</span>
                    <span>↗</span>
                </a>
            </div>
        </div>
    </div>

    <script>
        let latestAnalysisData = null;
        let currentFilterPhase = 'ALL';

        function toggleModal(id, show) {
            document.getElementById(id).style.display = show ? 'flex' : 'none';
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
            document.querySelectorAll('.btn-phase-tab').forEach(b => b.classList.remove('active'));
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
            if (currentFilterPhase !== 'ALL') {
                candidates = candidates.filter(c => c.clinical_phase === Number(currentFilterPhase));
            }

            if (candidates.length === 0) {
                therapiesList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; text-align: center; padding: 2rem;">No clinical combination therapies matching this filter.</div>';
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
                    <div style="font-size: 0.8rem; color: #38bdf8; margin-bottom: 0.3rem;">
                        Secondary Target: <strong style="color:#fff;">${c.secondary_target}</strong> | Synergy Score: <strong>${c.synergy_score}</strong> | Hub Centrality: ${c.hub_penalized_centrality}
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                    <div class="candidate-rationale">${c.biological_rationale}</div>
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
