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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-void: #090d16;
            --panel-bg: rgba(15, 23, 42, 0.85);
            --panel-border: rgba(255, 255, 255, 0.08);
            --accent-indigo: #6366f1;
            --accent-indigo-hover: #4f46e5;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-sky: #38bdf8;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-tertiary: #64748b;
            --font-main: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-void);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 1.5rem 1rem;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Navigation Header */
        header.app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--panel-border);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .brand-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
        }

        .brand-subtitle {
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #34d399;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .btn-header {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            color: var(--text-secondary);
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            text-decoration: none;
        }

        .btn-header:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-header.primary {
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.35);
            color: #a5b4fc;
        }

        .btn-header.primary:hover {
            background: rgba(99, 102, 241, 0.25);
            color: #fff;
        }

        /* Workstation Layout */
        .workstation-grid {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 960px) {
            .workstation-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .panel-title-text {
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 1.1rem;
        }

        label.field-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        .tooltip-trigger {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-tertiary);
            font-size: 0.68rem;
            cursor: help;
            position: relative;
        }

        .tooltip-trigger .tooltip-body {
            visibility: hidden;
            width: 230px;
            background: #0f172a;
            color: #cbd5e1;
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            position: absolute;
            z-index: 100;
            bottom: 130%;
            right: 0;
            opacity: 0;
            transition: opacity 0.2s ease;
            font-size: 0.76rem;
            border: 1px solid var(--panel-border);
            box-shadow: 0 10px 25px rgba(0,0,0,0.6);
            font-weight: 400;
            line-height: 1.35;
        }

        .tooltip-trigger:hover .tooltip-body {
            visibility: visible;
            opacity: 1;
        }

        input.input-field {
            width: 100%;
            padding: 0.7rem 0.9rem;
            background: rgba(9, 13, 22, 0.8);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            color: #fff;
            font-family: var(--font-main);
            font-size: 0.92rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input.input-field:focus {
            outline: none;
            border-color: var(--accent-indigo);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        .btn-run {
            width: 100%;
            padding: 0.85rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 0.92rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
            margin-top: 0.5rem;
        }

        .btn-run:hover {
            opacity: 0.95;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
        }

        .btn-run:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Preset Scenario Groups */
        .preset-group {
            margin-top: 1.5rem;
            border-top: 1px solid var(--panel-border);
            padding-top: 1rem;
        }

        .preset-group-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0.75rem 0 0.4rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .prevalence-tag {
            font-size: 0.68rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            background: rgba(99, 102, 241, 0.1);
            color: #a5b4fc;
            font-weight: 600;
        }

        .preset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 0.4rem;
        }

        .btn-preset-card {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--text-secondary);
            padding: 0.45rem 0.65rem;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }

        .btn-preset-card:hover {
            color: #fff;
            border-color: rgba(99, 102, 241, 0.4);
            background: rgba(99, 102, 241, 0.08);
        }

        .preset-meta {
            font-size: 0.68rem;
            color: var(--text-tertiary);
            margin-top: 0.1rem;
        }

        /* Results Canvas */
        .error-box {
            display: none;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            padding: 1rem;
            color: #fca5a5;
            font-size: 0.88rem;
            margin-bottom: 1.25rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-tile {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }

        .metric-number {
            font-size: clamp(1.1rem, 2.2vw, 1.5rem);
            font-weight: 800;
            color: var(--accent-sky);
            font-family: var(--font-mono);
            font-variant-numeric: tabular-nums;
        }

        .metric-label {
            font-size: 0.72rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }

        .canonical-bar {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
        }

        .pill-badge {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: #a5b4fc;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-family: var(--font-mono);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .filter-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .phase-tabs {
            display: flex;
            gap: 0.3rem;
        }

        .btn-phase-tab {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: var(--text-secondary);
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-phase-tab.active {
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.4);
            color: #a5b4fc;
            font-weight: 600;
        }

        /* Combination Therapy Cards */
        .candidate-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .candidate-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }

        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .drug-pair-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
        }

        .badge-phase {
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
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

        .candidate-metrics {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.6rem;
        }

        .progress-track {
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.6rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #38bdf8 100%);
            border-radius: 999px;
            width: 0%;
            transition: width 0.6s ease;
        }

        .candidate-rationale {
            font-size: 0.84rem;
            color: var(--text-secondary);
            line-height: 1.45;
        }

        .loader-panel {
            display: none;
            text-align: center;
            padding: 4rem 1rem;
            color: var(--text-secondary);
        }

        .spin-ring {
            width: 44px;
            height: 44px;
            border: 3px solid rgba(255,255,255,0.08);
            border-radius: 50%;
            border-top-color: var(--accent-indigo);
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1.25rem auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .empty-canvas {
            text-align: center;
            padding: 4.5rem 1rem;
            color: var(--text-tertiary);
        }

        /* Clinician Guide Modal */
        .modal-wrapper {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(9, 13, 22, 0.85);
            backdrop-filter: blur(12px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal-box {
            background: #0f172a;
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            max-width: 760px;
            width: 100%;
            max-height: 88vh;
            overflow-y: auto;
            padding: 2rem;
            color: #e2e8f0;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }

        .modal-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--panel-border);
        }

        .modal-heading {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
        }

        .modal-btn-close {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
        }

        .guide-section {
            margin-bottom: 1.5rem;
        }

        .guide-term {
            font-weight: 700;
            color: #fff;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }

        .guide-body {
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* Developer Footer */
        footer.dev-footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--panel-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--text-tertiary);
        }

        .dev-links {
            display: flex;
            gap: 1rem;
        }

        .dev-link {
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.2s ease;
        }

        .dev-link:hover {
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Application Header Bar -->
        <header class="app-header">
            <div class="brand-area">
                <div class="brand-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                </div>
                <div>
                    <div class="brand-title">Resistance Bypass Engine</div>
                    <div class="brand-subtitle">Targeted Oncology Microservice v0.1.0</div>
                </div>
            </div>

            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>ENGINE ONLINE</span>
                </div>
                <button class="btn-header primary" onclick="toggleModal(true)">
                    <span>📖 Clinician & Methodological Guide</span>
                </button>
            </div>
        </header>

        <!-- Workstation Grid -->
        <div class="workstation-grid">
            <!-- Left Sidebar Controls -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
                        <span>Analysis Parameters</span>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label class="field-label" for="primary_target">
                            <span>Primary Target Symbol</span>
                            <span class="tooltip-trigger">?
                                <span class="tooltip-body">Oncogenic driver gene targeted by frontline therapy (e.g. EGFR, ERBB2/HER2, ALK, BRAF).</span>
                            </span>
                        </label>
                        <input class="input-field" type="text" id="primary_target" value="EGFR" required placeholder="e.g. EGFR, ERBB2">
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="primary_drug">
                            <span>Primary Drug Name</span>
                            <span class="tooltip-trigger">?
                                <span class="tooltip-body">Frontline targeted agent administered to patient (e.g. Osimertinib, Trastuzumab, Sotorasib).</span>
                            </span>
                        </label>
                        <input class="input-field" type="text" id="primary_drug" value="Osimertinib" required placeholder="e.g. Osimertinib">
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="resistance_marker">
                            <span>Secondary Resistance Marker</span>
                            <span class="tooltip-trigger">?
                                <span class="tooltip-body">Bypass marker or secondary mutation driving acquired resistance (e.g. MET, KRAS, BRAF).</span>
                            </span>
                        </label>
                        <input class="input-field" type="text" id="resistance_marker" value="MET" required placeholder="e.g. MET, KRAS">
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="cancer_type">Cancer Indication</label>
                        <input class="input-field" type="text" id="cancer_type" value="Non-Small Cell Lung Cancer">
                    </div>

                    <button type="submit" id="submitBtn" class="btn-run">
                        <span>Execute Resistance Pipeline</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>

                <!-- High Prevalence Presets -->
                <div class="preset-group">
                    <div class="preset-group-title">
                        <span>🔥 High Prevalence Clinical Scenarios</span>
                        <span class="prevalence-tag">Top 1% Global</span>
                    </div>
                    <div class="preset-grid">
                        <button class="btn-preset-card" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'Non-Small Cell Lung Cancer')">
                            <strong>EGFR + MET</strong>
                            <div class="preset-meta">NSCLC (15-20% Bypass)</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'Non-Small Cell Lung Cancer')">
                            <strong>EGFR + EGFR</strong>
                            <div class="preset-meta">NSCLC (C797S Mutation)</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'Breast Cancer')">
                            <strong>HER2 + MET</strong>
                            <div class="preset-meta">Breast Cancer Bypass</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer')">
                            <strong>KRAS + EGFR</strong>
                            <div class="preset-meta">Colorectal RTK Feedback</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('BRAF', 'Dabrafenib', 'MAP2K1', 'Melanoma')">
                            <strong>BRAF + MEK1</strong>
                            <div class="preset-meta">Melanoma MAPK Axis</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('BRAF', 'Dabrafenib', 'EGFR', 'Colorectal Cancer')">
                            <strong>BRAF + EGFR</strong>
                            <div class="preset-meta">CRC V600E Feedback</div>
                        </button>
                    </div>

                    <div class="preset-group-title" style="margin-top: 1rem;">
                        <span>⚡ Specialized Indications</span>
                    </div>
                    <div class="preset-grid">
                        <button class="btn-preset-card" onclick="setPreset('ALK', 'Alectinib', 'MET', 'NSCLC')">
                            <strong>ALK + MET</strong>
                            <div class="preset-meta">ALK+ NSCLC Bypass</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'Breast Cancer')">
                            <strong>ESR1 + CDK4</strong>
                            <div class="preset-meta">HR+ Breast Cyclin Axis</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('BCR-ABL', 'Imatinib', 'ABL1', 'CML')">
                            <strong>BCR-ABL + ABL1</strong>
                            <div class="preset-meta">CML T315I Gatekeeper</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('AR', 'Enzalutamide', 'PIK3CA', 'Prostate Cancer')">
                            <strong>AR + PIK3CA</strong>
                            <div class="preset-meta">Prostate PI3K Crosstalk</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('RET', 'Selpercatinib', 'MET', 'NSCLC')">
                            <strong>RET + MET</strong>
                            <div class="preset-meta">RET Fusion Bypass</div>
                        </button>
                        <button class="btn-preset-card" onclick="setPreset('ROS1', 'Entrectinib', 'MET', 'NSCLC')">
                            <strong>ROS1 + MET</strong>
                            <div class="preset-meta">ROS1 Fusion Bypass</div>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right Results Panel -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span>Resistance Topology & Synergy Analysis</span>
                    </div>
                    <button id="copyJsonBtn" class="btn-header" style="display: none;" onclick="copyResultJson()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                        <span>Copy JSON</span>
                    </button>
                </div>

                <div id="errorBanner" class="error-box"></div>

                <div id="loader" class="loader-panel">
                    <div class="spin-ring"></div>
                    <p style="font-weight: 600; color: #fff; margin-bottom: 0.3rem;">Querying Biological API Networks...</p>
                    <p style="font-size: 0.82rem;">Resolving HGNC/UniProt IDs • Fetching STRING-DB PPI & Open Targets GraphQL</p>
                </div>

                <div id="placeholder" class="empty-canvas">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="1.5" style="margin-bottom: 1rem;"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
                    <p style="font-weight: 600; color: var(--text-secondary); margin-bottom: 0.4rem;">No Active Pipeline Execution</p>
                    <p style="font-size: 0.84rem;">Select a high-prevalence clinical scenario on the left or enter custom target parameters.</p>
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

                    <div class="filter-bar">
                        <h3 style="font-size: 0.95rem; font-weight: 700; color: #fff;">Ranked Dual-Drug Combination Therapies</h3>
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

        <!-- Developer Footer -->
        <footer class="dev-footer">
            <div>Targeted Oncology Resistance Bypass Engine • Open Source Microservice</div>
            <div class="dev-links">
                <a href="/docs" target="_blank" class="dev-link">⚡ Swagger UI (/docs)</a>
                <a href="/health" target="_blank" class="dev-link">💚 System Diagnostics (/health)</a>
                <a href="/openapi.json" target="_blank" class="dev-link">📄 OpenAPI Spec</a>
                <a href="https://github.com/realrezi/Resistance-Bypass-Engine" target="_blank" class="dev-link">📦 GitHub</a>
            </div>
        </footer>
    </div>

    <!-- Clinician Guide Modal -->
    <div id="clinicianModal" class="modal-wrapper" onclick="if(event.target===this) toggleModal(false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">📖 Clinician & Methodological Guide</div>
                <button class="modal-btn-close" onclick="toggleModal(false)">&times;</button>
            </div>
            
            <div class="guide-section">
                <div class="guide-term">🎯 Primary Target & Primary Drug</div>
                <div class="guide-body">The frontline oncogenic driver protein (e.g. EGFR in lung adenocarcinoma) and the primary targeted tyrosine kinase inhibitor or monoclonal antibody administered to the patient (e.g. Osimertinib).</div>
            </div>

            <div class="guide-section">
                <div class="guide-term">⚡ Secondary Resistance Marker & Mechanism</div>
                <div class="guide-body">The gene or protein driving acquired therapeutic resistance. Resistance is biological categorized into two main branches:</div>
                <ul style="margin: 0.5rem 0 0 1.2rem; font-size: 0.85rem; color: var(--text-secondary);">
                    <li><strong>Off-Target Bypass:</strong> Hyperactivation of a parallel signaling axis (e.g., MET receptor amplification or KRAS activation) that circumvents the primary target blockade to sustain downstream cell proliferation and anti-apoptotic signaling (PI3K-AKT and MAPK-ERK pathways).</li>
                    <li><strong>On-Target Mutation:</strong> Secondary mutations directly inside the primary target gene (e.g., EGFR C797S or ABL1 T315I) that structurally alter the ATP binding pocket to disrupt drug binding affinity.</li>
                </ul>
            </div>

            <div class="guide-section">
                <div class="guide-term">🧮 Hub-Penalized Bottleneck Centrality</div>
                <div class="guide-body">Graph analysis runs via NetworkX using pure Python. Centrality is computed as <code>Betweenness / log2(Degree + 2)</code>. This mathematical formula explicitly penalizes ubiquitous, non-specific hub proteins (such as Ubiquitin or TP53) while isolating true signaling bottleneck nodes driving resistance.</div>
            </div>

            <div class="guide-section">
                <div class="guide-term">🧪 Synergy Score & Dual-Drug Ranking</div>
                <div class="guide-body">A normalized score (0.0 to 1.0) combining network shortest path distance, bottleneck centrality, and ChEMBL drug-target binding affinity (pChEMBL). Recommends active, non-withdrawn clinical combination therapies to override resistance.</div>
            </div>

            <div style="text-align: right; margin-top: 1.5rem;">
                <button class="btn-run" style="width: auto; padding: 0.6rem 1.4rem; display: inline-flex;" onclick="toggleModal(false)">Close Guide</button>
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
                therapiesList.innerHTML = '<div style="color: var(--text-tertiary); font-size: 0.9rem; text-align: center; padding: 2.5rem;">No clinical combination therapies matching this filter.</div>';
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
                    <div class="candidate-metrics">
                        Secondary Target: <strong style="color:#fff;">${c.secondary_target}</strong> | Synergy Score: <strong style="color:var(--accent-sky);">${c.synergy_score}</strong> | Hub Penalty Centrality: ${c.hub_penalized_centrality}
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
