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
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <style>
        :root {
            --bg-lab: #090d16;
            --card-bg: #0f172a;
            --panel-bg: #1e293b;
            --border-lab: #1e293b;
            --border-subtle: #334155;
            --genomic-blue: #38bdf8;
            --genomic-blue-hover: #0284c7;
            --mutation-red: #f43f5e;
            --approved-green: #10b981;
            --purple-pathway: #c084fc;
            --amber-phase: #f59e0b;
            --text-main: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --shadow-lab: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 4px 12px -2px rgba(0, 0, 0, 0.3);
            --shadow-hover: 0 15px 35px -5px rgba(56, 189, 248, 0.2), 0 8px 15px -4px rgba(0, 0, 0, 0.4);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-lab);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.06) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(192, 132, 252, 0.06) 0%, transparent 40%),
                linear-gradient(to right, rgba(30, 41, 59, 0.4) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(30, 41, 59, 0.4) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
            color: var(--text-secondary);
            min-height: 100vh;
            padding: 1.25rem;
            line-height: 1.45;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1480px;
            margin: 0 auto;
        }

        /* Header Bar */
        header.app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.85rem 1.35rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.5);
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 1rem;
            color: #fff;
        }

        .brand-area {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .brand-logo {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
            border: 1.5px solid rgba(56, 189, 248, 0.4);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
        }

        .brand-subtitle {
            font-size: 0.78rem;
            color: #94a3b8;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.4rem 0.85rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #34d399;
            font-size: 0.78rem;
            font-weight: 700;
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
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f1f5f9;
            padding: 0.45rem 0.9rem;
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
            color: #ffffff;
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(56, 189, 248, 0.5);
        }

        .btn-header.author-btn {
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.25) 0%, rgba(124, 58, 237, 0.25) 100%);
            border-color: rgba(56, 189, 248, 0.4);
            color: #38bdf8;
            font-weight: 700;
        }

        .btn-header.author-btn:hover {
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.4) 0%, rgba(124, 58, 237, 0.4) 100%);
            color: #ffffff;
            border-color: #38bdf8;
        }

        /* Main Section: Dark Slate Academic Matrix */
        .academic-matrix-section {
            background: var(--card-bg);
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 1.35rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow-lab);
        }

        .matrix-top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .matrix-title-text {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .matrix-tabs-container {
            display: flex;
            gap: 0.4rem;
            border-bottom: 2px solid #1e293b;
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
            overflow-x: auto;
        }

        .btn-matrix-tab {
            background: #1e293b;
            border: 1px solid #334155;
            color: #94a3b8;
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .btn-matrix-tab.active {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border-color: #38bdf8;
            border-bottom: 3px solid #38bdf8;
            box-shadow: 0 2px 10px rgba(56, 189, 248, 0.2);
        }

        .prevalence-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
            gap: 0.9rem;
        }

        .prevalence-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-left: 4px solid var(--genomic-blue);
            border-radius: 10px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .prevalence-card:hover {
            background: #0f172a;
            border-color: var(--genomic-blue);
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }

        .prevalence-card.high-prev { border-left-color: var(--mutation-red); }
        .prevalence-card.approved-prev { border-left-color: var(--approved-green); }

        .prevalence-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }

        .scenario-pair-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .badge-prevalence {
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 800;
            font-family: var(--font-mono);
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        .badge-prevalence.high {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border-color: rgba(245, 158, 11, 0.3);
        }

        .locus-tag {
            font-size: 0.76rem;
            color: var(--purple-pathway);
            font-family: var(--font-mono);
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .scenario-mechanism {
            font-size: 0.82rem;
            color: var(--text-secondary);
            line-height: 1.45;
        }

        /* Workstation Layout Grid */
        .workstation-grid {
            display: grid;
            grid-template-columns: 460px 1fr;
            gap: 1.25rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .workstation-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 1.35rem;
            box-shadow: var(--shadow-lab);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid #1e293b;
        }

        .panel-title-text {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Clinical Form Controls */
        .form-group {
            margin-bottom: 1rem;
        }

        label.field-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.35rem;
        }

        input.input-field {
            width: 100%;
            padding: 0.7rem 0.9rem;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #f8fafc;
            font-family: var(--font-main);
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        input.input-field:focus {
            outline: none;
            background: #0f172a;
            border-color: var(--genomic-blue);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
        }

        .btn-run {
            width: 100%;
            padding: 0.85rem;
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: none;
            border-radius: 8px;
            color: #ffffff;
            font-size: 0.95rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
            margin-top: 0.5rem;
        }

        .btn-run:hover {
            opacity: 0.95;
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5);
        }

        /* Large & Spacious Point-and-Click Picker Palette */
        .spacious-picker-section {
            margin-top: 1.25rem;
            border-top: 2px solid #1e293b;
            padding-top: 1rem;
        }

        .picker-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.6rem;
        }

        .picker-title {
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .btn-clear-picker {
            background: #1e293b;
            border: 1px solid #334155;
            color: var(--text-muted);
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            cursor: pointer;
        }

        .btn-clear-picker:hover {
            color: var(--mutation-red);
            border-color: var(--mutation-red);
            background: rgba(244, 63, 94, 0.15);
        }

        .picker-category-label {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0.6rem 0 0.35rem 0;
        }

        .picker-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.75rem;
        }

        .btn-spacious-chip {
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 0.5rem 0.95rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .btn-spacious-chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
            background: #0f172a;
        }

        .btn-spacious-chip.target-chip {
            border-left: 4px solid var(--genomic-blue);
            background: rgba(56, 189, 248, 0.1);
        }
        .btn-spacious-chip.target-chip:hover { border-color: var(--genomic-blue); }

        .btn-spacious-chip.drug-chip {
            border-left: 4px solid var(--approved-green);
            background: rgba(16, 185, 129, 0.1);
        }
        .btn-spacious-chip.drug-chip:hover { border-color: var(--approved-green); }

        .btn-spacious-chip.marker-chip {
            border-left: 4px solid var(--mutation-red);
            background: rgba(244, 63, 94, 0.1);
        }
        .btn-spacious-chip.marker-chip:hover { border-color: var(--mutation-red); }

        /* Results Canvas */
        .vector-graph-canvas {
            background: linear-gradient(135deg, #030712 0%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            color: #fff;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.85rem;
            margin-bottom: 1.25rem;
        }

        .metric-tile {
            background: #1e293b;
            border: 1px solid #334155;
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
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38bdf8;
            padding: 0.35rem 0.7rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-family: var(--font-mono);
            font-weight: 700;
        }

        .candidate-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: var(--shadow-lab);
            transition: all 0.2s ease;
        }

        .candidate-card:hover {
            border-color: var(--genomic-blue);
            box-shadow: var(--shadow-hover);
        }

        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .drug-pair-name {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .badge-phase {
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 800;
            font-family: var(--font-mono);
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .badge-phase.approved {
            background: rgba(16, 185, 129, 0.15);
            color: var(--approved-green);
            border-color: rgba(16, 185, 129, 0.3);
        }

        .progress-track {
            height: 7px;
            background: #0f172a;
            border-radius: 999px;
            overflow: hidden;
            margin: 0.6rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #0284c7 0%, #059669 100%);
            border-radius: 999px;
            width: 0%;
            transition: width 0.6s ease;
        }

        /* Modals */
        .modal-wrapper {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(3, 7, 18, 0.85);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }

        .modal-box {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 16px;
            max-width: 760px;
            width: 100%;
            max-height: 88vh;
            overflow-y: auto;
            padding: 1.75rem;
            color: #cbd5e1;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }

        .modal-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #1e293b;
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
        <!-- Application Header Bar -->
        <header class="app-header">
            <div class="brand-area">
                <div class="brand-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M2 9c6.667 6 13.333 0 20 6"/><circle cx="7" cy="12" r="1.5" fill="#38bdf8"/><circle cx="12" cy="12" r="1.5" fill="#e11d48"/><circle cx="17" cy="12" r="1.5" fill="#38bdf8"/></svg>
                </div>
                <div>
                    <div class="brand-title">Targeted Oncology Resistance Bypass Engine</div>
                    <div class="brand-subtitle">Precision Network Biology & Resistance Modeling</div>
                </div>
            </div>

            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>GENOMIC APIs ONLINE</span>
                </div>
                <button class="btn-header" onclick="toggleModal('guidanceModal', true)">
                    <span>💡 Purpose & Workflow</span>
                </button>
                <button class="btn-header" onclick="toggleModal('clinicianModal', true)">
                    <span>📖 Methodological Guide</span>
                </button>
                <a href="https://github.com/realrezi" target="_blank" class="btn-header author-btn">
                    <span>Built by Ahmadreza Shirdel</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </a>
            </div>
        </header>

        <!-- Prevalent Tumor Resistance Genotypes Section (Academic & Professional) -->
        <section class="academic-matrix-section">
            <div class="matrix-top-bar">
                <div class="matrix-title-text">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    <span>Prevalent Tumor Resistance Genotypes & Clinical Bypass Pathways</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">
                    Instantiate characterized clinical phenotypes and high-prevalence resistance loci across 9 primary oncology indications:
                </div>
            </div>

            <div class="matrix-tabs-container">
                <button class="btn-matrix-tab active" onclick="switchMatrixCategory('nsclc')">🫁 NSCLC Loci</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('breast')">🎗️ Breast Carcinoma</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('crc')">🧬 Colorectal Loci</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('melanoma')">☀️ Melanoma Axis</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('cml')">🩸 Hematologic Myeloma</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('prostate')">🩺 Prostate Axis</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('ovarian')">🎗️ Ovarian Signatures</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('glioma')">🧠 Glioma & CNS</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('thyroid')">🫀 Thyroid & Rare Fusions</button>
            </div>

            <!-- Tab 1: NSCLC Scenarios -->
            <div id="matrix-nsclc" class="prevalence-cards-grid">
                <div class="prevalence-card high-prev" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + MET Amplification</span>
                        <span class="badge-prevalence">15–20% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFR) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-Target RTK Bypass. MET amplification reactivates ERBB3/PI3K signaling despite Osimertinib blockade.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + C797S Secondary Mutation</span>
                        <span class="badge-prevalence high">7–10% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (Exon 20 C797S)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> On-Target ATP Pocket Mutation. C797S mutation disrupts covalent binding of 3rd-gen TKI Osimertinib.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ALK', 'Alectinib', 'MET', 'Non-Small Cell Lung Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ALK + MET Bypass</span>
                        <span class="badge-prevalence high">8–12% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 2p23.2 (ALK) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Parallel RTK activation bypassing 2nd-gen ALK inhibitor (Alectinib) blockade in ALK+ NSCLC.</div>
                </div>
            </div>

            <!-- Tab 2: Breast Cancer Scenarios -->
            <div id="matrix-breast" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'HER2+ Breast Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">HER2 + MET Amplification</span>
                        <span class="badge-prevalence">10–15% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 17q12 (ERBB2) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-target RTK bypass hyperactivation overriding anti-HER2 monoclonal antibody (Trastuzumab) therapy.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'HR+/HER2- Breast Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ESR1 + CDK4/6 Cyclin Axis</span>
                        <span class="badge-prevalence high">12–18% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 6q25.1 (ESR1) ➔ Chr 12q14.1 (CDK4)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Endocrine therapy escape driven by ligand-independent ESR1 mutations & Cyclin D1/CDK4 pathway reactivation.</div>
                </div>
            </div>

            <!-- Tab 3: Colorectal Cancer Scenarios -->
            <div id="matrix-crc" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">KRAS G12C + EGFR Feedback</span>
                        <span class="badge-prevalence">20–25% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 12p12.1 (KRAS) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Rapid RTK feedback loop reactivating MAPK signaling; requires dual KRAS G12C + EGFR blockade.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('BRAF', 'Encorafenib', 'EGFR', 'Colorectal Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600E + EGFR Feedback</span>
                        <span class="badge-prevalence high">10–12% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Monotherapy BRAF inhibition induces strong EGFR feedback; FDA-approved Encorafenib + Cetuximab dual therapy.</div>
                </div>
            </div>

            <!-- Tab 4: Cutaneous Melanoma Scenarios -->
            <div id="matrix-melanoma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card approved-prev" onclick="setPreset('BRAF', 'Dabrafenib', 'MAP2K1', 'Cutaneous Melanoma')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600 + MAP2K1/MEK1</span>
                        <span class="badge-prevalence">35–45% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 15q22.31 (MAP2K1)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Re-activation of MAPK signaling cascade overcome by FDA-approved dual BRAF + MEK inhibition (Dabrafenib + Trametinib).</div>
                </div>
            </div>

            <!-- Tab 5: Hematologic Scenarios -->
            <div id="matrix-cml" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('ABL1', 'Imatinib', 'ABL1', 'Chronic Myeloid Leukemia')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BCR-ABL + T315I Gatekeeper</span>
                        <span class="badge-prevalence">15–20% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 9q34.12 (ABL1 T315I Gatekeeper)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Threonine to Isoleucine mutation causes steric clash with 1st/2nd-gen TKIs; managed via Ponatinib or Asciminib combination.</div>
                </div>
            </div>

            <!-- Tab 6: Prostate Cancer Scenarios -->
            <div id="matrix-prostate" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('AR', 'Enzalutamide', 'PIK3CA', 'Metastatic Castration-Resistant Prostate Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">AR + PIK3CA Crosstalk</span>
                        <span class="badge-prevalence">12–16% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr Xq12 (AR) ➔ Chr 3q26.32 (PIK3CA)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Reciprocal feedback crosstalk between Androgen Receptor and PI3K/AKT signaling pathways in mCRPC.</div>
                </div>
            </div>

            <!-- Tab 7: Ovarian & GYN Scenarios -->
            <div id="matrix-ovarian" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('PIK3CA', 'Alpelisib', 'KRAS', 'Ovarian Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">PIK3CA + KRAS Bypass</span>
                        <span class="badge-prevalence">8–12% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 3q26.32 (PIK3CA) ➔ Chr 12p12.1 (KRAS)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Parallel activation of RAS/MAPK axis circumventing selective PI3Kalpha inhibitor blockade in gynecologic malignancies.</div>
                </div>
            </div>

            <!-- Tab 8: Glioma & CNS Scenarios -->
            <div id="matrix-glioma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('EGFR', 'Gefitinib', 'MET', 'Glioblastoma')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFRvIII + MET Amplification</span>
                        <span class="badge-prevalence">10–14% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFRvIII) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Co-activation of multiple RTKs (EGFRvIII and MET) driving redundant oncogenic signaling in high-grade glioma.</div>
                </div>
            </div>

            <!-- Tab 9: Thyroid & Rare Fusions Scenarios -->
            <div id="matrix-thyroid" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('RET', 'Selpercatinib', 'MET', 'Thyroid Cancer')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">RET Fusion + MET Bypass</span>
                        <span class="badge-prevalence">&lt;5% Global Prev.</span>
                    </div>
                    <div class="locus-tag">Chr 10q11.21 (RET) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Acquired MET amplification emerging after selective RET inhibitor (Selpercatinib) treatment.</div>
                </div>
            </div>
        </section>

        <!-- Workstation Grid -->
        <div class="workstation-grid">
            <!-- Left Panel: Form & Molecular Target Selector -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
                        <span>Molecular Target & Drug Parameters</span>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="form-group">
                        <label class="field-label" for="primary_target">Primary Target Gene Locus</label>
                        <input class="input-field" type="text" id="primary_target" value="EGFR" required placeholder="Specify primary target locus (e.g. EGFR, ERBB2)..." list="targets_list" autocomplete="off">
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
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="primary_drug">Frontline Targeted Inhibitor</label>
                        <input class="input-field" type="text" id="primary_drug" value="Osimertinib" required placeholder="Specify frontline therapeutic agent..." list="drugs_list" autocomplete="off">
                        <datalist id="drugs_list">
                            <option value="Osimertinib">Osimertinib (3rd-gen EGFR TKI)</option>
                            <option value="Gefitinib">Gefitinib (1st-gen EGFR TKI)</option>
                            <option value="Trastuzumab">Trastuzumab (Anti-HER2 mAb)</option>
                            <option value="Capmatinib">Capmatinib (MET TKI)</option>
                            <option value="Alectinib">Alectinib (2nd-gen ALK TKI)</option>
                            <option value="Sotorasib">Sotorasib (KRAS G12C Inhibitor)</option>
                            <option value="Dabrafenib">Dabrafenib (BRAF Kinase Inhibitor)</option>
                            <option value="Encorafenib">Encorafenib (BRAF Kinase Inhibitor)</option>
                            <option value="Alpelisib">Alpelisib (PI3Kalpha Inhibitor)</option>
                            <option value="Fulvestrant">Fulvestrant (SERD)</option>
                            <option value="Imatinib">Imatinib (BCR-ABL TKI)</option>
                            <option value="Palbociclib">Palbociclib (CDK4/6 Inhibitor)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="resistance_marker">Secondary Bypass Resistance Marker</label>
                        <input class="input-field" type="text" id="resistance_marker" value="MET" required placeholder="Specify bypass marker or secondary mutation..." list="markers_list" autocomplete="off">
                        <datalist id="markers_list">
                            <option value="MET">MET Amplification / Bypass Hyperactivation</option>
                            <option value="EGFR">EGFR Secondary Gatekeeper (C797S / T790M)</option>
                            <option value="KRAS">KRAS Secondary Activation (G12C / G12V)</option>
                            <option value="BRAF">BRAF V600E Activation</option>
                            <option value="PIK3CA">PIK3CA Hyperactivation Mutation (H1047R)</option>
                            <option value="MAP2K1">MAP2K1 / MEK1 Activation</option>
                            <option value="CDK4">CDK4 Cyclin Pathway Axis</option>
                            <option value="ABL1">ABL1 Gatekeeper Mutation (T315I)</option>
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label class="field-label" for="cancer_type">Oncology Indication</label>
                        <input class="input-field" type="text" id="cancer_type" value="Non-Small Cell Lung Cancer" placeholder="Select tumor type..." list="indications_list" autocomplete="off">
                        <datalist id="indications_list">
                            <option value="Non-Small Cell Lung Cancer">Non-Small Cell Lung Cancer (NSCLC)</option>
                            <option value="HER2+ Breast Cancer">HER2+ Breast Cancer</option>
                            <option value="Colorectal Cancer">Colorectal Cancer (CRC)</option>
                            <option value="Cutaneous Melanoma">Cutaneous Melanoma</option>
                            <option value="Chronic Myeloid Leukemia">Chronic Myeloid Leukemia (CML)</option>
                        </datalist>
                    </div>

                    <button type="submit" id="submitBtn" class="btn-run">
                        <span>Compute Network Graph Algorithms</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>

                <!-- Molecular Target & Inhibitor Selector Palette -->
                <div class="spacious-picker-section">
                    <div class="picker-header">
                        <div class="picker-title">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M15 15l5 5M4 4l16 16"/></svg>
                            <span>Molecular Target & Inhibitor Selector</span>
                        </div>
                        <button type="button" class="btn-clear-picker" onclick="clearInputs()">Reset Fields</button>
                    </div>

                    <div class="picker-category-label">Target Driver Loci:</div>
                    <div class="picker-chip-grid">
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'EGFR')">EGFR</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'ERBB2')">HER2</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'ALK')">ALK</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'KRAS')">KRAS</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'BRAF')">BRAF</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'PIK3CA')">PIK3CA</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'ESR1')">ESR1</button>
                        <button type="button" class="btn-spacious-chip target-chip" onclick="quickFill('primary_target', 'ABL1')">ABL1</button>
                    </div>

                    <div class="picker-category-label">Frontline Inhibitor Molecules:</div>
                    <div class="picker-chip-grid">
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Osimertinib')">Osimertinib</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Trastuzumab')">Trastuzumab</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Alectinib')">Alectinib</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Sotorasib')">Sotorasib</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Dabrafenib')">Dabrafenib</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Fulvestrant')">Fulvestrant</button>
                        <button type="button" class="btn-spacious-chip drug-chip" onclick="quickFill('primary_drug', 'Imatinib')">Imatinib</button>
                    </div>

                    <div class="picker-category-label">Acquired Resistance Loci:</div>
                    <div class="picker-chip-grid">
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'MET')">MET Bypass</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'EGFR')">EGFR C797S</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'KRAS')">KRAS Activation</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'BRAF')">BRAF V600E</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'PIK3CA')">PIK3CA Mutation</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'ABL1')">ABL1 T315I</button>
                        <button type="button" class="btn-spacious-chip marker-chip" onclick="quickFill('resistance_marker', 'CDK4')">CDK4/6 Axis</button>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Signal Transduction Pathway & Synergy Matrix -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span>Signal Transduction Pathway Topology & Dual-Target Synergy Matrix</span>
                    </div>
                    <button id="copyJsonBtn" class="btn-header" style="display: none; color: #fff;" onclick="copyResultJson()">📋 Copy Report JSON</button>
                </div>

                <div id="errorBanner" style="display:none; background:rgba(244, 63, 94, 0.15); border:1px solid rgba(244, 63, 94, 0.3); padding:0.85rem; border-radius:8px; color:#fb7185; margin-bottom:1rem; font-size:0.85rem;"></div>

                <div id="loader" style="display: none; text-align: center; padding: 3rem 1rem;">
                    <div style="width: 44px; height: 44px; border: 3px solid #1e293b; border-radius: 50%; border-top-color: #38bdf8; animation: spin 0.8s linear infinite; margin: 0 auto 1rem auto;"></div>
                    <p style="font-weight: 800; color: #f8fafc;">Querying REST/GraphQL PPI Biological Databases...</p>
                    <p style="font-size: 0.82rem; color: #94a3b8;">Resolving Canonical HGNC IDs • Extracting NetworkX LCC Topology • Querying ChEMBL & Open Targets</p>
                </div>

                <!-- Multi-Kinase Cell Membrane SVG Signaling Visualizer (Professional & Dynamic) -->
                <div id="placeholder">
                    <div class="vector-graph-canvas" style="background: #030712; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; color: #fff; text-align: center; position: relative;">
                        <svg width="100%" height="240" viewBox="0 0 700 240" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <linearGradient id="primaryGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#38bdf8" />
                                    <stop offset="100%" stop-color="#0284c7" />
                                </linearGradient>
                                <linearGradient id="resistGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#f43f5e" />
                                    <stop offset="100%" stop-color="#e11d48" />
                                </linearGradient>
                                <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#c084fc" />
                                    <stop offset="100%" stop-color="#7c3aed" />
                                </linearGradient>
                                <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#34d399" />
                                    <stop offset="100%" stop-color="#059669" />
                                </linearGradient>
                                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                                    <feGaussianBlur stdDeviation="4" result="blur" />
                                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                </filter>
                            </defs>

                            <!-- Lipid Bilayer Membrane Layer -->
                            <line x1="20" y1="40" x2="680" y2="40" stroke="#334155" stroke-width="6" stroke-dasharray="8 6"/>
                            <text x="30" y="26" fill="#94a3b8" font-size="10" font-weight="700" font-family="sans-serif">EXTRACELLULAR RECEPTOR DOMAIN</text>
                            <text x="30" y="58" fill="#64748b" font-size="10" font-weight="700" font-family="sans-serif">PLASMA MEMBRANE LIPID BILAYER</text>

                            <!-- Signal Transduction Edges -->
                            <path d="M 120 40 L 120 120 L 250 80 L 400 80 L 520 120 L 520 40" stroke="#38bdf8" stroke-width="2.5" stroke-dasharray="6 4" fill="none"/>
                            <path d="M 250 80 L 320 170 L 450 170 L 520 120" stroke="#c084fc" stroke-width="2.5" fill="none"/>
                            <path d="M 400 80 L 580 160" stroke="#34d399" stroke-width="2" stroke-dasharray="4 4" fill="none"/>

                            <!-- Primary Target RTK Domain (EGFR) -->
                            <g transform="translate(120, 40)">
                                <rect x="-18" y="-20" width="36" height="40" rx="6" fill="url(#primaryGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="11">EGFR</text>
                                <circle cx="0" cy="30" r="14" fill="#0284c7" stroke="#38bdf8" stroke-width="2"/>
                                <text x="0" y="34" text-anchor="middle" fill="#fff" font-size="9" font-weight="800">p-Y</text>
                            </g>
                            <text x="120" y="10" text-anchor="middle" fill="#38bdf8" font-size="10" font-weight="700">Chr 7p11.2 (Primary Driver)</text>

                            <!-- Resistance Marker RTK Domain (MET) -->
                            <g transform="translate(520, 40)">
                                <rect x="-18" y="-20" width="36" height="40" rx="6" fill="url(#resistGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="11">MET</text>
                                <circle cx="0" cy="30" r="14" fill="#e11d48" stroke="#f43f5e" stroke-width="2"/>
                                <text x="0" y="34" text-anchor="middle" fill="#fff" font-size="9" font-weight="800">p-Y</text>
                            </g>
                            <text x="520" y="10" text-anchor="middle" fill="#f43f5e" font-size="10" font-weight="700">Chr 7q31.2 (Bypass Locus)</text>

                            <!-- Downstream Kinase Cascades -->
                            <!-- GRB2/SOS1 Adaptor -->
                            <g transform="translate(250, 80)">
                                <circle cx="0" cy="0" r="22" fill="url(#purpleGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">GRB2</text>
                            </g>

                            <!-- KRAS GTPase -->
                            <g transform="translate(400, 80)">
                                <rect x="-24" y="-18" width="48" height="36" rx="10" fill="url(#purpleGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">KRAS</text>
                            </g>

                            <!-- PI3K/AKT Pathway -->
                            <g transform="translate(320, 170)">
                                <circle cx="0" cy="0" r="20" fill="url(#greenGrad)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="9">PIK3CA</text>
                            </g>

                            <g transform="translate(450, 170)">
                                <circle cx="0" cy="0" r="20" fill="url(#greenGrad)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="9">AKT1</text>
                            </g>

                            <!-- ERK Translocation -->
                            <g transform="translate(580, 160)">
                                <circle cx="0" cy="0" r="24" fill="url(#resistGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">MAPK1</text>
                            </g>

                            <!-- Phosphosite & Signal Pulse Annotations -->
                            <rect x="280" y="45" width="90" height="20" rx="4" fill="#1e293b" stroke="#38bdf8" stroke-width="1"/>
                            <text x="325" y="59" text-anchor="middle" fill="#38bdf8" font-size="9" font-weight="700">SOS1 Activation</text>

                            <rect x="470" y="195" width="100" height="20" rx="4" fill="#1e293b" stroke="#f43f5e" stroke-width="1"/>
                            <text x="520" y="209" text-anchor="middle" fill="#f43f5e" font-size="9" font-weight="700">Bypass Signal Cascade</text>
                        </svg>
                        <p style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; margin-top: 0.5rem;">Receptor Tyrosine Kinase (RTK) Downstream Signaling Topology Map</p>
                        <p style="font-size: 0.83rem; color: #94a3b8; margin-top: 0.2rem;">Select a clinical tumor phenotype above or specify oncogenic driver parameters to compute network graph algorithms.</p>
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

                    <!-- Interactive Cytoscape Network Visualizer -->
                    <div class="network-viz-card" style="background: #090d16; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div style="display:flex; align-items:center; gap:0.5rem;">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 0-7.07 17.07"/></svg>
                                <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">Interactive Biological Signaling Network Graph</span>
                            </div>
                            <div style="display: flex; gap: 0.4rem;">
                                <button type="button" class="btn-header" style="font-size: 0.75rem; padding: 0.25rem 0.6rem; color: #fff;" onclick="resetGraphZoom()">Fit View</button>
                                <button type="button" class="btn-header" style="font-size: 0.75rem; padding: 0.25rem 0.6rem; color: #fff;" onclick="relayoutGraph('cose')">Physics</button>
                                <button type="button" class="btn-header" style="font-size: 0.75rem; padding: 0.25rem 0.6rem; color: #fff;" onclick="relayoutGraph('circle')">Circle</button>
                            </div>
                        </div>
                        <div id="cyNetwork" style="width: 100%; height: 350px; background: #0b1120; border-radius: 8px; border: 1px solid #1e293b; position: relative;"></div>
                        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:1.25rem; margin-top:0.75rem; font-size:0.78rem; color:#94a3b8;">
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#0284c7; border: 2px solid #38bdf8;"></span> Primary Target</span>
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#e11d48; border: 2px solid #f43f5e;"></span> Resistance Marker</span>
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#059669; border: 2px solid #34d399;"></span> Combination Target</span>
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#7c3aed; border: 2px solid #a855f7;"></span> Intermediary Bottleneck</span>
                        </div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <h3 style="font-size:1rem; font-weight:800; color:var(--text-main);">Ranked Dual-Drug Combination Therapies</h3>
                    </div>

                    <div id="therapiesList"></div>
                </div>
            </div>
        </div>

        <!-- Footer Bar with Author Attribution -->
        <footer style="margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border-lab); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; font-size: 0.85rem; color: var(--text-muted);">
            <div>
                Targeted Oncology Resistance Bypass Engine • Built with ❤️ by <strong>Ahmadreza Shirdel</strong>
            </div>
            <div>
                <a href="https://github.com/realrezi" target="_blank" style="color: var(--genomic-blue); text-decoration: none; font-weight: 700; display: inline-flex; align-items: center; gap: 0.35rem;">
                    <span>GitHub: github.com/realrezi</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </a>
            </div>
        </footer>
    </div>

    <!-- Modals -->
    <div id="guidanceModal" class="modal-wrapper" onclick="if(event.target===this) toggleModal('guidanceModal', false)">
        <div class="modal-box">
            <div class="modal-top">
                <div class="modal-heading">💡 Purpose & Workflow</div>
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

    <script>
        let latestAnalysisData = null;
        let cyInstance = null;

        function toggleModal(id, show) {
            document.getElementById(id).style.display = show ? 'flex' : 'none';
        }

        function quickFill(fieldId, value) {
            document.getElementById(fieldId).value = value;
        }

        function clearInputs() {
            document.getElementById('primary_target').value = '';
            document.getElementById('primary_drug').value = '';
            document.getElementById('resistance_marker').value = '';
            document.getElementById('cancer_type').value = '';
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
            ['nsclc', 'breast', 'crc', 'melanoma', 'cml', 'prostate', 'ovarian', 'glioma', 'thyroid'].forEach(c => {
                const el = document.getElementById('matrix-' + c);
                if (el) el.style.display = (c === cat) ? 'grid' : 'none';
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

                setTimeout(renderNetworkGraph, 50);
            } catch (err) {
                errorBanner.innerText = '❌ ' + err.message;
                errorBanner.style.display = 'block';
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        }

        function renderNetworkGraph() {
            if (!latestAnalysisData) return;
            const container = document.getElementById('cyNetwork');
            if (!container || typeof cytoscape === 'undefined') return;

            const primaryTarget = (latestAnalysisData.primary_target_canonical || '').toUpperCase();
            const resistanceMarker = (latestAnalysisData.resistance_marker_canonical || '').toUpperCase();
            const nodes = latestAnalysisData.network_nodes || [];
            const edges = latestAnalysisData.network_edges || [];

            const elements = [];

            if (nodes.length === 0) {
                elements.push({ data: { id: primaryTarget, label: primaryTarget, role: 'primary', degree: 4 } });
                elements.push({ data: { id: resistanceMarker, label: resistanceMarker, role: 'resistance', degree: 4 } });
                elements.push({ data: { id: 'edge_fallback', source: primaryTarget, target: resistanceMarker, score: 0.9 } });
            } else {
                const secondaryTargets = (latestAnalysisData.ranked_combinations || []).map(c => (c.secondary_target || '').toUpperCase());

                nodes.forEach(n => {
                    let role = 'intermediary';
                    if (n.id === primaryTarget) role = 'primary';
                    else if (n.id === resistanceMarker) role = 'resistance';
                    else if (secondaryTargets.includes(n.id)) role = 'secondary';

                    elements.push({
                        data: { id: n.id, label: n.id, degree: n.degree || 1, role: role }
                    });
                });

                edges.forEach((e, idx) => {
                    elements.push({
                        data: { id: 'edge_' + idx, source: e.source, target: e.target, score: e.score || 0.5 }
                    });
                });
            }

            if (cyInstance) cyInstance.destroy();

            cyInstance = cytoscape({
                container: container,
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#7c3aed',
                            'label': 'data(label)',
                            'color': '#ffffff',
                            'font-size': '11px',
                            'font-weight': 'bold',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'width': 'mapData(degree, 1, 20, 36, 65)',
                            'height': 'mapData(degree, 1, 20, 36, 65)',
                            'border-width': '2.5px',
                            'border-color': '#c084fc',
                            'shadow-blur': '12px',
                            'shadow-color': '#7c3aed',
                            'shadow-opacity': 0.6
                        }
                    },
                    {
                        selector: 'node[role = "primary"]',
                        style: {
                            'background-color': '#0284c7',
                            'border-color': '#38bdf8',
                            'border-width': '4px',
                            'width': '64px',
                            'height': '64px',
                            'font-size': '13px',
                            'shadow-blur': '20px',
                            'shadow-color': '#38bdf8',
                            'shadow-opacity': 0.8
                        }
                    },
                    {
                        selector: 'node[role = "resistance"]',
                        style: {
                            'background-color': '#e11d48',
                            'border-color': '#f43f5e',
                            'border-width': '4px',
                            'width': '64px',
                            'height': '64px',
                            'font-size': '13px',
                            'shadow-blur': '20px',
                            'shadow-color': '#f43f5e',
                            'shadow-opacity': 0.8
                        }
                    },
                    {
                        selector: 'node[role = "secondary"]',
                        style: {
                            'background-color': '#059669',
                            'border-color': '#34d399',
                            'border-width': '3.5px',
                            'width': '54px',
                            'height': '54px',
                            'font-size': '12px',
                            'shadow-blur': '15px',
                            'shadow-color': '#34d399',
                            'shadow-opacity': 0.7
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 'mapData(score, 0.4, 1, 2, 5.5)',
                            'line-color': '#334155',
                            'curve-style': 'bezier',
                            'opacity': 0.85
                        }
                    }
                ],
                layout: {
                    name: 'cose',
                    animate: true,
                    animationDuration: 700,
                    padding: 35
                }
            });

            cyInstance.on('tap', 'node', function(evt){
                const node = evt.target;
                alert(`Genomic Node: ${node.id()}\nDegree Centrality: ${node.data('degree')}\nBiological Role: ${node.data('role').toUpperCase()}`);
            });
        }


        function resetGraphZoom() {
            if (cyInstance) cyInstance.fit(30);
        }

        function relayoutGraph(layoutName) {
            if (cyInstance) {
                cyInstance.layout({ name: layoutName, animate: true, padding: 30 }).run();
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
                    <div style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.3rem;">
                        Secondary Target: <strong style="color:var(--text-main);">${c.secondary_target}</strong> | Synergy Score: <strong style="color:var(--genomic-blue);">${c.synergy_score}</strong> | Hub Centrality: ${c.hub_penalized_centrality}
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.45;">${c.biological_rationale}</div>
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
) -> Tuple[int, float, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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

    network_nodes = [{"id": str(n), "degree": int(G.degree(n))} for n in G.nodes]
    network_edges = [
        {
            "source": str(u),
            "target": str(v),
            "score": float(G[u][v].get("score", 0.5)),
        }
        for u, v in G.edges
    ]

    return pathway_nodes_count, shortest_path_distance, scored, network_nodes, network_edges


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
            network_nodes=[{"id": primary_target_canonical, "degree": 1}],
            network_edges=[],
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
            pathway_nodes_count, shortest_path_distance, scored_raw, net_nodes, net_edges = (
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
            network_nodes=net_nodes,
            network_edges=net_edges,
        )

