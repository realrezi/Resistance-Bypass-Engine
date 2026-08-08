import asyncio
import base64
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from src.clients.base import cache, close_connection_pools
from src.clients.chembl import ChEMBLClient
from src.clients.open_targets import OpenTargetsClient
from src.clients.string_db import StringDBClient
from src.engine.graph_builder import build_signaling_graph
from src.engine.scorer import PathwayScorer
from src.schemas.evidence import (
    ClaimType,
    EvidenceLevel,
    EvidenceSource,
    ScientificClaim,
)
from src.schemas.models import (
    CombinationCandidate,
    ReportMetadata,
    ResistanceBypassReport,
    ResistanceRequest,
    request_fingerprint,
)
from src.services.gene_annotation import get_gene_annotation
from src.services.id_mapper import IDMapper
from src.static.lab_b64 import LAB_MUTATION_B64

logger = logging.getLogger(__name__)
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
RATE_LIMIT_MAX_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60")))
MAX_REQUEST_BODY_BYTES = max(
    16_384, int(os.getenv("MAX_REQUEST_BODY_BYTES", str(256 * 1024)))
)
_request_windows: defaultdict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    yield
    await close_connection_pools()


app = FastAPI(
    title="Targeted Oncology Resistance Bypass Engine",
    description="Microservice modeling acquired drug resistance pathways in cancer",
    version="0.1.0",
    lifespan=app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://127.0.0.1:8765,http://localhost:8765"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_request_limits(request, call_next):
    """Bound expensive public requests without requiring an auth system."""
    if request.method == "POST" and request.url.path == "/api/v1/analyze-resistance":
        content_length = request.headers.get("content-length")
        try:
            body_size = int(content_length) if content_length else 0
        except ValueError:
            body_size = MAX_REQUEST_BODY_BYTES + 1
        if body_size > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds the configured limit."},
            )

        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _request_windows[client_key]
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(window[0] + RATE_LIMIT_WINDOW_SECONDS - now))
            return JSONResponse(
                status_code=429,
                content={"detail": "Analysis request rate limit exceeded."},
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)

        if len(_request_windows) > 2048:
            for key in list(_request_windows)[:512]:
                if not _request_windows[key]:
                    del _request_windows[key]
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Request-ID"] = request.headers.get(
        "x-request-id", uuid4().hex[:16]
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


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
    if LAB_MUTATION_B64:
        return Response(
            content=base64.b64decode(LAB_MUTATION_B64), media_type="image/png"
        )
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Avoid noisy 404s from browsers that request a favicon automatically."""
    return Response(status_code=204)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Targeted Oncology Resistance Bypass Engine | Clinical Genomic Laboratory</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script defer src="https://3dmol.org/build/3Dmol-min.js"></script>

    <style>
        :root {
            --bg-lab: #070a0f;
            --card-bg: #0d131a;
            --panel-bg: #121b24;
            --border-lab: #1c2a35;
            --border-subtle: #2a3b47;
            --genomic-blue: #67d8ff;
            --genomic-blue-hover: #22b8e6;
            --mutation-red: #ff6b6b;
            --approved-green: #55d6a3;
            --purple-pathway: #8fa9b8;
            --amber-phase: #f5bd62;
            --text-main: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --font-main: 'Space Grotesk', sans-serif;
            --font-mono: 'IBM Plex Mono', monospace;
            --shadow-lab: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 4px 12px -2px rgba(0, 0, 0, 0.3);
            --shadow-hover: 0 15px 35px -5px rgba(56, 189, 248, 0.2), 0 8px 15px -4px rgba(0, 0, 0, 0.4);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-lab);
            background-image: 
                radial-gradient(circle at 8% 12%, rgba(103, 216, 255, 0.09) 0%, transparent 32%),
                radial-gradient(circle at 92% 20%, rgba(255, 107, 107, 0.06) 0%, transparent 28%),
                linear-gradient(to right, rgba(42, 59, 71, 0.28) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(42, 59, 71, 0.28) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
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

        .genome-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(340px, 0.92fr);
            gap: 1rem;
            margin-bottom: 1.25rem;
            overflow: hidden;
            border: 1px solid var(--border-lab);
            border-radius: 12px;
            background: linear-gradient(118deg, #0b1218 0%, #0d1720 56%, #10161b 100%);
            box-shadow: var(--shadow-lab);
        }

        .genome-hero-copy { padding: clamp(1.5rem, 4vw, 3.5rem); }
        .hero-kicker {
            color: var(--genomic-blue);
            font-family: var(--font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }
        .genome-hero h1 {
            max-width: 720px;
            color: var(--text-main);
            font-size: clamp(2.05rem, 5vw, 4.4rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
            margin-bottom: 1rem;
        }
        .genome-hero h1 em { color: var(--mutation-red); font-style: normal; }
        .genome-hero-copy p { max-width: 58ch; color: var(--text-secondary); font-size: 0.98rem; }
        .hero-statline {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.35rem;
        }
        .hero-stat {
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 0.69rem;
            background: rgba(255,255,255,0.025);
        }
        .hero-stat strong { color: var(--genomic-blue); font-weight: 600; }
        .interpretation-strip { padding: 1rem 1.35rem; }
        .interpretation-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .interpretation-grid > div {
            border-left: 2px solid var(--genomic-blue);
            padding: 0.25rem 0.75rem;
        }
        .interpretation-grid strong { display: block; color: var(--text-main); font-size: 0.82rem; }
        .interpretation-grid span { display: block; color: var(--text-muted); font-size: 0.76rem; margin-top: 0.2rem; }
        .mutation-plate {
            position: relative;
            min-height: 300px;
            border-left: 1px solid var(--border-lab);
            background:
                linear-gradient(90deg, transparent 49.8%, rgba(103,216,255,0.12) 50%, transparent 50.2%),
                repeating-linear-gradient(0deg, transparent 0 28px, rgba(103,216,255,0.08) 29px 30px),
                #091016;
            overflow: hidden;
        }
        .mutation-plate::before, .mutation-plate::after {
            content: '';
            position: absolute;
            width: 120%;
            height: 44%;
            left: -10%;
            border-top: 2px solid rgba(103,216,255,0.8);
            border-bottom: 2px solid rgba(255,107,107,0.7);
            border-radius: 50%;
            transform: rotate(-14deg);
            box-shadow: 0 0 24px rgba(103,216,255,0.12);
        }
        .mutation-plate::before { top: 12%; }
        .mutation-plate::after { top: 43%; transform: rotate(14deg); }
        .mutation-readout {
            position: absolute;
            left: 1.35rem;
            right: 1.35rem;
            bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .mutation-readout strong { color: var(--mutation-red); }
        @media (max-width: 860px) {
            .genome-hero { grid-template-columns: 1fr; }
            .mutation-plate { min-height: 220px; border-left: 0; border-top: 1px solid var(--border-lab); }
            .interpretation-grid { grid-template-columns: 1fr; }
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
        .form-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
        @media (max-width: 720px) {
            .form-row { grid-template-columns: 1fr; }
            #structureWorkspace > div:last-child { grid-template-columns: 1fr !important; }
            #structureWorkspaceViewer { height: 300px !important; }
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

        input.input-field, select.input-field {
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

        input.input-field:focus, select.input-field:focus {
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
        .evidence-badges { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.55rem 0; }
        .evidence-badge {
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            padding: 0.2rem 0.48rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 0.64rem;
        }
        .evidence-badge.positive { color: var(--approved-green); border-color: rgba(85,214,163,0.4); }
        .evidence-badge.caution { color: var(--amber-phase); border-color: rgba(245,189,98,0.4); }
        .stage-rail { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 1rem; }
        .stage-step { display:inline-flex; align-items:center; gap:.35rem; border:1px solid var(--border-lab); border-radius:999px; padding:.3rem .55rem; color:var(--text-muted); font-family:var(--font-mono); font-size:.62rem; letter-spacing:.02em; }
        .stage-step.done { color:var(--approved-green); border-color:rgba(85,214,163,.42); background:rgba(85,214,163,.06); }
        .stage-step.partial { color:var(--amber-phase); border-color:rgba(245,189,98,.42); background:rgba(245,189,98,.06); }
        .home-workflow { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.65rem; margin-top:.8rem; text-align:left; }
        .home-workflow-step { min-height:112px; padding:.8rem; border:1px solid var(--border-lab); border-radius:10px; background:linear-gradient(145deg,rgba(15,23,42,.9),rgba(8,15,24,.96)); }
        .home-workflow-step strong { display:block; color:var(--text-main); font-size:.78rem; margin:.35rem 0 .25rem; }
        .home-workflow-step span { color:var(--text-muted); font-size:.72rem; line-height:1.4; }
        @media (max-width: 720px) { .home-workflow { grid-template-columns:1fr; } }
        .candidate-explain { border-top: 1px solid var(--border-lab); margin-top: 0.7rem; padding-top: 0.55rem; }
        .candidate-explain summary { cursor: pointer; color: var(--genomic-blue); font-family: var(--font-mono); font-size: 0.7rem; }
        .candidate-explain p, .candidate-explain li { color: var(--text-muted); font-size: 0.75rem; line-height: 1.45; }
        .candidate-explain ul { padding-left: 1.1rem; margin-top: 0.35rem; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
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

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
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

        <section class="genome-hero" aria-labelledby="hero-title">
            <div class="genome-hero-copy">
                <div class="hero-kicker">Clinical genomics / resistance atlas</div>
                <h1 id="hero-title">Trace the mutation.<br><em>Map the escape route.</em></h1>
                <p>Connect a treatment, alteration, and bypass marker to the evidence behind the resistance hypothesis—then inspect every source before you act on it.</p>
                <div class="hero-statline" aria-label="System capabilities">
                    <span class="hero-stat" title="HGNC, UniProt, STRING, Open Targets, and ChEMBL"><strong>05</strong> live source APIs</span>
                    <span class="hero-stat" title="On-target alteration and off-target bypass"><strong>02</strong> resistance modes</span>
                    <span class="hero-stat" title="A non-identifying hash of this request’s normalized inputs"><strong>01</strong> report fingerprint</span>
                </div>
            </div>
            <div class="mutation-plate" aria-label="Abstract DNA mutation visualization">
                <div class="mutation-readout"><span>sequence / <strong>acquired resistance</strong></span><span>EGFR → MET</span></div>
            </div>
        </section>

        <section class="academic-matrix-section interpretation-strip" aria-label="How to read the analysis">
            <div class="matrix-title-text">How to read the report</div>
            <div class="interpretation-grid">
                <div><strong>Heuristic Priority</strong><span>Ranks candidates for expert review. It is not measured drug synergy or a treatment recommendation.</span></div>
                <div><strong>Hub Centrality</strong><span>Belongs to the target node’s network position, so drugs sharing a target can share this component.</span></div>
                <div><strong>Evidence status</strong><span>Separates target, disease, pharmacology, and explicit pair-level evidence. Missing evidence stays visible.</span></div>
            </div>
        </section>

        <!-- Clinical resistance scenarios Section (Academic & Professional) -->
        <section class="academic-matrix-section">
            <div class="matrix-top-bar">
                <div class="matrix-title-text">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    <span>Clinical Resistance Scenarios & Bypass Pathways</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">
                    Explore characterized resistance hypotheses across 9 oncology indications. Each preset fills the alteration, treatment-line, and disease context so the live analysis can retrieve evidence that is relevant to the scenario:
                </div>
            </div>

            <div class="matrix-tabs-container">
                <button class="btn-matrix-tab active" onclick="switchMatrixCategory('nsclc', this)">🫁 NSCLC Loci</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('breast', this)">🎗️ Breast Carcinoma</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('crc', this)">🧬 Colorectal Loci</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('melanoma', this)">☀️ Melanoma Axis</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('cml', this)">🩸 Hematologic Myeloma</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('prostate', this)">🩺 Prostate Axis</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('ovarian', this)">🎗️ Ovarian Signatures</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('glioma', this)">🧠 Glioma & CNS</button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('thyroid', this)">🫀 Thyroid & Rare Fusions</button>

            </div>

            <!-- Tab 1: NSCLC Scenarios -->
            <div id="matrix-nsclc" class="prevalence-cards-grid">
                <div class="prevalence-card high-prev" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'Non-Small Cell Lung Cancer', 'L858R', 'MET amplification', 'amplification', 'after progression on osimertinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + MET Amplification</span>
                        <span class="badge-prevalence">MET bypass • verify tissue/assay context</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFR) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-Target RTK Bypass. MET amplification reactivates ERBB3/PI3K signaling despite 3rd-gen Osimertinib blockade.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'Non-Small Cell Lung Cancer', 'L858R', 'C797S', 'mutation', 'after progression on osimertinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + C797S Secondary Mutation</span>
                        <span class="badge-prevalence high">On-target binding change • verify genotype</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (Exon 20 C797S)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> On-Target ATP Pocket Mutation. C797S mutation disrupts covalent binding of Osimertinib.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ALK', 'Alectinib', 'MET', 'Non-Small Cell Lung Cancer', 'EML4-ALK', 'MET amplification', 'amplification', 'after progression on alectinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ALK + MET Bypass</span>
                        <span class="badge-prevalence high">Parallel RTK bypass • compare ALK line</span>
                    </div>
                    <div class="locus-tag">Chr 2p23.2 (ALK) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Parallel RTK activation bypassing 2nd-gen ALK inhibitor (Alectinib) blockade in ALK+ NSCLC.</div>
                </div>
            </div>

            <!-- Tab 2: Breast Cancer Scenarios -->
            <div id="matrix-breast" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'HER2+ Breast Cancer', 'ERBB2 amplification', 'MET amplification', 'amplification', 'after progression on trastuzumab')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">HER2 + MET Amplification</span>
                        <span class="badge-prevalence">RTK bypass hypothesis • verify HER2 status</span>
                    </div>
                    <div class="locus-tag">Chr 17q12 (ERBB2) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Off-target RTK bypass hyperactivation overriding anti-HER2 monoclonal antibody (Trastuzumab) therapy.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'HR+/HER2- Breast Cancer', 'ESR1 mutation', 'CDK4/6 pathway activation', 'activation', 'after endocrine therapy progression')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ESR1 + CDK4/6 Cyclin Axis</span>
                        <span class="badge-prevalence high">Endocrine escape • specify prior therapy</span>
                    </div>
                    <div class="locus-tag">Chr 6q25.1 (ESR1) ➔ Chr 12q14.1 (CDK4)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Endocrine therapy escape post-Aromatase Inhibitor (AI) failure driven by ligand-independent ESR1 mutations & Cyclin D1/CDK4 pathway reactivation.</div>
                </div>

            </div>

            <!-- Tab 3: Colorectal Cancer Scenarios -->
            <div id="matrix-crc" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer', 'G12C', 'EGFR feedback activation', 'activation', 'after progression on KRAS G12C inhibition')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">KRAS G12C + EGFR Feedback</span>
                        <span class="badge-prevalence">EGFR feedback loop • assess prior KRAS therapy</span>
                    </div>
                    <div class="locus-tag">Chr 12p12.1 (KRAS) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Rapid RTK feedback loop reactivating MAPK signaling; provides a rationale for evaluating KRAS G12C plus EGFR blockade.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('BRAF', 'Encorafenib', 'EGFR', 'Colorectal Cancer', 'V600E', 'EGFR feedback activation', 'activation', 'specified metastatic BRAF V600E CRC indication')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600E + EGFR Feedback</span>
                        <span class="badge-prevalence high">EGFR feedback loop • approved-pair context</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> BRAF inhibition can induce EGFR feedback; Encorafenib plus cetuximab has FDA authorization in specified BRAF V600E metastatic colorectal-cancer indications.</div>
                </div>
            </div>

            <!-- Tab 4: Cutaneous Melanoma Scenarios -->
            <div id="matrix-melanoma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card approved-prev" onclick="setPreset('BRAF', 'Dabrafenib', 'MAP2K1', 'Cutaneous Melanoma', 'V600E', 'MAP2K1 resistance alteration', 'mutation', 'after progression on BRAF/MEK-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600 + MAP2K1/MEK1</span>
                        <span class="badge-prevalence">MAPK reactivation • confirm acquired alteration</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 15q22.31 (MAP2K1)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> MAPK reactivation is a documented resistance biology; dabrafenib plus trametinib is authorized only for specified BRAF V600 melanoma, NSCLC, thyroid, and other labeled indications—not as a universal resistance rescue.</div>
                </div>
            </div>

            <!-- Tab 5: Hematologic Scenarios -->
            <div id="matrix-cml" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('ABL1', 'Imatinib', 'ABL1', 'Chronic Myeloid Leukemia', 'BCR-ABL1', 'T315I', 'mutation', 'after progression on first/second-generation TKI')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BCR-ABL + T315I Gatekeeper</span>
                        <span class="badge-prevalence">Gatekeeper mutation • specify prior TKI generations</span>
                    </div>
                    <div class="locus-tag">Chr 9q34.12 (ABL1 T315I Gatekeeper)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Threonine-to-isoleucine substitution alters the kinase binding site; ponatinib or asciminib are approved options in specified CML/Ph+ ALL settings, while combination use requires its own evidence.</div>
                </div>
            </div>

            <!-- Tab 6: Prostate Cancer Scenarios -->
            <div id="matrix-prostate" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('AR', 'Enzalutamide', 'PIK3CA', 'Metastatic Castration-Resistant Prostate Cancer', 'AR alteration', 'PIK3CA/PTEN pathway alteration', 'activation', 'after progression on androgen-receptor pathway inhibition')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">AR + PIK3CA Crosstalk</span>
                        <span class="badge-prevalence">AR–PI3K crosstalk • confirm PTEN/PI3K assay</span>
                    </div>
                    <div class="locus-tag">Chr Xq12 (AR) ➔ Chr 3q26.32 (PIK3CA)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Reciprocal feedback crosstalk between Androgen Receptor and PI3K/AKT signaling pathways in mCRPC.</div>
                </div>
            </div>

            <!-- Tab 7: Ovarian & GYN Scenarios -->
            <div id="matrix-ovarian" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('PIK3CA', 'Alpelisib', 'KRAS', 'Ovarian Cancer', 'PIK3CA alteration', 'KRAS alteration', 'mutation', 'after progression on PI3K-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">PIK3CA + KRAS Bypass</span>
                        <span class="badge-prevalence">RAS/MAPK co-alteration • confirm tumor subtype</span>
                    </div>
                    <div class="locus-tag">Chr 3q26.32 (PIK3CA) ➔ Chr 12p12.1 (KRAS)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Parallel activation of RAS/MAPK axis circumventing selective PI3Kalpha inhibitor blockade in gynecologic malignancies.</div>
                </div>
            </div>

            <!-- Tab 8: Glioma & CNS Scenarios -->
            <div id="matrix-glioma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('EGFR', 'Gefitinib', 'MET', 'Glioblastoma', 'EGFRvIII', 'MET amplification', 'amplification', 'after progression on EGFR-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFRvIII + MET Amplification</span>
                        <span class="badge-prevalence">RTK redundancy • specify EGFR assay and line</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFRvIII) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Co-activation of multiple RTKs (EGFRvIII and MET) driving redundant oncogenic signaling in high-grade glioma.</div>
                </div>
            </div>

            <!-- Tab 9: Thyroid & Rare Fusions Scenarios -->
            <div id="matrix-thyroid" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('RET', 'Selpercatinib', 'MET', 'Thyroid Cancer', 'RET fusion', 'MET amplification', 'amplification', 'after progression on selpercatinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">RET Fusion + MET Bypass</span>
                        <span class="badge-prevalence">MET bypass hypothesis • confirm RET progression</span>
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

                    <div class="form-row">
                        <div class="form-group">
                            <label class="field-label" for="primary_alteration">Primary Alteration <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="primary_alteration" placeholder="e.g. L858R, G12C, fusion" autocomplete="off">
                        </div>
                        <div class="form-group">
                            <label class="field-label" for="resistance_alteration">Resistance Alteration <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="resistance_alteration" placeholder="e.g. amplification, C797S" autocomplete="off">
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="field-label" for="resistance_alteration_type">Alteration Type <span style="color:#64748b;">(optional)</span></label>
                            <select class="input-field" id="resistance_alteration_type">
                                <option value="">Not specified</option>
                                <option value="mutation">Mutation</option>
                                <option value="amplification">Amplification</option>
                                <option value="deletion">Deletion</option>
                                <option value="fusion">Fusion</option>
                                <option value="overexpression">Overexpression</option>
                                <option value="splice_variant">Splice variant</option>
                                <option value="activation">Pathway activation</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="field-label" for="treatment_line">Treatment Context <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="treatment_line" placeholder="e.g. after progression on osimertinib" autocomplete="off">
                        </div>
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
                </form>
            </div>


            <!-- Right Panel: Signal Transduction Pathway & Evidence Priority Matrix -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span>Signal Transduction Pathway Topology & Evidence Priority Matrix</span>
                    </div>
                    <div style="display:flex; gap:0.45rem; flex-wrap:wrap; justify-content:flex-end;">
                        <button id="compareBtn" class="btn-header" style="display: none; color: #fff;" onclick="compareReports()">Compare last two</button>
                        <button id="copyJsonBtn" class="btn-header" style="display: none; color: #fff;" onclick="copyResultJson()">Copy report JSON</button>
                    </div>
                </div>

                <div id="errorBanner" style="display:none; background:rgba(244, 63, 94, 0.15); border:1px solid rgba(244, 63, 94, 0.3); padding:0.85rem; border-radius:8px; color:#fb7185; margin-bottom:1rem; font-size:0.85rem;"></div>
                <div id="warningBanner" style="display:none; background:rgba(251, 191, 36, 0.12); border:1px solid rgba(251, 191, 36, 0.35); padding:0.85rem; border-radius:8px; color:#fde68a; margin-bottom:1rem; font-size:0.85rem;"></div>

                <div id="loader" style="display: none; text-align: center; padding: 3rem 1rem;">
                    <div style="width: 44px; height: 44px; border: 3px solid #1e293b; border-radius: 50%; border-top-color: #38bdf8; animation: spin 0.8s linear infinite; margin: 0 auto 1rem auto;"></div>
                    <p style="font-weight: 800; color: #f8fafc;">Querying REST/GraphQL PPI Biological Databases...</p>
                    <p style="font-size: 0.82rem; color: #94a3b8;">Resolving Canonical HGNC IDs • Extracting NetworkX LCC Topology • Querying ChEMBL & Open Targets</p>
                </div>

                <!-- Multi-Kinase Cell Membrane SVG Signaling Visualizer (Professional & Dynamic) -->
                <div id="placeholder">
                    <div class="vector-graph-canvas" style="background: #030712; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; color: #fff; text-align: center; position: relative;">
                        <div class="home-workflow" aria-label="Analysis workflow">
                            <div class="home-workflow-step"><span style="color:var(--genomic-blue);font-family:var(--font-mono);">01 / RESOLVE</span><strong>Canonical identifiers</strong><span>Map gene aliases to HGNC, UniProt, Ensembl, and ChEMBL records.</span></div>
                            <div class="home-workflow-step"><span style="color:var(--mutation-red);font-family:var(--font-mono);">02 / CONNECT</span><strong>Evidence network</strong><span>Retrieve PPI, clinical, disease, and pharmacology context from live sources.</span></div>
                            <div class="home-workflow-step"><span style="color:var(--approved-green);font-family:var(--font-mono);">03 / QUALIFY</span><strong>Rank with abstention</strong><span>Separate computational priority from clinical evidence and expose missing data.</span></div>
                        </div>
                        <svg style="display:none" width="100%" height="240" viewBox="0 0 700 240" fill="none" xmlns="http://www.w3.org/2000/svg">
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
                            <g transform="translate(120, 40)" style="cursor: pointer;" onclick="openNodeInspector('EGFR')">
                                <rect x="-18" y="-20" width="36" height="40" rx="6" fill="url(#primaryGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="11">EGFR</text>
                                <circle cx="0" cy="30" r="14" fill="#0284c7" stroke="#38bdf8" stroke-width="2"/>
                                <text x="0" y="34" text-anchor="middle" fill="#fff" font-size="9" font-weight="800">p-Y</text>
                            </g>
                            <text x="120" y="10" text-anchor="middle" fill="#38bdf8" font-size="10" font-weight="700">Chr 7p11.2 (Primary Driver)</text>

                            <!-- Resistance Marker RTK Domain (MET) -->
                            <g transform="translate(520, 40)" style="cursor: pointer;" onclick="openNodeInspector('MET')">
                                <rect x="-18" y="-20" width="36" height="40" rx="6" fill="url(#resistGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="11">MET</text>
                                <circle cx="0" cy="30" r="14" fill="#e11d48" stroke="#f43f5e" stroke-width="2"/>
                                <text x="0" y="34" text-anchor="middle" fill="#fff" font-size="9" font-weight="800">p-Y</text>
                            </g>
                            <text x="520" y="10" text-anchor="middle" fill="#f43f5e" font-size="10" font-weight="700">Chr 7q31.2 (Bypass Locus)</text>

                            <!-- Downstream Kinase Cascades -->
                            <!-- GRB2/SOS1 Adaptor -->
                            <g transform="translate(250, 80)" style="cursor: pointer;" onclick="openNodeInspector('GRB2')">
                                <circle cx="0" cy="0" r="22" fill="url(#purpleGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">GRB2</text>
                            </g>

                            <!-- KRAS GTPase -->
                            <g transform="translate(400, 80)" style="cursor: pointer;" onclick="openNodeInspector('KRAS')">
                                <rect x="-24" y="-18" width="48" height="36" rx="10" fill="url(#purpleGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">KRAS</text>
                            </g>

                            <!-- PI3K/AKT Pathway -->
                            <g transform="translate(320, 170)" style="cursor: pointer;" onclick="openNodeInspector('PIK3CA')">
                                <circle cx="0" cy="0" r="20" fill="url(#greenGrad)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="9">PIK3CA</text>
                            </g>

                            <g transform="translate(450, 170)" style="cursor: pointer;" onclick="openNodeInspector('AKT1')">
                                <circle cx="0" cy="0" r="20" fill="url(#greenGrad)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="9">AKT1</text>
                            </g>

                            <!-- ERK Translocation -->
                            <g transform="translate(580, 160)" style="cursor: pointer;" onclick="openNodeInspector('MAPK1')">
                                <circle cx="0" cy="0" r="24" fill="url(#resistGrad)" filter="url(#glow)"/>
                                <text x="0" y="4" text-anchor="middle" fill="#fff" font-weight="800" font-size="10">MAPK1</text>
                            </g>


                            <!-- Phosphosite & Signal Pulse Annotations -->
                            <rect x="280" y="45" width="90" height="20" rx="4" fill="#1e293b" stroke="#38bdf8" stroke-width="1"/>
                            <text x="325" y="59" text-anchor="middle" fill="#38bdf8" font-size="9" font-weight="700">SOS1 Activation</text>

                            <rect x="470" y="195" width="100" height="20" rx="4" fill="#1e293b" stroke="#f43f5e" stroke-width="1"/>
                            <text x="520" y="209" text-anchor="middle" fill="#f43f5e" font-size="9" font-weight="700">Bypass Signal Cascade</text>
                        </svg>
                        <p style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; margin-top: 0.5rem;">Evidence-first analysis workflow</p>
                        <p style="font-size: 0.83rem; color: #94a3b8; margin-top: 0.2rem;">The target-specific network graph and molecular structure workspace appear after you run an analysis.</p>
                    </div>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div id="comparisonPanel" class="candidate-card" style="display:none; margin-bottom:1rem;"></div>
                    <div class="canonical-bar">
                        <span class="pill-badge" id="primaryTag">Primary Target: -</span>
                        <span class="pill-badge" id="resistanceTag">Resistance Marker: -</span>
                        <span class="pill-badge" id="timingTag" style="color:var(--text-muted);">Source timing: —</span>
                    </div>
                    <div id="analysisStageBar" class="stage-rail" aria-label="Analysis stages">
                        <span class="stage-step done">01 · IDs resolved</span>
                        <span class="stage-step done">02 · Network built</span>
                        <span class="stage-step done">03 · Evidence merged</span>
                        <span class="stage-step done">04 · Candidates scored</span>
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

                    <div id="structureWorkspace" class="network-viz-card" style="background: linear-gradient(135deg, #0b1218, #0b111b); border: 1px solid #1c3b4b; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; margin-bottom:.75rem;">
                            <div>
                                <div style="display:flex; align-items:center; gap:.5rem;">
                                    <span style="color:var(--genomic-blue); font-size:1.1rem;">⌬</span>
                                    <span style="font-weight:800; font-size:.95rem; color:var(--text-main);">Molecular structure workspace</span>
                                </div>
                                <p style="font-size:.78rem; color:var(--text-muted); margin-top:.25rem;">Inspect a curated experimental structure for the selected network node. A missing structure stays visible as missing evidence.</p>
                            </div>
                            <div id="structureStatusBadge" class="evidence-badge caution">Select a node</div>
                        </div>
                        <div style="display:grid; grid-template-columns:minmax(0, 1fr) 220px; gap:1rem; align-items:stretch;">
                            <div id="structureWorkspaceViewer" style="height:360px; min-height:280px; width:100%; position:relative; background:#070a0f; border:1px solid #1c2a35; border-radius:9px; overflow:hidden; display:flex; align-items:center; justify-content:center; color:var(--text-muted); font-size:.82rem;">Run an analysis to load a curated structure mapping.</div>
                            <div id="structureWorkspaceMeta" style="border:1px solid #1c2a35; border-radius:9px; padding:.85rem; background:#0d131a; font-size:.77rem; color:var(--text-secondary);">
                                <div style="font-family:var(--font-mono); color:var(--genomic-blue); font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;">Structure evidence</div>
                                <div id="structureWorkspaceTarget" style="font-weight:800; color:var(--text-main); font-size:1.2rem; margin:.55rem 0 .25rem;">—</div>
                                <div id="structureWorkspacePdb" style="font-family:var(--font-mono); color:var(--approved-green);">PDB —</div>
                                <p id="structureWorkspaceNote" style="margin-top:.7rem; line-height:1.45;">This panel distinguishes verified structures from unavailable mappings.</p>
                                <button type="button" class="btn-header" style="margin-top:.8rem; width:100%; justify-content:center; color:var(--genomic-blue);" onclick="openSelectedNodeInspector()">Open full inspector</button>
                            </div>
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
    <div id="guidanceModal" class="modal-wrapper" role="dialog" aria-modal="true" aria-hidden="true" aria-labelledby="guidanceHeading" onclick="if(event.target===this) toggleModal('guidanceModal', false)">
        <div class="modal-box" style="max-width: 600px; background: #0f172a; border: 1px solid #334155; border-radius: 14px; color: #f8fafc;">
            <div class="modal-top" style="border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; margin-bottom: 1rem;">
                <div id="guidanceHeading" class="modal-heading" style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">💡 Purpose & Workflow</div>
                <button class="modal-close" style="color: #94a3b8;" onclick="toggleModal('guidanceModal', false)">&times;</button>
            </div>
            <p style="margin-bottom: 1rem; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                This clinical engine models acquired therapeutic drug resistance in cancer using real-time REST/GraphQL biological APIs (HGNC, UniProt, STRING-DB, Open Targets, ChEMBL v4) and pure Python NetworkX graph algorithms.
            </p>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px; margin-bottom: 0.75rem;">
                <div style="font-weight: 700; color: #38bdf8;">1. Resolve Canonical Identifiers</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Resolves alias symbols (e.g. HER2 ➔ ERBB2) to official HGNC IDs and UniProt Accession codes.</div>
            </div>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px; margin-bottom: 0.75rem;">
                <div style="font-weight: 700; color: #c084fc;">2. Build Biological PPI Signaling Graph</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Queries STRING-DB for protein-protein interaction networks and extracts the Largest Connected Component (LCC).</div>
            </div>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px;">
                <div style="font-weight: 700; color: #34d399;">3. Hub-Penalized Centrality & Therapy Ranking</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Computes <code>Betweenness / log2(Degree + 2)</code> to isolate non-generic bottleneck targets and ranks active clinical combinations.</div>
            </div>
        </div>
    </div>

    <div id="clinicianModal" class="modal-wrapper" role="dialog" aria-modal="true" aria-hidden="true" aria-labelledby="clinicianHeading" onclick="if(event.target===this) toggleModal('clinicianModal', false)">
        <div class="modal-box" style="max-width: 600px; background: #0f172a; border: 1px solid #334155; border-radius: 14px; color: #f8fafc;">
            <div class="modal-top" style="border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; margin-bottom: 1rem;">
                <div id="clinicianHeading" class="modal-heading" style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">📖 Methodological & Clinical Guide</div>
                <button class="modal-close" style="color: #94a3b8;" onclick="toggleModal('clinicianModal', false)">&times;</button>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: #cbd5e1;">
                <p style="margin-bottom: 0.75rem;"><strong style="color: #38bdf8;">Off-Target Bypass:</strong> Hyperactivation of a parallel signaling pathway (e.g., MET amplification) that bypasses frontline drug blockade.</p>
                <p style="margin-bottom: 0.75rem;"><strong style="color: #f43f5e;">On-Target Mutation:</strong> Secondary mutations directly inside the primary target gene (e.g., EGFR C797S or ABL1 T315I) altering drug binding affinity.</p>
                <p><strong style="color: #34d399;">Hub-Penalized Bottleneck Centrality:</strong> Evaluated as <code>Betweenness / log2(Degree + 2)</code> to strip non-specific hub proteins (like TP53 or Ubiquitin) while pinpointing critical resistance signaling nodes.</p>
            </div>
        </div>
    </div>


    <!-- Deep Multi-Omics Node Inspector Modal -->
    <div id="nodeModal" class="modal-wrapper" role="dialog" aria-modal="true" aria-hidden="true" aria-labelledby="nodeModalTitle" onclick="if(event.target===this) toggleModal('nodeModal', false)">
        <div class="modal-box" style="max-width: 680px; background: #0f172a; border: 1px solid #334155; border-radius: 14px; color: #f8fafc;">
            <div class="modal-top" style="border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; margin-bottom: 1rem;">
                <div>
                    <div id="nodeModalTitle" style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">EGFR</div>
                    <div id="nodeModalFullName" style="font-size: 0.88rem; color: #38bdf8; font-weight: 600;">Epidermal Growth Factor Receptor</div>
                </div>
                <button class="modal-close" style="color: #94a3b8;" onclick="toggleModal('nodeModal', false)">&times;</button>
            </div>

            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <span class="pill-badge" id="nodeModalLocus" style="background: rgba(56, 189, 248, 0.15); border-color: rgba(56, 189, 248, 0.35); color: #38bdf8;">Chr 7p11.2</span>
                <span class="pill-badge" id="nodeModalDruggability" style="background: rgba(52, 211, 153, 0.15); border-color: rgba(52, 211, 153, 0.35); color: #34d399;">Clinically targeted; regulatory status is indication-specific</span>
            </div>

            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem; font-size: 0.85rem;">
                <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 0.35rem;">Biological Function & Signal Role</div>
                <div id="nodeModalRole" style="color: #f8fafc; font-weight: 600;">Receptor Tyrosine Kinase (RTK) Initiator</div>
            </div>

            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 0.5rem;">Curated resistance annotations (not live COSMIC frequencies)</div>
                <div id="nodeModalHotspots" style="display: flex; flex-direction: column; gap: 0.4rem;"></div>
            </div>

            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem;">Interactive 3D Macromolecular Structure (PDB: <span id="nodeModalPdbTag" style="color:#34d399;">1M17</span>)</div>
                    <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600;">Rotate 360° • Zoom • Pan</div>
                </div>
                <div id="pdb3dViewer" style="height: 280px; width: 100%; position: relative; background: #090d16; border-radius: 8px; border: 1px solid #334155; overflow: hidden;"></div>
            </div>


            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1.25rem; display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div id="nodeModalEnsembl" style="font-size: 0.85rem; font-family: monospace; font-weight: 700; color: #38bdf8;">ENSG00000146648</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.2rem;">Ensembl ID</div>
                </div>
                <div>
                    <div id="nodeModalUniProt" style="font-size: 0.85rem; font-family: monospace; font-weight: 700; color: #c084fc;">P00533</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.2rem;">UniProt Accession</div>
                </div>
                <div>
                    <div id="nodeModalDegree" style="font-size: 0.85rem; font-family: monospace; font-weight: 700; color: #34d399;">14 Edges</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.2rem;">Degree Centrality</div>
                </div>
            </div>

            <div style="display: flex; gap: 0.6rem; justify-content: flex-end; flex-wrap: wrap;">
                <a id="linkUniProt" href="#" target="_blank" class="btn-header" style="font-size: 0.78rem; color: #38bdf8; text-decoration: none;">🔗 Open UniProt KB</a>
                <a id="linkEnsembl" href="#" target="_blank" class="btn-header" style="font-size: 0.78rem; color: #c084fc; text-decoration: none;">🔗 Open Ensembl Browser</a>
                <a id="linkPDB" href="#" target="_blank" class="btn-header" style="font-size: 0.78rem; color: #34d399; text-decoration: none;">🔬 View PDB 3D Model</a>
            </div>
        </div>
    </div>

    <script>
        let latestAnalysisData = null;
        let analysisHistory = [];
        let cyInstance = null;
        let selectedStructureNode = null;
        let structureViewer = null;

        function toggleModal(id, show) {
            const modal = document.getElementById(id);
            if (!modal) return;
            modal.style.display = show ? 'flex' : 'none';
            modal.setAttribute('aria-hidden', show ? 'false' : 'true');
            if (show) {
                modal.dataset.previousFocus = document.activeElement?.id || '';
                modal.querySelector('.modal-close')?.focus();
            } else if (modal.dataset.previousFocus) {
                document.getElementById(modal.dataset.previousFocus)?.focus();
            }
        }

        document.addEventListener('keydown', event => {
            if (event.key === 'Escape') {
                document.querySelectorAll('.modal-wrapper[aria-hidden="false"]').forEach(modal => toggleModal(modal.id, false));
            }
        });

        document.querySelectorAll('.prevalence-card').forEach(card => {
            card.setAttribute('role', 'button');
            card.setAttribute('tabindex', '0');
            card.addEventListener('keydown', event => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    card.click();
                }
            });
        });

        function quickFill(fieldId, value) {
            document.getElementById(fieldId).value = value;
        }

        function clearInputs() {
            document.getElementById('primary_target').value = '';
            document.getElementById('primary_drug').value = '';
            document.getElementById('resistance_marker').value = '';
            document.getElementById('cancer_type').value = '';
            document.getElementById('primary_alteration').value = '';
            document.getElementById('resistance_alteration').value = '';
            document.getElementById('resistance_alteration_type').value = '';
            document.getElementById('treatment_line').value = '';
        }

        function setPreset(target, drug, marker, indication, primaryAlteration, resistanceAlteration, alterationType, treatmentLine) {
            document.getElementById('primary_target').value = target;
            document.getElementById('primary_drug').value = drug;
            document.getElementById('resistance_marker').value = marker;
            if (indication) document.getElementById('cancer_type').value = indication;
            document.getElementById('primary_alteration').value = primaryAlteration || '';
            document.getElementById('resistance_alteration').value = resistanceAlteration || '';
            document.getElementById('resistance_alteration_type').value = alterationType || '';
            document.getElementById('treatment_line').value = treatmentLine || '';
            executePipeline();
        }

        function switchMatrixCategory(cat, btn) {
            document.querySelectorAll('.btn-matrix-tab').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            else if (window.event && window.event.target) window.event.target.classList.add('active');
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
            const warningBanner = document.getElementById('warningBanner');
            const copyJsonBtn = document.getElementById('copyJsonBtn');
            const compareBtn = document.getElementById('compareBtn');
            const comparisonPanel = document.getElementById('comparisonPanel');
            const timingTag = document.getElementById('timingTag');

            submitBtn.disabled = true;
            errorBanner.style.display = 'none';
            warningBanner.style.display = 'none';
            placeholder.style.display = 'none';
            resultsContent.style.display = 'none';
            copyJsonBtn.style.display = 'none';
            comparisonPanel.style.display = 'none';
            loader.style.display = 'block';

            const payload = {
                primary_target: document.getElementById('primary_target').value.trim(),
                primary_drug: document.getElementById('primary_drug').value.trim(),
                resistance_marker: document.getElementById('resistance_marker').value.trim(),
                cancer_type: document.getElementById('cancer_type').value.trim(),
                primary_alteration: document.getElementById('primary_alteration').value.trim() || null,
                resistance_alteration: document.getElementById('resistance_alteration').value.trim() || null,
                resistance_alteration_type: document.getElementById('resistance_alteration_type').value || null,
                treatment_line: document.getElementById('treatment_line').value.trim() || null
            };

            let requestTimeout;
            try {
                const controller = new AbortController();
                requestTimeout = setTimeout(() => controller.abort(), 60000);
                const response = await fetch('/api/v1/analyze-resistance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: controller.signal
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Analysis pipeline failed');

                latestAnalysisData = data;
                analysisHistory = [...analysisHistory, data].slice(-2);
                compareBtn.style.display = analysisHistory.length > 1 ? 'inline-flex' : 'none';
                const warnings = Array.isArray(data.warnings) ? data.warnings : [];
                if (warnings.length > 0) {
                    warningBanner.innerText = '⚠️ ' + warnings.join(' • ');
                    warningBanner.style.display = 'block';
                    document.querySelectorAll('#analysisStageBar .stage-step').forEach(step => {
                        if (step.textContent.includes('Evidence merged')) step.classList.add('partial');
                    });
                }
                document.getElementById('primaryTag').innerText = 'Primary Target: ' + data.primary_target_canonical;
                document.getElementById('resistanceTag').innerText = 'Resistance Marker: ' + data.resistance_marker_canonical;
                document.getElementById('resTypeVal').innerText = data.resistance_type;
                document.getElementById('nodesCountVal').innerText = data.pathway_nodes_count;
                
                const distNum = typeof data.shortest_path_distance === 'number' 
                    ? data.shortest_path_distance 
                    : Number(data.shortest_path_distance) || 0;
                document.getElementById('distVal').innerText = distNum.toFixed(3);
                const timings = data.metadata && data.metadata.source_timings_ms ? data.metadata.source_timings_ms : {};
                const totalMs = Object.values(timings).reduce((sum, value) => sum + Number(value || 0), 0);
                timingTag.innerText = totalMs ? `Source timing: ${Math.round(totalMs)} ms` : 'Source timing: unavailable';

                renderTherapies();
                resultsContent.style.display = 'block';
                copyJsonBtn.style.display = 'inline-flex';

                setTimeout(() => {
                    renderNetworkGraph();
                    renderStructureWorkspace();
                }, 50);
            } catch (err) {
                errorBanner.innerText = '❌ ' + (err.name === 'AbortError' ? 'The live-source query exceeded 60 seconds. Try again or use a cached request.' : err.message);
                errorBanner.style.display = 'block';
                placeholder.style.display = 'block';
            } finally {
                clearTimeout(requestTimeout);
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
                container.innerHTML = '<div style="padding:2rem;text-align:center;color:#94a3b8;">No validated network topology was returned for this request.</div>';
                return;
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
                openNodeInspector(node.id());
            });

            cyInstance.on('tap', 'edge', function(evt){
                const edge = evt.target;
                const score = edge.data('score') || 0.5;
                alert(`Biological PPI Interaction Edge: ${edge.data('source')} ⟷ ${edge.data('target')}\nSTRING-DB Confidence Score: ${score.toFixed(3)}`);
            });

            cyInstance.on('mouseover', 'node', function(evt){
                const node = evt.target;
                node.style('border-width', '6px');
                node.style('shadow-blur', '25px');
            });

            cyInstance.on('mouseout', 'node', function(evt){
                const node = evt.target;
                const isKey = node.data('role') === 'primary' || node.data('role') === 'resistance';
                node.style('border-width', isKey ? '4px' : '2.5px');
                node.style('shadow-blur', isKey ? '20px' : '12px');
            });
        }


        function openNodeInspector(nodeId) {
            const nodeClean = (nodeId || '').toUpperCase();
            let nodeData = null;
            if (latestAnalysisData && latestAnalysisData.network_nodes) {
                const found = latestAnalysisData.network_nodes.find(n => (n.id || '').toUpperCase() === nodeClean);
                if (found) nodeData = found;
            }

            const ann = (nodeData && nodeData.annotation) ? nodeData.annotation : {
                symbol: nodeClean,
                name: nodeClean + " Human Protein Target",
                locus: "Genomic Locus",
                ensembl_id: null,
                uniprot_id: null,
                pdb_id: null,
                structure_status: "unavailable",
                druggability: "Not annotated in the curated local target panel",
                role: "Unannotated network node",
                hotspots: []
            };

            selectedStructureNode = { id: nodeClean, annotation: ann, degree: nodeData ? nodeData.degree : 0 };

            document.getElementById('nodeModalTitle').innerText = ann.symbol || nodeClean;
            document.getElementById('nodeModalFullName').innerText = ann.name || nodeClean;
            document.getElementById('nodeModalLocus').innerText = ann.locus || 'Locus Tag';
            document.getElementById('nodeModalDruggability').innerText = ann.druggability || 'Targeted Candidate';
            document.getElementById('nodeModalRole').innerText = ann.role || 'Signal Transduction Interactor';
            document.getElementById('nodeModalEnsembl').innerText = ann.ensembl_id || 'Not mapped';
            document.getElementById('nodeModalUniProt').innerText = ann.uniprot_id || 'Not mapped';
            document.getElementById('nodeModalDegree').innerText = (nodeData ? nodeData.degree : 4) + ' Connections';

            const hotspotsContainer = document.getElementById('nodeModalHotspots');
            hotspotsContainer.innerHTML = '';
            (ann.hotspots || []).forEach(h => {
                const item = document.createElement('div');
                item.style.cssText = 'background: #0f172a; border: 1px solid #334155; padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 0.82rem; color: #fb7185; font-weight: 600; display: flex; align-items: center; gap: 0.45rem;';
                item.innerHTML = `<span>⚠️</span> <span>${escapeHtml(h)}</span>`;
                hotspotsContainer.appendChild(item);
            });

            const pdbId = ann.pdb_id;
            document.getElementById('nodeModalPdbTag').innerText = pdbId || 'unavailable';
            const viewerContainer = document.getElementById('pdb3dViewer');
            viewerContainer.innerHTML = '';

            const uniProtUrl = ann.uniprot_id ? 'https://www.uniprot.org/uniprotkb/' + ann.uniprot_id : '#';
            const ensemblUrl = ann.ensembl_id ? 'https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=' + ann.ensembl_id : '#';
            const pdbUrl = pdbId ? 'https://www.rcsb.org/structure/' + pdbId : '#';

            document.getElementById('linkUniProt').href = uniProtUrl;
            document.getElementById('linkEnsembl').href = ensemblUrl;
            document.getElementById('linkPDB').href = pdbUrl;

            toggleModal('nodeModal', true);

            loadStructureViewer(viewerContainer, ann, 'modal');
        }

        function openSelectedNodeInspector() {
            if (selectedStructureNode) openNodeInspector(selectedStructureNode.id);
        }

        function loadStructureViewer(container, ann, mode) {
            container.innerHTML = '';
            const pdbId = ann && ann.pdb_id;
            if (!pdbId) {
                container.innerHTML = '<div style="padding:1.25rem;text-align:center;color:#94a3b8;line-height:1.5;">No curated experimental structure mapping is available for this target. This is an evidence gap, not a predicted structure.</div>';
                return;
            }
            if (!window.$3Dmol) {
                container.innerHTML = '<div style="padding:1.25rem;text-align:center;color:#94a3b8;">3Dmol could not be loaded. Open the RCSB record to inspect the structure.</div>';
                return;
            }
            container.innerHTML = '<div style="padding:1.25rem;text-align:center;color:#94a3b8;">Loading PDB ' + escapeHtml(pdbId) + '…</div>';
            try {
                const viewer = $3Dmol.createViewer(container, { backgroundColor: '0x070a0f' });
                if (mode === 'workspace') structureViewer = viewer;
                $3Dmol.download('pdb:' + pdbId, viewer, { multichannel: true }, function() {
                    viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
                    viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.18, color: 'white' });
                    viewer.zoomTo();
                    viewer.render();
                });
            } catch (error) {
                container.innerHTML = '<div style="padding:1.25rem;text-align:center;color:#ff6b6b;">Structure viewer failed to initialize. The RCSB record remains available.</div>';
            }
        }

        function renderStructureWorkspace() {
            const panel = document.getElementById('structureWorkspace');
            if (!panel || !latestAnalysisData) return;
            const primary = (latestAnalysisData.primary_target_canonical || '').toUpperCase();
            const node = (latestAnalysisData.network_nodes || []).find(item => (item.id || '').toUpperCase() === primary) || (latestAnalysisData.network_nodes || [])[0];
            if (!node) return;
            selectedStructureNode = node;
            const ann = node.annotation || {};
            const hasStructure = Boolean(ann.pdb_id);
            document.getElementById('structureWorkspaceTarget').innerText = ann.symbol || node.id || primary;
            document.getElementById('structureWorkspacePdb').innerText = hasStructure ? 'PDB ' + ann.pdb_id : 'PDB unavailable';
            document.getElementById('structureWorkspaceNote').innerText = hasStructure
                ? 'Curated PDB mapping from the local annotation panel. Confirm chain, construct, ligand, and alteration coverage in the linked RCSB record; this does not prove that the supplied alteration is present in this entry.'
                : 'No verified experimental structure is mapped locally for this node. The engine does not substitute a fabricated or unverified identifier.';
            const badge = document.getElementById('structureStatusBadge');
            badge.innerText = hasStructure ? 'Curated PDB mapping' : 'Structure unavailable';
            badge.className = 'evidence-badge ' + (hasStructure ? 'positive' : 'caution');
            loadStructureViewer(document.getElementById('structureWorkspaceViewer'), ann, 'workspace');
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
                const displayRank = c.rank ?? (idx + 1);
                const tiedRank = c.tie_group && candidates.filter(item => item.tie_group === c.tie_group).length > 1;
                const rankLabel = tiedRank ? `T${displayRank}` : `#${displayRank}`;
                const targetUnavailable = c.target_in_graph === false || c.scoring_status === 'abstained_target_not_in_validated_topology';
                const isHighestReportedStage = c.clinical_phase === 4;
                const phaseLabel = c.clinical_phase == null ? 'Clinical phase not reported' : (isHighestReportedStage ? 'Phase 4 / highest reported stage' : 'Phase ' + c.clinical_phase);
                const components = c.score_components || {};
                const componentText = [
                    ['Topology', components.topology],
                    ['Proximity', components.proximity],
                    ['Pharmacology', components.pharmacology],
                    ['Clinical evidence', components.clinical_evidence]
                ].filter(([, value]) => value !== null && value !== undefined)
                 .map(([label, value]) => `${label}: ${Number(value).toFixed(2)}`).join(' • ');
                const evidenceLinks = (c.evidence || []).map(source => {
                    const url = safeExternalUrl(source.url);
                    const label = escapeHtml(source.name || 'Evidence source');
                    const stableId = source.stable_id ? ` (${escapeHtml(source.stable_id)})` : '';
                    return url
                        ? `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color:#7dd3fc;">${label}${stableId}</a>`
                        : `<span>${label}${stableId}</span>`;
                }).join(' • ');
                const evidenceFlags = [
                    c.indication_match === true ? 'Indication match' : null,
                    c.combination_evidence === true ? 'Pair co-mention found' : 'No pair-level report found',
                    c.clinical_status ? escapeHtml(c.clinical_status) : null
                ].filter(Boolean).join(' • ');
                const evidenceStatusLabels = {
                    abstained: ['Abstained: topology missing', 'caution'],
                    pair_co_mention: ['Pair co-mention only', 'caution'],
                    pharmacology_available: ['Pharmacology available', 'positive'],
                    computational_hypothesis: ['Computational hypothesis', 'caution']
                };
                const [evidenceStatusLabel, evidenceStatusKind] = evidenceStatusLabels[c.evidence_status] || ['Evidence status unavailable', 'caution'];
                const badges = [
                    c.indication_match === true ? ['Indication matched', 'positive'] : ['Check disease context', 'caution'],
                    c.combination_evidence === true ? ['Pair co-mention found', 'positive'] : ['No pair-level report', 'caution'],
                    [evidenceStatusLabel, evidenceStatusKind],
                    c.clinical_status === 'stopped_or_withdrawn' ? ['Stopped / withdrawn', 'caution'] : null
                ].filter(Boolean).map(([label, kind]) => `<span class="evidence-badge ${kind}">${label}</span>`).join('');
                const evidenceRows = (c.evidence || []).map(source => {
                    const date = source.retrieved_at ? `retrieved ${escapeHtml(source.retrieved_at)}` : 'date unavailable';
                    const excerpt = source.excerpt_or_field ? `<br><span>${escapeHtml(source.excerpt_or_field)}</span>` : '';
                    return `<li>${escapeHtml(source.name || 'Source')} — ${date}${excerpt}</li>`;
                }).join('');
                const tieNote = tiedRank
                    ? `<span class="evidence-badge caution">Tied rank · insufficient drug-specific evidence</span>`
                    : '';
                const topologyNote = targetUnavailable
                    ? '<span class="evidence-badge caution">Not scored · add/verify target in network</span>'
                    : '';
                const card = document.createElement('div');
                card.className = 'candidate-card';
                card.innerHTML = `
                    <div class="candidate-header">
                        <span class="drug-pair-name">${rankLabel} ${escapeHtml(c.secondary_drug)} + ${escapeHtml(primaryDrug)}</span>
                        <span class="badge-phase ${isHighestReportedStage ? 'approved' : ''}">${phaseLabel}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 0.35rem;">
                        Secondary Target: <strong style="color:#f8fafc;">${escapeHtml(c.secondary_target)}</strong> | Heuristic Priority: <strong style="color:#38bdf8;">${c.synergy_score}</strong> | Hub Centrality: <strong style="color:#c084fc;">${(c.hub_penalized_centrality || 0).toFixed(3)}</strong>
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.5rem;">${componentText || 'Component evidence not available'}</div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Evidence:</strong> ${evidenceFlags || 'Not reported'}${evidenceLinks ? `<br>${evidenceLinks}` : ''}</div>
                    <div class="evidence-badges">${badges}${tieNote}${topologyNote}</div>




                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.45;">${escapeHtml(c.biological_rationale)}</div>
                    <details class="candidate-explain">
                        <summary>Why this rank?</summary>
                        <p>This is a target-level network and evidence priority, not a measured combination effect.</p>
                        <ul>
                            <li>Target: ${escapeHtml(c.secondary_target)}; hub centrality belongs to this target node.</li>
                            <li>Priority components: ${escapeHtml(componentText || 'not available')}.</li>
                            <li>Clinical interpretation: ${c.evidence_notes?.length ? escapeHtml(c.evidence_notes.join(' ')) : (c.combination_evidence === true ? 'a returned report co-mentioned the primary drug; inspect the linked record for arm and outcome details.' : 'no pair-level report was identified; treat this as target/network evidence only.')}</li>
                            ${evidenceRows}
                        </ul>
                    </details>
                `;
                therapiesList.appendChild(card);
            });
        }

        function escapeHtml(value) {
            return String(value ?? '').replace(/[&<>'"]/g, character => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
            }[character]));
        }

        function safeExternalUrl(value) {
            try {
                const url = new URL(value);
                return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
            } catch (_) {
                return '';
            }
        }

        function copyResultJson() {
            if (!latestAnalysisData) return;
            navigator.clipboard.writeText(JSON.stringify(latestAnalysisData, null, 2))
                .then(() => alert('Structured JSON response copied to clipboard!'))
                .catch(() => alert('Failed to copy to clipboard'));
        }

        function compareReports() {
            if (analysisHistory.length < 2) return;
            const [previous, current] = analysisHistory;
            const top = report => (report.ranked_combinations || [])[0] || {};
            const row = (label, left, right) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(left ?? '—')}</td><td>${escapeHtml(right ?? '—')}</td></tr>`;
            const panel = document.getElementById('comparisonPanel');
            panel.innerHTML = `<details open><summary style="cursor:pointer;color:var(--genomic-blue);font-family:var(--font-mono);font-size:.72rem;">Hypothesis comparison</summary>
                <table style="width:100%;margin-top:.65rem;border-collapse:collapse;font-size:.76rem;color:var(--text-secondary);">
                    <thead><tr><th style="text-align:left;padding:.35rem 0;">Field</th><th style="text-align:left;">Previous</th><th style="text-align:left;">Current</th></tr></thead>
                    <tbody>${row('Resistance type', previous.resistance_type, current.resistance_type)}${row('Network nodes', previous.pathway_nodes_count, current.pathway_nodes_count)}${row('Top candidate', top(previous).secondary_drug || 'No candidate', top(current).secondary_drug || 'No candidate')}${row('Candidates returned', (previous.ranked_combinations || []).length, (current.ranked_combinations || []).length)}</tbody>
                </table></details>`;
            panel.style.display = 'block';
        }
    </script>
</body>
</html>
"""


def _sync_build_and_score(
    interactions: list[dict[str, Any]],
    primary_target: str,
    resistance_target: str,
    raw_candidates: list[dict[str, Any]],
) -> tuple[
    int, float | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """CPU-bound worker function offloaded via asyncio.to_thread."""
    G = build_signaling_graph(interactions)
    if len(G.nodes) < 2:
        raise ValueError("NoPathwayFound: Insufficient biological interactions.")

    scored = PathwayScorer.score_candidates(G, primary_target, raw_candidates)
    pathway_nodes_count = len(G.nodes)

    try:
        shortest_path_distance = PathwayScorer.calculate_shortest_distance(
            PathwayScorer.extract_lcc(G), primary_target, resistance_target
        )
    except ValueError:
        shortest_path_distance = None

    network_nodes = [
        {
            "id": str(n),
            "degree": int(G.degree(n)),
            "annotation": get_gene_annotation(str(n)),
        }
        for n in G.nodes
    ]
    network_edges = [
        {
            "source": str(u),
            "target": str(v),
            "score": float(G[u][v].get("score", 0.5)),
        }
        for u, v in G.edges
    ]

    return (
        pathway_nodes_count,
        shortest_path_distance,
        scored,
        network_nodes,
        network_edges,
    )


@app.get("/", response_class=HTMLResponse)
async def root_dashboard() -> str:
    """Serve the interactive web UI dashboard for the Resistance Bypass Engine."""
    return INDEX_HTML


@app.get("/api/v1/structure/{symbol}")
async def structure_lookup(symbol: str) -> dict[str, Any]:
    """Return curated structure metadata without inventing identifiers."""
    annotation = get_gene_annotation(symbol)
    pdb_id = annotation.get("pdb_id")
    if not pdb_id:
        return {
            "symbol": annotation.get("symbol", symbol.upper()),
            "status": "unavailable",
            "message": "No verified local PDB mapping is available for this target.",
            "annotation": annotation,
        }
    return {
        "symbol": annotation.get("symbol", symbol.upper()),
        "status": "curated_structure_mapping",
        "pdb_id": pdb_id,
        "source": "RCSB Protein Data Bank",
        "url": f"https://www.rcsb.org/structure/{pdb_id}",
        "annotation": annotation,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check diagnostic endpoint."""
    return {
        "status": "ok",
        "service": "Targeted Oncology Resistance Bypass Engine",
        "version": "0.1.0",
        "environment": os.getenv("APP_ENV", "development"),
        "network_clients": {
            "hgnc_rest": "configured_not_probed",
            "uniprot_kb": "configured_not_probed",
            "string_db": "configured_not_probed",
            "chembl_v4": "configured_not_probed",
            "open_targets_v4": "configured_not_probed",
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


def _is_drug_withdrawn(drug_row: dict[str, Any]) -> bool:
    """Check if an Open Targets drug row indicates a withdrawn status."""
    status = (drug_row.get("status") or "").strip().lower()
    return status in {"withdrawn", "stopped_or_withdrawn"}


def _network_evidence_claim(
    primary_target: str, resistance_marker: str
) -> ScientificClaim:
    """Describe network retrieval as provenance, not as clinical proof."""
    return ScientificClaim(
        claim_id="runtime-network-context",
        claim_text=(
            f"A STRING physical-association network was retrieved for {primary_target} "
            f"and {resistance_marker}."
        ),
        claim_type=ClaimType.COMPUTATIONAL,
        evidence=[
            EvidenceSource(
                name="STRING",
                url="https://string-db.org/",
                retrieved_at=datetime.now(UTC).date(),
                level=EvidenceLevel.CURATED_DATABASE,
                excerpt_or_field="network_type=physical",
                limitations=[
                    "Undirected association data are not proof of causal signaling.",
                    "Tissue and treatment context are not established by this record.",
                ],
            )
        ],
        unresolved_questions=[
            "Is the observed association active in the requested disease and alteration context?",
            "Is there direct experimental evidence for the proposed resistance mechanism?",
        ],
    )


def _report_metadata(
    request: ResistanceRequest,
    trace_id: str | None = None,
    source_timings_ms: dict[str, float] | None = None,
    partial_sources: list[str] | None = None,
) -> ReportMetadata:
    return ReportMetadata(
        schema_version="0.1.0",
        methodology_version="evidence-priority-0.3",
        generated_at=datetime.now(UTC),
        request_fingerprint=request_fingerprint(request),
        sources=["HGNC", "UniProt", "STRING", "Open Targets", "ChEMBL"],
        trace_id=trace_id,
        source_timings_ms=source_timings_ms or {},
        partial_sources=partial_sources or [],
    )


async def _timed_call(name: str, awaitable: Any, timings: dict[str, float]) -> Any:
    started = time.perf_counter()
    try:
        return await awaitable
    finally:
        timings[name] = round((time.perf_counter() - started) * 1000, 1)


async def _bounded_timed_call(
    name: str,
    awaitable: Any,
    timings: dict[str, float],
    timeout_seconds: float,
) -> Any:
    """Time a live-source call and bound provider latency per source.

    A single unavailable upstream should produce a clearly marked partial report,
    not make every independent source wait for its retry budget to expire.
    """
    return await _timed_call(
        name,
        asyncio.wait_for(awaitable, timeout=timeout_seconds),
        timings,
    )


@app.post("/api/v1/analyze-resistance", response_model=ResistanceBypassReport)
async def analyze_resistance(req: ResistanceRequest) -> ResistanceBypassReport:
    """Analyze drug resistance pathways and rank dual-drug combination candidates."""
    trace_id = uuid4().hex[:16]
    source_timings: dict[str, float] = {}
    partial_sources: list[str] = []
    id_mapper = IDMapper()
    live_source_timeout = max(
        5.0, float(os.getenv("LIVE_SOURCE_TIMEOUT_SECONDS", "15"))
    )

    try:
        mapped_primary, mapped_resistance = await _timed_call(
            "HGNC/UniProt/ChEMBL ID mapping",
            asyncio.gather(
                id_mapper.map_identifier(req.primary_target),
                id_mapper.map_identifier(req.resistance_marker),
            ),
            source_timings,
        )
    except Exception as exc:  # noqa: BLE001 - redact all upstream failures at the API boundary
        # Do not disclose upstream URLs, credentials, or implementation details.
        logger.warning(
            "ID resolution failed for trace %s: %s", trace_id, type(exc).__name__
        )
        raise HTTPException(
            status_code=422,
            detail=f"ID Resolution failed. Trace ID: {trace_id}",
        )

    primary_target_canonical = mapped_primary.canonical_symbol
    resistance_marker_canonical = mapped_resistance.canonical_symbol

    chembl_client = ChEMBLClient()

    # Branching Evaluation
    if primary_target_canonical == resistance_marker_canonical:
        # On-Target Mutation Branching
        resistance_type = "On-Target Mutation"
        pathway_nodes_count = 1
        shortest_path_distance = 0.0

        molecules = await _timed_call(
            "ChEMBL clinical molecules",
            chembl_client.get_clinical_molecules(
                target_chembl_id=mapped_primary.chembl_target_id,
                max_phase_gte=2,
                withdrawn_flag=False,
            ),
            source_timings,
        )

        ranked_combinations: list[CombinationCandidate] = []
        warnings: list[str] = []
        warnings.append(
            "On-target candidates are target-linked clinical records; variant-specific resistance reversal and pair-level efficacy are not established by this result."
        )
        for mol in molecules[:10]:
            drug_name = mol.get("pref_name") or mol.get(
                "molecule_chembl_id", "Unknown Drug"
            )
            max_phase = mol.get("max_phase")

            ranked_combinations.append(
                CombinationCandidate(
                    secondary_drug=drug_name.upper(),
                    secondary_target=primary_target_canonical,
                    mechanism_of_action="Next-Generation Inhibitor",
                    clinical_phase=int(max_phase) if max_phase is not None else None,
                    is_withdrawn=False,
                    synergy_score=0.0,
                    hub_penalized_centrality=0.0,
                    chembl_ic50_nm=None,
                    biological_rationale=f"On-target mutation in {primary_target_canonical}. Next-generation inhibitor overrides resistance.",
                    evidence=[
                        EvidenceSource(
                            name="ChEMBL",
                            url="https://www.ebi.ac.uk/chembl/",
                            release="ChEMBL API",
                            retrieved_at=datetime.now(UTC).date(),
                            level=EvidenceLevel.CURATED_DATABASE,
                            stable_id=mol.get("molecule_chembl_id"),
                            excerpt_or_field="clinical molecule target record",
                            limitations=[
                                "Target-linked clinical status does not prove resistance reversal or combination benefit."
                            ],
                        )
                    ],
                )
            )

        if not ranked_combinations:
            warnings.append(
                "No target-linked clinical molecule was returned; no candidate was fabricated."
            )

        return ResistanceBypassReport(
            primary_target_canonical=primary_target_canonical,
            resistance_marker_canonical=resistance_marker_canonical,
            resistance_type=resistance_type,
            pathway_nodes_count=pathway_nodes_count,
            shortest_path_distance=shortest_path_distance,
            ranked_combinations=ranked_combinations,
            network_nodes=[
                {
                    "id": primary_target_canonical,
                    "degree": 1,
                    "annotation": get_gene_annotation(primary_target_canonical),
                }
            ],
            network_edges=[],
            evidence_claims=[
                _network_evidence_claim(
                    primary_target_canonical, resistance_marker_canonical
                )
            ],
            warnings=warnings,
            primary_alteration=req.primary_alteration,
            resistance_alteration=req.resistance_alteration,
            resistance_alteration_type=req.resistance_alteration_type,
            treatment_line=req.treatment_line,
            metadata=_report_metadata(req, trace_id, source_timings, partial_sources),
        )

    else:
        # Off-Target Bypass Branching
        resistance_type = "Off-Target Bypass"

        string_client = StringDBClient()
        ot_client = OpenTargetsClient()

        async def _fetch_activities() -> dict[str, float]:
            if mapped_resistance.chembl_target_id:
                return await chembl_client.get_target_activities(
                    mapped_resistance.chembl_target_id
                )
            return {}

        # Fetch independent external APIs concurrently and keep partial results visible.
        results = await asyncio.gather(
            _bounded_timed_call(
                "STRING PPI network",
                string_client.get_network(
                    primary_target_canonical, resistance_marker_canonical
                ),
                source_timings,
                live_source_timeout,
            ),
            _bounded_timed_call(
                "Open Targets clinical candidates",
                ot_client.get_known_drugs(
                    mapped_resistance.ensembl_id,
                    cancer_type=req.cancer_type,
                    primary_drug=req.primary_drug,
                ),
                source_timings,
                live_source_timeout,
            ),
            _bounded_timed_call(
                "ChEMBL activity",
                _fetch_activities(),
                source_timings,
                live_source_timeout,
            ),
            return_exceptions=True,
        )
        interactions = results[0] if not isinstance(results[0], Exception) else []
        ot_drugs = results[1] if not isinstance(results[1], Exception) else []
        activity_map = results[2] if not isinstance(results[2], Exception) else {}
        for label, result in zip(
            ["STRING", "Open Targets", "ChEMBL"], results, strict=True
        ):
            if isinstance(result, Exception):
                partial_sources.append(label)

        # Filter out withdrawn drugs — AGENTS.md: "ranks active, non-withdrawn clinical dual-drug combination therapies"
        ot_drugs = [d for d in ot_drugs if not _is_drug_withdrawn(d)]

        raw_candidates: list[dict[str, Any]] = []
        warnings: list[str] = []
        if partial_sources:
            warnings.append(
                "Some live sources failed; the report contains a partial result: "
                + ", ".join(partial_sources)
            )

        if ot_drugs:
            for drug in ot_drugs:
                drug_name = drug.get("prefName") or drug.get("drugId") or "Unknown"
                moa = drug.get("mechanismOfAction") or "Bypass Pathway Inhibitor"
                phase = drug.get("phase")
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
                        "clinical_status": drug.get("clinicalStatus"),
                        "indication_match": drug.get("indicationMatch"),
                        "combination_evidence": drug.get("combinationEvidence"),
                        "indications": drug.get("diseaseNames", []),
                        "evidence": (drug.get("evidence") or [])
                        + [
                            EvidenceSource(
                                name="Open Targets",
                                url="https://platform.opentargets.org/",
                                release="Open Targets Platform API v4",
                                retrieved_at=datetime.now(UTC).date(),
                                level=EvidenceLevel.CURATED_DATABASE,
                                stable_id=drug.get("drugId"),
                                excerpt_or_field="drugAndClinicalCandidates",
                                limitations=[
                                    "Target-linked clinical evidence does not establish pair-level efficacy."
                                ],
                            ),
                            EvidenceSource(
                                name="ChEMBL",
                                url="https://www.ebi.ac.uk/chembl/",
                                release="ChEMBL API",
                                retrieved_at=datetime.now(UTC).date(),
                                level=EvidenceLevel.CURATED_DATABASE,
                                stable_id=drug.get("drugId"),
                                excerpt_or_field="target activity lookup",
                                limitations=[
                                    "Binding activity is not equivalent to clinical response."
                                ],
                            ),
                        ],
                    }
                )
        else:
            warnings.append(
                "No clinical candidate was returned for the resistance marker; no drug identity was fabricated."
            )

        # THREAD SAFETY: Offload heavy NetworkX CPU-bound math via asyncio.to_thread
        try:
            (
                pathway_nodes_count,
                shortest_path_distance,
                scored_raw,
                net_nodes,
                net_edges,
            ) = await asyncio.to_thread(
                _sync_build_and_score,
                interactions,
                primary_target_canonical,
                resistance_marker_canonical,
                raw_candidates,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        ranked_combinations: list[CombinationCandidate] = [
            CombinationCandidate(
                secondary_drug=c.get("secondary_drug", "Unknown"),
                secondary_target=c.get("secondary_target", resistance_marker_canonical),
                mechanism_of_action=c.get("mechanism_of_action", "Combination Therapy"),
                clinical_phase=(
                    int(c["clinical_phase"])
                    if c.get("clinical_phase") is not None
                    else None
                ),
                is_withdrawn=bool(c.get("is_withdrawn", False)),
                synergy_score=float(c.get("synergy_score", 0.0)),
                hub_penalized_centrality=float(c.get("hub_penalized_centrality", 0.0)),
                chembl_ic50_nm=c.get("chembl_ic50_nm"),
                score_components=c.get("score_components"),
                evidence=c.get("evidence", []),
                clinical_status=c.get("clinical_status"),
                indication_match=c.get("indication_match"),
                combination_evidence=c.get("combination_evidence"),
                indications=c.get("indications", []),
                rank=c.get("rank"),
                tie_group=c.get("tie_group"),
                tie_reason=c.get("tie_reason"),
                evidence_completeness=c.get("evidence_completeness"),
                shortest_path_distance=c.get("shortest_path_distance"),
                target_in_graph=c.get("target_in_graph"),
                scoring_status=c.get("scoring_status"),
                evidence_status=c.get("evidence_status"),
                evidence_notes=c.get("evidence_notes", []),
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
            evidence_claims=[
                _network_evidence_claim(
                    primary_target_canonical, resistance_marker_canonical
                )
            ],
            warnings=warnings,
            primary_alteration=req.primary_alteration,
            resistance_alteration=req.resistance_alteration,
            resistance_alteration_type=req.resistance_alteration_type,
            treatment_line=req.treatment_line,
            metadata=_report_metadata(req, trace_id, source_timings, partial_sources),
        )
