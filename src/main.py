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
        "src/static/network.png",
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
        "src/static/lab_mutation.png",
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
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .prevalence-card {
            position: relative;
            display: flex;
            flex-direction: column;
            min-height: 215px;
            overflow: hidden;
            background:
                radial-gradient(circle at 94% 10%, rgba(67, 190, 218, 0.11), transparent 28%),
                linear-gradient(145deg, #10232d, #0b1821);
            border: 1px solid #2b4855;
            border-top: 2px solid var(--genomic-blue);
            border-radius: 9px;
            padding: 1.05rem;
            cursor: pointer;
            transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
        }

        .scenario-organ-mark {
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            display: grid;
            place-items: center;
            width: 46px;
            height: 46px;
            border: 1px solid color-mix(in srgb, var(--indication-accent, #65d5ea) 36%, transparent);
            border-radius: 50%;
            background: color-mix(in srgb, var(--indication-accent, #65d5ea) 8%, #0c1921);
            color: var(--indication-accent, #65d5ea);
            opacity: 0.78;
            pointer-events: none;
        }
        .scenario-organ-mark svg { width: 27px; height: 27px; }

        .prevalence-card:hover {
            background:
                radial-gradient(circle at 94% 10%, rgba(67, 190, 218, 0.17), transparent 30%),
                linear-gradient(145deg, #13303b, #0c1b24);
            border-color: #63cce3;
            transform: translateY(-3px);
            box-shadow: 0 16px 30px rgba(1, 8, 12, 0.24);
        }
        .prevalence-card:hover .scenario-organ-mark { opacity: 1; }

        .prevalence-card:focus-visible { outline: 2px solid #78dcef; outline-offset: 3px; }

        .prevalence-card.high-prev { border-top-color: var(--mutation-red); }
        .prevalence-card.approved-prev { border-top-color: var(--approved-green); }

        .scenario-explorer {
            border: 1px solid #294250;
            border-radius: 8px;
            background: #10212b;
            padding: 0.85rem;
            margin-bottom: 1rem;
        }
        .scenario-library-panel .matrix-tabs-container {
            display: grid;
            grid-template-columns: repeat(4, minmax(150px, 1fr));
            gap: 0.55rem;
            border-bottom: 0;
            padding: 0;
            overflow: visible;
        }
        .scenario-library-panel .btn-matrix-tab {
            position: relative;
            min-height: 84px;
            flex-direction: column;
            align-items: flex-start;
            justify-content: center;
            gap: 0.22rem;
            overflow: hidden;
            border-radius: 9px;
            background:
                radial-gradient(circle at 88% 50%, color-mix(in srgb, var(--indication-accent, #6fd9eb) 9%, transparent), transparent 33%),
                #101e28;
            border-color: color-mix(in srgb, var(--indication-accent, #6fd9eb) 32%, #263e49);
            border-left: 2px solid var(--indication-accent, #6fd9eb);
            color: #d6e5e9;
            padding: 0.75rem 3.7rem 0.75rem 0.9rem;
            box-shadow: none;
            transition: transform .18s ease, border-color .18s ease, background .18s ease;
        }
        .scenario-library-panel .btn-matrix-tab:hover {
            transform: translateY(-2px);
            border-color: var(--indication-accent, #6fd9eb);
            background: color-mix(in srgb, var(--indication-accent, #6fd9eb) 8%, #10222c);
        }
        .scenario-library-panel .btn-matrix-tab.active {
            background: color-mix(in srgb, var(--indication-accent, #6fd9eb) 14%, #10242e);
            color: #f3fcfd;
            border-color: var(--indication-accent, #6fd9eb);
            border-bottom-width: 1px;
            box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--indication-accent, #6fd9eb) 18%, transparent);
        }
        .scenario-tab-code { color: var(--indication-accent, #6fd3e8); font: 0.63rem var(--font-mono); letter-spacing: .12em; text-transform: uppercase; }
        .scenario-tab-label { color: inherit; font-size: .88rem; font-weight: 720; letter-spacing: -.012em; }
        .scenario-tab-organ {
            position: absolute;
            top: 50%;
            right: .85rem;
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            transform: translateY(-50%);
            border: 1px solid color-mix(in srgb, var(--indication-accent, #6fd9eb) 34%, transparent);
            border-radius: 50%;
            background: color-mix(in srgb, var(--indication-accent, #6fd9eb) 7%, #0c1921);
            color: var(--indication-accent, #6fd9eb);
            opacity: .88;
        }
        .scenario-tab-organ svg { width: 26px; height: 26px; }
        .btn-matrix-tab.active .scenario-tab-organ { opacity: 1; background: color-mix(in srgb, var(--indication-accent, #6fd9eb) 13%, #0c1921); }
        .scenario-explorer-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: 0.7rem; }
        .scenario-explorer-head strong { color: #effcff; font-size: 0.9rem; }
        .scenario-explorer-head span { color: #91aeb9; font: 0.68rem var(--font-mono); }
        .scenario-filter-row { display: grid; grid-template-columns: minmax(220px, 1.4fr) repeat(2, minmax(150px, 0.7fr)) auto; gap: 0.5rem; }
        .scenario-filter-row input, .scenario-filter-row select {
            min-width: 0;
            border: 1px solid #385563;
            border-radius: 6px;
            padding: 0.58rem 0.65rem;
            background: #0b1821;
            color: #e8f5f7;
            font: 0.75rem var(--font-main);
        }
        .scenario-filter-row input:focus, .scenario-filter-row select:focus { outline: 2px solid rgba(75, 196, 224, 0.35); outline-offset: 1px; border-color: #63d4e9; }
        .scenario-reset { border: 1px solid #385563; border-radius: 6px; padding: 0.55rem 0.7rem; background: transparent; color: #aad0d9; cursor: pointer; font: 600 0.72rem var(--font-main); }
        .scenario-reset:hover { border-color: #63d4e9; color: #effcff; }
        .scenario-result-count { color: #75d4ee; font: 0.68rem var(--font-mono); margin-top: 0.65rem; }
        .scenario-empty { display: none; border: 1px dashed #41616e; border-radius: 7px; padding: 1.2rem; color: #a8c0c8; text-align: center; font-size: 0.8rem; }
        .scenario-library-panel.filtering .prevalence-cards-grid { display: grid !important; }
        .scenario-library-panel.filtering .prevalence-cards-grid > .prevalence-card.is-hidden { display: none; }
        .scenario-library-panel.filtering .prevalence-cards-grid > .prevalence-card { display: flex; }
        .prevalence-card .scenario-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; margin-top: auto; padding-top: 0.85rem; }
        .prevalence-card .scenario-meta span { border: 1px solid #365563; border-radius: 4px; padding: 0.2rem 0.42rem; color: #9bc0ca; font: 0.59rem var(--font-mono); letter-spacing: .025em; }
        .prevalence-card .scenario-meta span:last-child { margin-left: auto; border-color: transparent; color: #77d9ec; font-family: var(--font-main); font-weight: 700; }
        .scenario-evidence-link { display: inline-flex; margin-top: 0.7rem; color: #7edff0; font: 0.66rem var(--font-mono); text-decoration: none; }
        .scenario-evidence-link:hover { color: #effcff; text-decoration: underline; }
        .scenario-evidence-note { margin: -0.25rem 0 0.9rem; color: #8faab4; font-size: 0.72rem; }
        @media (max-width: 1100px) { .scenario-library-panel .matrix-tabs-container { grid-template-columns: repeat(3, minmax(140px, 1fr)); } .prevalence-cards-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 900px) { .scenario-filter-row { grid-template-columns: 1fr 1fr; } .scenario-filter-row input { grid-column: 1 / -1; } .scenario-library-panel .matrix-tabs-container { grid-template-columns: repeat(2, minmax(130px, 1fr)); } }
        @media (max-width: 680px) { .prevalence-cards-grid { grid-template-columns: 1fr; } }
        @media (max-width: 560px) { .scenario-filter-row { grid-template-columns: 1fr; } }

        .prevalence-header {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.48rem;
            margin-bottom: 0.65rem;
            padding-right: 2.8rem;
        }

        .scenario-pair-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: var(--text-main);
            letter-spacing: -0.018em;
        }

        .badge-prevalence {
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.62rem;
            font-weight: 650;
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
            display: inline-flex;
            align-items: center;
            width: fit-content;
            font-size: 0.66rem;
            color: #9bc3cc;
            font-family: var(--font-mono);
            font-weight: 500;
            margin-bottom: 0.65rem;
            padding-left: 0.7rem;
            border-left: 2px solid #765da6;
        }

        .scenario-mechanism {
            padding-top: 0.68rem;
            border-top: 1px solid #27424e;
            font-size: 0.75rem;
            color: #9eb5bd;
            line-height: 1.52;
        }
        .scenario-mechanism strong { color: #dcebed; font-weight: 650; }

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

        /* Institutional oncology lab refresh: quiet canvas, explicit hierarchy. */
        body {
            background-color: #081018;
            background-image:
                radial-gradient(circle at 84% 0%, rgba(75, 196, 224, 0.10), transparent 30%),
                radial-gradient(circle at 0% 40%, rgba(44, 101, 128, 0.09), transparent 32%);
            padding: clamp(0.75rem, 2vw, 1.5rem);
        }

        header.app-header {
            background: rgba(13, 24, 34, 0.92);
            border-color: #294250;
            border-radius: 10px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
            padding: 0.75rem 1rem;
            position: relative;
        }

        .brand-area { gap: 0.7rem; }
        .brand-logo {
            width: 40px;
            height: 40px;
            border-radius: 9px;
            background: #102d3b;
            border-color: #287f9e;
        }
        .brand-title {
            font-size: clamp(1rem, 1.8vw, 1.22rem);
            letter-spacing: -0.035em;
        }
        .brand-subtitle {
            font-family: var(--font-mono);
            font-size: 0.64rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .header-actions { gap: 0.35rem; }
        .status-pill {
            padding: 0.34rem 0.65rem;
            background: rgba(34, 197, 164, 0.10);
            border-color: rgba(34, 197, 164, 0.35);
            color: #72e1c5;
            font-size: 0.67rem;
        }
        .btn-header {
            background: transparent;
            border-color: #334b59;
            border-radius: 7px;
            padding: 0.42rem 0.68rem;
            font-size: 0.75rem;
        }
        .btn-header:hover { background: #172b37; border-color: #55c8e8; }
        .btn-header.author-btn {
            background: #123747;
            border-color: #287f9e;
            color: #7ddcf2;
        }

        .genome-hero {
            grid-template-columns: minmax(0, 1.2fr) minmax(290px, 0.8fr);
            border-color: #294250;
            border-radius: 10px;
            background: #0e1b25;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
        }
        .genome-hero-copy { padding: clamp(1.5rem, 4vw, 3.25rem); }
        .hero-kicker { color: #77d4ea; margin-bottom: 0.8rem; }
        .genome-hero h1 { max-width: 650px; font-size: clamp(2.1rem, 5vw, 4.1rem); }
        .genome-hero h1 em { color: #75d4ee; }
        .genome-hero-copy p { max-width: 52ch; font-size: 1rem; color: #c7d6dc; }
        .hero-actions { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.35rem; }
        .hero-primary, .hero-secondary {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 7px;
            padding: 0.65rem 0.9rem;
            font: 600 0.78rem var(--font-main);
            text-decoration: none;
            cursor: pointer;
        }
        .hero-primary { background: #4bc4e0; color: #061018; border: 1px solid #7edff0; }
        .hero-secondary { background: transparent; color: #b9d6df; border: 1px solid #3c5a67; }
        .hero-primary:hover { background: #8be4f1; }
        .hero-secondary:hover { border-color: #73cfe4; color: #effcff; }
        .hero-statline {
            margin-top: 1.2rem;
            gap: 0;
            color: #8ba7b1;
        }
        .hero-stat {
            border: 0;
            border-radius: 0;
            background: transparent;
            padding: 0 0.7rem;
        }
        .hero-stat:first-child { padding-left: 0; }
        .hero-stat + .hero-stat { border-left: 1px solid #35505c; }
        .mutation-plate {
            min-height: 270px;
            border-left-color: #294250;
            background:
                radial-gradient(circle at 68% 40%, rgba(62, 195, 222, 0.13), transparent 32%),
                linear-gradient(145deg, #102632, #0b151d);
        }
        .mutation-plate::before, .mutation-plate::after { display: none; }
        .mutation-plate::marker { display: none; }
        .mutation-readout { left: 1.1rem; right: 1.1rem; bottom: 1rem; }
        .evidence-plate-label {
            position: absolute;
            top: 1rem;
            left: 1.15rem;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: .5rem;
            color: #92b5bf;
            font: .58rem var(--font-mono);
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .evidence-plate-label::before { content: ''; width: 18px; height: 1px; background: #66d4e9; }
        .evidence-map-graphic {
            position: absolute;
            inset: 1.2rem 1.2rem 2.7rem;
            width: calc(100% - 2.4rem);
            height: calc(100% - 3.9rem);
            opacity: 0.46;
        }
        .evidence-map-line { fill: none; stroke: #326273; stroke-width: 1.3; stroke-dasharray: 4 6; }
        .evidence-map-helix { fill: none; stroke: #54cce5; stroke-width: 1.5; opacity: 0.34; }
        .evidence-map-node { fill: #0d202a; stroke: #65d5ea; stroke-width: 1.5; }
        .evidence-map-node.core { fill: #61d3e9; stroke: #b7f4ff; }
        .evidence-chain {
            position: absolute;
            inset: 3rem 1.1rem 3.3rem;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            align-items: center;
            gap: .42rem;
            color: #90aeb9;
        }
        .evidence-chain::before {
            content: '';
            position: absolute;
            left: 12%;
            right: 12%;
            top: 50%;
            height: 1px;
            background: linear-gradient(90deg, #315766, #6dd9ec, #315766);
        }
        .evidence-chain-step {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: .38rem;
            min-height: 150px;
            padding: .7rem .45rem;
            border: 1px solid rgba(84, 174, 194, .26);
            border-radius: 8px;
            background: rgba(7, 22, 29, .82);
            backdrop-filter: blur(8px);
            text-align: center;
        }
        .evidence-step-number {
            position: absolute;
            top: .52rem;
            left: .55rem;
            color: #9be8f4;
            font: 0.56rem var(--font-mono);
        }
        .evidence-step-icon {
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            border: 1px solid #4e9aae;
            border-radius: 50%;
            background: #0c2731;
            color: #79dceb;
        }
        .evidence-step-icon svg { width: 23px; height: 23px; }
        .evidence-step-copy { display: flex; flex-direction: column; min-width: 0; }
        .evidence-step-copy strong { color: #f0fbfd; font: 650 0.79rem var(--font-main); }
        .evidence-step-copy small { color: #91acb5; font: 0.54rem var(--font-mono); letter-spacing: .04em; text-transform: uppercase; margin-top: .15rem; }
        .evidence-step-source { color: #62bfd2; font: 0.49rem var(--font-mono); letter-spacing: .06em; text-transform: uppercase; }

        .interpretation-strip { background: #0c1821; border-color: #263f4c; }
        .scenario-library { margin-bottom: 1.25rem; }
        .scenario-library-summary {
            list-style: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid #294250;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            background: #0e1b25;
            color: #e6f5f7;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
        }
        .scenario-library-summary::-webkit-details-marker { display: none; }
        .scenario-library-summary::after { content: '+'; color: #70d9ee; font: 1.3rem var(--font-mono); }
        .scenario-library[open] .scenario-library-summary::after { content: '−'; }
        .scenario-library-summary small { display: block; color: #91aeb9; font: 0.72rem var(--font-mono); margin-top: 0.25rem; }
        .scenario-library-panel { margin-top: 0.6rem; }
        .scenario-library-panel > .matrix-top-bar { display: none; }
        .scenario-library-panel.academic-matrix-section { border-radius: 10px; }
        .academic-matrix-section { border-color: #263f4c; border-radius: 10px; background: #0d1922; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16); }
        .workstation-grid { scroll-margin-top: 1rem; }
        .route-page { display: none; }
        .route-page h2 { color: #f0fbfd; font-size: clamp(1.6rem, 3vw, 2.4rem); letter-spacing: -0.04em; margin-bottom: 0.45rem; }
        .route-page p { max-width: 68ch; color: #b7cbd2; }
        .route-page .route-eyebrow { color: #77d4ea; font: 0.68rem var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.75rem; }
        .route-page .route-card-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin-top: 1.25rem; }
        .route-page .route-card { border: 1px solid #294250; border-radius: 8px; padding: 1rem; background: #10212b; }
        .route-page .route-card strong { display: block; color: #effcff; margin-bottom: 0.35rem; }
        .route-page .route-card span { color: #9bb4bd; font-size: 0.8rem; }
        body[data-page="home"] .workstation-grid { display: none; }
        body[data-page="home"] .route-page { display: none; }
        body[data-page="home"] .interpretation-strip,
        body[data-page="home"] .scenario-library { display: none; }
        body[data-page="analyze"] .genome-hero,
        body[data-page="analyze"] .interpretation-strip,
        body[data-page="analyze"] .scenario-library { display: none; }
        body[data-page="analyze"] .analysis-route { display: block; margin-bottom: 1rem; }
        body[data-page="scenarios"] .genome-hero,
        body[data-page="scenarios"] .interpretation-strip,
        body[data-page="scenarios"] .workstation-grid { display: none; }
        body[data-page="scenarios"] .scenario-library { display: block; }
        body[data-page="scenarios"] .scenario-library-panel { margin-top: 0.75rem; }
        body[data-page="method"] .genome-hero,
        body[data-page="method"] .interpretation-strip,
        body[data-page="method"] .scenario-library,
        body[data-page="method"] .workstation-grid { display: none; }
        body[data-page="method"] .method-route { display: block; }
        body[data-page="sources"] .genome-hero,
        body[data-page="sources"] .interpretation-strip,
        body[data-page="sources"] .scenario-library,
        body[data-page="sources"] .workstation-grid { display: none; }
        body[data-page="sources"] .sources-route { display: block; }
        .home-showcase { display: none; }
        body[data-page="home"] .home-showcase { display: block; margin-bottom: 0; }
        .home-showcase-head { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: 1rem; }
        .home-showcase .route-eyebrow { color: #77d4ea; font: 0.66rem var(--font-mono); letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem; }
        .home-showcase-head h2 { color: #f2fbfc; font-size: clamp(1.35rem, 2.5vw, 2rem); letter-spacing: -0.04em; }
        .showcase-invitation { display: flex; align-items: center; gap: .75rem; max-width: 330px; }
        .showcase-invitation strong { color: #78dced; font: 500 clamp(2.3rem, 4vw, 3.4rem)/.8 var(--font-main); letter-spacing: -.07em; }
        .showcase-invitation span { color: #9fb7bf; font-size: .72rem; line-height: 1.45; }
        .featured-scenario-grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(280px, .75fr); gap: 0.75rem; }
        .featured-scenario {
            position: relative;
            min-height: 190px;
            overflow: hidden;
            border: 1px solid #294654;
            border-radius: 9px;
            padding: 1.1rem;
            background: linear-gradient(145deg, #10232d, #0c1820);
            color: inherit;
            text-decoration: none;
            transition: transform .2s ease, border-color .2s ease, background .2s ease;
        }
        .featured-scenario:first-child { min-height: 255px; background: radial-gradient(circle at 84% 24%, rgba(84, 207, 230, .11), transparent 28%), linear-gradient(145deg, #12303b, #0b1a22); }
        .featured-scenario:hover { transform: translateY(-3px); border-color: #68d2e7; background: #122934; }
        .featured-scenario .feature-index { color: #6ccfe4; font: 0.65rem var(--font-mono); letter-spacing: .1em; }
        .featured-scenario h3 { color: #f1fbfc; font-size: clamp(1.1rem, 2vw, 1.55rem); margin: .8rem 0 .35rem; letter-spacing: -.025em; }
        .featured-scenario p { color: #9db7c0; font-size: .78rem; line-height: 1.5; max-width: 42ch; }
        .feature-route-map { display: grid; grid-template-columns: auto minmax(90px, 1fr) auto; align-items: center; gap: .7rem; margin-top: 1rem; max-width: 480px; }
        .feature-gene-node { display: grid; place-items: center; width: 62px; height: 62px; border: 1px solid #55bdd3; border-radius: 50%; background: #0b2029; }
        .feature-gene-node small { color: #779aa5; font: .45rem var(--font-mono); letter-spacing: .08em; }
        .feature-gene-node strong { color: #dff9fc; font: 700 .78rem var(--font-main); }
        .feature-gene-node.bypass { border-color: #ef6570; box-shadow: 0 0 0 5px rgba(239, 101, 112, .06); }
        .feature-route-line { position: relative; height: 1px; background: linear-gradient(90deg, #4dbed6, #ef6570); }
        .feature-route-line::after { content: ''; position: absolute; right: -1px; top: -3px; border-left: 6px solid #ef6570; border-top: 3px solid transparent; border-bottom: 3px solid transparent; }
        .feature-route-line span { position: absolute; left: 50%; bottom: .45rem; transform: translateX(-50%); white-space: nowrap; color: #86a9b3; font: .5rem var(--font-mono); text-transform: uppercase; }
        .feature-action { position: absolute; right: 1rem; bottom: .9rem; color: #7ee2f1; font: 650 .68rem var(--font-main); }
        .scenario-index-card {
            display: flex;
            flex-direction: column;
            min-height: 255px;
            border: 1px solid #294654;
            border-radius: 9px;
            padding: 1rem;
            background: #0b171f;
        }
        .scenario-index-card > span { color: #79d7e9; font: .64rem var(--font-mono); letter-spacing: .1em; text-transform: uppercase; }
        .scenario-index-card > strong { color: #f1fbfc; font-size: 1.1rem; margin: .35rem 0 .75rem; }
        .scenario-index-links { display: grid; gap: .35rem; }
        .scenario-index-links a {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #b8d0d7;
            border-top: 1px solid #253e49;
            padding: .55rem 0;
            font-size: .74rem;
            text-decoration: none;
        }
        .scenario-index-links a:hover { color: #75daed; }
        .scenario-index-card > a { margin-top: auto; color: #78d8eb; font: 600 .72rem var(--font-main); text-decoration: none; }
        @media (max-width: 900px) { .featured-scenario-grid { grid-template-columns: 1fr; } .featured-scenario:first-child { min-height: 245px; } .home-showcase-head { align-items: flex-start; flex-direction: column; } }
        @media (max-width: 760px) { .route-page .route-card-grid { grid-template-columns: 1fr; } }
        @media (max-width: 860px) {
            .header-actions { width: 100%; overflow-x: auto; padding-bottom: 0.2rem; }
            .genome-hero { grid-template-columns: 1fr; }
            .mutation-plate { border-left: 0; border-top: 1px solid #294250; }
        }
        @media (max-width: 560px) {
            .mutation-plate { min-height: 440px; }
            .evidence-chain {
                inset: 2.8rem .9rem 3.2rem;
                grid-template-columns: 1fr;
                gap: .35rem;
            }
            .evidence-chain::before {
                top: 10%;
                bottom: 10%;
                left: 31px;
                right: auto;
                width: 1px;
                height: auto;
                background: linear-gradient(#315766, #6dd9ec, #315766);
            }
            .evidence-chain-step {
                min-height: 100px;
                flex-direction: row;
                justify-content: flex-start;
                gap: .65rem;
                padding: .6rem .65rem .6rem 1rem;
                text-align: left;
            }
            .evidence-step-number { top: .42rem; left: .45rem; }
            .evidence-step-source { margin-left: auto; text-align: right; }
            .feature-route-map { grid-template-columns: auto minmax(50px, 1fr) auto; }
            .feature-gene-node { width: 54px; height: 54px; }
        }

        /* Analyze page — clinical instrument panel */
        body[data-page="analyze"] {
            background-image: radial-gradient(circle at 76% 18%, rgba(83, 205, 226, .075), transparent 30%), radial-gradient(circle at 15% 58%, rgba(239, 101, 112, .045), transparent 28%);
            background-size: 100% 100%;
        }
        body[data-page="analyze"] .header-actions a[href="/analyze"] { border-color:#5ec9dc; background:#14303a; color:#f0fbfc; box-shadow:inset 0 -2px #63d2e4; }
        body[data-page="analyze"] .analysis-route {
            position: relative; min-height: 250px; overflow: hidden;
            padding: clamp(1.5rem, 3vw, 2.4rem); border-color: #31515e;
            background: radial-gradient(circle at 86% 22%, rgba(93, 210, 230, .12), transparent 23%), linear-gradient(115deg, #102630 0%, #0d1c25 62%, #0a151d 100%);
        }
        body[data-page="analyze"] .analysis-route::before {
            content: ''; position: absolute; inset: 0; pointer-events: none;
            background: repeating-linear-gradient(90deg, transparent 0 71px, rgba(119, 212, 234, .035) 72px 73px);
            mask-image: linear-gradient(90deg, transparent, #000 55%);
        }
        .analysis-route-shell { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(310px, .75fr); gap: clamp(2rem, 5vw, 5rem); align-items: end; }
        .analysis-route-copy h2 { max-width: 720px; font-size: clamp(2.25rem, 4.8vw, 4.7rem); line-height: .98; text-wrap: balance; }
        .analysis-route-copy p { margin-top: 1rem; font-size: .96rem; line-height: 1.65; text-wrap: pretty; }
        .analysis-contract { position: relative; padding: 1.1rem 1.15rem 1.15rem; border: 1px solid rgba(104, 205, 225, .3); border-radius: 8px; background: rgba(8, 22, 29, .72); box-shadow: inset 0 1px rgba(255,255,255,.035), 0 18px 50px rgba(0, 8, 13, .22); backdrop-filter: blur(12px); }
        .analysis-contract::before { content:''; position:absolute; left:-1px; top:1rem; bottom:1rem; width:2px; background:#70d7e9; }
        .analysis-contract-label { color:#7edbed; font:.6rem var(--font-mono); letter-spacing:.12em; text-transform:uppercase; }
        .analysis-contract strong { display:block; margin:.5rem 0 .25rem; color:#f1fbfc; font-size:1rem; }
        .analysis-contract p { color:#9eb7bf; font-size:.73rem; line-height:1.5; }
        .analysis-contract-meta { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.75rem; }
        .analysis-contract-meta span { padding:.25rem .42rem; border:1px solid #34515c; border-radius:4px; color:#a9c2c9; font:.57rem var(--font-mono); }

        body[data-page="analyze"] .workstation-grid { grid-template-columns: minmax(360px, 430px) minmax(0, 1fr); gap: 1rem; }
        body[data-page="analyze"] .panel { border-color: #29434f; border-radius: 10px; background: #0c1821; box-shadow: 0 18px 46px rgba(0, 7, 12, .28); }
        .analysis-input-panel { position: sticky; top: 1rem; padding: 0 !important; overflow: hidden; }
        .analysis-output-panel { min-height: 680px; padding: 0 !important; overflow: hidden; }
        .analysis-input-panel .panel-header, .analysis-output-panel .panel-header { min-height: 68px; margin: 0; padding: 1rem 1.2rem; border-bottom-color: #29434f; background: rgba(14, 31, 40, .84); }
        .panel-kicker { display:block; margin-bottom:.2rem; color:#68cfe2; font:.56rem var(--font-mono); letter-spacing:.11em; text-transform:uppercase; }
        .analysis-input-panel .panel-title-text, .analysis-output-panel .panel-title-text { font-size: 1rem; letter-spacing: -.01em; }
        .analysis-input-panel form { padding: 1.15rem 1.2rem 1.25rem; }
        .analysis-output-body { padding: 1rem; }
        .analysis-core-fields { display:grid; gap:.65rem; }
        .analysis-field-card { position: relative; padding: .72rem .8rem .78rem 3.25rem; border: 1px solid #2b4551; border-radius: 8px; background: linear-gradient(110deg, #10232d, #0d1b24); transition: border-color .2s ease, transform .2s ease, background .2s ease; }
        .analysis-field-card:focus-within { transform: translateX(2px); border-color:#65cde1; background:#102832; }
        .field-sequence { position:absolute; left:.7rem; top:.75rem; display:grid; place-items:center; width:1.85rem; height:1.85rem; border:1px solid #447381; border-radius:50%; color:#8be3ef; font:.58rem var(--font-mono); box-shadow:0 0 0 5px rgba(88, 202, 224, .035); }
        .analysis-field-card.resistance .field-sequence { color:#ff9299; border-color:#96535c; box-shadow:0 0 0 5px rgba(239, 101, 112, .035); }
        .analysis-field-card .form-group { margin:0; }
        .analysis-field-card label.field-label { margin:0 0 .18rem; color:#9fb8c0; font-size:.67rem; font-weight:600; }
        .analysis-field-card input.input-field { min-height:0; padding:0; border:0; border-radius:0; background:transparent; box-shadow:none; color:#f3fbfc; font-size:1rem; letter-spacing:-.015em; }
        .analysis-field-card input.input-field:focus { background:transparent; box-shadow:none; }
        .analysis-field-card input.input-field::placeholder { color:#58727c; }
        .analysis-optional { margin-top:.8rem; border:1px solid #273f4a; border-radius:8px; background:#0a151d; }
        .analysis-optional summary { list-style:none; cursor:pointer; display:flex; align-items:center; justify-content:space-between; padding:.72rem .8rem; color:#d6e8eb; font-size:.73rem; font-weight:650; }
        .analysis-optional summary::-webkit-details-marker { display:none; }
        .analysis-optional summary::after { content:'+'; color:#6fd7e9; font:1rem var(--font-mono); }
        .analysis-optional[open] summary::after { content:'−'; }
        .analysis-optional-copy { color:#73909a; font:.57rem var(--font-mono); font-weight:400; margin-left:auto; margin-right:.7rem; }
        .analysis-optional-fields { padding:.1rem .8rem .8rem; border-top:1px solid #203640; }
        .analysis-optional-fields .form-group { margin-top:.72rem; margin-bottom:0; }
        .analysis-optional-fields label.field-label { color:#a9bec5; font-size:.68rem; }
        .analysis-optional-fields input.input-field, .analysis-optional-fields select.input-field, .analysis-cancer-field input.input-field { min-height:42px; padding:.62rem .72rem; border-color:#304b57; border-radius:6px; background:#10212a; font-size:.78rem; font-weight:550; }
        .analysis-cancer-field { margin:.8rem 0 0 !important; padding:.78rem .8rem; border:1px solid #29434f; border-radius:8px; background:#0d1d26; }
        .analysis-cancer-field label.field-label { color:#d8e9ec; font-size:.7rem; }
        .analysis-cancer-field label.field-label::after { content:'disease context'; color:#63838e; font:.52rem var(--font-mono); letter-spacing:.06em; text-transform:uppercase; }
        .analysis-form-note { display:flex; align-items:flex-start; gap:.55rem; margin:.8rem 0 .25rem; color:#829da6; font-size:.65rem; line-height:1.45; }
        .analysis-form-note::before { content:'i'; flex:0 0 auto; display:grid; place-items:center; width:1rem; height:1rem; border:1px solid #3d6674; border-radius:50%; color:#78d8e9; font:.58rem var(--font-mono); }
        .analysis-input-panel .btn-run { min-height:50px; margin-top:.8rem; border:1px solid #76d8e7; border-radius:7px; background:#72d4e5; color:#071319; box-shadow:0 10px 26px rgba(61, 183, 207, .16), inset 0 1px rgba(255,255,255,.38); }
        .analysis-input-panel .btn-run:hover { transform:translateY(-2px); background:#8be0ec; box-shadow:0 15px 34px rgba(61,183,207,.24); }
        .analysis-input-panel .btn-run:active { transform:translateY(0) scale(.99); }

        .analysis-empty-stage { position:relative; min-height:540px; overflow:hidden; border:1px solid #263f4a; border-radius:8px; background:radial-gradient(circle at 76% 28%, rgba(86, 205, 225, .09), transparent 28%), linear-gradient(145deg, #09161e, #081119); }
        .analysis-empty-stage::before { content:''; position:absolute; inset:0; opacity:.16; background-image:radial-gradient(circle, #68cfe2 1px, transparent 1px); background-size:28px 28px; mask-image:linear-gradient(to bottom, #000, transparent 72%); }
        .analysis-empty-head { position:relative; z-index:1; display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding:1.1rem 1.15rem; }
        .analysis-empty-head span { color:#6fd5e8; font:.58rem var(--font-mono); letter-spacing:.11em; text-transform:uppercase; }
        .analysis-empty-head p { margin-top:.3rem; color:#9ab4bc; font-size:.72rem; }
        .analysis-ready-state { display:flex; align-items:center; gap:.4rem; color:#75d7a9; font:.57rem var(--font-mono); white-space:nowrap; }
        .analysis-ready-state::before { content:''; width:6px; height:6px; border-radius:50%; background:#55d6a3; box-shadow:0 0 0 5px rgba(85,214,163,.08); }
        .analysis-pathway-preview { position:relative; z-index:1; display:grid; grid-template-columns:auto minmax(80px,1fr) auto minmax(80px,1fr) auto; align-items:center; gap:.65rem; max-width:720px; margin:3.2rem auto 0; padding:0 1.4rem; }
        .preview-node { position:relative; display:grid; place-items:center; width:92px; height:92px; border:1px solid #55bcd1; border-radius:50%; background:radial-gradient(circle at 35% 28%, #163d49, #0a1d26 66%); box-shadow:0 0 0 8px rgba(82,188,209,.035), 0 20px 42px rgba(0,0,0,.24); }
        .preview-node.resistance { border-color:#df6973; background:radial-gradient(circle at 35% 28%, #49262d, #211219 68%); box-shadow:0 0 0 8px rgba(239,101,112,.035), 0 20px 42px rgba(0,0,0,.24); }
        .preview-node small { color:#789aa5; font:.48rem var(--font-mono); letter-spacing:.08em; }
        .preview-node strong { max-width:72px; overflow:hidden; text-overflow:ellipsis; color:#f0fbfc; font-size:.85rem; white-space:nowrap; }
        .preview-node em { position:absolute; top:calc(100% + .55rem); color:#77939d; font:normal .52rem var(--font-mono); white-space:nowrap; }
        .preview-drug { display:flex; flex-direction:column; align-items:center; gap:.38rem; min-width:100px; }
        .preview-drug-mark { display:grid; place-items:center; width:42px; height:42px; border:1px solid #c6a85d; border-radius:50% 50% 44% 56%; background:#2a2518; color:#f2ca6e; font-size:1rem; transform:rotate(-8deg); }
        .preview-drug strong { max-width:120px; overflow:hidden; text-overflow:ellipsis; color:#d9c78f; font:.58rem var(--font-mono); white-space:nowrap; }
        .preview-route { position:relative; height:1px; background:linear-gradient(90deg,#477c89,#6ed3e6); }
        .preview-route:last-of-type { background:linear-gradient(90deg,#6ed3e6,#d96670); }
        .preview-route::after { content:''; position:absolute; right:-1px; top:-3px; border-left:6px solid currentColor; border-top:3px solid transparent; border-bottom:3px solid transparent; color:#67c8dc; }
        .preview-route span { position:absolute; left:50%; bottom:.55rem; transform:translateX(-50%); color:#6c8993; font:.48rem var(--font-mono); white-space:nowrap; text-transform:uppercase; }
        .analysis-workflow-strip { position:absolute; z-index:1; left:1rem; right:1rem; bottom:1rem; display:grid; grid-template-columns:repeat(3,1fr); border:1px solid #27434e; border-radius:7px; background:rgba(7,18,25,.9); backdrop-filter:blur(10px); }
        .analysis-workflow-step { position:relative; min-height:105px; padding:.85rem .85rem .8rem 2.7rem; border-right:1px solid #27434e; }
        .analysis-workflow-step:last-child { border-right:0; }
        .analysis-workflow-step b { position:absolute; left:.75rem; top:.85rem; color:#77d9eb; font:.6rem var(--font-mono); }
        .analysis-workflow-step strong { display:block; color:#e9f7f9; font-size:.73rem; }
        .analysis-workflow-step span { display:block; margin-top:.3rem; color:#819da6; font-size:.62rem; line-height:1.45; }
        .analysis-workflow-step small { display:block; margin-top:.42rem; color:#597782; font:.5rem var(--font-mono); letter-spacing:.04em; }
        .analysis-output-panel #placeholder > .vector-graph-canvas { display:none; }
        .analysis-loader { position:relative; min-height:520px; padding:1.2rem !important; text-align:left !important; overflow:hidden; }
        .analysis-loader::after { content:''; position:absolute; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,#74d9e9,transparent); animation:analysisScan 1.7s ease-in-out infinite; }
        @keyframes analysisScan { from{top:8%;opacity:0} 20%{opacity:1} to{top:92%;opacity:0} }
        .analysis-loader-head { display:flex; align-items:center; justify-content:space-between; padding-bottom:1rem; border-bottom:1px solid #28414c; }
        .analysis-loader-head strong { color:#edf9fb; font-size:.9rem; }
        .analysis-loader-head span { color:#69cfdf; font:.58rem var(--font-mono); }
        .analysis-skeleton-grid { display:grid; grid-template-columns:1.25fr .75fr; gap:.75rem; margin-top:1rem; }
        .analysis-skeleton { min-height:155px; border:1px solid #243b46; border-radius:7px; background:linear-gradient(100deg,#0b1820 25%,#10242d 42%,#0b1820 60%); background-size:240% 100%; animation:analysisShimmer 1.4s linear infinite; }
        .analysis-skeleton.wide { grid-column:1/-1; min-height:210px; }
        @keyframes analysisShimmer { to{background-position:-240% 0} }

        body[data-page="analyze"] .canonical-bar { gap:.45rem; padding:.2rem 0 .8rem; margin:0; }
        body[data-page="analyze"] .pill-badge { border-radius:4px; background:#10232c; border-color:#31515e; color:#9edfea; font-size:.63rem; }
        body[data-page="analyze"] .stage-rail { display:grid; grid-template-columns:repeat(4,1fr); gap:0; margin:0 0 1rem; border:1px solid #29434f; border-radius:7px; overflow:hidden; }
        body[data-page="analyze"] .stage-step { justify-content:center; min-height:44px; border:0; border-right:1px solid #29434f; border-radius:0; background:#0d1d25; text-align:center; }
        body[data-page="analyze"] .stage-step:last-child { border-right:0; }
        body[data-page="analyze"] .stage-step.done { color:#8fe2b8; background:rgba(50,126,93,.1); }
        body[data-page="analyze"] .metrics-grid { grid-template-columns:1.35fr .75fr .75fr; gap:.65rem; margin-bottom:1rem; }
        body[data-page="analyze"] .metric-tile { position:relative; overflow:hidden; min-height:92px; padding:1rem; border-color:#2b4753; border-radius:7px; background:linear-gradient(145deg,#10242d,#0c1921); text-align:left; }
        body[data-page="analyze"] .metric-tile::after { content:''; position:absolute; right:-20px; bottom:-28px; width:90px; height:90px; border:1px solid rgba(102,211,231,.09); border-radius:50%; }
        body[data-page="analyze"] .metric-number { font-size:1.55rem; letter-spacing:-.04em; }
        body[data-page="analyze"] .metric-label { margin-top:.45rem; color:#78949e; font-size:.57rem; }
        body[data-page="analyze"] .network-viz-card { border-color:#294650 !important; border-radius:8px !important; background:#08151c !important; box-shadow:inset 0 1px rgba(255,255,255,.025); }
        body[data-page="analyze"] #cyNetwork { border-color:#26434e !important; border-radius:6px !important; background:radial-gradient(circle at 50% 45%,#0f2630,#071117 72%) !important; }
        body[data-page="analyze"] .candidate-card { position:relative; border-color:#2d4b57; border-radius:8px; background:linear-gradient(120deg,#10232c,#0c1921); box-shadow:none; }
        body[data-page="analyze"] .candidate-card::before { content:''; position:absolute; top:.8rem; bottom:.8rem; left:-1px; width:2px; background:#65d2e5; }
        body[data-page="analyze"] .candidate-card:hover { transform:translateY(-2px); border-color:#61cadd; box-shadow:0 16px 36px rgba(0,8,13,.28); }
        .candidate-header { align-items:flex-start; gap:.8rem; margin-bottom:.8rem; }
        .candidate-identity { display:grid; grid-template-columns:42px minmax(0,1fr); gap:.72rem; align-items:center; min-width:0; }
        .candidate-rank { display:grid; place-items:center; width:42px; height:42px; border:1px solid #4c8290; border-radius:50%; background:#0b2028; color:#80deed; font:700 .72rem var(--font-mono); box-shadow:0 0 0 5px rgba(102,205,224,.035); }
        .candidate-overline { display:block; color:#698791; font:.52rem var(--font-mono); letter-spacing:.09em; text-transform:uppercase; margin-bottom:.18rem; }
        .candidate-target-route { display:flex; align-items:center; gap:.5rem; margin:.15rem 0 .8rem 3.35rem; color:#91abb3; font-size:.68rem; }
        .candidate-target-route strong { color:#f1fafb; }
        .candidate-target-route span { color:#5f7b85; font-family:var(--font-mono); }
        .candidate-scoreboard { display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; margin:.7rem 0; }
        .candidate-score { min-height:68px; padding:.65rem .7rem; border:1px solid #294650; border-radius:6px; background:#0a1920; }
        .candidate-score small { display:block; color:#718d97; font:.52rem var(--font-mono); letter-spacing:.06em; text-transform:uppercase; }
        .candidate-score strong { display:block; margin-top:.26rem; color:#e7f6f8; font:650 1rem var(--font-mono); }
        .candidate-score.primary strong { color:#76dced; }
        .candidate-score .progress-track { height:3px; margin:.45rem 0 0; }
        .candidate-rationale { margin-top:.7rem; padding:.72rem .8rem; border-left:2px solid #4ca8bb; background:#0b1b23; color:#b5c9ce; font-size:.76rem; line-height:1.5; }
        .candidate-source-details, .candidate-explain { border-top:1px solid #29434e; margin-top:.65rem; padding-top:.55rem; }
        .candidate-source-details summary, .candidate-explain summary { cursor:pointer; color:#78d8e8; font:.63rem var(--font-mono); }
        .candidate-source-links { display:flex; flex-wrap:wrap; gap:.38rem; margin-top:.6rem; }
        .candidate-source-links a, .candidate-source-links span { max-width:100%; overflow:hidden; text-overflow:ellipsis; padding:.28rem .42rem; border:1px solid #31515d; border-radius:4px; color:#9ed9e3 !important; background:#0b1a21; font:.56rem var(--font-mono); text-decoration:none; white-space:nowrap; }
        .candidate-source-links a:hover { border-color:#66cfdf; background:#102831; }
        @media (max-width: 1100px) {
            body[data-page="analyze"] .workstation-grid { grid-template-columns:1fr; }
            .analysis-input-panel { position:relative; top:auto; }
            .analysis-input-panel form { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
            .analysis-core-fields, .analysis-optional { margin:0; }
            .analysis-cancer-field, .analysis-form-note, .analysis-input-panel .btn-run { grid-column:1/-1; }
        }
        @media (max-width: 760px) {
            .analysis-route-shell { grid-template-columns:1fr; gap:1.2rem; }
            .analysis-route-copy h2 { font-size:clamp(2rem,12vw,3.25rem); }
            .analysis-input-panel form { display:block; }
            .analysis-optional { margin-top:.8rem; }
            .analysis-pathway-preview { grid-template-columns:1fr; gap:.9rem; margin-top:1rem; }
            .preview-route { width:1px; height:28px; justify-self:center; background:linear-gradient(#477c89,#6ed3e6); }
            .preview-route::after { right:-3px; top:auto; bottom:-1px; border-left:3px solid transparent; border-right:3px solid transparent; border-top:6px solid #67c8dc; }
            .preview-route span { left:.75rem; bottom:50%; transform:translateY(50%); }
            .analysis-empty-stage { min-height:770px; }
            .analysis-workflow-strip { grid-template-columns:1fr; }
            .analysis-workflow-step { min-height:74px; border-right:0; border-bottom:1px solid #27434e; }
            .analysis-workflow-step:last-child { border-bottom:0; }
            body[data-page="analyze"] .stage-rail { grid-template-columns:1fr 1fr; }
            body[data-page="analyze"] .stage-step:nth-child(2) { border-right:0; }
            body[data-page="analyze"] .stage-step:nth-child(-n+2) { border-bottom:1px solid #29434f; }
            body[data-page="analyze"] .metrics-grid { grid-template-columns:1fr; }
            .analysis-skeleton-grid { grid-template-columns:1fr; }
            .analysis-skeleton.wide { grid-column:auto; }
            .candidate-scoreboard { grid-template-columns:1fr; }
            .candidate-target-route { margin-left:0; }
        }

    </style>
</head>
<body data-page="home">
    <div class="container">
        <!-- Application Header Bar -->
        <header class="app-header">
            <div class="brand-area">
                <div class="brand-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M2 9c6.667 6 13.333 0 20 6"/><circle cx="7" cy="12" r="1.5" fill="#38bdf8"/><circle cx="12" cy="12" r="1.5" fill="#e11d48"/><circle cx="17" cy="12" r="1.5" fill="#38bdf8"/></svg>
                </div>
                <div>
                    <div class="brand-title">Resistance Bypass Engine</div>
                    <div class="brand-subtitle">Evidence-first oncology resistance analysis</div>
                </div>
            </div>

            <div class="header-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>5 DATA SOURCES AVAILABLE</span>
                </div>
                <a href="/" class="btn-header"><span>Home</span></a>
                <a href="/analyze" class="btn-header"><span>Analyze</span></a>
                <a href="/scenarios" class="btn-header"><span>Scenarios</span></a>
                <a href="/method" class="btn-header"><span>Method</span></a>
                <a href="/sources" class="btn-header"><span>Sources</span></a>
                <a href="https://github.com/realrezi" target="_blank" class="btn-header author-btn">
                    <span>About the project</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </a>
            </div>
        </header>

        <section class="genome-hero" aria-labelledby="hero-title">
            <div class="genome-hero-copy">
                <div class="hero-kicker">Acquired resistance / evidence review</div>
                <h1 id="hero-title">Examine resistance.<br><em>Review the evidence.</em></h1>
                <p>Enter a treatment and a resistance-related gene change. The report checks the gene names, maps related proteins, and shows the supporting records and missing evidence.</p>
                <div class="hero-actions">
                    <a class="hero-primary" href="/analyze">Start an analysis <span aria-hidden="true">→</span></a>
                    <a class="hero-secondary" href="/scenarios">Browse resistance scenarios</a>
                </div>
                <div class="hero-statline" aria-label="System capabilities">
                    <span class="hero-stat" title="HGNC, UniProt, STRING, Open Targets, and ChEMBL"><strong>05</strong> biological databases</span>
                    <span class="hero-stat" title="A change in the treated target or activation of another pathway"><strong>02</strong> resistance patterns</span>
                    <span class="hero-stat" title="A non-identifying code that can be used to reproduce the same request"><strong>01</strong> reproducible report ID</span>
                </div>
            </div>
            <div class="mutation-plate" aria-label="Evidence chain visualization">
                <div class="evidence-plate-label">How the report is built</div>
                <svg class="evidence-map-graphic" viewBox="0 0 540 240" aria-hidden="true">
                    <path class="evidence-map-helix" d="M18 38 C82 10 112 70 174 38 S272 10 330 38" />
                    <path class="evidence-map-helix" d="M18 72 C82 100 112 40 174 72 S272 100 330 72" />
                    <path class="evidence-map-line" d="M74 150 C152 112 190 183 270 143 S389 104 468 145" />
                    <path class="evidence-map-line" d="M74 150 L151 198 L270 143 L362 188 L468 145" />
                    <circle class="evidence-map-node" cx="74" cy="150" r="8" />
                    <circle class="evidence-map-node" cx="151" cy="198" r="6" />
                    <circle class="evidence-map-node core" cx="270" cy="143" r="9" />
                    <circle class="evidence-map-node" cx="362" cy="188" r="6" />
                    <circle class="evidence-map-node" cx="468" cy="145" r="8" />
                </svg>
                <div class="evidence-chain">
                    <div class="evidence-chain-step">
                        <span class="evidence-step-number">01</span>
                        <span class="evidence-step-icon" aria-hidden="true">
                            <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 3c10 5 6 21 16 26M24 3C14 8 18 24 8 29M10 8h12M9 15h14M10 22h12"/></svg>
                        </span>
                        <span class="evidence-step-copy"><strong>Check the genes</strong><small>Official gene and protein records</small></span>
                        <span class="evidence-step-source">HGNC · UniProt</span>
                    </div>
                    <div class="evidence-chain-step">
                        <span class="evidence-step-number">02</span>
                        <span class="evidence-step-icon" aria-hidden="true">
                            <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="16" r="3"/><circle cx="17" cy="7" r="3"/><circle cx="25" cy="19" r="3"/><circle cx="14" cy="26" r="2.5"/><path d="m9 14 6-5m5 1 3 6m-1 6-6 3m-3-1-5-6"/></svg>
                        </span>
                        <span class="evidence-step-copy"><strong>Map related proteins</strong><small>Known protein interactions</small></span>
                        <span class="evidence-step-source">STRING</span>
                    </div>
                    <div class="evidence-chain-step">
                        <span class="evidence-step-number">03</span>
                        <span class="evidence-step-icon" aria-hidden="true">
                            <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 4h14v24H9z"/><path d="M13 4V2h6v2M13 10h6M13 15h6M13 20h4"/><path d="m20 23 2 2 4-5"/></svg>
                        </span>
                        <span class="evidence-step-copy"><strong>Review the evidence</strong><small>Disease, drug, and study records</small></span>
                        <span class="evidence-step-source">Clinical data</span>
                    </div>
                </div>
                <div class="mutation-readout"><span>sequence / <strong>acquired resistance</strong></span><span>EGFR → MET</span></div>
            </div>
        </section>

        <section class="home-showcase academic-matrix-section" aria-labelledby="featured-scenarios-title">
            <div class="home-showcase-head">
                <div>
                    <div class="route-eyebrow">Evidence-based starting points</div>
                    <h2 id="featured-scenarios-title">Begin with a documented resistance pattern.</h2>
                </div>
                <div class="showcase-invitation"><strong>19</strong><span>reviewed examples across 11 cancer types. Start with a known resistance pattern, then examine the current database records.</span></div>
            </div>
            <div class="featured-scenario-grid">
                <a class="featured-scenario" href="/analyze?target=EGFR&amp;drug=Osimertinib&amp;marker=MET&amp;indication=Non-Small+Cell+Lung+Cancer&amp;primary_alteration=L858R&amp;resistance_alteration=MET+amplification&amp;alteration_type=amplification&amp;treatment_line=after+progression+on+osimertinib&amp;autorun=1">
                    <span class="feature-index">FEATURED / NSCLC</span>
                    <h3>EGFR inhibition → MET amplification</h3>
                    <p>Review reported MET-associated alternative signaling after progression on osimertinib.</p>
                    <div class="feature-route-map" aria-hidden="true">
                        <span class="feature-gene-node"><small>PRIMARY</small><strong>EGFR</strong></span>
                        <span class="feature-route-line"><span>after treatment</span></span>
                        <span class="feature-gene-node bypass"><small>RESISTANCE</small><strong>MET</strong></span>
                    </div>
                    <span class="feature-action">Open live analysis →</span>
                </a>
                <aside class="scenario-index-card" aria-label="Resistance example overview">
                    <span>Example library</span>
                    <strong>19 reviewed scenarios</strong>
                    <div class="scenario-index-links">
                        <a href="/scenarios?query=ESR1%20Y537S"><span>Breast · ESR1 Y537S</span><span>→</span></a>
                        <a href="/scenarios?query=FGFR2%20V565F"><span>Biliary · FGFR2 V565F</span><span>→</span></a>
                        <a href="/scenarios?query=ROS1%20G2032R"><span>Lung · ROS1 G2032R</span><span>→</span></a>
                    </div>
                    <a href="/scenarios">Explore all 11 cancer types →</a>
                </aside>
            </div>
        </section>

        <section class="route-page route-card-section analysis-route academic-matrix-section" aria-labelledby="analysis-route-title">
            <div class="analysis-route-shell">
                <div class="analysis-route-copy">
                    <div class="route-eyebrow">Evidence review / new analysis</div>
                    <h2 id="analysis-route-title">Build a resistance evidence report.</h2>
                    <p>Describe the treated target and the change observed after therapy. The report connects gene records, protein interactions, drug activity, and clinical sources without turning a research score into a treatment recommendation.</p>
                </div>
                <aside class="analysis-contract" aria-label="Report contents">
                    <div class="analysis-contract-label">Every report includes</div>
                    <strong>Sources, limits, and missing evidence</strong>
                    <p>A reproducible record of what the databases returned—and what they did not establish.</p>
                    <div class="analysis-contract-meta"><span>5 databases</span><span>timed queries</span><span>report ID</span></div>
                </aside>
            </div>
        </section>

        <section class="route-page route-card-section method-route academic-matrix-section" aria-labelledby="method-route-title">
            <div class="route-eyebrow">Method / limits</div>
            <h2 id="method-route-title">What the analysis can—and cannot—show.</h2>
            <p>The tool distinguishes a biological link from evidence that a treatment works. It can organize findings for expert review, but it cannot determine patient suitability, dose, expected response, or treatment benefit.</p>
            <div class="route-card-grid">
                <div class="route-card"><strong>Check names</strong><span>Match common gene names, such as HER2, to their official HGNC, UniProt, Ensembl, and ChEMBL records.</span></div>
                <div class="route-card"><strong>Map interactions</strong><span>Build a protein-interaction map from live databases and state when a source is unavailable.</span></div>
                <div class="route-card"><strong>Assess support</strong><span>Show what supports the result, what conflicts with it, and which drug-activity data are missing.</span></div>
            </div>
        </section>

        <section class="route-page route-card-section sources-route academic-matrix-section" aria-labelledby="sources-route-title">
            <div class="route-eyebrow">Data sources / report details</div>
            <h2 id="sources-route-title">Five biological databases are checked.</h2>
            <p>Each report shows which databases responded, when they were checked, what evidence they returned, and a non-identifying ID for repeating the same request.</p>
            <div class="route-card-grid">
                <div class="route-card"><strong>HGNC + UniProt</strong><span>Official gene names and reviewed human protein records.</span></div>
                <div class="route-card"><strong>STRING-DB</strong><span>Reported physical interactions between proteins.</span></div>
                <div class="route-card"><strong>Open Targets + ChEMBL</strong><span>Drug, disease, clinical-development, and laboratory activity records.</span></div>
            </div>
        </section>

        <section class="academic-matrix-section interpretation-strip" aria-label="How to read the analysis">
            <div class="matrix-title-text">How to read the report</div>
            <div class="interpretation-grid">
                <div><strong>Research priority</strong><span>Orders results for expert review. It is not measured drug synergy or a treatment recommendation.</span></div>
                <div><strong>Network influence</strong><span>Describes where a protein sits in the interaction map. Drugs aimed at the same protein can therefore receive the same value.</span></div>
                <div><strong>Supporting evidence</strong><span>Shows whether the result has relevant disease, drug-activity, and drug-pair records. Missing evidence remains visible.</span></div>
            </div>
        </section>

        <!-- Clinical resistance scenarios Section (Academic & Professional) -->
        <details class="scenario-library" id="scenario-library">
            <summary class="scenario-library-summary">
                <span><strong>Clinical resistance scenarios</strong><small>Browse 19 reviewed examples across 11 cancer types.</small></span>
                <span class="hero-secondary">Open library</span>
            </summary>
        <section class="academic-matrix-section scenario-library-panel">
            <div class="matrix-top-bar">
                <div class="matrix-title-text">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    <span>Reported resistance patterns</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">
                    Choose a documented resistance example from 11 cancer types. Each example fills in the gene change, prior treatment, and cancer type before opening the live analysis.
                </div>
            </div>

            <div class="matrix-tabs-container">
                <button class="btn-matrix-tab active" onclick="switchMatrixCategory('all', this)"><span class="scenario-tab-code">Library</span><span class="scenario-tab-label">All cancer types</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('nsclc', this)"><span class="scenario-tab-code">Lung</span><span class="scenario-tab-label">NSCLC</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('breast', this)"><span class="scenario-tab-code">Breast</span><span class="scenario-tab-label">Breast carcinoma</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('crc', this)"><span class="scenario-tab-code">GI</span><span class="scenario-tab-label">Colorectal cancer</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('melanoma', this)"><span class="scenario-tab-code">Skin</span><span class="scenario-tab-label">Melanoma</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('cml', this)"><span class="scenario-tab-code">Myeloid</span><span class="scenario-tab-label">CML / Ph+ ALL</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('prostate', this)"><span class="scenario-tab-code">GU</span><span class="scenario-tab-label">Prostate cancer</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('ovarian', this)"><span class="scenario-tab-code">GYN</span><span class="scenario-tab-label">Ovarian cancer</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('glioma', this)"><span class="scenario-tab-code">CNS</span><span class="scenario-tab-label">Glioma</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('thyroid', this)"><span class="scenario-tab-code">Endocrine</span><span class="scenario-tab-label">Thyroid / RET</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('gist', this)"><span class="scenario-tab-code">Sarcoma</span><span class="scenario-tab-label">GIST</span></button>
                <button class="btn-matrix-tab" onclick="switchMatrixCategory('cholangiocarcinoma', this)"><span class="scenario-tab-code">Biliary</span><span class="scenario-tab-label">Cholangiocarcinoma</span></button>

            </div>

            <div class="scenario-explorer" aria-label="Scenario explorer filters">
                <div class="scenario-explorer-head">
                    <strong>Find a resistance pattern</strong>
                    <span>19 reviewed scenarios · 11 cancer types</span>
                </div>
                <div class="scenario-filter-row">
                    <input id="scenarioSearch" type="search" placeholder="Search mutation, gene, drug, or cancer…" aria-label="Search scenarios">
                    <select id="scenarioCancerFilter" aria-label="Filter by cancer type">
                        <option value="all">All cancer types</option>
                        <option value="nsclc">NSCLC</option>
                        <option value="breast">Breast cancer</option>
                        <option value="crc">Colorectal cancer</option>
                        <option value="melanoma">Melanoma</option>
                        <option value="cml">CML / Ph+ ALL</option>
                        <option value="prostate">Prostate cancer</option>
                        <option value="ovarian">Ovarian / GYN</option>
                        <option value="glioma">Glioma / CNS</option>
                        <option value="thyroid">Thyroid / rare fusions</option>
                        <option value="gist">GIST</option>
                        <option value="cholangiocarcinoma">Cholangiocarcinoma</option>
                    </select>
                    <select id="scenarioMechanismFilter" aria-label="Filter by resistance mechanism">
                        <option value="all">All mechanisms</option>
                        <option value="bypass">Alternative signaling pathway</option>
                        <option value="feedback">Signaling reactivation</option>
                        <option value="ontarget">Change affecting drug binding</option>
                        <option value="gatekeeper">Drug-binding site change</option>
                        <option value="pathway">Return of the inhibited pathway</option>
                    </select>
                    <button type="button" class="scenario-reset" onclick="resetScenarioFilters()">Reset</button>
                </div>
                <div id="scenarioResultCount" class="scenario-result-count" aria-live="polite">Showing 19 reviewed scenarios</div>
            </div>
            <div id="scenarioEmpty" class="scenario-empty">No scenario matches those filters. Try a gene, mutation, drug, or cancer type.</div>

            <!-- Tab 1: NSCLC Scenarios -->
            <div id="matrix-nsclc" data-cancer="nsclc" class="prevalence-cards-grid">
                <div class="prevalence-card high-prev" onclick="setPreset('EGFR', 'Osimertinib', 'MET', 'Non-Small Cell Lung Cancer', 'L858R', 'MET amplification', 'amplification', 'after progression on osimertinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + MET Amplification</span>
                        <span class="badge-prevalence">MET amplification after treatment · check sample and assay</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFR) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> MET amplification may support signaling through ERBB3 and PI3K despite EGFR inhibition with osimertinib.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('EGFR', 'Osimertinib', 'EGFR', 'Non-Small Cell Lung Cancer', 'L858R', 'C797S', 'mutation', 'after progression on osimertinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFR + C797S Secondary Mutation</span>
                        <span class="badge-prevalence high">Drug-binding change · confirm the reported variant</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (Exon 20 C797S)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> EGFR C797S changes the drug-binding site and can reduce covalent binding of osimertinib.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ALK', 'Alectinib', 'MET', 'Non-Small Cell Lung Cancer', 'EML4-ALK', 'MET amplification', 'amplification', 'after progression on alectinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ALK + MET signaling</span>
                        <span class="badge-prevalence high">Alternative receptor signaling · check prior ALK therapy</span>
                    </div>
                    <div class="locus-tag">Chr 2p23.2 (ALK) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> MET activation may provide an alternative signaling route during alectinib treatment in ALK-positive NSCLC.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('ROS1', 'Crizotinib', 'ROS1', 'ROS1 fusion-positive Non-Small Cell Lung Cancer', 'ROS1 fusion', 'G2032R solvent-front mutation', 'mutation', 'after progression on a ROS1 tyrosine kinase inhibitor')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ROS1 + G2032R</span>
                        <span class="badge-prevalence">Solvent-front mutation · inspect TKI history</span>
                    </div>
                    <div class="locus-tag">Chr 6q22.1 (ROS1 G2032R)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> A solvent-front substitution can impair binding of earlier ROS1 inhibitors. Drug sensitivity remains inhibitor-specific and must be checked against the current clinical record.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10283448/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary translational evidence ↗</a>
                </div>
            </div>

            <!-- Tab 2: Breast Cancer Scenarios -->
            <div id="matrix-breast" data-cancer="breast" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('HER2', 'Trastuzumab', 'MET', 'HER2+ Breast Cancer', 'ERBB2 amplification', 'MET amplification', 'amplification', 'after progression on trastuzumab')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">HER2 + MET Amplification</span>
                        <span class="badge-prevalence">Possible MET-associated resistance · confirm HER2 and MET status</span>
                    </div>
                    <div class="locus-tag">Chr 17q12 (ERBB2) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> Increased MET signaling may provide an alternative pathway during trastuzumab treatment.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('ESR1', 'Fulvestrant', 'CDK4', 'HR+/HER2- Breast Cancer', 'ESR1 mutation', 'CDK4/6 pathway activation', 'activation', 'after endocrine therapy progression')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ESR1 + CDK4/6 signaling</span>
                        <span class="badge-prevalence high">After endocrine therapy · specify prior treatment</span>
                    </div>
                    <div class="locus-tag">Chr 6q25.1 (ESR1) ➔ Chr 12q14.1 (CDK4)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> Ligand-independent ESR1 variants and Cyclin D1/CDK4 activity may support growth after aromatase-inhibitor treatment.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('ESR1', 'Letrozole', 'ESR1', 'HR-positive metastatic Breast Cancer', 'estrogen receptor-positive disease', 'Y537S ligand-binding-domain mutation', 'mutation', 'after progression on aromatase-inhibitor therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">ESR1 + Y537S</span>
                        <span class="badge-prevalence">Ligand-independent activation · variant-specific</span>
                    </div>
                    <div class="locus-tag">Chr 6q25.1 (ESR1 Y537S)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Y537S stabilizes an active estrogen-receptor conformation and is associated with acquired endocrine resistance; it does not establish sensitivity to a particular subsequent regimen.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4821807/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary structural evidence ↗</a>
                </div>

            </div>

            <!-- Tab 3: Colorectal Cancer Scenarios -->
            <div id="matrix-crc" data-cancer="crc" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('KRAS', 'Sotorasib', 'EGFR', 'Colorectal Cancer', 'G12C', 'EGFR feedback activation', 'activation', 'after progression on KRAS G12C inhibition')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">KRAS G12C + EGFR signaling</span>
                        <span class="badge-prevalence">Return of EGFR signaling · check prior KRAS therapy</span>
                    </div>
                    <div class="locus-tag">Chr 12p12.1 (KRAS) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> EGFR activity can restore MAPK signaling during KRAS G12C inhibition, providing a reason to study both targets together.</div>
                </div>

                <div class="prevalence-card approved-prev" onclick="setPreset('BRAF', 'Encorafenib', 'EGFR', 'Colorectal Cancer', 'V600E', 'EGFR feedback activation', 'activation', 'specified metastatic BRAF V600E CRC indication')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BRAF V600E + EGFR signaling</span>
                        <span class="badge-prevalence high">Approved only in specified clinical settings</span>
                    </div>
                    <div class="locus-tag">Chr 7q34 (BRAF) ➔ Chr 7p11.2 (EGFR)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> EGFR activity can restore MAPK signaling during BRAF inhibition. Encorafenib plus cetuximab is FDA-authorized in specified BRAF V600E metastatic colorectal cancer settings.</div>
                </div>
            </div>

            <!-- Tab 4: Cutaneous Melanoma Scenarios -->
            <div id="matrix-melanoma" data-cancer="melanoma" class="prevalence-cards-grid" style="display: none;">
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
            <div id="matrix-cml" data-cancer="cml" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card high-prev" onclick="setPreset('ABL1', 'Imatinib', 'ABL1', 'Chronic Myeloid Leukemia', 'BCR-ABL1', 'T315I', 'mutation', 'after progression on first/second-generation TKI')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">BCR–ABL1 + T315I</span>
                        <span class="badge-prevalence">Drug-binding change · specify previous TKIs</span>
                    </div>
                    <div class="locus-tag">Chr 9q34.12 (ABL1 T315I Gatekeeper)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Threonine-to-isoleucine substitution alters the kinase binding site; ponatinib or asciminib are approved options in specified CML/Ph+ ALL settings, while combination use requires its own evidence.</div>
                </div>
            </div>

            <!-- Tab 6: Prostate Cancer Scenarios -->
            <div id="matrix-prostate" data-cancer="prostate" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('AR', 'Enzalutamide', 'PIK3CA', 'Metastatic Castration-Resistant Prostate Cancer', 'AR alteration', 'PIK3CA/PTEN pathway alteration', 'activation', 'after progression on androgen-receptor pathway inhibition')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">AR + PI3K signaling</span>
                        <span class="badge-prevalence">Confirm PTEN/PI3K test result and prior therapy</span>
                    </div>
                    <div class="locus-tag">Chr Xq12 (AR) ➔ Chr 3q26.32 (PIK3CA)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> Androgen-receptor and PI3K–AKT signaling can regulate one another in metastatic castration-resistant prostate cancer.</div>
                </div>
            </div>

            <!-- Tab 7: Ovarian & GYN Scenarios -->
            <div id="matrix-ovarian" data-cancer="ovarian" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('PIK3CA', 'Alpelisib', 'KRAS', 'Ovarian Cancer', 'PIK3CA alteration', 'KRAS alteration', 'mutation', 'after progression on PI3K-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">PIK3CA + KRAS signaling</span>
                        <span class="badge-prevalence">RAS/MAPK co-alteration • confirm tumor subtype</span>
                    </div>
                    <div class="locus-tag">Chr 3q26.32 (PIK3CA) ➔ Chr 12p12.1 (KRAS)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> RAS–MAPK activation may maintain signaling during selective PI3Kα inhibition.</div>
                </div>

                <div class="prevalence-card" onclick="setPreset('PARP1', 'Olaparib', 'BRCA2', 'BRCA-mutated Ovarian Cancer', 'BRCA2 pathogenic loss-of-function alteration', 'BRCA2 reversion restoring the open reading frame', 'mutation', 'after progression on PARP-inhibitor therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">PARP inhibition + BRCA2 reversion</span>
                        <span class="badge-prevalence">HR restoration · confirm paired sequencing</span>
                    </div>
                    <div class="locus-tag">Chr 13q13.1 (BRCA2 reversion)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> A secondary reversion can restore BRCA2 open-reading-frame function and homologous recombination, providing a documented route to PARP-inhibitor resistance.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4991495/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary mechanistic evidence ↗</a>
                </div>
            </div>

            <!-- Tab 8: Glioma & CNS Scenarios -->
            <div id="matrix-glioma" data-cancer="glioma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('EGFR', 'Gefitinib', 'MET', 'Glioblastoma', 'EGFRvIII', 'MET amplification', 'amplification', 'after progression on EGFR-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">EGFRvIII + MET Amplification</span>
                        <span class="badge-prevalence">Confirm EGFR and MET results in the relevant sample</span>
                    </div>
                    <div class="locus-tag">Chr 7p11.2 (EGFRvIII) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> Concurrent signaling through EGFRvIII and MET may reduce dependence on EGFR alone in high-grade glioma.</div>
                </div>
            </div>

            <!-- Tab 9: Thyroid & Rare Fusions Scenarios -->
            <div id="matrix-thyroid" data-cancer="thyroid" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('RET', 'Selpercatinib', 'MET', 'Thyroid Cancer', 'RET fusion', 'MET amplification', 'amplification', 'after progression on selpercatinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">RET fusion + MET signaling</span>
                        <span class="badge-prevalence">Possible MET-associated resistance · confirm progression on RET therapy</span>
                    </div>
                    <div class="locus-tag">Chr 10q11.21 (RET) ➔ Chr 7q31.2 (MET)</div>
                    <div class="scenario-mechanism"><strong>Proposed mechanism:</strong> MET amplification has been reported after treatment with the selective RET inhibitor selpercatinib.</div>
                </div>
                <div class="prevalence-card" onclick="setPreset('RET', 'Selpercatinib', 'RET', 'RET-altered Thyroid Cancer', 'RET fusion or activating mutation', 'G810R/S/C solvent-front mutation', 'mutation', 'after progression on selective RET inhibition')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">RET + G810 solvent-front mutation</span>
                        <span class="badge-prevalence">On-target binding change · substitution-specific</span>
                    </div>
                    <div class="locus-tag">Chr 10q11.21 (RET G810R/S/C)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> G810 solvent-front substitutions can sterically interfere with selective RET-inhibitor binding. The returned report must not be read as evidence for a replacement therapy.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7430178/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary clinical evidence ↗</a>
                </div>
            </div>

            <!-- Tab 10: Gastrointestinal Stromal Tumor -->
            <div id="matrix-gist" data-cancer="gist" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('KIT', 'Imatinib', 'KIT', 'Gastrointestinal Stromal Tumor', 'KIT exon 11 activating mutation', 'V654A secondary ATP-pocket mutation', 'mutation', 'after progression on imatinib')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">KIT + V654A</span>
                        <span class="badge-prevalence">Secondary ATP-pocket mutation · exon-specific</span>
                    </div>
                    <div class="locus-tag">Chr 4q12 (KIT V654A)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> The secondary KIT V654A substitution alters the ATP-binding pocket and is associated with acquired imatinib resistance in GIST.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7718339/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary model evidence ↗</a>
                </div>
            </div>

            <!-- Tab 11: FGFR2-altered Cholangiocarcinoma -->
            <div id="matrix-cholangiocarcinoma" data-cancer="cholangiocarcinoma" class="prevalence-cards-grid" style="display: none;">
                <div class="prevalence-card" onclick="setPreset('FGFR2', 'Pemigatinib', 'FGFR2', 'FGFR2 fusion-positive Intrahepatic Cholangiocarcinoma', 'FGFR2 fusion or rearrangement', 'V565F gatekeeper mutation', 'mutation', 'after progression on FGFR-directed therapy')">
                    <div class="prevalence-header">
                        <span class="scenario-pair-title">FGFR2 + V565F</span>
                        <span class="badge-prevalence">Gatekeeper mutation · polyclonal resistance possible</span>
                    </div>
                    <div class="locus-tag">Chr 10q26.13 (FGFR2 V565F)</div>
                    <div class="scenario-mechanism"><strong>Mechanism:</strong> Secondary FGFR2 kinase-domain mutations, including V565 gatekeeper substitutions, are documented after FGFR-inhibitor therapy in FGFR2-altered cholangiocarcinoma.</div>
                    <a class="scenario-evidence-link" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10767308/" target="_blank" rel="noopener" onclick="event.stopPropagation()">Primary resistance landscape ↗</a>
                </div>
            </div>

        </section>
        </details>

        <!-- Workstation Grid -->
        <div class="workstation-grid" id="workstation">
            <!-- Left Panel: Form & Molecular Target Selector -->
            <div class="panel analysis-input-panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
                        <span><small class="panel-kicker">01 / Define the case</small>Treatment and resistance details</span>
                    </div>
                </div>

                <form id="analyzeForm" onsubmit="runAnalysis(event)">
                    <div class="analysis-core-fields">
                    <div class="analysis-field-card">
                        <span class="field-sequence">A</span>
                        <div class="form-group">
                            <label class="field-label" for="primary_target">Primary target gene</label>
                            <input class="input-field" type="text" id="primary_target" value="EGFR" required placeholder="e.g. EGFR or ERBB2" list="targets_list" autocomplete="off">
                        </div>
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
                            <option value="KIT">KIT Receptor Tyrosine Kinase (Chr 4q12)</option>
                            <option value="FGFR2">Fibroblast Growth Factor Receptor 2 (Chr 10q26.13)</option>
                            <option value="PARP1">Poly(ADP-Ribose) Polymerase 1 (Chr 1q42.12)</option>
                            <option value="BRCA2">BRCA2 DNA Repair Associated (Chr 13q13.1)</option>
                        </datalist>
                    </div>

                    <div class="analysis-field-card">
                        <span class="field-sequence">Rx</span>
                        <div class="form-group">
                            <label class="field-label" for="primary_drug">Targeted therapy</label>
                            <input class="input-field" type="text" id="primary_drug" value="Osimertinib" required placeholder="e.g. osimertinib" list="drugs_list" autocomplete="off">
                        </div>
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
                            <option value="Crizotinib">Crizotinib (ALK / ROS1 / MET TKI)</option>
                            <option value="Letrozole">Letrozole (Aromatase Inhibitor)</option>
                            <option value="Olaparib">Olaparib (PARP Inhibitor)</option>
                            <option value="Selpercatinib">Selpercatinib (Selective RET Inhibitor)</option>
                            <option value="Pemigatinib">Pemigatinib (FGFR1–3 Inhibitor)</option>
                        </datalist>
                    </div>

                    <div class="analysis-field-card resistance">
                        <span class="field-sequence">B</span>
                        <div class="form-group">
                            <label class="field-label" for="resistance_marker">Resistance-associated gene</label>
                            <input class="input-field" type="text" id="resistance_marker" value="MET" required placeholder="e.g. MET or a secondary EGFR change" list="markers_list" autocomplete="off">
                        </div>
                        <datalist id="markers_list">
                            <option value="MET">MET amplification / alternative pathway activation</option>
                            <option value="EGFR">EGFR resistance variants (C797S / T790M)</option>
                            <option value="KRAS">KRAS Secondary Activation (G12C / G12V)</option>
                            <option value="BRAF">BRAF V600E Activation</option>
                            <option value="PIK3CA">PIK3CA activating mutation (H1047R)</option>
                            <option value="MAP2K1">MAP2K1 / MEK1 Activation</option>
                            <option value="CDK4">Cyclin D–CDK4/6 pathway activation</option>
                            <option value="ABL1">ABL1 drug-binding change (T315I)</option>
                            <option value="ROS1">ROS1 Solvent-Front Mutation (G2032R)</option>
                            <option value="RET">RET Solvent-Front Mutation (G810 substitutions)</option>
                            <option value="KIT">KIT Secondary Mutation (V654A / N822K)</option>
                            <option value="FGFR2">FGFR2 drug-binding change (V565 substitutions)</option>
                            <option value="BRCA2">BRCA2 Reversion / Homologous-Recombination Restoration</option>
                        </datalist>
                    </div>
                    </div>

                    <details class="analysis-optional" open>
                    <summary>Describe the observed changes <span class="analysis-optional-copy">optional but useful</span></summary>
                    <div class="analysis-optional-fields">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="field-label" for="primary_alteration">Initial gene change <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="primary_alteration" placeholder="e.g. L858R, G12C, fusion" autocomplete="off">
                        </div>
                        <div class="form-group">
                            <label class="field-label" for="resistance_alteration">Resistance-related change <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="resistance_alteration" placeholder="e.g. amplification, C797S" autocomplete="off">
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="field-label" for="resistance_alteration_type">Type of gene change <span style="color:#64748b;">(optional)</span></label>
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
                            <label class="field-label" for="treatment_line">When resistance was identified <span style="color:#64748b;">(optional)</span></label>
                            <input class="input-field" type="text" id="treatment_line" placeholder="e.g. after progression on osimertinib" autocomplete="off">
                        </div>
                    </div>
                    </div>
                    </details>

                    <div class="form-group analysis-cancer-field">
                        <label class="field-label" for="cancer_type">Cancer type</label>
                        <input class="input-field" type="text" id="cancer_type" value="Non-Small Cell Lung Cancer" placeholder="Select cancer type..." list="indications_list" autocomplete="off">
                        <datalist id="indications_list">
                            <option value="Non-Small Cell Lung Cancer">Non-Small Cell Lung Cancer (NSCLC)</option>
                            <option value="HER2+ Breast Cancer">HER2+ Breast Cancer</option>
                            <option value="Colorectal Cancer">Colorectal Cancer (CRC)</option>
                            <option value="Cutaneous Melanoma">Cutaneous Melanoma</option>
                            <option value="Chronic Myeloid Leukemia">Chronic Myeloid Leukemia (CML)</option>
                        </datalist>
                    </div>

                    <p class="analysis-form-note">Use this report to organize research evidence. It does not determine treatment suitability or predict response.</p>

                    <button type="submit" id="submitBtn" class="btn-run">
                        <span>Analyze resistance evidence</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>
            </div>


            <!-- Right Panel: Signal Transduction Pathway & Evidence Priority Matrix -->
            <div class="panel analysis-output-panel">
                <div class="panel-header">
                    <div class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                        <span><small class="panel-kicker">02 / Evidence report</small>Protein interactions and research priorities</span>
                    </div>
                    <div style="display:flex; gap:0.45rem; flex-wrap:wrap; justify-content:flex-end;">
                        <button id="compareBtn" class="btn-header" style="display: none; color: #fff;" onclick="compareReports()">Compare last two</button>
                        <button id="copyJsonBtn" class="btn-header" style="display: none; color: #fff;" onclick="copyResultJson()">Copy report JSON</button>
                    </div>
                </div>

                <div class="analysis-output-body">

                <div id="errorBanner" style="display:none; background:rgba(244, 63, 94, 0.15); border:1px solid rgba(244, 63, 94, 0.3); padding:0.85rem; border-radius:8px; color:#fb7185; margin-bottom:1rem; font-size:0.85rem;"></div>
                <div id="warningBanner" style="display:none; background:rgba(251, 191, 36, 0.12); border:1px solid rgba(251, 191, 36, 0.35); padding:0.85rem; border-radius:8px; color:#fde68a; margin-bottom:1rem; font-size:0.85rem;"></div>

                <div id="loader" class="analysis-loader" style="display: none;">
                    <div class="analysis-loader-head"><strong>Building the evidence report</strong><span>LIVE DATABASE REVIEW</span></div>
                    <div class="analysis-skeleton-grid" aria-hidden="true">
                        <div class="analysis-skeleton wide"></div>
                        <div class="analysis-skeleton"></div>
                        <div class="analysis-skeleton"></div>
                    </div>
                </div>

                <!-- Multi-Kinase Cell Membrane SVG Signaling Visualizer (Professional & Dynamic) -->
                <div id="placeholder">
                    <div class="analysis-empty-stage">
                        <div class="analysis-empty-head">
                            <div><span>Case preview</span><p>The report will follow this treatment-to-resistance relationship.</p></div>
                            <div class="analysis-ready-state">Ready to analyze</div>
                        </div>
                        <div class="analysis-pathway-preview" aria-label="Selected treatment and resistance relationship">
                            <div class="preview-node"><small>TARGET</small><strong id="previewPrimary">EGFR</strong><em>treated protein</em></div>
                            <div class="preview-route"><span>targeted by</span></div>
                            <div class="preview-drug"><div class="preview-drug-mark">Rx</div><strong id="previewDrug">Osimertinib</strong></div>
                            <div class="preview-route"><span>resistance observed</span></div>
                            <div class="preview-node resistance"><small>CHANGE</small><strong id="previewResistance">MET</strong><em id="previewCancer">NSCLC context</em></div>
                        </div>
                        <div class="analysis-workflow-strip" aria-label="Analysis workflow">
                            <div class="analysis-workflow-step"><b>01</b><strong>Confirm the records</strong><span>Match gene and protein names to reviewed database entries.</span><small>HGNC · UNIPROT</small></div>
                            <div class="analysis-workflow-step"><b>02</b><strong>Retrieve interactions</strong><span>Build a connected map from reported protein interactions.</span><small>STRING</small></div>
                            <div class="analysis-workflow-step"><b>03</b><strong>Review the support</strong><span>Separate network findings from drug and clinical evidence.</span><small>OPEN TARGETS · CHEMBL</small></div>
                        </div>
                    </div>
                    <div class="vector-graph-canvas" style="background: #030712; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; color: #fff; text-align: center; position: relative;">
                        <div class="home-workflow" aria-label="Analysis workflow">
                            <div class="home-workflow-step"><span style="color:var(--genomic-blue);font-family:var(--font-mono);">01 / CHECK</span><strong>Official gene records</strong><span>Match common gene names to HGNC, UniProt, Ensembl, and ChEMBL records.</span></div>
                            <div class="home-workflow-step"><span style="color:var(--mutation-red);font-family:var(--font-mono);">02 / MAP</span><strong>Protein interactions</strong><span>Retrieve reported protein interactions and relevant drug and disease records.</span></div>
                            <div class="home-workflow-step"><span style="color:var(--approved-green);font-family:var(--font-mono);">03 / REVIEW</span><strong>Strength of evidence</strong><span>Keep the research score separate from clinical evidence and clearly show missing data.</span></div>
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
                            <text x="520" y="10" text-anchor="middle" fill="#f43f5e" font-size="10" font-weight="700">MET · chromosome 7q31.2</text>

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
                            <text x="520" y="209" text-anchor="middle" fill="#f43f5e" font-size="9" font-weight="700">Alternative Signaling</text>
                        </svg>
                        <p style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; margin-top: 0.5rem;">How the analysis works</p>
                        <p style="font-size: 0.83rem; color: #94a3b8; margin-top: 0.2rem;">Run an analysis to view reported protein interactions, evidence records, and any linked experimental protein structure.</p>
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
                        <span class="stage-step done">01 · Gene records confirmed</span>
                        <span class="stage-step done">02 · Interactions retrieved</span>
                        <span class="stage-step done">03 · Evidence reviewed</span>
                        <span class="stage-step done">04 · Research priorities calculated</span>
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
                            <div class="metric-label" title="Shortest calculated route between the two proteins in the retrieved interaction network">Network Distance</div>
                        </div>
                    </div>

                    <!-- Interactive Cytoscape Network Visualizer -->
                    <div class="network-viz-card" style="background: #090d16; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <div style="display:flex; align-items:center; gap:0.5rem;">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 0-7.07 17.07"/></svg>
                                <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">Reported protein interaction network</span>
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
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#059669; border: 2px solid #34d399;"></span> Additional Drug Target</span>
                            <span style="display:inline-flex; align-items:center; gap:0.35rem;"><span style="width:10px; height:10px; border-radius:50%; background:#7c3aed; border: 2px solid #a855f7;"></span> Intermediate Protein</span>
                        </div>
                    </div>

                    <div id="structureWorkspace" class="network-viz-card" style="background: linear-gradient(135deg, #0b1218, #0b111b); border: 1px solid #1c3b4b; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; margin-bottom:.75rem;">
                            <div>
                                <div style="display:flex; align-items:center; gap:.5rem;">
                                    <span style="color:var(--genomic-blue); font-size:1.1rem;">⌬</span>
                                    <span style="font-weight:800; font-size:.95rem; color:var(--text-main);">Experimental protein structure</span>
                                </div>
                                <p style="font-size:.78rem; color:var(--text-muted); margin-top:.25rem;">Inspect an experimental structure linked to the selected protein. If none is available, the report states this clearly.</p>
                            </div>
                            <div id="structureStatusBadge" class="evidence-badge caution">Select a node</div>
                        </div>
                        <div style="display:grid; grid-template-columns:minmax(0, 1fr) 220px; gap:1rem; align-items:stretch;">
                            <div id="structureWorkspaceViewer" style="height:360px; min-height:280px; width:100%; position:relative; background:#070a0f; border:1px solid #1c2a35; border-radius:9px; overflow:hidden; display:flex; align-items:center; justify-content:center; color:var(--text-muted); font-size:.82rem;">Run an analysis to check for a linked experimental structure.</div>
                            <div id="structureWorkspaceMeta" style="border:1px solid #1c2a35; border-radius:9px; padding:.85rem; background:#0d131a; font-size:.77rem; color:var(--text-secondary);">
                                <div style="font-family:var(--font-mono); color:var(--genomic-blue); font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;">Structure record</div>
                                <div id="structureWorkspaceTarget" style="font-weight:800; color:var(--text-main); font-size:1.2rem; margin:.55rem 0 .25rem;">—</div>
                                <div id="structureWorkspacePdb" style="font-family:var(--font-mono); color:var(--approved-green);">PDB —</div>
                                <p id="structureWorkspaceNote" style="margin-top:.7rem; line-height:1.45;">This panel shows whether a reviewed PDB record is linked to the selected protein.</p>
                                <button type="button" class="btn-header" style="margin-top:.8rem; width:100%; justify-content:center; color:var(--genomic-blue);" onclick="openSelectedNodeInspector()">View protein details</button>
                            </div>
                        </div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <h3 style="font-size:1rem; font-weight:800; color:var(--text-main);">Potential additional therapies for research review</h3>
                    </div>

                    <div id="therapiesList"></div>
                </div>
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
                <div id="guidanceHeading" class="modal-heading" style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">Purpose and method</div>
                <button class="modal-close" style="color: #94a3b8;" onclick="toggleModal('guidanceModal', false)">&times;</button>
            </div>
            <p style="margin-bottom: 1rem; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                This research tool examines acquired resistance using current records from HGNC, UniProt, STRING-DB, Open Targets, and ChEMBL. It organizes those records with a reproducible protein-network analysis.
            </p>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px; margin-bottom: 0.75rem;">
                <div style="font-weight: 700; color: #38bdf8;">1. Confirm official gene records</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Matches common names, such as HER2, to the official gene symbol ERBB2 and its reviewed database records.</div>
            </div>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px; margin-bottom: 0.75rem;">
                <div style="font-weight: 700; color: #c084fc;">2. Build the protein-interaction map</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Retrieves reported protein interactions from STRING-DB and analyzes the connected part of the network that contains usable data.</div>
            </div>
            <div style="background: #1e293b; border: 1px solid #334155; padding: 0.85rem; border-radius: 8px;">
                <div style="font-weight: 700; color: #34d399;">3. Calculate research priorities</div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.2rem;">Combines network position, distance, and available drug records. The result supports research review; it does not measure drug synergy or clinical benefit.</div>
            </div>
        </div>
    </div>

    <div id="clinicianModal" class="modal-wrapper" role="dialog" aria-modal="true" aria-hidden="true" aria-labelledby="clinicianHeading" onclick="if(event.target===this) toggleModal('clinicianModal', false)">
        <div class="modal-box" style="max-width: 600px; background: #0f172a; border: 1px solid #334155; border-radius: 14px; color: #f8fafc;">
            <div class="modal-top" style="border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; margin-bottom: 1rem;">
                <div id="clinicianHeading" class="modal-heading" style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">Clinical interpretation guide</div>
                <button class="modal-close" style="color: #94a3b8;" onclick="toggleModal('clinicianModal', false)">&times;</button>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: #cbd5e1;">
                <p style="margin-bottom: 0.75rem;"><strong style="color: #38bdf8;">Alternative-pathway resistance:</strong> Another signaling route, such as MET activation, may maintain growth signals despite treatment of the original target.</p>
                <p style="margin-bottom: 0.75rem;"><strong style="color: #f43f5e;">Change in the treated target:</strong> A secondary alteration in the original target, such as EGFR C797S or ABL1 T315I, may reduce drug binding.</p>
                <p><strong style="color: #34d399;">Network position score:</strong> Estimates how strongly a protein connects different parts of the retrieved interaction network while reducing the influence of proteins that connect to almost everything. It is a research measure, not clinical evidence.</p>
            </div>
        </div>
    </div>


    <!-- Protein details modal -->
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
                <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 0.35rem;">Biological function</div>
                <div id="nodeModalRole" style="color: #f8fafc; font-weight: 600;">Receptor Tyrosine Kinase (RTK) Initiator</div>
            </div>

            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 0.5rem;">Reviewed resistance notes (not live COSMIC frequencies)</div>
                <div id="nodeModalHotspots" style="display: flex; flex-direction: column; gap: 0.4rem;"></div>
            </div>

            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.72rem;">Experimental protein structure (PDB: <span id="nodeModalPdbTag" style="color:#34d399;">1M17</span>)</div>
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
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.2rem;">UniProt ID</div>
                </div>
                <div>
                    <div id="nodeModalDegree" style="font-size: 0.85rem; font-family: monospace; font-weight: 700; color: #34d399;">14 Edges</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.2rem;">Network Connections</div>
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
        const currentPage = document.body.dataset.page || 'home';
        if (currentPage === 'scenarios') {
            document.getElementById('scenario-library')?.setAttribute('open', '');
        }

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
            syncAnalysisPreview();
        }

        function setPreset(target, drug, marker, indication, primaryAlteration, resistanceAlteration, alterationType, treatmentLine) {
            if (currentPage !== 'analyze') {
                const params = new URLSearchParams({
                    target, drug, marker, indication: indication || '',
                    primary_alteration: primaryAlteration || '', resistance_alteration: resistanceAlteration || '',
                    alteration_type: alterationType || '', treatment_line: treatmentLine || '', autorun: '1'
                });
                window.location.href = `/analyze?${params.toString()}`;
                return;
            }
            document.getElementById('primary_target').value = target;
            document.getElementById('primary_drug').value = drug;
            document.getElementById('resistance_marker').value = marker;
            if (indication) document.getElementById('cancer_type').value = indication;
            document.getElementById('primary_alteration').value = primaryAlteration || '';
            document.getElementById('resistance_alteration').value = resistanceAlteration || '';
            document.getElementById('resistance_alteration_type').value = alterationType || '';
            document.getElementById('treatment_line').value = treatmentLine || '';
            syncAnalysisPreview();
            executePipeline();
        }

        const scenarioCategories = ['nsclc', 'breast', 'crc', 'melanoma', 'cml', 'prostate', 'ovarian', 'glioma', 'thyroid', 'gist', 'cholangiocarcinoma'];
        const scenarioCategoryLabels = {
            nsclc: 'NSCLC', breast: 'Breast cancer', crc: 'Colorectal cancer', melanoma: 'Melanoma',
            cml: 'CML / Ph+ ALL', prostate: 'Prostate cancer', ovarian: 'Ovarian / GYN',
            glioma: 'Glioma / CNS', thyroid: 'Thyroid / rare fusions', gist: 'GIST',
            cholangiocarcinoma: 'Cholangiocarcinoma'
        };
        const scenarioCategoryColors = {
            all: '#6fd9eb', nsclc: '#6fd9eb', breast: '#e78ca7', crc: '#d5a269', melanoma: '#9d8bd8',
            cml: '#e56f78', prostate: '#75a9df', ovarian: '#d58ccf', glioma: '#a29be8',
            thyroid: '#79cbb9', gist: '#daa36f', cholangiocarcinoma: '#b8c86f'
        };
        const scenarioOrganIcons = {
            all: '<path d="M17 9c20 9 10 37 30 46M47 9C27 18 37 46 17 55M21 17h22M18 27h28M18 37h28M21 47h22"/>',
            nsclc: '<path d="M31 8v18M31 17c-5 0-8 4-10 9M33 17c5 0 8 4 10 9"/><path d="M27 18c-9-2-15 8-15 21 0 8 5 12 12 9 5-2 6-8 6-15V18M37 18c9-2 15 8 15 21 0 8-5 12-12 9-5-2-6-8-6-15V18"/>',
            breast: '<circle cx="32" cy="32" r="19"/><circle cx="32" cy="32" r="5"/><path d="M32 13v14M16 24l12 6M48 24l-12 6M18 43l10-7M46 43l-10-7"/>',
            crc: '<path d="M18 13c-5 0-7 4-7 9v19c0 6 4 10 10 10h22c6 0 10-4 10-10V22c0-5-2-9-7-9M22 13v9c0 3 2 5 5 5h10c3 0 5-2 5-5v-9M22 51v-9c0-3 2-5 5-5h10c3 0 5 2 5 5v9"/>',
            melanoma: '<path d="M8 24h48M8 34h48M8 44h48"/><path d="M15 18c4-7 8-7 12 0s8 7 12 0 8-7 12 0"/><circle cx="32" cy="29" r="3"/><path d="m32 32-5 8m5-8 5 8"/>',
            cml: '<circle cx="21" cy="24" r="10"/><circle cx="42" cy="20" r="7"/><circle cx="40" cy="43" r="11"/><circle cx="17" cy="45" r="5"/><path d="M35 39c4-4 8-4 11 0"/>',
            prostate: '<path d="M24 10h16v12c0 5-3 9-8 9s-8-4-8-9V10Z"/><path d="M32 31v7"/><path d="M20 42c0-5 4-8 12-8s12 3 12 8-5 10-12 10-12-5-12-10Z"/>',
            ovarian: '<path d="M32 19v28M22 26c0 8 4 12 10 12M42 26c0 8-4 12-10 12"/><path d="M22 28c-6 0-10-4-10-9M42 28c6 0 10-4 10-9"/><circle cx="10" cy="17" r="5"/><circle cx="54" cy="17" r="5"/><path d="M26 47h12"/>',
            glioma: '<path d="M29 12c-7-5-15 0-14 8-7 2-7 12-1 15-4 7 3 15 10 12 3 7 13 5 13-2 8 3 15-5 10-12 7-5 4-14-3-16 0-8-10-12-15-7Z"/><path d="M32 13v34M20 22c5 0 8 3 8 7M44 21c-5 0-8 3-8 7M19 39c5 0 8-3 8-7M45 39c-5 0-8-3-8-7"/>',
            thyroid: '<path d="M29 24c-4-8-14-10-17-3-4 9 4 23 17 17M35 24c4-8 14-10 17-3 4 9-4 23-17 17"/><path d="M29 22h6v19h-6zM32 12v10"/>',
            gist: '<path d="M23 10c0 10-2 16-8 20-7 5-5 19 5 23 12 5 27-4 29-18 1-8-5-12-13-10-8 2-10-4-13-15Z"/><path d="M24 19c5 6 9 9 16 10"/>',
            cholangiocarcinoma: '<path d="M10 20c11-9 31-11 44 1-4 15-14 24-27 24-10 0-17-7-17-25Z"/><path d="M33 15v22m0-10 9 6m-9-2-8 7"/><path d="M42 33v12c0 5 3 7 7 7"/>'
        };
        const scenarioCategoryOrder = ['all', ...scenarioCategories];
        document.querySelectorAll('.scenario-library-panel .btn-matrix-tab').forEach((button, index) => {
            const category = scenarioCategoryOrder[index] || 'all';
            button.dataset.scenarioCategory = category;
            button.style.setProperty('--indication-accent', scenarioCategoryColors[category] || '#6fd9eb');
            const organMark = document.createElement('span');
            organMark.className = 'scenario-tab-organ';
            organMark.setAttribute('aria-hidden', 'true');
            organMark.innerHTML = `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${scenarioOrganIcons[category] || scenarioOrganIcons.all}</svg>`;
            button.appendChild(organMark);
        });
        let activeScenarioCategory = 'all';

        function switchMatrixCategory(cat, btn) {
            activeScenarioCategory = cat;
            document.querySelectorAll('.btn-matrix-tab').forEach(b => b.classList.remove('active'));
            const activeButton = btn || document.querySelector(`[data-scenario-category="${cat}"]`);
            activeButton?.classList.add('active');
            document.querySelectorAll('.scenario-filter-row select').forEach(select => {
                if (select.id === 'scenarioCancerFilter' && select.value !== 'all' && select.value !== cat) select.value = 'all';
            });
            scenarioCategories.forEach(c => {
                const el = document.getElementById('matrix-' + c);
                if (el) el.style.display = (cat === 'all' || c === cat) ? 'grid' : 'none';
            });
            document.querySelector('.scenario-library-panel')?.classList.remove('filtering');
            document.querySelectorAll('.prevalence-card').forEach(card => card.classList.remove('is-hidden'));
            document.getElementById('scenarioEmpty')?.style.setProperty('display', 'none');
            const count = cat === 'all' ? document.querySelectorAll('.prevalence-card').length : document.querySelectorAll(`#matrix-${cat} .prevalence-card`).length;
            const counter = document.getElementById('scenarioResultCount');
            if (counter) counter.textContent = `Showing ${count} reviewed scenario${count === 1 ? '' : 's'} · ${cat === 'all' ? 'all cancer types' : scenarioCategoryLabels[cat]}`;
        }

        function mechanismMatches(text, filter) {
            if (filter === 'all') return true;
            if (filter === 'bypass') return /bypass|crosstalk|redundan/.test(text);
            if (filter === 'feedback') return /feedback|reactivation/.test(text);
            if (filter === 'ontarget') return /on-target|covalent|secondary mutation/.test(text);
            if (filter === 'gatekeeper') return /gatekeeper|binding site|binding change/.test(text);
            if (filter === 'pathway') return /pathway|mapk|endocrine|pi3k/.test(text);
            return true;
        }

        function filterScenarioCards() {
            const search = (document.getElementById('scenarioSearch')?.value || '').trim().toLowerCase();
            const searchTokens = search.split(/\\s+/).filter(Boolean);
            const cancer = document.getElementById('scenarioCancerFilter')?.value || 'all';
            const mechanism = document.getElementById('scenarioMechanismFilter')?.value || 'all';
            const filtering = Boolean(search || cancer !== 'all' || mechanism !== 'all');
            const panel = document.querySelector('.scenario-library-panel');
            panel?.classList.toggle('filtering', filtering);
            if (!filtering) { switchMatrixCategory(activeScenarioCategory); return; }
            let visible = 0;
            scenarioCategories.forEach(category => {
                const grid = document.getElementById('matrix-' + category);
                if (!grid) return;
                grid.style.display = 'grid';
                grid.querySelectorAll('.prevalence-card').forEach(card => {
                    const text = card.textContent.toLowerCase();
                    const searchableText = `${text} ${category}`;
                    const matchesSearch = searchTokens.every(token => searchableText.includes(token));
                    const matchesCancer = cancer === 'all' || category === cancer;
                    const matchesMechanism = mechanismMatches(text, mechanism);
                    const isVisible = matchesSearch && matchesCancer && matchesMechanism;
                    card.classList.toggle('is-hidden', !isVisible);
                    if (isVisible) visible += 1;
                });
            });
            const counter = document.getElementById('scenarioResultCount');
            if (counter) counter.textContent = `Showing ${visible} matching scenario${visible === 1 ? '' : 's'}`;
            document.getElementById('scenarioEmpty')?.style.setProperty('display', visible ? 'none' : 'block');
        }

        function resetScenarioFilters() {
            const search = document.getElementById('scenarioSearch');
            const cancer = document.getElementById('scenarioCancerFilter');
            const mechanism = document.getElementById('scenarioMechanismFilter');
            if (search) search.value = '';
            if (cancer) cancer.value = 'all';
            if (mechanism) mechanism.value = 'all';
            switchMatrixCategory('all', document.querySelector('.btn-matrix-tab'));
        }

        function setupScenarioExplorer() {
            document.querySelectorAll('.prevalence-card').forEach(card => {
                const category = card.closest('[data-cancer]')?.dataset.cancer;
                if (!category || card.querySelector('.scenario-meta')) return;
                card.dataset.indication = category;
                card.style.setProperty('--indication-accent', scenarioCategoryColors[category] || '#65d5ea');
                const organMark = document.createElement('span');
                organMark.className = 'scenario-organ-mark';
                organMark.setAttribute('aria-hidden', 'true');
                organMark.innerHTML = `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${scenarioOrganIcons[category] || scenarioOrganIcons.nsclc}</svg>`;
                card.prepend(organMark);
                const meta = document.createElement('div');
                meta.className = 'scenario-meta';
                const locusText = card.querySelector('.locus-tag')?.textContent || '';
                const alteration = locusText.includes('(') ? locusText.split('(')[1].split(')')[0] : 'gene-change details';
                meta.innerHTML = `<span>${escapeHtml(scenarioCategoryLabels[category] || category)}</span><span>${escapeHtml(alteration)}</span><span>open analysis</span>`;
                card.appendChild(meta);
            });
            document.querySelectorAll('#scenarioSearch, #scenarioCancerFilter, #scenarioMechanismFilter').forEach(element => {
                element.addEventListener(element.tagName === 'INPUT' ? 'input' : 'change', filterScenarioCards);
            });
            const query = new URLSearchParams(window.location.search).get('query');
            const search = document.getElementById('scenarioSearch');
            if (query && search) {
                search.value = query;
                filterScenarioCards();
            }
        }
        if (currentPage === 'scenarios') switchMatrixCategory('all', document.querySelector('.btn-matrix-tab'));
        setupScenarioExplorer();

        function applyAnalysisPresetFromUrl() {
            if (currentPage !== 'analyze') return;
            const params = new URLSearchParams(window.location.search);
            const mappings = {
                target: 'primary_target', drug: 'primary_drug', marker: 'resistance_marker', indication: 'cancer_type',
                primary_alteration: 'primary_alteration', resistance_alteration: 'resistance_alteration',
                alteration_type: 'resistance_alteration_type', treatment_line: 'treatment_line'
            };
            Object.entries(mappings).forEach(([param, field]) => {
                const value = params.get(param);
                const element = document.getElementById(field);
                if (value && element) element.value = value;
            });
            syncAnalysisPreview();
            if (params.get('autorun') === '1' && params.get('target') && params.get('marker')) {
                setTimeout(() => executePipeline(), 120);
            }
        }

        function syncAnalysisPreview() {
            if (currentPage !== 'analyze') return;
            const values = {
                previewPrimary: document.getElementById('primary_target')?.value.trim() || 'Target',
                previewDrug: document.getElementById('primary_drug')?.value.trim() || 'Therapy',
                previewResistance: document.getElementById('resistance_marker')?.value.trim() || 'Change',
                previewCancer: document.getElementById('cancer_type')?.value.trim() || 'Cancer type not specified'
            };
            Object.entries(values).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            });
        }

        ['primary_target', 'primary_drug', 'resistance_marker', 'cancer_type'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', syncAnalysisPreview);
            document.getElementById(id)?.addEventListener('change', syncAnalysisPreview);
        });
        applyAnalysisPresetFromUrl();


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
                const resistanceTypeLabels = {
                    'Off-Target Bypass': 'Alternative pathway',
                    'On-Target Mutation': 'Change in treated target'
                };
                document.getElementById('resTypeVal').innerText = resistanceTypeLabels[data.resistance_type] || data.resistance_type;
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
                container.innerHTML = '<div style="padding:2rem;text-align:center;color:#94a3b8;">No suitable protein-interaction network was found for this request.</div>';
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
                druggability: "Not described in the local target panel",
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
                container.innerHTML = '<div style="padding:1.25rem;text-align:center;color:#94a3b8;line-height:1.5;">No reviewed experimental structure is linked to this protein. A predicted structure is not shown as a substitute.</div>';
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
                ? 'Linked PDB record from the reviewed local list. Confirm the chain, construct, ligand, and variant in RCSB; this link does not show that the submitted alteration is present in the structure.'
                : 'No reviewed experimental structure is linked locally to this protein. An unverified PDB record is not shown.';
            const badge = document.getElementById('structureStatusBadge');
            badge.innerText = hasStructure ? 'Linked PDB record' : 'Structure unavailable';
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
                therapiesList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; text-align: center; padding: 2rem;">No matching clinical-stage drug records were found for this target and cancer type.</div>';
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
                const combinedPhaseLabels = {12: 'Phase 1/2', 23: 'Phase 2/3'};
                const phaseLabel = c.clinical_phase == null
                    ? 'Clinical phase not reported'
                    : (combinedPhaseLabels[c.clinical_phase] || (isHighestReportedStage ? 'Phase 4 / highest reported stage' : 'Phase ' + c.clinical_phase));
                const components = c.score_components || {};
                const componentText = [
                    ['Protein network', components.topology],
                    ['Network distance', components.proximity],
                    ['Drug activity data', components.pharmacology],
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
                const evidenceStatusLabels = {
                    abstained: ['Not ranked: target absent from network', 'caution'],
                    pair_co_mention: ['A source mentions both drugs; verify the study design', 'caution'],
                    pharmacology_available: ['Drug activity data available', 'positive'],
                    computational_hypothesis: ['Based on network data only', 'caution']
                };
                const [evidenceStatusLabel, evidenceStatusKind] = evidenceStatusLabels[c.evidence_status] || ['Evidence status unavailable', 'caution'];
                const badges = [
                    c.indication_match === true ? ['Cancer type matched', 'positive'] : ['Check the cancer type', 'caution'],
                    c.combination_evidence === true ? ['A source mentions both drugs', 'positive'] : ['No study of this drug pair found', 'caution'],
                    [evidenceStatusLabel, evidenceStatusKind],
                    c.clinical_status === 'stopped_or_withdrawn' ? ['Stopped / withdrawn', 'caution'] : null
                ].filter(Boolean).map(([label, kind]) => `<span class="evidence-badge ${kind}">${label}</span>`).join('');
                const evidenceRows = (c.evidence || []).map(source => {
                    const date = source.retrieved_at ? `retrieved ${escapeHtml(source.retrieved_at)}` : 'date unavailable';
                    const excerpt = source.excerpt_or_field ? `<br><span>${escapeHtml(source.excerpt_or_field)}</span>` : '';
                    return `<li>${escapeHtml(source.name || 'Source')} — ${date}${excerpt}</li>`;
                }).join('');
                const tieNote = tiedRank
                    ? `<span class="evidence-badge caution">Tied rank · not enough drug-specific evidence to separate these results</span>`
                    : '';
                const topologyNote = targetUnavailable
                    ? '<span class="evidence-badge caution">Not ranked · target absent from the retrieved protein network</span>'
                    : '';
                const card = document.createElement('div');
                card.className = 'candidate-card';
                card.innerHTML = `
                    <div class="candidate-header">
                        <div class="candidate-identity">
                            <span class="candidate-rank">${rankLabel}</span>
                            <div><span class="candidate-overline">Candidate combination</span><span class="drug-pair-name">${escapeHtml(c.secondary_drug)} + ${escapeHtml(primaryDrug)}</span></div>
                        </div>
                        <span class="badge-phase ${isHighestReportedStage ? 'approved' : ''}">${phaseLabel}</span>
                    </div>
                    <div class="candidate-target-route">Additional target <strong>${escapeHtml(c.secondary_target)}</strong><span>→</span> reviewed with <strong>${escapeHtml(primaryDrug)}</strong></div>
                    <div class="candidate-scoreboard">
                        <div class="candidate-score primary"><small>Research priority</small><strong>${c.synergy_score}</strong><div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div></div>
                        <div class="candidate-score"><small>Network position</small><strong>${(c.hub_penalized_centrality || 0).toFixed(3)}</strong></div>
                        <div class="candidate-score"><small>Network distance</small><strong>${c.shortest_path_distance == null ? '—' : Number(c.shortest_path_distance).toFixed(2)}</strong></div>
                    </div>
                    <div class="evidence-badges">${badges}${tieNote}${topologyNote}</div>
                    <div class="candidate-rationale">${escapeHtml(c.biological_rationale)}</div>
                    <details class="candidate-source-details">
                        <summary>Review ${c.evidence?.length || 0} linked source record${c.evidence?.length === 1 ? '' : 's'}</summary>
                        <div class="candidate-source-links">${evidenceLinks || '<span>No linked source record</span>'}</div>
                    </details>
                    <details class="candidate-explain">
                        <summary>How was this research priority calculated?</summary>
                        <p>This ordering combines protein-network information with available database evidence. It does not measure drug synergy or predict clinical benefit.</p>
                        <ul>
                            <li>Target: ${escapeHtml(c.secondary_target)}. The network-position value describes this protein, so drugs aimed at the same protein may share it.</li>
                            <li>Information used: ${escapeHtml(componentText || 'not available')}.</li>
                            <li>Clinical interpretation: ${c.evidence_notes?.length ? escapeHtml(c.evidence_notes.join(' ')) : (c.combination_evidence === true ? 'a returned source mentions the primary drug; inspect the linked record to confirm that the drugs were studied together and review the reported outcomes.' : 'no study of this specific drug pair was found; interpret this as protein-network evidence only.')}</li>
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


def _render_page(page: str) -> str:
    """Render a route-aware view from the shared evidence-first UI shell."""
    allowed_pages = {"home", "analyze", "scenarios", "method", "sources"}
    selected_page = page if page in allowed_pages else "home"
    return INDEX_HTML.replace(
        '<body data-page="home">', f'<body data-page="{selected_page}">', 1
    )


@app.get("/", response_class=HTMLResponse)
async def root_dashboard() -> str:
    """Serve the concise landing page for the Resistance Bypass Engine."""
    return _render_page("home")


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page() -> str:
    """Serve the full resistance analysis workbench."""
    return _render_page("analyze")


@app.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page() -> str:
    """Serve the curated clinical resistance scenario library."""
    return _render_page("scenarios")


@app.get("/method", response_class=HTMLResponse)
async def method_page() -> str:
    """Serve the scientific methodology and guardrails page."""
    return _render_page("method")


@app.get("/sources", response_class=HTMLResponse)
async def sources_page() -> str:
    """Serve the live source and provenance overview page."""
    return _render_page("sources")


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
            detail=f"The gene or drug record could not be confirmed. Report ID: {trace_id}",
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

        molecules: list[dict[str, Any]] = []
        if mapped_primary.chembl_target_id:
            try:
                molecules = await _bounded_timed_call(
                    "ChEMBL clinical molecules",
                    chembl_client.get_clinical_molecules(
                        target_chembl_id=mapped_primary.chembl_target_id,
                        max_phase_gte=2,
                        withdrawn_flag=False,
                    ),
                    source_timings,
                    live_source_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - upstream failure becomes a partial report
                logger.warning(
                    "ChEMBL lookup failed for trace %s: %s",
                    trace_id,
                    type(exc).__name__,
                )
                partial_sources.append("ChEMBL")
        else:
            partial_sources.append("ChEMBL target mapping")

        ranked_combinations: list[CombinationCandidate] = []
        warnings: list[str] = []
        warnings.append(
            "On-target candidates are target-linked clinical records; variant-specific resistance reversal and pair-level efficacy are not established by this result."
        )
        if partial_sources:
            warnings.append(
                "The report loaded without complete ChEMBL results because the live source was unavailable or the target record was missing."
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
                    biological_rationale=(
                        f"This drug record is linked to {primary_target_canonical}. "
                        "Activity against the reported resistance alteration is not established by this result."
                    ),
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
                "No matching clinical-stage drug record was found for this target, so no result is shown."
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
                moa = drug.get("mechanismOfAction") or (
                    "Drug targeting the resistance-associated protein"
                )
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
                        "biological_rationale": (
                            f"Targets {target_sym.upper()}, which may be involved in resistance to therapy directed at "
                            f"{primary_target_canonical}. This result does not establish restored sensitivity."
                        ),
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
                "No matching clinical-stage drug record was found for the resistance-associated gene, so no result is shown."
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
        except ValueError:
            # Missing interaction data is an evidence gap, not a malformed
            # request. Preserve verified drug records without inventing a path.
            if "STRING" not in partial_sources:
                partial_sources.append("STRING network")
            warnings.append(
                "No connected protein-interaction network was returned for this gene pair. Drug records are shown without a network-based priority."
            )
            pathway_nodes_count = 0
            shortest_path_distance = None
            scored_raw = []
            for candidate in raw_candidates:
                unscored = dict(candidate)
                unscored.update(
                    {
                        "synergy_score": 0.0,
                        "hub_penalized_centrality": 0.0,
                        "score_components": {
                            "topology": 0.0,
                            "proximity": 0.0,
                            "pharmacology": None,
                        },
                        "target_in_graph": False,
                        "scoring_status": "network_unavailable",
                        "evidence_status": "network evidence unavailable",
                        "evidence_notes": [
                            "No connected interaction network was available, so this candidate was not ranked by network position."
                        ],
                    }
                )
                scored_raw.append(unscored)
            net_nodes = [
                {
                    "id": symbol,
                    "degree": 0,
                    "annotation": get_gene_annotation(symbol),
                }
                for symbol in (primary_target_canonical, resistance_marker_canonical)
            ]
            net_edges = []

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
                    (
                        f"Targets {c.get('secondary_target')}, a protein in the retrieved resistance network. "
                        f"Clinical benefit for {resistance_marker_canonical}-associated resistance is not established."
                    ),
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
