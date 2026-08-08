import asyncio
import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import _bounded_timed_call, _request_windows, app
from src.schemas.models import AlterationType, IDMappingResult, ResistanceRequest

client = TestClient(app)


@pytest.mark.asyncio
async def test_live_source_call_is_bounded_and_records_timing():
    timings: dict[str, float] = {}

    async def slow_source() -> None:
        await asyncio.sleep(0.05)

    with pytest.raises(asyncio.TimeoutError):
        await _bounded_timed_call("STRING PPI network", slow_source(), timings, 0.01)

    assert "STRING PPI network" in timings
    assert timings["STRING PPI network"] < 1000


def test_vercel_entrypoint_exports_fastapi_app():
    from api.index import app as vercel_app

    assert vercel_app is app


def test_health_endpoint():
    """Verify GET /health returns status ok and cache metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cache_size_bytes" in data
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_homepage_contains_mutation_focused_workspace():
    response = client.get("/")
    assert response.status_code == 200
    assert "Examine resistance." in response.text
    assert "mutation-plate" in response.text
    assert "Experimental protein structure" in response.text
    assert "structureWorkspaceViewer" in response.text
    assert 'href="/" class="btn-header"' in response.text
    assert "Begin with a documented resistance pattern." in response.text
    assert 'class="featured-scenario" href="/analyze?' in response.text
    assert "target=EGFR&amp;drug=Osimertinib&amp;marker=MET" in response.text
    assert "autorun=1" in response.text


def test_scenario_explorer_includes_expanded_evidence_based_catalog():
    response = client.get("/scenarios")
    assert response.status_code == 200
    assert "19 reviewed scenarios" in response.text
    assert "scenarioCancerFilter" in response.text
    assert "ROS1 + G2032R" in response.text
    assert "KIT + V654A" in response.text
    assert "FGFR2 + V565F" in response.text
    assert "Primary clinical evidence" in response.text
    assert "searchTokens.every" in response.text
    assert len(re.findall(r'class="prevalence-card(?: |")', response.text)) == 19
    assert response.text.count('data-cancer="') == 11
    assert "across 9 oncology indications" not in response.text
    assert "scenarioOrganIcons" in response.text
    assert "scenarioCategoryColors" in response.text
    assert "scenario-tab-organ" in response.text
    assert "activeButton?.classList.add('active')" in response.text


def test_every_scenario_uses_a_locally_resolvable_gene_pair():
    from src.services.id_mapper import CURATED_GENE_ALIASES, CURATED_GENE_IDENTIFIERS

    html = client.get("/scenarios").text
    gene_pairs = re.findall(
        r"setPreset\('([^']+)', '[^']+', '([^']+)'",
        html,
    )
    assert len(gene_pairs) == 19
    supported = set(CURATED_GENE_IDENTIFIERS) | set(CURATED_GENE_ALIASES)
    assert {gene for pair in gene_pairs for gene in pair} <= supported


def test_frontend_uses_plain_language_for_core_workflow():
    response = client.get("/analyze")
    assert response.status_code == 200
    assert "Treatment and resistance details" in response.text
    assert "Analyze resistance evidence" in response.text
    assert "Gene records confirmed" in response.text
    assert "Interactions retrieved" in response.text
    assert "Research priorities calculated" in response.text
    for jargon in (
        "Canonical identity",
        "Canonical IDs",
        "PPI topology",
        "Heuristic Priority",
        "Hub Centrality",
    ):
        assert jargon not in response.text
    assert re.search(r'id="submitBtn".*?</button>\s*</form>', response.text, re.DOTALL)


def test_frontend_has_focused_routes():
    routes = {
        "/analyze": '<body data-page="analyze">',
        "/scenarios": '<body data-page="scenarios">',
        "/method": '<body data-page="method">',
        "/sources": '<body data-page="sources">',
    }
    for route, marker in routes.items():
        response = client.get(route)
        assert response.status_code == 200
        assert marker in response.text
        assert response.text.count(marker) == 1


def test_frontend_inline_actions_reference_defined_functions():
    html = client.get("/").text
    onclick_functions = set(
        re.findall(
            r'onclick="(?:if\([^\"]+\)\s*)?([A-Za-z_$][\w$]*)\(',
            html,
        )
    )
    defined_functions = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", html))
    assert onclick_functions <= defined_functions


def test_structure_lookup_does_not_invent_unknown_identifiers():
    response = client.get("/api/v1/structure/UNKNOWN_TARGET")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["annotation"]["pdb_id"] is None
    assert data["annotation"]["uniprot_id"] is None


def test_input_sanitization_validator():
    """Verify ResistanceRequest validator strips and uppercases target symbols."""
    req = ResistanceRequest(
        primary_drug="Osimertinib",
        primary_target="  egfr  ",
        resistance_marker="  met  ",
    )
    assert req.primary_target == "EGFR"
    assert req.resistance_marker == "MET"


def test_input_length_is_bounded():
    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "x" * 161,
            "primary_target": "EGFR",
            "resistance_marker": "MET",
        },
    )
    assert response.status_code == 422


def test_analysis_rate_limit_returns_retry_after(monkeypatch):
    import src.main as main_module

    _request_windows.clear()
    monkeypatch.setattr(main_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    first = client.post(
        "/api/v1/analyze-resistance",
        json={"primary_drug": "x", "primary_target": "", "resistance_marker": "MET"},
    )
    second = client.post(
        "/api/v1/analyze-resistance",
        json={"primary_drug": "x", "primary_target": "", "resistance_marker": "MET"},
    )
    assert first.status_code in {400, 422}
    assert second.status_code == 429
    assert "retry-after" in second.headers
    _request_windows.clear()


def test_structured_alteration_context_is_preserved():
    req = ResistanceRequest(
        primary_target="egfr",
        primary_drug="Osimertinib",
        resistance_marker="met",
        primary_alteration=" L858R ",
        resistance_alteration=" amplification ",
        resistance_alteration_type="amplification",
        treatment_line="after progression",
    )
    assert req.primary_alteration == "L858R"
    assert req.resistance_alteration == "amplification"
    assert req.resistance_alteration_type is AlterationType.AMPLIFICATION
    assert req.treatment_line == "after progression"


@patch("src.main.IDMapper")
@patch("src.main.ChEMBLClient")
def test_on_target_mutation_branching(mock_chembl_cls, mock_id_mapper_cls):
    """Verify POST /api/v1/analyze-resistance executes On-Target Mutation branch when T_primary == T_resistance."""
    mock_id_mapper = mock_id_mapper_cls.return_value
    mock_id_mapper.map_identifier = AsyncMock(
        side_effect=lambda sym: IDMappingResult(
            original_input=sym,
            canonical_symbol="EGFR",
            ensembl_id="ENSG00000146648",
            uniprot_id="P00533",
            chembl_target_id="CHEMBL203",
        )
    )

    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_clinical_molecules = AsyncMock(
        return_value=[
            {
                "pref_name": "Lazertinib",
                "molecule_chembl_id": "CHEMBL4298782",
                "max_phase": 4,
            }
        ]
    )

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "EGFR",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["primary_target_canonical"] == "EGFR"
    assert data["resistance_marker_canonical"] == "EGFR"
    assert data["resistance_type"] == "On-Target Mutation"
    assert data["pathway_nodes_count"] == 1
    assert data["shortest_path_distance"] == 0.0
    assert len(data["ranked_combinations"]) > 0
    assert data["ranked_combinations"][0]["secondary_drug"] == "LAZERTINIB"


@patch("src.main.IDMapper")
@patch("src.main.ChEMBLClient")
def test_on_target_empty_evidence_does_not_fabricate_candidate(
    mock_chembl_cls, mock_id_mapper_cls
):
    mock_id_mapper_cls.return_value.map_identifier = AsyncMock(
        return_value=IDMappingResult(
            original_input="EGFR",
            canonical_symbol="EGFR",
            ensembl_id="ENSG00000146648",
            uniprot_id="P00533",
            chembl_target_id="CHEMBL203",
        )
    )
    mock_chembl_cls.return_value.get_clinical_molecules = AsyncMock(return_value=[])

    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Osimertinib",
            "primary_target": "EGFR",
            "resistance_marker": "EGFR",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ranked_combinations"] == []
    assert any("no result is shown" in warning.lower() for warning in data["warnings"])


@patch("src.main.IDMapper")
@patch("src.main.ChEMBLClient")
def test_on_target_source_failure_returns_partial_report(
    mock_chembl_cls, mock_id_mapper_cls
):
    mock_id_mapper_cls.return_value.map_identifier = AsyncMock(
        return_value=IDMappingResult(
            original_input="KIT",
            canonical_symbol="KIT",
            ensembl_id="ENSG00000157404",
            uniprot_id="P10721",
            chembl_target_id="CHEMBL1936",
        )
    )
    mock_chembl_cls.return_value.get_clinical_molecules = AsyncMock(
        side_effect=TimeoutError
    )

    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Imatinib",
            "primary_target": "KIT",
            "resistance_marker": "KIT",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ranked_combinations"] == []
    assert "ChEMBL" in data["metadata"]["partial_sources"]
    assert any("without complete chembl" in item.lower() for item in data["warnings"])


@patch("src.main.IDMapper")
@patch("src.main.StringDBClient")
@patch("src.main.OpenTargetsClient")
@patch("src.main.ChEMBLClient")
def test_off_target_bypass_branching(
    mock_chembl_cls, mock_ot_cls, mock_string_cls, mock_id_mapper_cls
):
    """Verify POST /api/v1/analyze-resistance executes Off-Target Bypass branch when T_primary != T_resistance."""
    mock_id_mapper = mock_id_mapper_cls.return_value

    async def mock_map(sym):
        if "EGFR" in sym.upper():
            return IDMappingResult(
                original_input=sym,
                canonical_symbol="EGFR",
                ensembl_id="ENSG00000146648",
                uniprot_id="P00533",
                chembl_target_id="CHEMBL203",
            )
        else:
            return IDMappingResult(
                original_input=sym,
                canonical_symbol="MET",
                ensembl_id="ENSG00000105976",
                uniprot_id="P08581",
                chembl_target_id="CHEMBL3714",
            )

    mock_id_mapper.map_identifier = AsyncMock(side_effect=mock_map)

    mock_string = mock_string_cls.return_value
    mock_string.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900}
        ]
    )

    mock_ot = mock_ot_cls.return_value
    mock_ot.get_known_drugs = AsyncMock(
        return_value=[
            {
                "prefName": "Capmatinib",
                "drugId": "CHEMBL3545380",
                "phase": 4,
                "mechanismOfAction": "MET Kinase Inhibitor",
                "targetSymbol": "MET",
            }
        ]
    )

    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_target_activities = AsyncMock(return_value={"CAPMATINIB": 8.5})

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "MET",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["primary_target_canonical"] == "EGFR"
    assert data["resistance_marker_canonical"] == "MET"
    assert data["resistance_type"] == "Off-Target Bypass"
    assert data["pathway_nodes_count"] == 2
    assert len(data["ranked_combinations"]) > 0
    assert data["ranked_combinations"][0]["secondary_drug"] == "CAPMATINIB"
    assert data["evidence_claims"][0]["claim_type"] == "computational"
    assert data["evidence_claims"][0]["review_state"] == "unreviewed"
    assert data["evidence_claims"][0]["evidence"][0]["name"] == "STRING"
    assert data["metadata"]["methodology_version"] == "evidence-priority-0.3"
    assert len(data["metadata"]["request_fingerprint"]) == 64
    assert len(data["metadata"]["trace_id"]) == 16
    assert "STRING PPI network" in data["metadata"]["source_timings_ms"]
    assert data["metadata"]["partial_sources"] == []
    assert data["metadata"]["sources"] == [
        "HGNC",
        "UniProt",
        "STRING",
        "Open Targets",
        "ChEMBL",
    ]
    assert {
        source["name"] for source in data["ranked_combinations"][0]["evidence"]
    } == {"Open Targets", "ChEMBL"}
