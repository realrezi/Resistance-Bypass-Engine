from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.main import _request_windows, app
from src.schemas.models import AlterationType, IDMappingResult, ResistanceRequest

client = TestClient(app)


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
    assert "Trace the mutation." in response.text
    assert "mutation-plate" in response.text
    assert "Molecular structure workspace" in response.text
    assert "structureWorkspaceViewer" in response.text


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
    assert any("no candidate was fabricated" in warning for warning in data["warnings"])


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
