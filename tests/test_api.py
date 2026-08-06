from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas.models import IDMappingResult, ResistanceRequest

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns status ok and cache metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cache_size_bytes" in data


def test_input_sanitization_validator():
    """Verify ResistanceRequest validator strips and uppercases target symbols."""
    req = ResistanceRequest(
        primary_drug="Osimertinib",
        primary_target="  egfr  ",
        resistance_marker="  met  ",
    )
    assert req.primary_target == "EGFR"
    assert req.resistance_marker == "MET"


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
            {"pref_name": "Lazertinib", "molecule_chembl_id": "CHEMBL4298782", "max_phase": 4}
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
