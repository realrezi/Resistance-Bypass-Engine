from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schemas.models import IDMappingResult, ResistanceRequest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def mapping(symbol: str) -> IDMappingResult:
    values = {
        "EGFR": ("ENSG00000146648", "P00533", "CHEMBL203"),
        "MET": ("ENSG00000105976", "P08581", "CHEMBL3714"),
    }
    ensembl, uniprot, chembl = values[symbol]
    return IDMappingResult(
        original_input=symbol,
        canonical_symbol=symbol,
        ensembl_id=ensembl,
        uniprot_id=uniprot,
        chembl_target_id=chembl,
    )


def candidate(target: str = "MET") -> dict:
    return {
        "secondary_drug": "CAPMATINIB",
        "secondary_target": target,
        "mechanism_of_action": "MET inhibitor",
        "clinical_phase": 4,
        "clinical_status": "active_or_completed",
        "is_withdrawn": False,
        "indication_match": True,
        "combination_evidence": True,
        "median_pchembl": 8.2,
        "activity_measurements": 3,
        "biological_rationale": "Evidence-linked target inhibition.",
        "evidence": [
            {
                "source": "AACT",
                "record_id": "NCT1",
                "url": "https://clinicaltrials.gov/study/NCT1",
            }
        ],
    }


def test_health_is_truthful_and_has_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["upstream_status"] == "not_checked"
    assert response.json()["version"] == "0.2.0"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_input_validation_and_normalization(client):
    request = ResistanceRequest(
        primary_drug="  Osimertinib ",
        primary_target=" egfr ",
        resistance_marker=" met ",
        cancer_type=" Non-Small   Cell Lung Cancer ",
    )
    assert request.primary_target == "EGFR"
    assert request.resistance_marker == "MET"
    assert request.primary_drug == "Osimertinib"
    assert request.cancer_type == "Non-Small Cell Lung Cancer"
    assert (
        client.post(
            "/api/v1/analyze-resistance",
            json={"primary_drug": "", "primary_target": "", "resistance_marker": ""},
        ).status_code
        == 422
    )


@patch("src.main._fetch_candidates_for_target", new_callable=AsyncMock)
@patch("src.main.IDMapper")
def test_on_target_branch_is_honest_and_never_fabricates_candidate(
    mapper_cls, fetch_candidates, client
):
    mapper_cls.return_value.map_identifier = AsyncMock(return_value=mapping("EGFR"))
    fetch_candidates.return_value = (
        [{**candidate("EGFR"), "secondary_drug": "LAZERTINIB"}],
        [],
    )
    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Osimertinib",
            "primary_target": "EGFR",
            "resistance_marker": "EGFR",
            "resistance_alteration": "C797S",
        },
    )
    assert response.status_code == 200
    report = response.json()
    assert report["resistance_type"] == "On-Target Alteration"
    assert report["ranked_combinations"][0]["combination_priority_score"] < 1
    assert "variant-specific" in " ".join(
        report["ranked_combinations"][0]["limitations"]
    )
    assert report["score_label"].startswith("Heuristic")


@patch("src.main._fetch_candidates_for_target", new_callable=AsyncMock)
@patch("src.main.StringDBClient")
@patch("src.main.IDMapper")
def test_off_target_branch_uses_relevant_physical_component(
    mapper_cls, string_cls, fetch_candidates, client
):
    async def map_symbol(symbol):
        return mapping(symbol.upper())

    mapper_cls.return_value.map_identifier = AsyncMock(side_effect=map_symbol)
    string_cls.return_value.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900}
        ]
    )
    fetch_candidates.return_value = ([candidate()], [])
    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Osimertinib",
            "primary_target": "EGFR",
            "resistance_marker": "MET",
            "cancer_type": "Non-Small Cell Lung Cancer",
        },
    )
    assert response.status_code == 200
    report = response.json()
    assert report["pathway_nodes_count"] == 2
    assert report["shortest_path_distance"] == pytest.approx(0.105)
    assert report["primary_drug"] == "Osimertinib"
    assert report["ranked_combinations"][0]["secondary_drug"] == "CAPMATINIB"
    assert any(item.get("network_type") == "physical" for item in report["provenance"])
    string_cls.return_value.get_network.assert_awaited_once_with(
        "EGFR", "MET", network_type="physical"
    )


@patch("src.main.StringDBClient")
@patch("src.main.IDMapper")
def test_disconnected_requested_targets_return_clear_400(
    mapper_cls, string_cls, client
):
    async def map_symbol(symbol):
        return mapping(symbol.upper())

    mapper_cls.return_value.map_identifier = AsyncMock(side_effect=map_symbol)
    string_cls.return_value.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "GRB2", "score": 900},
            {"preferredName_A": "MET", "preferredName_B": "GAB1", "score": 900},
        ]
    )
    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Osimertinib",
            "primary_target": "EGFR",
            "resistance_marker": "MET",
        },
    )
    assert response.status_code == 400
    assert "not connected" in response.json()["detail"]


@patch("src.main.IDMapper")
def test_invalid_symbol_is_422_and_upstream_failure_is_502(mapper_cls, client):
    mapper_cls.return_value.map_identifier = AsyncMock(
        side_effect=ValueError("not found")
    )
    payload = {
        "primary_drug": "Drug",
        "primary_target": "BAD",
        "resistance_marker": "MET",
    }
    assert client.post("/api/v1/analyze-resistance", json=payload).status_code == 422

    mapper_cls.return_value.map_identifier = AsyncMock(
        side_effect=RuntimeError("offline")
    )
    assert client.post("/api/v1/analyze-resistance", json=payload).status_code == 502
