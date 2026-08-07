from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import _fetch_candidates_for_target, app
from src.schemas.models import IDMappingResult

client = TestClient(app)


def mapped(symbol: str) -> IDMappingResult:
    ids = {
        "EGFR": ("ENSG_EGFR", "P_EGFR", "CHEMBL_EGFR"),
        "MET": ("ENSG_MET", "P_MET", "CHEMBL_MET"),
        "GRB2": ("ENSG_GRB2", "P_GRB2", "CHEMBL_GRB2"),
    }
    ensembl, uniprot, chembl = ids[symbol]
    return IDMappingResult(
        original_input=symbol,
        canonical_symbol=symbol,
        ensembl_id=ensembl,
        uniprot_id=uniprot,
        chembl_target_id=chembl,
    )


def raw_candidate(target: str, drug: str) -> dict:
    return {
        "secondary_drug": drug,
        "secondary_target": target,
        "mechanism_of_action": f"{target} inhibitor",
        "clinical_phase": 3,
        "clinical_status": "active_or_completed",
        "is_withdrawn": False,
        "indication_match": target == "MET",
        "combination_evidence": target == "MET",
        "median_pchembl": 8.0,
        "activity_measurements": 2,
        "biological_rationale": f"Evidence-linked {target} agent.",
        "evidence": [],
    }


@patch("src.main._fetch_candidates_for_target", new_callable=AsyncMock)
@patch("src.main.StringDBClient")
@patch("src.main.IDMapper")
def test_pipeline_discovers_and_queries_intermediary_targets(
    mapper_cls, string_cls, fetch_candidates
):
    mapper_cls.return_value.map_identifier = AsyncMock(
        side_effect=lambda symbol: mapped(symbol.upper())
    )
    string_cls.return_value.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "GRB2", "score": 900},
            {"preferredName_A": "GRB2", "preferredName_B": "MET", "score": 850},
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 700},
        ]
    )

    async def candidates_for(mapping, *_):
        target = mapping["canonical_symbol"]
        return [raw_candidate(target, f"{target}-DRUG")], []

    fetch_candidates.side_effect = candidates_for
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
    targets = {item["secondary_target"] for item in report["ranked_combinations"]}
    assert targets == {"MET", "GRB2"}
    assert {node["id"] for node in report["network_nodes"]} == {"EGFR", "GRB2", "MET"}
    assert report["pathway_nodes_count"] == 3


@patch("src.main._fetch_candidates_for_target", new_callable=AsyncMock)
@patch("src.main.StringDBClient")
@patch("src.main.IDMapper")
def test_no_evidence_returns_empty_result_not_fabricated_drug(
    mapper_cls, string_cls, fetch_candidates
):
    mapper_cls.return_value.map_identifier = AsyncMock(
        side_effect=lambda symbol: mapped(symbol.upper())
    )
    string_cls.return_value.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900}
        ]
    )
    fetch_candidates.return_value = ([], [])
    response = client.post(
        "/api/v1/analyze-resistance",
        json={
            "primary_drug": "Osimertinib",
            "primary_target": "EGFR",
            "resistance_marker": "MET",
        },
    )
    assert response.status_code == 200
    assert response.json()["ranked_combinations"] == []
    assert any("No active" in warning for warning in response.json()["warnings"])


@pytest.mark.asyncio
@patch("src.main.ChEMBLClient")
@patch("src.main.OpenTargetsClient")
async def test_candidate_fetch_filters_primary_stopped_withdrawn_and_phase_one(
    ot_cls, chembl_cls
):
    def drug(name, phase, status="reported", withdrawn=False):
        return {
            "prefName": name,
            "drugId": name,
            "phase": phase,
            "clinicalStatus": status,
            "isWithdrawn": withdrawn,
            "mechanismOfAction": "MET inhibitor",
            "indicationMatch": True,
            "combinationEvidence": False,
            "evidence": [],
        }

    ot_cls.return_value.get_known_drugs = AsyncMock(
        return_value=[
            drug("Osimertinib", 4),
            drug("PHASE1", 1),
            drug("STOPPED", 3, "stopped"),
            drug("WITHDRAWN", 4, withdrawn=True),
            drug("CAPMATINIB", 4, "active_or_completed"),
        ]
    )
    chembl_cls.return_value.get_target_activities = AsyncMock(
        return_value={
            "CAPMATINIB": {
                "median_pchembl": 8.1,
                "measurement_count": 2,
                "measurement_types": ["IC50"],
                "molecule_chembl_id": "CHEMBL1",
            }
        }
    )
    candidates, warnings = await _fetch_candidates_for_target(
        mapped("MET").model_dump(), "Osimertinib", "EGFR", "NSCLC"
    )
    assert warnings == []
    assert [item["secondary_drug"] for item in candidates] == ["CAPMATINIB"]
    assert candidates[0]["median_pchembl"] == 8.1
