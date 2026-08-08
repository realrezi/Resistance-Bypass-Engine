from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.main import app
from src.schemas.models import IDMappingResult

client = TestClient(app)


@patch("src.main.IDMapper")
@patch("src.main.StringDBClient")
@patch("src.main.OpenTargetsClient")
@patch("src.main.ChEMBLClient")
def test_e2e_egfr_met_resistance_pipeline(
    mock_chembl_cls, mock_ot_cls, mock_string_cls, mock_id_mapper_cls
):
    """End-to-end pipeline test for Osimertinib (EGFR) + MET bypass scenario."""
    mock_id_mapper = mock_id_mapper_cls.return_value

    async def mock_map(sym):
        clean = sym.strip().upper()
        if "EGFR" in clean:
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
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900},
            {"preferredName_A": "MET", "preferredName_B": "ERBB3", "score": 850},
            {"preferredName_A": "EGFR", "preferredName_B": "ERBB3", "score": 800},
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
            },
            {
                "prefName": "Crizotinib",
                "drugId": "CHEMBL1201585",
                "phase": 4,
                "mechanismOfAction": "MET/ALK Inhibitor",
                "targetSymbol": "MET",
            },
            {
                "prefName": "Tepotinib",
                "drugId": "CHEMBL3989849",
                "phase": 4,
                "mechanismOfAction": "MET Inhibitor",
                "targetSymbol": "MET",
            },
        ]
    )

    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_target_activities = AsyncMock(
        return_value={"CAPMATINIB": 9.0, "CRIZOTINIB": 8.2, "TEPOTINIB": 8.7}
    )

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "MET",
        "cancer_type": "Non-Small Cell Lung Cancer",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    report = response.json()

    assert report["primary_target_canonical"] == "EGFR"
    assert report["resistance_marker_canonical"] == "MET"
    assert report["resistance_type"] == "Off-Target Bypass"
    assert report["pathway_nodes_count"] == 3
    assert report["shortest_path_distance"] > 0.0

    ranked = report["ranked_combinations"]
    assert len(ranked) == 3

    # Check that clinical secondary drugs are present
    drug_names = [c["secondary_drug"] for c in ranked]
    assert "CAPMATINIB" in drug_names
    assert "CRIZOTINIB" in drug_names
    assert "TEPOTINIB" in drug_names

    # Check synergy score bounds
    for candidate in ranked:
        score = candidate["synergy_score"]
        assert 0.0 <= score <= 1.0
        assert candidate["hub_penalized_centrality"] >= 0.0


@patch("src.main.IDMapper")
@patch("src.main.StringDBClient")
@patch("src.main.OpenTargetsClient")
@patch("src.main.ChEMBLClient")
def test_alias_resolution_her2_to_erbb2(
    mock_chembl_cls, mock_ot_cls, mock_string_cls, mock_id_mapper_cls
):
    """Verify gene symbol alias HER2 maps to canonical ERBB2 end-to-end."""
    mock_id_mapper = mock_id_mapper_cls.return_value

    async def mock_map(sym):
        clean = sym.strip().upper()
        if "EGFR" in clean:
            return IDMappingResult(
                original_input=sym,
                canonical_symbol="EGFR",
                ensembl_id="ENSG00000146648",
                uniprot_id="P00533",
                chembl_target_id="CHEMBL203",
            )
        else:
            # HER2 maps to canonical ERBB2
            return IDMappingResult(
                original_input=sym,
                canonical_symbol="ERBB2",
                ensembl_id="ENSG00000141736",
                uniprot_id="P04626",
                chembl_target_id="CHEMBL1824",
            )

    mock_id_mapper.map_identifier = AsyncMock(side_effect=mock_map)

    mock_string = mock_string_cls.return_value
    mock_string.get_network = AsyncMock(
        return_value=[
            {"preferredName_A": "EGFR", "preferredName_B": "ERBB2", "score": 950}
        ]
    )

    mock_ot = mock_ot_cls.return_value
    mock_ot.get_known_drugs = AsyncMock(return_value=[])
    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_target_activities = AsyncMock(return_value={})

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "HER2",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    report = response.json()

    assert report["primary_target_canonical"] == "EGFR"
    assert report["resistance_marker_canonical"] == "ERBB2"
    assert report["resistance_type"] == "Off-Target Bypass"


@patch("src.main.IDMapper")
def test_invalid_gene_symbol_error_handling(mock_id_mapper_cls):
    """Verify invalid gene symbol raises 422 Unprocessable Entity."""
    mock_id_mapper = mock_id_mapper_cls.return_value
    mock_id_mapper.map_identifier = AsyncMock(
        side_effect=ValueError("Gene symbol 'INVALIDGENEXXX' not found in HGNC.")
    )

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "INVALIDGENEXXX",
        "resistance_marker": "MET",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "could not be confirmed" in data["detail"]
    assert "INVALIDGENEXXX" not in data["detail"]


@patch("src.main.IDMapper")
@patch("src.main.StringDBClient")
@patch("src.main.OpenTargetsClient")
@patch("src.main.ChEMBLClient")
def test_no_pathway_returns_an_explicit_partial_report(
    mock_chembl_cls, mock_ot_cls, mock_string_cls, mock_id_mapper_cls
):
    """An unavailable network is reported as missing evidence, not a broken page."""
    mock_id_mapper = mock_id_mapper_cls.return_value
    mock_id_mapper.map_identifier = AsyncMock(
        side_effect=lambda sym: IDMappingResult(
            original_input=sym,
            canonical_symbol=sym.upper(),
            ensembl_id="ENSG00000000000",
            uniprot_id="P00000",
        )
    )

    mock_string = mock_string_cls.return_value
    mock_string.get_network = AsyncMock(return_value=[])  # Empty network

    mock_ot = mock_ot_cls.return_value
    mock_ot.get_known_drugs = AsyncMock(return_value=[])

    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_target_activities = AsyncMock(return_value={})

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "MET",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pathway_nodes_count"] == 0
    assert data["network_edges"] == []
    assert "STRING network" in data["metadata"]["partial_sources"]
    assert any(
        "No connected protein-interaction network" in item for item in data["warnings"]
    )


@patch("src.main.IDMapper")
@patch("src.main.StringDBClient")
@patch("src.main.OpenTargetsClient")
@patch("src.main.ChEMBLClient")
def test_withdrawn_drug_filtering(
    mock_chembl_cls, mock_ot_cls, mock_string_cls, mock_id_mapper_cls
):
    """Verify withdrawn drugs from Open Targets are filtered out."""
    mock_id_mapper = mock_id_mapper_cls.return_value

    async def mock_map(sym):
        clean = sym.strip().upper()
        if "EGFR" in clean:
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
                "prefName": "ActiveDrug",
                "drugId": "CHEMBL001",
                "phase": 3,
                "mechanismOfAction": "MET Inhibitor",
                "targetSymbol": "MET",
                "status": "Active",
            },
            {
                "prefName": "WithdrawnDrug",
                "drugId": "CHEMBL002",
                "phase": 4,
                "mechanismOfAction": "MET Inhibitor",
                "targetSymbol": "MET",
                "status": "Withdrawn",
            },
        ]
    )

    mock_chembl = mock_chembl_cls.return_value
    mock_chembl.get_target_activities = AsyncMock(return_value={"ACTIVEDRUG": 8.0})

    payload = {
        "primary_drug": "Osimertinib",
        "primary_target": "EGFR",
        "resistance_marker": "MET",
    }
    response = client.post("/api/v1/analyze-resistance", json=payload)
    assert response.status_code == 200
    report = response.json()

    drug_names = [c["secondary_drug"] for c in report["ranked_combinations"]]
    assert "ACTIVEDRUG" in drug_names
    assert "WITHDRAWNDRUG" not in drug_names
