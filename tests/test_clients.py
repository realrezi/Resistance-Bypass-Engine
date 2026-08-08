from unittest.mock import AsyncMock, patch

import pytest

from src.clients.base import USER_AGENT, BaseHTTPClient, _stable_cache_key
from src.clients.chembl import ChEMBLClient
from src.clients.open_targets import OpenTargetsClient
from src.clients.string_db import StringDBClient
from src.services.id_mapper import CURATED_GENE_IDENTIFIERS, IDMapper


@pytest.mark.asyncio
async def test_scenario_gene_identifiers_resolve_without_live_api_calls():
    """Built-in scenarios must still open when identifier services are unavailable."""
    mapper = IDMapper()
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mock_get:
        for symbol, expected in CURATED_GENE_IDENTIFIERS.items():
            result = await mapper.map_identifier(symbol)
            assert result.canonical_symbol == symbol
            assert result.ensembl_id == expected["ensembl_id"]
            assert result.uniprot_id == expected["uniprot_id"]

        her2 = await mapper.map_identifier("HER2")
        assert her2.canonical_symbol == "ERBB2"
        mock_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensembl_version_stripping():
    """Verify that Ensembl IDs with version suffixes (e.g., .15) are stripped."""
    mapper = IDMapper()
    mock_hgnc_response = {
        "response": {
            "docs": [
                {
                    "symbol": "EGFR",
                    "ensembl_gene_id": "ENSG00000146648.15",
                }
            ]
        }
    }
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_hgnc_response
        symbol, ensembl_id = await mapper.resolve_hgnc("EGFR")
        assert symbol == "EGFR"
        assert ensembl_id == "ENSG00000146648"


@pytest.mark.asyncio
async def test_chembl_single_protein_filtering():
    """Verify ChEMBL target resolution filters strictly for target_type == 'SINGLE PROTEIN'."""
    mapper = IDMapper()
    mock_chembl_response = {
        "targets": [
            {
                "target_chembl_id": "CHEMBL12345",
                "target_type": "PROTEIN COMPLEX",
            },
            {
                "target_chembl_id": "CHEMBL203",
                "target_type": "SINGLE PROTEIN",
            },
            {
                "target_chembl_id": "CHEMBL99999",
                "target_type": "SELECTIVITY GROUP",
            },
        ]
    }
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_chembl_response
        chembl_id = await mapper.resolve_chembl_target("P00533")
        assert chembl_id == "CHEMBL203"


@pytest.mark.asyncio
async def test_chembl_exhaustive_pagination():
    """Verify ChEMBL client follows page_meta.next until all pages are fetched."""
    client = ChEMBLClient()
    page1 = {
        "molecules": [{"molecule_chembl_id": "CHEMBL1", "pref_name": "Drug A"}],
        "page_meta": {"next": "/api/data/molecule.json?page=2"},
    }
    page2 = {
        "molecules": [{"molecule_chembl_id": "CHEMBL2", "pref_name": "Drug B"}],
        "page_meta": {"next": None},
    }

    with patch.object(client, "get_json", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [page1, page2]
        molecules = await client.get_clinical_molecules()
        assert len(molecules) == 2
        assert molecules[0]["pref_name"] == "Drug A"
        assert molecules[1]["pref_name"] == "Drug B"
        assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_user_agent_header():
    """Verify base HTTP client includes mandatory User-Agent header."""
    base_client = BaseHTTPClient()
    assert base_client.headers["User-Agent"] == USER_AGENT
    assert "ResistanceBypassEngine/1.0" in USER_AGENT
    assert base_client.headers["mailto"] == "developer@example.com"


@pytest.mark.asyncio
async def test_string_db_client():
    """Verify STRING-DB client parameters and payload handling."""
    client = StringDBClient()
    mock_network = [
        {
            "stringId_A": "9606.ENSP00000275493",
            "preferredName_A": "EGFR",
            "preferredName_B": "MET",
            "score": 900,
        }
    ]
    with patch.object(client, "get_json", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_network
        res = await client.get_network("EGFR", "MET")
        assert len(res) == 1
        assert res[0]["preferredName_A"] == "EGFR"
        mock_get.assert_called_once_with(
            "https://string-db.org/api/json/network",
            params={
                "identifiers": "EGFR\rMET",
                "species": 9606,
                "required_score": 400,
                "add_nodes": 25,
            },
        )


@pytest.mark.asyncio
async def test_open_targets_client():
    """Verify Open Targets GraphQL query execution."""
    client = OpenTargetsClient()
    mock_ot_response = {
        "data": {
            "target": {
                "id": "ENSG00000146648",
                "approvedSymbol": "EGFR",
                "drugAndClinicalCandidates": {
                    "count": 1,
                    "rows": [
                        {
                            "id": "mock_id",
                            "maxClinicalStage": "PHASE_4",
                            "drug": {
                                "id": "CHEMBL1201585",
                                "name": "OSIMERTINIB",
                                "drugType": "Small molecule",
                                "maximumClinicalStage": "PHASE_4",
                                "mechanismsOfAction": {
                                    "rows": [
                                        {
                                            "mechanismOfAction": "EGFR inhibitor",
                                            "targetName": "EGFR",
                                        }
                                    ]
                                },
                            },
                        }
                    ],
                },
            }
        }
    }
    with patch.object(client, "post_json", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_ot_response
        drugs = await client.get_known_drugs("ENSG00000146648.15")
        assert len(drugs) == 1
        assert drugs[0]["prefName"] == "OSIMERTINIB"
        assert drugs[0]["phase"] == 4


@pytest.mark.asyncio
async def test_open_targets_parses_disease_and_pair_evidence():
    client = OpenTargetsClient()
    response = {
        "data": {
            "target": {
                "approvedSymbol": "MET",
                "drugAndClinicalCandidates": {
                    "rows": [
                        {
                            "maxClinicalStage": "PHASE_3",
                            "diseases": [
                                {
                                    "diseaseFromSource": "NSCLC",
                                    "disease": {
                                        "id": "EFO_0003060",
                                        "name": "non-small cell lung cancer",
                                    },
                                }
                            ],
                            "clinicalReports": [
                                {
                                    "id": "nct00000001",
                                    "source": "AACT",
                                    "clinicalStage": "PHASE_2",
                                    "trialOverallStatus": "RECRUITING",
                                    "url": "https://clinicaltrials.gov/study/NCT00000001",
                                    "title": "Osimertinib plus capmatinib in NSCLC",
                                }
                            ],
                            "drug": {
                                "id": "CHEMBL1",
                                "name": "CAPMATINIB",
                                "drugType": "Small molecule",
                                "mechanismsOfAction": {"rows": []},
                            },
                        }
                    ]
                },
            }
        }
    }
    with patch.object(client, "post_json", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        drugs = await client.get_known_drugs(
            "ENSG00000105976",
            cancer_type="Non-Small Cell Lung Cancer",
            primary_drug="Osimertinib",
        )

    assert drugs[0]["indicationMatch"] is True
    assert drugs[0]["combinationEvidence"] is True
    assert drugs[0]["clinicalStatus"] == "active_or_completed"
    assert drugs[0]["evidence"][0].stable_id == "nct00000001"


@pytest.mark.asyncio
async def test_open_targets_does_not_fabricate_unknown_phase_or_status():
    client = OpenTargetsClient()
    response = {
        "data": {
            "target": {
                "approvedSymbol": "MET",
                "drugAndClinicalCandidates": {
                    "rows": [{"drug": {"id": "CHEMBL1", "name": "Example"}}]
                },
            }
        }
    }
    with patch.object(client, "post_json", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        drugs = await client.get_known_drugs("ENSG00000105976")
    assert drugs[0]["phase"] is None
    assert drugs[0]["clinicalStatus"] == "status_not_reported"


def test_cache_key_determinism():
    """Verify cache keys are stable regardless of dict key insertion order."""
    params_a = {"species": 9606, "identifiers": "EGFR", "required_score": 400}
    params_b = {"required_score": 400, "species": 9606, "identifiers": "EGFR"}
    key_a = _stable_cache_key("GET", "https://example.com/api", params_a)
    key_b = _stable_cache_key("GET", "https://example.com/api", params_b)
    assert key_a == key_b

    # None payload produces stable key
    key_none = _stable_cache_key("POST", "https://example.com/api", None)
    assert "null" in key_none
