from unittest.mock import AsyncMock, patch

import pytest

from src.clients.base import (
    CACHE_TTL_SECONDS,
    USER_AGENT,
    BaseHTTPClient,
    _is_json_value,
    _stable_cache_key,
)
from src.clients.chembl import ChEMBLClient
from src.clients.open_targets import OpenTargetsClient
from src.clients.string_db import StringDBClient
from src.services.id_mapper import IDMapper


@pytest.mark.asyncio
async def test_ensembl_version_stripping():
    mapper = IDMapper()
    response = {
        "response": {
            "docs": [{"symbol": "EGFR", "ensembl_gene_id": "ENSG00000146648.15"}]
        }
    }
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mocked:
        mocked.return_value = response
        symbol, ensembl_id = await mapper.resolve_hgnc("EGFR")
    assert (symbol, ensembl_id) == ("EGFR", "ENSG00000146648")


@pytest.mark.asyncio
async def test_hgnc_alias_search_prefers_exact_alias():
    mapper = IDMapper()
    responses = [
        {"response": {"docs": []}},
        {
            "response": {
                "docs": [
                    {"symbol": "OTHER", "alias_symbol": ["X"]},
                    {"symbol": "ERBB2", "alias_symbol": ["HER2"]},
                ]
            }
        },
        {"response": {"docs": [{"symbol": "ERBB2", "ensembl_gene_id": "ENSG1"}]}},
    ]
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mocked:
        mocked.side_effect = responses
        symbol, _ = await mapper.resolve_hgnc("HER2")
    assert symbol == "ERBB2"


@pytest.mark.asyncio
async def test_chembl_single_protein_human_filtering():
    mapper = IDMapper()
    response = {
        "targets": [
            {
                "target_chembl_id": "WRONG",
                "target_type": "SINGLE PROTEIN",
                "organism": "Mus musculus",
            },
            {
                "target_chembl_id": "CHEMBL203",
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
            },
        ]
    }
    with patch.object(mapper, "get_json", new_callable=AsyncMock) as mocked:
        mocked.return_value = response
        target = await mapper.resolve_chembl_target("P00533")
    assert target == "CHEMBL203"


@pytest.mark.asyncio
async def test_chembl_exhaustive_activity_pagination_and_metadata():
    client = ChEMBLClient()
    page1 = {
        "activities": [
            {
                "molecule_pref_name": "Drug A",
                "pchembl_value": "8.0",
                "standard_type": "IC50",
                "molecule_chembl_id": "CHEMBL1",
                "assay_chembl_id": "ASSAY1",
                "document_chembl_id": "DOC1",
            }
        ],
        "page_meta": {"next": "/chembl/api/data/activity.json?offset=1"},
    }
    page2 = {
        "activities": [
            {
                "molecule_pref_name": "Drug A",
                "pchembl_value": "6.0",
                "standard_type": "Ki",
                "molecule_chembl_id": "CHEMBL1",
                "assay_chembl_id": "ASSAY2",
                "document_chembl_id": "DOC2",
            }
        ],
        "page_meta": {"next": None},
    }
    with patch.object(client, "get_json", new_callable=AsyncMock) as mocked:
        mocked.side_effect = [page1, page2]
        result = await client.get_target_activities("CHEMBL3714")
    assert mocked.call_count == 2
    assert result["DRUG A"]["median_pchembl"] == 7.0
    assert result["DRUG A"]["measurement_count"] == 2
    assert result["DRUG A"]["measurement_types"] == ["IC50", "Ki"]


@pytest.mark.asyncio
async def test_string_client_requests_physical_network_and_identity():
    client = StringDBClient()
    with patch.object(client, "get_json", new_callable=AsyncMock) as mocked:
        mocked.return_value = []
        await client.get_network("EGFR", "MET")
    _, kwargs = mocked.call_args
    assert kwargs["params"]["network_type"] == "physical"
    assert kwargs["params"]["required_score"] == 400
    assert kwargs["params"]["caller_identity"].startswith("mailto:")


@pytest.mark.asyncio
async def test_open_targets_parses_indication_trial_and_withdrawal_evidence():
    client = OpenTargetsClient()
    response = {
        "data": {
            "target": {
                "approvedSymbol": "MET",
                "drugAndClinicalCandidates": {
                    "rows": [
                        {
                            "id": "row1",
                            "maxClinicalStage": "PHASE_4",
                            "diseases": [
                                {
                                    "diseaseFromSource": "NSCLC",
                                    "disease": {
                                        "id": "EFO_1",
                                        "name": "non-small cell lung cancer",
                                    },
                                }
                            ],
                            "clinicalReports": [
                                {
                                    "id": "NCT1",
                                    "source": "AACT",
                                    "clinicalStage": "PHASE_3",
                                    "trialOverallStatus": "RECRUITING",
                                    "url": "https://clinicaltrials.gov/study/NCT1",
                                    "title": "Osimertinib plus capmatinib",
                                }
                            ],
                            "drug": {
                                "id": "CHEMBL1",
                                "name": "CAPMATINIB",
                                "drugType": "Small molecule",
                                "maximumClinicalStage": "PHASE_4",
                                "drugWarnings": [],
                                "mechanismsOfAction": {
                                    "rows": [
                                        {
                                            "mechanismOfAction": "MET inhibitor",
                                            "targetName": "MET",
                                        }
                                    ]
                                },
                            },
                        }
                    ]
                },
            }
        }
    }
    with patch.object(client, "post_json", new_callable=AsyncMock) as mocked:
        mocked.return_value = response
        drugs = await client.get_known_drugs(
            "ENSG00000105976", "Non-Small Cell Lung Cancer", "Osimertinib"
        )
    assert drugs[0]["indicationMatch"] is True
    assert drugs[0]["combinationEvidence"] is True
    assert drugs[0]["clinicalStatus"] == "active_or_completed"
    assert drugs[0]["evidence"][0]["record_id"] == "NCT1"


@pytest.mark.asyncio
async def test_open_targets_surfaces_graphql_errors():
    client = OpenTargetsClient()
    with patch.object(client, "post_json", new_callable=AsyncMock) as mocked:
        mocked.return_value = {"errors": [{"message": "schema changed"}]}
        with pytest.raises(RuntimeError, match="schema changed"):
            await client.get_known_drugs("ENSG1")


def test_cache_contract_and_key_determinism():
    a = {"species": 9606, "identifiers": "EGFR"}
    b = {"identifiers": "EGFR", "species": 9606}
    assert _stable_cache_key("GET", "https://example.test", a) == _stable_cache_key(
        "GET", "https://example.test", b
    )
    assert _is_json_value({"a": [1, None, True]})
    assert not _is_json_value({"a": object()})
    assert CACHE_TTL_SECONDS == 604800
    assert "mailto:" in USER_AGENT


def test_base_client_has_compliance_headers():
    client = BaseHTTPClient()
    assert client.headers["User-Agent"] == USER_AGENT
    assert client.headers["From"]
