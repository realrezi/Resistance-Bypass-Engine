from datetime import date

import pytest
from pydantic import ValidationError

from src.schemas.evidence import (
    ClaimType,
    EvidenceDirection,
    EvidenceLevel,
    EvidenceSource,
    ReviewState,
    ScientificClaim,
)
from src.schemas.models import ResistanceRequest, request_fingerprint


def test_scientific_claim_records_provenance_and_context():
    claim = ScientificClaim(
        claim_id="CLM-001",
        claim_text="MET amplification may mediate acquired resistance.",
        claim_type=ClaimType.RESISTANCE_MECHANISM,
        gene_symbol="MET",
        alteration="amplification",
        alteration_type="amplification",
        disease_context="Non-Small Cell Lung Cancer",
        treatment_context="EGFR inhibitor after progression",
        evidence=[
            EvidenceSource(
                name="Example registry",
                stable_id="NCT00000000",
                url="https://clinicaltrials.gov/study/NCT00000000",
                retrieved_at=date(2026, 8, 8),
                level=EvidenceLevel.PROSPECTIVE_CLINICAL,
                direction=EvidenceDirection.SUPPORTS,
            )
        ],
    )

    assert claim.review_state is ReviewState.UNREVIEWED
    assert claim.evidence[0].stable_id == "NCT00000000"


def test_claim_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ScientificClaim(
            claim_id="CLM-002",
            claim_text="Unknown field must not be silently accepted.",
            claim_type=ClaimType.STRUCTURAL,
            unsupported_field=True,
        )


def test_request_fingerprint_is_stable_and_non_identifying():
    first = ResistanceRequest(
        primary_drug="Osimertinib",
        primary_target="EGFR",
        resistance_marker="MET",
    )
    second = ResistanceRequest(
        primary_drug="Osimertinib",
        primary_target=" egfr ",
        resistance_marker=" met ",
    )
    assert request_fingerprint(first) == request_fingerprint(second)
    assert len(request_fingerprint(first)) == 64
