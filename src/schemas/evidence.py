"""Structured evidence and review metadata for scientific claims.

These models are additive. They do not replace the existing response schema until
the claim inventory has been reviewed and the migration is approved.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ClaimType(StrEnum):
    RESISTANCE_MECHANISM = "resistance_mechanism"
    ALTERATION_PREVALENCE = "alteration_prevalence"
    DRUG_TARGET = "drug_target"
    TARGET_DEPENDENCY = "target_dependency"
    PHARMACOLOGY = "pharmacology"
    DISEASE_RELEVANCE = "disease_relevance"
    PAIR_COMBINATION = "pair_combination"
    SAFETY_FEASIBILITY = "safety_feasibility"
    STRUCTURAL = "structural"
    COMPUTATIONAL = "computational"


class ReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTED = "conflicted"
    OUTDATED = "outdated"
    UNSUPPORTED = "unsupported"
    RETIRED = "retired"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceLevel(StrEnum):
    RANDOMIZED_CLINICAL = "randomized_clinical"
    PROSPECTIVE_CLINICAL = "prospective_clinical"
    RETROSPECTIVE_CLINICAL = "retrospective_clinical"
    TRANSLATIONAL_MODEL = "translational_model"
    CONTROLLED_EXPERIMENT = "controlled_experiment"
    CURATED_DATABASE = "curated_database"
    COMPUTATIONAL = "computational"


class EvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    stable_id: str | None = Field(default=None, max_length=240)
    url: HttpUrl | None = None
    release: str | None = Field(default=None, max_length=120)
    published_on: date | None = None
    retrieved_at: date
    direction: EvidenceDirection = EvidenceDirection.SUPPORTS
    level: EvidenceLevel
    excerpt_or_field: str | None = Field(default=None, max_length=2000)
    limitations: list[str] = Field(default_factory=list, max_length=20)


class ScientificClaim(BaseModel):
    """Auditable claim record; does not assert that a claim is true by itself."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=100)
    claim_text: str = Field(min_length=1, max_length=2000)
    claim_type: ClaimType
    review_state: ReviewState = ReviewState.UNREVIEWED
    approved_wording: str | None = Field(default=None, max_length=2000)
    gene_symbol: str | None = Field(default=None, max_length=32)
    alteration: str | None = Field(default=None, max_length=200)
    alteration_type: str | None = Field(default=None, max_length=64)
    disease_context: str | None = Field(default=None, max_length=240)
    treatment_context: str | None = Field(default=None, max_length=240)
    population: str | None = Field(default=None, max_length=500)
    evidence: list[EvidenceSource] = Field(default_factory=list, max_length=50)
    reviewer: str | None = Field(default=None, max_length=160)
    reviewed_on: date | None = None
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
