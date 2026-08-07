from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlterationType(str, Enum):
    mutation = "mutation"
    amplification = "amplification"
    fusion = "fusion"
    overexpression = "overexpression"
    loss = "loss"
    unknown = "unknown"


class IDMappingResult(BaseModel):
    original_input: str
    canonical_symbol: str
    ensembl_id: str
    uniprot_id: str
    chembl_target_id: str | None = None


class ResistanceRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    primary_drug: str = Field(min_length=1, max_length=120)
    primary_target: str = Field(min_length=1, max_length=40)
    resistance_marker: str = Field(min_length=1, max_length=40)
    cancer_type: str = Field(
        default="Non-Small Cell Lung Cancer", min_length=1, max_length=160
    )
    primary_alteration: str | None = Field(default=None, max_length=160)
    resistance_alteration: str | None = Field(default=None, max_length=160)
    resistance_alteration_type: AlterationType = AlterationType.unknown

    @field_validator("primary_target", "resistance_marker")
    @classmethod
    def clean_target_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator(
        "primary_drug", "cancer_type", "primary_alteration", "resistance_alteration"
    )
    @classmethod
    def clean_free_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None


class EvidenceReference(BaseModel):
    source: str
    record_id: str | None = None
    url: str | None = None
    title: str | None = None
    status: str | None = None


class ScoreComponents(BaseModel):
    topology: float = Field(ge=0.0, le=1.0)
    proximity: float = Field(ge=0.0, le=1.0)
    pharmacology: float | None = Field(default=None, ge=0.0, le=1.0)
    clinical_evidence: float = Field(ge=0.0, le=1.0)


class CombinationCandidate(BaseModel):
    secondary_drug: str
    secondary_target: str
    mechanism_of_action: str
    clinical_phase: int = Field(ge=0, le=4)
    clinical_status: str = "unknown"
    indication_match: bool = False
    combination_evidence: bool = False
    is_withdrawn: bool = False
    combination_priority_score: float = Field(ge=0.0, le=1.0)
    # Retained for API compatibility. This is a heuristic priority score, not
    # experimental drug synergy.
    synergy_score: float = Field(ge=0.0, le=1.0)
    score_components: ScoreComponents
    hub_penalized_centrality: float = Field(ge=0.0, le=1.0)
    shortest_path_distance: float | None = Field(default=None, ge=0.0)
    median_pchembl: float | None = None
    activity_measurements: int = Field(default=0, ge=0)
    biological_rationale: str
    evidence: list[EvidenceReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResistanceBypassReport(BaseModel):
    primary_drug: str
    cancer_type: str
    primary_target_canonical: str
    resistance_marker_canonical: str
    primary_alteration: str | None = None
    resistance_alteration: str | None = None
    resistance_type: str
    pathway_nodes_count: int = Field(ge=1)
    shortest_path_distance: float
    score_label: str = "Heuristic Combination Priority Score"
    methodology_version: str = "0.2.0"
    ranked_combinations: list[CombinationCandidate]
    network_nodes: list[dict[str, Any]] = Field(default_factory=list)
    network_edges: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
