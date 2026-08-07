import json
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.schemas.evidence import EvidenceSource, ScientificClaim


class AlterationType(StrEnum):
    MUTATION = "mutation"
    AMPLIFICATION = "amplification"
    DELETION = "deletion"
    FUSION = "fusion"
    OVEREXPRESSION = "overexpression"
    SPLICE_VARIANT = "splice_variant"
    ACTIVATION = "activation"
    UNKNOWN = "unknown"


class ReportMetadata(BaseModel):
    schema_version: str
    methodology_version: str
    generated_at: datetime
    request_fingerprint: str
    sources: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    source_timings_ms: dict[str, float] = Field(default_factory=dict)
    partial_sources: list[str] = Field(default_factory=list)


class IDMappingResult(BaseModel):
    original_input: str
    canonical_symbol: str
    ensembl_id: str
    uniprot_id: str
    chembl_target_id: str | None = None


class ResistanceRequest(BaseModel):
    primary_drug: str = Field(min_length=1, max_length=160)
    primary_target: str = Field(min_length=1, max_length=64)
    resistance_marker: str = Field(min_length=1, max_length=64)
    # Omission must not silently apply an NSCLC-specific clinical filter.
    cancer_type: str | None = Field(default=None, max_length=200)
    primary_alteration: str | None = Field(default=None, max_length=200)
    resistance_alteration: str | None = Field(default=None, max_length=200)
    resistance_alteration_type: AlterationType | None = None
    treatment_line: str | None = Field(default=None, max_length=200)

    @field_validator(
        "primary_target",
        "resistance_marker",
        "primary_alteration",
        "resistance_alteration",
        "cancer_type",
        "treatment_line",
    )
    @classmethod
    def clean_input(cls, v: str, info: ValidationInfo) -> str:
        if v:
            normalized = v.strip()
            if info.field_name in {"primary_target", "resistance_marker"}:
                return normalized.upper()
            return normalized
        return v


class CombinationCandidate(BaseModel):
    secondary_drug: str
    secondary_target: str
    mechanism_of_action: str
    clinical_phase: int | None = None
    is_withdrawn: bool = False
    synergy_score: float
    hub_penalized_centrality: float
    chembl_ic50_nm: float | None = None
    biological_rationale: str
    score_components: dict[str, float | None] | None = None
    evidence: list[EvidenceSource] = Field(default_factory=list)
    clinical_status: str | None = None
    indication_match: bool | None = None
    combination_evidence: bool | None = None
    indications: list[str] = Field(default_factory=list)
    rank: int | None = None
    tie_group: int | None = None
    tie_reason: str | None = None
    evidence_completeness: float | None = None
    shortest_path_distance: float | None = None
    target_in_graph: bool | None = None
    scoring_status: str | None = None
    evidence_status: str | None = None
    evidence_notes: list[str] = Field(default_factory=list)


class ResistanceBypassReport(BaseModel):
    primary_target_canonical: str
    resistance_marker_canonical: str
    resistance_type: str
    pathway_nodes_count: int
    shortest_path_distance: float | None = None
    ranked_combinations: list[CombinationCandidate]
    network_nodes: list[dict[str, Any]] | None = None
    network_edges: list[dict[str, Any]] | None = None
    evidence_claims: list[ScientificClaim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    primary_alteration: str | None = None
    resistance_alteration: str | None = None
    resistance_alteration_type: AlterationType | None = None
    treatment_line: str | None = None
    metadata: ReportMetadata | None = None


def request_fingerprint(request: ResistanceRequest) -> str:
    """Create a stable, non-identifying fingerprint for reproducibility."""
    payload = request.model_dump(mode="json", exclude_none=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
