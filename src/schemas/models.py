from typing import List, Optional
from pydantic import BaseModel, field_validator


class IDMappingResult(BaseModel):
    original_input: str
    canonical_symbol: str
    ensembl_id: str
    uniprot_id: str
    chembl_target_id: Optional[str] = None


class ResistanceRequest(BaseModel):
    primary_drug: str
    primary_target: str
    resistance_marker: str
    cancer_type: Optional[str] = "Non-Small Cell Lung Cancer"

    @field_validator("primary_target", "resistance_marker")
    @classmethod
    def clean_target_symbol(cls, v: str) -> str:
        if v:
            return v.strip().upper()
        return v


class CombinationCandidate(BaseModel):
    secondary_drug: str
    secondary_target: str
    mechanism_of_action: str
    clinical_phase: int
    is_withdrawn: bool = False
    synergy_score: float
    hub_penalized_centrality: float
    chembl_ic50_nm: Optional[float] = None
    biological_rationale: str


class ResistanceBypassReport(BaseModel):
    primary_target_canonical: str
    resistance_marker_canonical: str
    resistance_type: str
    pathway_nodes_count: int
    shortest_path_distance: float
    ranked_combinations: List[CombinationCandidate]
