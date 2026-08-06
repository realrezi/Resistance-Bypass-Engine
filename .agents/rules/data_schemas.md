# Rule: Pydantic Data Models & Validation Schemas
Write Python code in `src/schemas/models.py` for:
- `IDMappingResult`: `original_input: str`, `canonical_symbol: str`, `ensembl_id: str`, `uniprot_id: str`, `chembl_target_id: Optional[str] = None`.
- `ResistanceRequest`: `primary_drug: str`, `primary_target: str`, `resistance_marker: str`, `cancer_type: Optional[str] = "Non-Small Cell Lung Cancer"`. Use `@field_validator` on targets to run `.strip().upper()`.
- `CombinationCandidate`: `secondary_drug: str`, `secondary_target: str`, `mechanism_of_action: str`, `clinical_phase: int`, `is_withdrawn: bool = False`, `synergy_score: float`, `hub_penalized_centrality: float`, `chembl_ic50_nm: Optional[float] = None`, `biological_rationale: str`.
- `ResistanceBypassReport`: `primary_target_canonical: str`, `resistance_marker_canonical: str`, `resistance_type: str`, `pathway_nodes_count: int`, `shortest_path_distance: float`, `ranked_combinations: List[CombinationCandidate]`.
