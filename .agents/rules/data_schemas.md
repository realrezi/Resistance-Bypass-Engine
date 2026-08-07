# Rule: Pydantic Data Models & Validation Schemas

- Validate non-empty, bounded primary drug, target, resistance marker, and cancer type.
- Normalize canonical target inputs but preserve human-readable drug/cancer text.
- Accept optional primary/resistance alteration text and a structured alteration type.
- Candidate responses must contain a heuristic combination-priority score, decomposed score components, clinical status, indication/pair-evidence flags, provenance references, and candidate limitations.
- Keep the legacy `synergy_score` only as a compatibility mirror; documentation and UI must not describe it as experimental synergy.
- Reports must include request context, methodology version, research-safety warnings, source provenance, relevant component nodes/edges, and evidence-linked candidates.
- Unknown scientific values must be `None` or empty collections, never synthetic placeholders.
