# Scientific Claims Audit

Status: initial inventory; no claim has been silently rewritten or deleted.

This manifest records the current project claims that require source verification. A claim marked `unreviewed` is not necessarily false; it means that the repository does not yet provide enough structured provenance to support the exact wording, population, and context.

## Review states

- `unreviewed`: present in the current application and awaiting verification.
- `supported`: verified against a relevant source with adequate context.
- `partially_supported`: directionally supported but wording, population, or context needs qualification.
- `conflicted`: credible sources disagree.
- `outdated`: previously supported but no longer current for the stated context.
- `unsupported`: no adequate source identified after a documented search.
- `retired`: removed from user-facing content only after recording the rationale and reviewer.

## High-priority claims requiring review

| ID | Current claim or behavior | Location | Review state | Required review |
|---|---|---|---|---|
| CLM-001 | EGFR L858R with MET amplification is reported as a 15–20% acquired-resistance scenario in NSCLC. | `src/main.py` clinical scenario matrix | partially_supported | Prospective and retrospective studies support MET amplification as a recurrent mechanism, but reported rates vary by treatment line, specimen, assay, and denominator. Do not publish a universal 15–20% estimate without specifying context. |
| CLM-002 | EGFR C797S is reported as a 7–10% gatekeeper-resistance scenario. | `src/main.py` clinical scenario matrix | partially_supported | C797S is a recognized mechanism, but reported frequency varies substantially by first-line versus later-line osimertinib and tissue versus plasma testing. The current range needs a named cohort. |
| CLM-003 | MET bypass after ALK inhibitor treatment is reported as 8–12%. | `src/main.py` clinical scenario matrix | unreviewed | Verify ALK fusion population, inhibitor generation, progression setting, and evidence type. |
| CLM-004 | HER2/MET bypass is reported as 10–15% in HER2-positive breast cancer. | `src/main.py` clinical scenario matrix | unreviewed | Verify disease subtype, treatment line, assay, and whether this is amplification, overexpression, or pathway activation. |
| CLM-005 | ESR1/CDK4 resistance after aromatase-inhibitor failure is reported as 25–40%. | `src/main.py` clinical scenario matrix | unreviewed | Separate ESR1 mutation prevalence from CDK4/6 pathway escape and identify the denominator. |
| CLM-006 | KRAS G12C colorectal-cancer feedback is reported as 70–85%. | `src/main.py` clinical scenario matrix | unsupported | Primary studies support EGFR/RAS-MAPK feedback as a resistance mechanism, but the reviewed sources do not support this exact 70–85% prevalence statement. It must be replaced with a context-specific metric or labeled as a mechanistic hypothesis. |
| CLM-007 | BRAF V600E colorectal-cancer feedback is reported as 75–85%. | `src/main.py` clinical scenario matrix | unreviewed | Verify population, treatment regimen, and whether the number is a response/resistance rate rather than a mutation prevalence. |
| CLM-008 | Dabrafenib plus trametinib is described as FDA-approved dual BRAF/MEK therapy. | `src/main.py` clinical scenario matrix | partially_supported | FDA labeling supports specific BRAF V600 melanoma and NSCLC indications, not every resistance context. The UI must retain disease, biomarker, and jurisdiction qualifiers. |
| CLM-009 | Encorafenib plus cetuximab is described as FDA-approved dual therapy. | `src/main.py` clinical scenario matrix | partially_supported | FDA supports the combination for specified BRAF V600E metastatic CRC indications; the claim must not imply universal approval or efficacy for other diseases or resistance settings. |
| CLM-010 | ABL1 T315I is described as a gatekeeper/master resistance mutation and ponatinib or asciminib as management options. | `src/main.py`, `src/services/gene_annotation.py` | unreviewed | Verify disease, treatment line, mutation context, and regulatory wording. |
| CLM-011 | MET, EGFR, ERBB2, ALK, KRAS, BRAF, PIK3CA, ESR1, ABL1, CDK4, MAP2K1, AKT1, ROS1, and RET are labeled as Tier 1 FDA-approved targets. | `src/services/gene_annotation.py` | unreviewed | Replace gene-level blanket labels with drug-, indication-, and jurisdiction-specific regulatory evidence. |
| CLM-012 | Static PDB IDs are displayed as representative structures for annotated genes. | `src/services/gene_annotation.py`, `src/main.py` | unreviewed | Verify that each structure represents the relevant protein/domain/state and record chain, ligand, and experimental method. |
| CLM-013 | COSMIC is presented as a source for clinical resistance hotspots and variant frequencies. | `src/main.py` | unreviewed | Add a concrete COSMIC release, field, access date, and attribution; do not imply that the current UI has live COSMIC frequency data. |
| CLM-014 | On-target analysis creates a fallback next-generation inhibitor when no evidence is returned. | `src/main.py` | unreviewed | Replace fabricated candidates with an explicit no-evidence result after medical review. Preserve the behavior in this audit until approved. |
| CLM-015 | Off-target analysis creates a fallback `${marker} Inhibitor` candidate when no clinical record is returned. | `src/main.py` | unreviewed | Remove fabricated candidate identity from user-facing output after adding an abstention response and regression coverage. |
| CLM-016 | A candidate with clinical phase 4 is rendered as FDA Approved. | `src/main.py` | unreviewed | Verify phase semantics and indication-specific authorization instead of equating global development stage with approval. |
| CLM-017 | Network proximity and hub-penalized centrality are used to rank dual-drug combinations. | `src/engine/scorer.py`, `README.md` | unreviewed | Preserve as computational prioritization, but label it as hypothesis generation and validate against a locked benchmark. |
| CLM-018 | STRING physical associations are treated as a signaling network for resistance interpretation. | `src/clients/string_db.py`, `src/engine/graph_builder.py` | unreviewed | Verify network type, confidence threshold, directionality limitations, tissue context, and causal interpretation. |

## Data inventories

### Curated application content

- Clinical scenario cards and prevalence ranges are embedded in `src/main.py`.
- Gene names, loci, stable identifiers, PDB IDs, druggability labels, hotspots, and pathways are embedded in `src/services/gene_annotation.py`.
- Structural images and static visual assets are stored under `src/static/` and `api/static/`.
- README and UI explanatory text contain additional mechanism and regulatory claims.

### Live evidence sources

- HGNC and UniProt provide identity and canonicalization data.
- STRING provides interaction data used for graph construction.
- Open Targets provides clinical candidate and trial-related data.
- ChEMBL provides target activity data.

The live sources currently do not, by themselves, prove that a proposed pair is effective in a specific patient population. Pair-level claims require direct trial or experimental evidence.

## Required evidence record for each reviewed claim

Each claim should eventually have:

- source name and stable identifier;
- publication, trial, guideline, or database release date;
- retrieval date;
- disease and biomarker context;
- treatment line and comparator where relevant;
- population and denominator for prevalence estimates;
- evidence type and level;
- directness: direct, indirect, or computational;
- supporting and contradicting sources;
- reviewer and review date;
- approved wording and limitations.

## Immediate correction candidates

These behaviors are high risk because they can manufacture clinical certainty even when upstream data are absent:

1. fabricated next-generation on-target candidates;
2. fabricated `${marker} Inhibitor` candidates;
3. gene-level FDA-approved labels;
4. phase-4-equals-approval rendering;
5. prevalence numbers without source or denominator;
6. PDB and COSMIC labels without record-level provenance.

The first four behaviors have been replaced with abstention or scoped wording. Remaining prevalence and structure claims are retained as review items, not presented as universal clinical facts. This audit intentionally records the original claims so that correction does not destroy the original medical context.

## Review gate

No claim in this manifest should be marked `supported` solely by automated retrieval. A medically consequential change requires source review, documented rationale, and regression tests demonstrating that the useful scenario remains represented.

## Verification pass: 2026-08-08

The following sources were reviewed for the initial high-priority pass:

- [FDA: Encorafenib traditional approval for BRAF V600E metastatic colorectal cancer](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-traditional-approval-encorafenib-metastatic-colorectal-cancer-braf-v600e-mutation) — supports a scoped regulatory statement for CLM-009 and identifies the BREAKWATER population and regimen.
- [FDA Tafinlar prescribing information](https://www.accessdata.fda.gov/drugsatfda_docs/label/2026/202806s40lbl.pdf) — supports indication-specific dabrafenib/trametinib labeling for CLM-008.
- [AURA3 acquired-resistance analysis](https://www.nature.com/articles/s41467-023-35962-x) — supports MET amplification and EGFR C797S as recurrent osimertinib-resistance mechanisms, while documenting assay and sampling limitations.
- [Prospective osimertinib resistance study](https://www.nature.com/articles/s41416-023-02475-9) — reports mechanism counts in a defined cohort and demonstrates why a single universal prevalence range is unsafe.
- [Prospective multicenter EGFR resistance study](https://www.nature.com/articles/s41392-025-02481-8) — reports different C797S and MET amplification percentages by treatment line and specimen type.
- [KRAS G12C/EGFR resistance characterization](https://pmc.ncbi.nlm.nih.gov/articles/PMC9827113/) — supports feedback biology in CRC but does not support the current 70–85% prevalence wording in CLM-006.

These sources support retaining the scenarios as hypotheses while replacing unqualified percentages with source-specific estimates, denominators, and assay context.

## Follow-up audit: 2026-08-08

The scenario matrix no longer displays unsupported universal percentages. It now labels frequency as cohort-, treatment-line-, and assay-dependent until a denominator and source record are attached. Clicking a scenario now passes its named alteration and treatment context into the request; previously those fields were left blank despite the card text.

The candidate UI now distinguishes a tied rank from a meaningful ordering. When candidates share the same target-level topology/proximity score and lack drug-specific pharmacology, the report returns the same rank and an explicit insufficiency reason. This prevents source-order sorting from being mistaken for comparative clinical evidence.

Static structure identifiers are now described as curated local PDB mappings rather than verified coverage of the supplied alteration. Users are directed to confirm chain, construct, ligand, and variant coverage in the linked RCSB record.

The following claims remain intentionally unresolved and require a named cohort or primary source before numeric wording is restored: ALK/MET 8–12%, HER2/MET 10–15%, ESR1/CDK4 25–40%, BRAF/MAP2K1 35–45%, AR/PI3K 40–50%, ovarian PIK3CA/KRAS 15–25%, glioma EGFR/MET 10–15%, and RET/MET 10–15%.

Additional software/interpretation findings fixed in this pass:

- Scenario cards now transmit the alteration and treatment context they display.
- Candidate ordering now exposes tied ranks when the available evidence cannot distinguish drugs.
- Dynamic candidate and annotation text is HTML-escaped before insertion into the workstation.
- The node inspector no longer labels its local hotspot list as live COSMIC frequencies.
- The illustrative pathway diagram is labeled as illustrative rather than as the computed patient-specific network.
