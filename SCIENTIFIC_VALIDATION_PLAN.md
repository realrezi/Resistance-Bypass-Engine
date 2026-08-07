# Scientific and Clinical Validation Plan

## Purpose

This document defines how to make the Targeted Oncology Resistance Bypass Engine more scientifically defensible and clinically useful without reducing it to an empty application or deleting its existing oncology knowledge.

The project is a research and evidence-prioritization system. It is not a treatment recommendation system, a diagnostic device, or proof that a drug combination benefits patients. Its purpose is to organize resistance biology and clinical evidence so that qualified experts can review hypotheses efficiently.

## Core principle: preserve, audit, improve

Existing clinical scenarios, mechanisms, prevalence statements, gene annotations, structural references, UI explanations, and live data integrations are valuable project assets. They must not be deleted merely because they are difficult to validate.

They also must not be treated as automatically correct.

Every existing medical claim will move through an explicit review state:

- **Unreviewed:** inherited from the current project and awaiting source verification.
- **Supported:** backed by an identifiable, relevant, and current source.
- **Partially supported:** directionally supported, but missing specificity, population, or clinical context.
- **Conflicted:** credible sources disagree or report materially different estimates.
- **Outdated:** once supported but no longer current or clinically representative.
- **Unsupported:** no adequate source was found after a documented search.
- **Retired:** removed from user-facing recommendations only after recording the claim, reason, reviewer, and replacement or uncertainty statement.

The review process improves or qualifies content before considering retirement. No bulk deletion, replacement with generic text, or silent rewriting is allowed.

## Scope and intended use

The engine may:

- organize acquired-resistance mechanisms;
- connect targets, alterations, pathways, drugs, diseases, and trials;
- prioritize evidence-linked research hypotheses;
- expose uncertainty and evidence gaps;
- support expert literature and trial review.

The engine must not:

- claim that a target association proves combination efficacy;
- present a computational rank as a clinical recommendation;
- infer patient suitability, dose, schedule, or expected response;
- hide contradictory, negative, withdrawn, or missing evidence;
- use a prevalence estimate without population and source context.

## Scientific claims model

Every displayed medical claim should be represented as a structured record with:

- claim text;
- claim type;
- disease and treatment context;
- gene, protein, alteration, and transcript where applicable;
- evidence source and stable identifier;
- publication or database date;
- evidence direction: supports, contradicts, or is neutral;
- evidence level;
- population, model system, and sample size when available;
- reviewer and review date;
- limitations and unresolved questions.

The minimum claim types are:

1. resistance mechanism;
2. alteration prevalence;
3. drug–target relationship;
4. target dependency or pathway activity;
5. pharmacologic activity;
6. disease relevance;
7. pair-level combination evidence;
8. safety or feasibility;
9. structural or molecular interpretation.

## Evidence hierarchy

Evidence level must describe directness, not merely prestige:

1. randomized clinical evidence for the relevant disease, alteration, and combination;
2. prospective clinical combination or basket-trial evidence;
3. well-controlled retrospective or real-world clinical evidence;
4. patient-derived, organoid, xenograft, or other translational models;
5. controlled cell-line or molecular experiments;
6. curated pathway, target, or pharmacology databases;
7. computational inference or network proximity alone.

Evidence levels are not interchangeable. A high-level study in a different disease or alteration may be less relevant than a lower-level study in the exact context. Reports must show both evidence strength and context match.

## Directness rules

The system will separate the following claims:

- a drug binds or inhibits a target;
- a target is associated with a resistance mechanism;
- the mechanism occurs in the requested disease;
- both drugs were evaluated together;
- the pair produced a clinical or experimental response.

Only explicit pair-level evidence may support a strong combination-evidence flag. A drug targeting a bypass node must not be described as clinically validated in combination merely because both drugs appear in the same disease or pathway.

## Existing content audit

The first implementation phase is an inventory, not a rewrite.

For each existing scenario and annotation:

1. extract the exact claim;
2. identify the current source, if any;
3. classify the claim type and alteration context;
4. search for primary literature, guidelines, trials, or authoritative databases;
5. record supporting and contradicting evidence;
6. check whether the wording overstates causality, prevalence, approval, or efficacy;
7. propose corrected wording and confidence;
8. obtain expert review for medically consequential changes.

The original wording and provenance remain available in the audit history. User-facing text may be corrected, qualified, or retired, but never silently discarded.

## Variant and alteration normalization

Mutations, amplifications, fusions, deletions, copy-number changes, expression changes, and pathway activation must be modeled separately.

Where possible, records should include:

- HGVS genomic and protein notation;
- gene symbol and stable identifier;
- transcript accession and version;
- genome build and genomic coordinates;
- alteration type;
- zygosity or copy-number information;
- biomarker assay and cutoff;
- resistance direction and treatment exposure;
- normalized disease and therapy context.

Free-text input may remain available for usability, but normalized structured data must drive matching and evidence retrieval.

## Source and provenance requirements

Each source record should include:

- source name and URL;
- DOI, PMID, NCT, database identifier, or equivalent stable ID;
- source version or release date;
- retrieval timestamp;
- exact structured fields or excerpt used;
- licensing and attribution requirements;
- parser or mapping version.

Live API data must be cached with a timestamp and source version where available. Reports should disclose when a result was generated and whether a source has changed since a previous report.

## Conflict, negative, and missing evidence

The system must not collapse disagreement into an average score.

- Contradictory evidence is displayed with direction, context, and study quality.
- Negative studies remain visible when relevant to the requested context.
- Withdrawn, terminated, or failed trials are not treated as positive clinical evidence.
- Missing evidence produces an uncertainty flag or abstention, not a default positive value.
- A lack of evidence is never described as evidence of no effect.

## Ranking and uncertainty

The engine may retain a computational prioritization score, but it must be decomposed into interpretable dimensions:

- mechanism relevance;
- disease and biomarker match;
- target engagement and pharmacology;
- pair-level evidence;
- clinical feasibility;
- evidence recency;
- uncertainty and contradiction penalties.

Scores must be calibrated against a locked benchmark where possible. The output must include the evidence components, their missingness, confidence intervals or ranges when available, and an abstention state when evidence is too weak or contradictory.

No score may be labeled synergy unless it is calculated from an explicitly defined experimental synergy assay and validated for that use.

## Benchmark and validation design

The benchmark must include positive, negative, ambiguous, and no-evidence cases across multiple tumor types and alteration classes.

To reduce bias:

- benchmark labels are frozen before model or rule tuning;
- cases are separated by time to prevent information leakage;
- publications used for tuning are not reused for final evaluation;
- the benchmark includes failed and non-replicating hypotheses;
- performance is reported by disease, alteration type, and evidence level.

Metrics should include precision at useful review cutoffs, recall, ranking stability, calibration, abstention quality, and error analysis. A benchmark is evidence about system behavior, not proof of patient benefit.

## Clinical and pharmacologic feasibility

When data are available, the engine should flag:

- overlapping or dose-limiting toxicities;
- pharmacokinetic or pharmacodynamic conflicts;
- inadequate exposure or tissue penetration;
- target expression or dependency concerns;
- incompatible schedules;
- disease-stage or line-of-therapy mismatch;
- lack of combination feasibility despite biological rationale.

Unknown safety or feasibility must be reported as unknown rather than inferred to be acceptable.

## Expert review and change control

Medical-content changes require a review record containing:

- original claim;
- proposed change;
- evidence considered;
- reviewer identity or role;
- decision and rationale;
- date and version;
- unresolved disagreement.

Code review and medical review are separate concerns. Passing tests does not approve a clinical claim, and clinical approval does not replace software testing.

## Reproducibility and release process

Every release should provide:

- source and dependency versions;
- schema and scoring version;
- query timestamp;
- reproducible request payload;
- evidence manifest;
- deterministic computation settings;
- known limitations and changed claims.

Release stages are:

1. audit and inventory;
2. schema and provenance implementation;
3. content correction and expert review;
4. deterministic tests and benchmark evaluation;
5. internal clinical review;
6. staging deployment and smoke testing;
7. explicit approval for production deployment.

No GitHub push, merge, or Vercel deployment occurs automatically as part of an implementation step.

## Acceptance criteria

The project is ready for the next stage only when:

- existing clinical content has been inventoried;
- every user-facing claim has a review state;
- sources and retrieval dates are available for supported claims;
- unsupported or conflicted claims are visibly qualified;
- pair-level evidence is not confused with target-level evidence;
- alteration types are structurally distinguishable;
- missing and negative evidence are handled explicitly;
- benchmark cases and evaluation rules are documented;
- regression tests show that useful existing workflows still work;
- an expert reviewer has approved the medically consequential wording;
- deployment remains a separate, explicitly approved action.

## Initial implementation sequence

The first coding phase should make no ranking change. It should:

1. add a claim/provenance schema;
2. inventory existing curated content into an auditable manifest;
3. add review-state and evidence-status fields;
4. add regression tests for existing scenarios and API compatibility;
5. produce a review report listing unsupported, outdated, conflicted, and high-priority claims.

Only after that report is reviewed should the project change medical wording, evidence weighting, or candidate ranking.

## Final standard

The project should be judged by whether a qualified reviewer can answer, for every important output:

> What exactly is being claimed, which evidence supports it, how direct and current is that evidence, what contradicts it, what is computational inference, and what remains unknown?

If the system cannot answer those questions, it should lower confidence or abstain rather than manufacture certainty.
