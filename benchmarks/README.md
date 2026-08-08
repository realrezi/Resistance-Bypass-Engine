# Validation Benchmark

This directory is reserved for a versioned, clinician-reviewed benchmark. It is intentionally not populated with invented labels.

`cases.template.json` is deliberately empty. A case must not be added until its label, source identifiers, cutoff date, and rationale have been reviewed.

Each case should eventually contain:

- a stable case identifier;
- disease and treatment context;
- standardized primary and resistance alterations;
- data-availability date;
- the evidence sources available at that date;
- expected class: positive, negative, ambiguous, or no-evidence;
- reviewer rationale and disagreement notes;
- a temporal split assignment.

## Rules

1. Freeze labels before tuning ranking rules.
2. Keep positive and negative cases from the same disease and treatment contexts where possible.
3. Do not use a publication for tuning and final evaluation simultaneously.
4. Preserve failed, withdrawn, and non-replicating hypotheses.
5. Report performance by disease, alteration type, evidence level, and how often the tool declined to rank a result because evidence was insufficient.
6. Do not treat benchmark performance as proof of patient benefit.

The first dataset should be assembled from reviewed clinical and translational sources, with explicit permission and licensing checks. Until that review is complete, the application must not report benchmark metrics.
