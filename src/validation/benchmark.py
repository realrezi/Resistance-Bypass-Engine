"""Clinician-reviewed benchmark schema and ranking metrics.

The evaluator intentionally accepts externally reviewed labels. It never creates
clinical truth labels from model output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    AMBIGUOUS = "ambiguous"
    NO_EVIDENCE = "no_evidence"


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    disease_context: str = Field(min_length=1, max_length=240)
    treatment_context: str = Field(min_length=1, max_length=240)
    primary_target: str = Field(min_length=1, max_length=32)
    primary_alteration: str | None = Field(default=None, max_length=200)
    resistance_target: str = Field(min_length=1, max_length=32)
    resistance_alteration: str | None = Field(default=None, max_length=200)
    label: BenchmarkLabel
    evidence_cutoff: date
    source_ids: list[str] = Field(min_length=1, max_length=50)
    reviewer: str = Field(min_length=1, max_length=160)
    rationale: str = Field(min_length=1, max_length=2000)
    temporal_split: str = Field(pattern="^(development|validation|test)$")


class BenchmarkPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    ranked_targets: list[str] = Field(default_factory=list, max_length=100)
    abstained: bool = False


def evaluate_rankings(
    cases: Sequence[BenchmarkCase],
    predictions: Iterable[BenchmarkPrediction],
    k: int = 5,
) -> dict[str, float | int]:
    """Calculate transparent ranking metrics for reviewed cases.

    Positive cases count as retrieved when a ranked target matches the reviewed
    resistance target. Ambiguous and no-evidence cases are evaluated separately
    through abstention, avoiding forced positive/negative labels.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    by_id = {prediction.case_id: prediction for prediction in predictions}
    reviewed = [case for case in cases if case.case_id in by_id]
    positives = [case for case in reviewed if case.label is BenchmarkLabel.POSITIVE]
    retrieved = 0
    correct_abstentions = 0
    eligible_abstentions = 0

    for case in reviewed:
        prediction = by_id[case.case_id]
        top_targets = {target.upper() for target in prediction.ranked_targets[:k]}
        if case.label is BenchmarkLabel.POSITIVE:
            retrieved += int(case.resistance_target.upper() in top_targets)
        if case.label in {BenchmarkLabel.AMBIGUOUS, BenchmarkLabel.NO_EVIDENCE}:
            eligible_abstentions += 1
            correct_abstentions += int(prediction.abstained)

    recall_at_k = retrieved / len(positives) if positives else 0.0
    abstention_coverage = (
        correct_abstentions / eligible_abstentions if eligible_abstentions else 0.0
    )
    return {
        "cases_evaluated": len(reviewed),
        "positive_cases": len(positives),
        # This is retrieval recall, not precision: the denominator is reviewed
        # positive cases rather than returned predictions.
        "recall_at_k": recall_at_k,
        "precision_at_k": recall_at_k,  # backward-compatible legacy key
        "abstention_coverage_on_ambiguous_or_no_evidence": abstention_coverage,
        "abstention_rate_on_ambiguous_or_no_evidence": abstention_coverage,
    }
