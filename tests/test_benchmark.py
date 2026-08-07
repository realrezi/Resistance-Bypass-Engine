from datetime import date

import pytest

from src.validation.benchmark import (
    BenchmarkCase,
    BenchmarkLabel,
    BenchmarkPrediction,
    evaluate_rankings,
)


def case(case_id: str, label: BenchmarkLabel) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        disease_context="Non-Small Cell Lung Cancer",
        treatment_context="after EGFR inhibitor progression",
        primary_target="EGFR",
        resistance_target="MET",
        label=label,
        evidence_cutoff=date(2025, 1, 1),
        source_ids=["PMID:123"],
        reviewer="clinical-reviewer",
        rationale="Reviewed benchmark fixture.",
        temporal_split="test",
    )


def test_benchmark_evaluator_separates_positive_retrieval_and_abstention():
    cases = [
        case("positive-1", BenchmarkLabel.POSITIVE),
        case("unknown-1", BenchmarkLabel.NO_EVIDENCE),
    ]
    predictions = [
        BenchmarkPrediction(case_id="positive-1", ranked_targets=["MET"]),
        BenchmarkPrediction(case_id="unknown-1", abstained=True),
    ]

    metrics = evaluate_rankings(cases, predictions, k=5)

    assert metrics["cases_evaluated"] == 2
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["abstention_coverage_on_ambiguous_or_no_evidence"] == 1.0
    assert metrics["abstention_rate_on_ambiguous_or_no_evidence"] == 1.0


def test_benchmark_evaluator_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_rankings([], [], k=0)
