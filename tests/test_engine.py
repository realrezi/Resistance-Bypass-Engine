import networkx as nx
import pytest

from src.engine.graph_builder import HUB_EXCLUSION_SET, build_signaling_graph
from src.engine.scorer import PathwayScorer, clamp01


def test_graph_strips_self_loops_and_keeps_highest_confidence_duplicate():
    graph = build_signaling_graph(
        [
            {"preferredName_A": "EGFR", "preferredName_B": "EGFR", "score": 900},
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 700},
            {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900},
        ]
    )
    assert list(nx.selfloop_edges(graph)) == []
    assert graph.number_of_edges() == 1
    assert graph["EGFR"]["MET"]["score"] == pytest.approx(0.9)
    assert graph["EGFR"]["MET"]["weight"] == pytest.approx(0.105)


def test_graph_excludes_only_technical_hubs_and_retains_tp53():
    graph = build_signaling_graph(
        [
            {"preferredName_A": "EGFR", "preferredName_B": "TP53", "score": 950},
            {"preferredName_A": "MET", "preferredName_B": "UBB", "score": 900},
        ]
    )
    assert "TP53" in graph
    assert "UBB" not in graph
    assert HUB_EXCLUSION_SET == {"UBC", "UBB", "RPS27A"}


def test_invalid_or_noncanonical_interactions_are_skipped():
    graph = build_signaling_graph(
        [
            {"stringId_A": "9606.A", "stringId_B": "9606.B", "score": 900},
            {"preferredName_A": "A", "preferredName_B": "B", "score": "bad"},
            {"preferredName_A": "C", "preferredName_B": "D"},
        ]
    )
    assert graph.number_of_nodes() == 0


def test_relevant_component_must_contain_both_requested_targets():
    graph = nx.Graph()
    graph.add_edge("EGFR", "MET", weight=0.1)
    graph.add_edge("KRAS", "BRAF", weight=0.1)
    relevant = PathwayScorer.extract_relevant_component(graph, {"EGFR", "MET"})
    assert set(relevant) == {"EGFR", "MET"}

    with pytest.raises(ValueError, match="not connected"):
        PathwayScorer.extract_relevant_component(graph, {"EGFR", "BRAF"})
    with pytest.raises(ValueError, match="absent"):
        PathwayScorer.extract_relevant_component(graph, {"EGFR", "ALK"})


def test_target_discovery_includes_resistance_marker_and_is_deterministic():
    graph = nx.Graph()
    graph.add_edge("EGFR", "GRB2", weight=0.2)
    graph.add_edge("GRB2", "MET", weight=0.2)
    graph.add_edge("GRB2", "SOS1", weight=0.3)
    targets = PathwayScorer.rank_target_nodes(graph, "EGFR", "MET", limit=3)
    assert targets[0] == "MET"
    assert len(targets) == 3


def test_missing_candidate_target_is_rejected_instead_of_rewarded():
    graph = nx.Graph()
    graph.add_edge("EGFR", "MET", weight=0.2)
    assert (
        PathwayScorer.score_candidates(
            graph, "EGFR", [{"secondary_target": "NOT_IN_GRAPH"}]
        )
        == []
    )


def test_priority_score_is_decomposed_bounded_and_not_pool_normalized():
    graph = nx.Graph()
    graph.add_edge("EGFR", "MET", weight=0.2)
    base = {
        "secondary_target": "MET",
        "secondary_drug": "CAPMATINIB",
        "mechanism_of_action": "MET inhibitor",
        "clinical_phase": 4,
        "clinical_status": "active_or_completed",
        "indication_match": True,
        "combination_evidence": True,
        "median_pchembl": 8.0,
        "biological_rationale": "Evidence-linked MET inhibition.",
    }
    single = PathwayScorer.score_candidates(graph, "EGFR", [base])[0]
    pooled = PathwayScorer.score_candidates(
        graph,
        "EGFR",
        [base, {**base, "secondary_drug": "OTHER", "median_pchembl": 6.0}],
    )
    same = next(item for item in pooled if item["secondary_drug"] == "CAPMATINIB")
    assert same["combination_priority_score"] == single["combination_priority_score"]
    assert 0 <= single["combination_priority_score"] <= 1
    assert set(single["score_components"]) == {
        "topology",
        "proximity",
        "pharmacology",
        "clinical_evidence",
    }
    assert "not an experimental synergy" in single["limitations"][-1]


def test_missing_pharmacology_is_explicit_and_weights_are_renormalized():
    graph = nx.Graph()
    graph.add_edge("EGFR", "MET", weight=0.2)
    result = PathwayScorer.score_candidates(
        graph,
        "EGFR",
        [
            {
                "secondary_target": "MET",
                "secondary_drug": "DRUG",
                "clinical_phase": 2,
                "biological_rationale": "Target-linked agent.",
            }
        ],
    )[0]
    assert result["score_components"]["pharmacology"] is None
    assert any("pChEMBL" in limitation for limitation in result["limitations"])


@pytest.mark.parametrize("value, expected", [(-1, 0), (0.5, 0.5), (2, 1)])
def test_clamp01(value, expected):
    assert clamp01(value) == expected
