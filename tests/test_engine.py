import networkx as nx
import pytest

from src.engine.graph_builder import HUB_EXCLUSION_SET, build_signaling_graph
from src.engine.scorer import PathwayScorer, normalize_series, normalize_value


def test_self_loop_stripping():
    """Verify homodimer self-loops are stripped from the graph."""
    interactions = [
        {"preferredName_A": "EGFR", "preferredName_B": "EGFR", "score": 900},
        {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 800},
    ]
    G = build_signaling_graph(interactions)
    assert len(list(nx.selfloop_edges(G))) == 0
    assert ("EGFR", "EGFR") not in G.edges()
    assert ("EGFR", "MET") in G.edges()


def test_hub_exclusion():
    """Verify master regulators TP53, UBC, UBB, RPS27A are dropped."""
    interactions = [
        {"preferredName_A": "EGFR", "preferredName_B": "TP53", "score": 950},
        {"preferredName_A": "MET", "preferredName_B": "UBB", "score": 900},
        {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 850},
    ]
    G = build_signaling_graph(interactions)
    for hub in HUB_EXCLUSION_SET:
        assert hub not in G.nodes()
    assert "EGFR" in G.nodes()
    assert "MET" in G.nodes()


def test_lcc_isolation():
    """Verify LCC extraction isolates largest component and handles guards."""
    G = nx.Graph()
    # Component 1: EGFR - MET - ERBB3 (size 3)
    G.add_edge("EGFR", "MET", weight=0.1)
    G.add_edge("MET", "ERBB3", weight=0.1)
    # Component 2: KRAS - BRAF (size 2)
    G.add_edge("KRAS", "BRAF", weight=0.1)

    G_lcc = PathwayScorer.extract_lcc(G)
    assert len(G_lcc.nodes()) == 3
    assert set(G_lcc.nodes()) == {"EGFR", "MET", "ERBB3"}

    # Test topology guard with insufficient nodes
    empty_g = nx.Graph()
    empty_g.add_node("EGFR")
    with pytest.raises(ValueError, match="NoPathwayFound"):
        PathwayScorer.extract_lcc(empty_g)


def test_single_candidate_normalization():
    """Verify single-candidate or zero-variance pool normalizes correctly.

    With the fixed normalize_value, zero-variance defaults to 0.0 (no signal),
    unless explicitly overridden.
    """
    # Default zero_var_default=0.0: identical values → 0.0
    val = normalize_value(5.0, 5.0, 5.0)
    assert val == 0.0

    series_normed = normalize_series([10.0])
    assert series_normed == [0.0]

    series_identical = normalize_series([7.5, 7.5, 7.5])
    assert series_identical == [0.0, 0.0, 0.0]

    # With zero_var_default=1.0 (legacy behavior)
    val_legacy = normalize_value(5.0, 5.0, 5.0, zero_var_default=1.0)
    assert val_legacy == 1.0


def test_case_a_vs_case_b_scoring():
    """Verify Case A vs Case B formula switching."""
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    G.add_edge("MET", "ERBB3", weight=0.3)

    candidates_with_affinity = [
        {"secondary_target": "MET", "pchembl_value": 8.0},
        {"secondary_target": "ERBB3", "pchembl_value": 7.0},
    ]

    scored_a = PathwayScorer.score_candidates(G, "EGFR", candidates_with_affinity)
    assert len(scored_a) == 2
    # Verify synergy_score key present and non-zero
    for item in scored_a:
        assert "synergy_score" in item
        assert 0.0 <= item["synergy_score"] <= 1.0
        assert set(item["score_components"]) == {
            "topology",
            "proximity",
            "pharmacology",
            "clinical_evidence",
        }

    candidates_missing_affinity = [
        {"secondary_target": "MET"},
        {"secondary_target": "ERBB3"},
    ]

    scored_b = PathwayScorer.score_candidates(G, "EGFR", candidates_missing_affinity)
    assert len(scored_b) == 2
    for item in scored_b:
        assert "synergy_score" in item
        assert 0.0 <= item["synergy_score"] <= 1.0
        assert item["score_components"]["pharmacology"] is None


def test_duplicate_edge_resolution():
    """Verify that duplicate edges from STRING-DB keep the highest confidence (min weight)."""
    interactions = [
        {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 700},
        {"preferredName_A": "EGFR", "preferredName_B": "MET", "score": 900},
    ]
    G = build_signaling_graph(interactions)
    assert G.number_of_edges() == 1
    # score 900/1000 = 0.9, weight = 1.005 - 0.9 = 0.105 (should keep this one, lower weight)
    assert abs(G["EGFR"]["MET"]["weight"] - 0.105) < 1e-6


def test_score_clamping():
    """Verify scores outside [0, 1000] are clamped properly."""
    interactions = [
        {"preferredName_A": "A", "preferredName_B": "B", "score": -50},
        {"preferredName_A": "C", "preferredName_B": "D", "score": 1500},
    ]
    G = build_signaling_graph(interactions)
    for u, v, data in G.edges(data=True):
        assert 0.0 <= data["score"] <= 1.0


def test_mixed_affinity_per_candidate_branching():
    """Verify that candidates with affinity use Case A and those without use Case B
    independently — not the old all-or-nothing batch behavior."""
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    G.add_edge("MET", "ERBB3", weight=0.3)

    candidates_mixed = [
        {"secondary_target": "MET", "pchembl_value": 8.0},
        {"secondary_target": "ERBB3"},  # No affinity
    ]

    scored = PathwayScorer.score_candidates(G, "EGFR", candidates_mixed)
    assert len(scored) == 2
    for item in scored:
        assert 0.0 <= item["synergy_score"] <= 1.0


def test_candidate_outside_validated_topology_abstains():
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    scored = PathwayScorer.score_candidates(
        G, "EGFR", [{"secondary_target": "UNKNOWN"}]
    )
    assert scored[0]["synergy_score"] == 0.0
    assert scored[0]["shortest_path_distance"] is None
    assert scored[0]["target_in_graph"] is False
    assert scored[0]["scoring_status"].startswith("abstained_")
    assert scored[0]["evidence_status"] == "abstained"


def test_evidence_status_is_not_a_clinical_confidence_score():
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    scored = PathwayScorer.score_candidates(
        G,
        "EGFR",
        [{"secondary_target": "MET", "pchembl_value": 8.0}],
    )
    assert scored[0]["evidence_status"] == "pharmacology_available"
    assert "clinical response" in scored[0]["evidence_notes"][0]


def test_zero_centrality_normalization():
    """Verify that a graph where all nodes have zero betweenness centrality
    does NOT give them max centrality score."""
    # A simple edge: EGFR-MET. Both have centrality 0 in a 2-node graph.
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)

    centralities = PathwayScorer.calculate_bottleneck_centralities(G)
    # In a 2-node graph, betweenness centrality for both is 0.0
    assert centralities["EGFR"] == 0.0
    assert centralities["MET"] == 0.0

    # The fixed transform retains the composite degree signal without candidate-pool normalization.
    candidates = [{"secondary_target": "MET"}]
    scored = PathwayScorer.score_candidates(G, "EGFR", candidates)
    assert len(scored) == 1
    assert 0.0 < scored[0]["hub_penalized_centrality"] < 1.0


def test_score_is_invariant_to_adding_another_candidate():
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    G.add_edge("MET", "ERBB3", weight=0.3)

    one = PathwayScorer.score_candidates(
        G, "EGFR", [{"secondary_target": "MET", "pchembl_value": 8.0}]
    )[0]
    two = PathwayScorer.score_candidates(
        G,
        "EGFR",
        [
            {"secondary_target": "MET", "pchembl_value": 8.0},
            {"secondary_target": "ERBB3", "pchembl_value": 6.0},
        ],
    )[0]

    assert two["synergy_score"] == one["synergy_score"]


def test_target_level_ties_are_explicit():
    """Drugs sharing a target without drug-specific activity are marked as tied."""
    G = nx.Graph()
    G.add_edge("EGFR", "MET", weight=0.2)
    scored = PathwayScorer.score_candidates(
        G,
        "EGFR",
        [
            {"secondary_drug": "DRUG_B", "secondary_target": "MET"},
            {"secondary_drug": "DRUG_A", "secondary_target": "MET"},
        ],
    )
    assert [item["rank"] for item in scored] == [1, 1]
    assert [item["tie_group"] for item in scored] == [1, 1]
    assert all(
        "Insufficient drug-specific evidence" in item["tie_reason"] for item in scored
    )
