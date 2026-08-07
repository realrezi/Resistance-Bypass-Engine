from typing import Any

import networkx as nx

HUB_EXCLUSION_SET: set[str] = {"TP53", "UBC", "UBB", "RPS27A"}


def build_signaling_graph(string_interactions: list[dict[str, Any]]) -> nx.Graph:
    """Build a weighted, undirected NetworkX graph from STRING-DB interactor payloads.

    Excludes master regulator hubs (TP53, UBC, UBB, RPS27A), calculates Dijkstra weights,
    strips homodimer self-loops, and retains the highest-confidence edge when duplicates exist.
    """
    G = nx.Graph()

    for interaction in string_interactions:
        node_a = interaction.get("preferredName_A") or interaction.get("stringId_A")
        node_b = interaction.get("preferredName_B") or interaction.get("stringId_B")

        if not node_a or not node_b:
            continue

        node_a = str(node_a).strip().upper()
        node_b = str(node_b).strip().upper()

        # Skip self-loops before edge addition
        if node_a == node_b:
            continue

        # Hub Exclusion List: Drop TP53, UBC, UBB, RPS27A
        if node_a in HUB_EXCLUSION_SET or node_b in HUB_EXCLUSION_SET:
            continue

        # Extract confidence score
        raw_score = interaction.get("score")
        if raw_score is None:
            raw_score = interaction.get("combined_score", 400)

        try:
            score_val = float(raw_score)
        except (ValueError, TypeError):
            score_val = 400.0

        # Score normalization
        score_norm = score_val if score_val <= 1.0 else score_val / 1000.0
        # Clamp to [0, 1]
        score_norm = min(max(score_norm, 0.0), 1.0)
        # Edge Weight formula: w(e) = 1.005 - S_norm
        weight = 1.005 - score_norm

        # Duplicate edge resolution: keep highest confidence (lowest weight)
        if G.has_edge(node_a, node_b):
            existing_weight = G[node_a][node_b].get("weight", weight)
            if weight < existing_weight:
                G[node_a][node_b]["weight"] = weight
                G[node_a][node_b]["score"] = score_norm
        else:
            G.add_edge(node_a, node_b, weight=weight, score=score_norm)

    # CRITICAL TOPOLOGY GUARD: Strip any remaining self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    return G
