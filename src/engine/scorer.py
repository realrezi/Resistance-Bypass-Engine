import math
from typing import Any, Dict, List, Optional
import networkx as nx


def normalize_value(
    x: float, min_val: float, max_val: float, zero_var_default: float = 0.0
) -> float:
    """Normalize value x to [0.0, 1.0] with configurable zero-variance guard.

    Args:
        x: The value to normalize.
        min_val: Minimum value in the series.
        max_val: Maximum value in the series.
        zero_var_default: Value to return when min_val == max_val (zero variance).
            Use 0.0 for metrics where zero-variance means "no signal" (e.g. centrality).
            Use 1.0 for metrics where equal values should be treated as "maximum" (legacy).
    """
    if abs(max_val - min_val) < 1e-9:
        return zero_var_default
    val = (x - min_val) / (max_val - min_val)
    return min(max(val, 0.0), 1.0)


def normalize_series(
    values: List[float], zero_var_default: float = 0.0
) -> List[float]:
    """Normalize a list of float values to [0.0, 1.0].

    Args:
        values: List of raw values.
        zero_var_default: Default when all values are identical (zero variance).
    """
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    return [normalize_value(v, min_v, max_v, zero_var_default) for v in values]



class PathwayScorer:
    @staticmethod
    def extract_lcc(G: nx.Graph) -> nx.Graph:
        """Guard topology and extract Largest Connected Component (LCC)."""
        if len(G.nodes) < 2:
            raise ValueError("NoPathwayFound: Insufficient biological interactions.")

        connected_comps = list(nx.connected_components(G))
        if not connected_comps:
            raise ValueError("NoPathwayFound: Insufficient biological interactions.")

        lcc_nodes = max(connected_comps, key=len)
        if len(lcc_nodes) < 2:
            raise ValueError("NoPathwayFound: Insufficient biological interactions.")

        return G.subgraph(lcc_nodes).copy()

    @staticmethod
    def calculate_bottleneck_centralities(G_lcc: nx.Graph) -> Dict[str, float]:
        """Calculate betweenness centrality with degree penalization:

        C_B,adjusted(v) = C_B(v) / log2(degree(v) + 2)
        """
        C_B = nx.betweenness_centrality(G_lcc, weight="weight")
        adjusted_centrality: Dict[str, float] = {}

        for node in G_lcc.nodes():
            cb_val = C_B.get(node, 0.0)
            deg = G_lcc.degree(node)
            deg_penalty = math.log2(deg + 2)
            adjusted_centrality[node] = cb_val / deg_penalty

        return adjusted_centrality


    @staticmethod
    def calculate_shortest_distance(
        G_lcc: nx.Graph, source_node: str, target_node: str
    ) -> float:
        """Calculate Dijkstra shortest path distance d between source_node and target_node."""
        if (
            source_node in G_lcc
            and target_node in G_lcc
            and nx.has_path(G_lcc, source_node, target_node)
        ):
            return float(
                nx.dijkstra_path_length(
                    G_lcc, source_node, target_node, weight="weight"
                )
            )
        return 2.0

    @classmethod
    def score_candidates(
        cls,
        G: nx.Graph,
        primary_target: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score candidate target combinations on the network graph.

        Each candidate dict should have keys:
        - secondary_target: str
        - chembl_ic50_nm: Optional[float] or pchembl_value: Optional[float]
        """
        if not candidates:
            return []

        # 1. Topology Guard & LCC extraction
        G_lcc = cls.extract_lcc(G)

        # 2. Centrality calculation
        adjusted_centralities = cls.calculate_bottleneck_centralities(G_lcc)

        raw_scores: List[Dict[str, Any]] = []

        # 3. Calculate raw distance, composite centrality, and affinity for each candidate
        for cand in candidates:
            sec_target = cand.get("secondary_target", "").strip().upper()

            # Adjusted Betweenness Centrality
            cb_adj = adjusted_centralities.get(sec_target, 0.0)

            # Degree Centrality in G_lcc with Hub Penalty
            deg_centrality = 0.0
            if sec_target in G_lcc:
                node_deg = G_lcc.degree(sec_target)
                max_deg = max([G_lcc.degree(n) for n in G_lcc.nodes()], default=1)
                deg_penalty = math.log2(node_deg + 2)
                deg_centrality = (node_deg / max(max_deg, 1)) / deg_penalty

            # Composite Hub-Penalized Centrality
            composite_centrality = cb_adj + 0.5 * deg_centrality

            # Shortest Distance
            dist = cls.calculate_shortest_distance(G_lcc, primary_target, sec_target)

            # Affinity (pChEMBL value if available)
            pchembl = cand.get("pchembl_value")
            if pchembl is None and cand.get("chembl_ic50_nm") is not None:
                # Convert IC50 in nM to pChEMBL = -log10(IC50 * 1e-9)
                try:
                    ic50_nm = float(cand["chembl_ic50_nm"])
                    if ic50_nm > 0:
                        pchembl = -math.log10(ic50_nm * 1e-9)
                except (ValueError, TypeError):
                    pchembl = None

            raw_scores.append(
                {
                    "candidate": cand,
                    "target": sec_target,
                    "cb_adj": composite_centrality,
                    "distance": dist,
                    "affinity": pchembl,
                }
            )


        # 4. Batch Normalization across candidate pool
        # Centrality: zero-variance → 0.0 (no signal detected)
        cb_raw = [item["cb_adj"] for item in raw_scores]
        cb_normed = normalize_series(cb_raw, zero_var_default=0.0)

        # Distance: zero-variance → 0.0 so (1.0 - d_norm) = 1.0 (no penalty)
        dist_raw = [item["distance"] for item in raw_scores]
        dist_normed = normalize_series(dist_raw, zero_var_default=0.0)

        # Affinity: per-candidate branching
        # Collect non-None affinities for normalization range
        non_none_affinities = [
            item["affinity"] for item in raw_scores if item["affinity"] is not None
        ]
        aff_min = min(non_none_affinities) if non_none_affinities else 0.0
        aff_max = max(non_none_affinities) if non_none_affinities else 0.0

        scored_results: List[Dict[str, Any]] = []

        for i, item in enumerate(raw_scores):
            cb_n = cb_normed[i]
            d_n = dist_normed[i]

            if item["affinity"] is not None:
                # Case A (has affinity): Synergy = 0.40 * C_B + 0.30 * (1 - d) + 0.30 * Aff
                aff_n = normalize_value(item["affinity"], aff_min, aff_max, zero_var_default=1.0)
                synergy = 0.40 * cb_n + 0.30 * (1.0 - d_n) + 0.30 * aff_n
            else:
                # Case B (missing affinity): Synergy = 0.55 * C_B + 0.45 * (1 - d)
                synergy = 0.55 * cb_n + 0.45 * (1.0 - d_n)

            candidate_res = dict(item["candidate"])
            candidate_res["synergy_score"] = round(synergy, 4)
            candidate_res["hub_penalized_centrality"] = round(cb_n, 4)
            candidate_res["shortest_path_distance"] = round(item["distance"], 4)



            scored_results.append(candidate_res)

        # Sort candidates descending by synergy_score
        scored_results.sort(key=lambda x: x["synergy_score"], reverse=True)
        return scored_results
