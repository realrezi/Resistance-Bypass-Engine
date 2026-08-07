import math
from typing import Any

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


def normalize_series(values: list[float], zero_var_default: float = 0.0) -> list[float]:
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
    def calculate_bottleneck_centralities(G_lcc: nx.Graph) -> dict[str, float]:
        """Calculate betweenness centrality with degree penalization:

        C_B,adjusted(v) = C_B(v) / log2(degree(v) + 2)
        """
        C_B = nx.betweenness_centrality(G_lcc, weight="weight")
        adjusted_centrality: dict[str, float] = {}

        for node in G_lcc.nodes():
            cb_val = C_B.get(node, 0.0)
            deg = G_lcc.degree(node)
            deg_penalty = math.log2(deg + 2)
            adjusted_centrality[node] = cb_val / deg_penalty

        return adjusted_centrality

    @staticmethod
    def calculate_shortest_distance(
        G_lcc: nx.Graph, source_node: str, target_node: str
    ) -> float | None:
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
        return None

    @classmethod
    def score_candidates(
        cls,
        G: nx.Graph,
        primary_target: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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

        raw_scores: list[dict[str, Any]] = []

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

        scored_results: list[dict[str, Any]] = []

        for item in raw_scores:
            # Fixed, candidate-pool-independent transforms. Existing scores do not
            # change merely because another candidate is added to the request.
            cb_n = 1.0 - math.exp(-4.0 * max(item["cb_adj"], 0.0))
            target_present = item["distance"] is not None
            proximity = math.exp(-max(item["distance"], 0.0)) if target_present else 0.0
            if item["affinity"] is not None:
                # Logistic transform centered at pChEMBL 7 (~100 nM).
                aff_n = 1.0 / (1.0 + math.exp(-1.0 * (item["affinity"] - 7.0)))
                synergy = 0.40 * cb_n + 0.30 * proximity + 0.30 * aff_n
            else:
                # Missing pharmacology is an evidence gap, not a reason to
                # increase the weight of the remaining heuristic components.
                synergy = 0.40 * cb_n + 0.30 * proximity
            if not target_present:
                synergy = 0.0

            candidate_res = dict(item["candidate"])
            candidate_res["synergy_score"] = round(synergy, 4)
            candidate_res["score_components"] = {
                "topology": round(cb_n, 4),
                "proximity": round(proximity, 4),
                "pharmacology": round(aff_n, 4)
                if item["affinity"] is not None
                else None,
                "clinical_evidence": None,
            }
            evidence_parts = [
                item["candidate"].get("pchembl_value") is not None
                or item["candidate"].get("chembl_ic50_nm") is not None,
                item["candidate"].get("indication_match") is True,
                item["candidate"].get("combination_evidence") is True,
            ]
            candidate_res["evidence_completeness"] = round(
                sum(evidence_parts) / len(evidence_parts), 4
            )
            candidate_res["hub_penalized_centrality"] = round(cb_n, 4)
            candidate_res["shortest_path_distance"] = (
                round(item["distance"], 4) if item["distance"] is not None else None
            )
            candidate_res["scoring_status"] = (
                "scored"
                if target_present
                else "abstained_target_not_in_validated_topology"
            )
            candidate_res["target_in_graph"] = target_present
            if not target_present:
                candidate_res["evidence_status"] = "abstained"
                candidate_res["evidence_notes"] = [
                    "The target was not present in the validated network topology."
                ]
            elif item["candidate"].get("combination_evidence") is True:
                candidate_res["evidence_status"] = "pair_co_mention"
                candidate_res["evidence_notes"] = [
                    "A returned clinical record co-mentioned the primary drug; this does not prove same-arm efficacy."
                ]
            elif item["affinity"] is not None:
                candidate_res["evidence_status"] = "pharmacology_available"
                candidate_res["evidence_notes"] = [
                    "Drug-specific activity was available, but activity is not clinical response."
                ]
            else:
                candidate_res["evidence_status"] = "computational_hypothesis"
                candidate_res["evidence_notes"] = [
                    "The result is supported by network heuristics without drug-specific activity."
                ]

            scored_results.append(candidate_res)

        # Sort by computational score, then transparent evidence completeness.
        # The final name sort is only deterministic ordering inside a true tie.
        scored_results.sort(
            key=lambda x: (
                x["synergy_score"],
                x.get("evidence_completeness", 0.0),
                x.get("secondary_drug", ""),
            ),
            reverse=True,
        )
        previous_key = None
        rank = 0
        tie_group = 0
        for index, candidate in enumerate(scored_results):
            tie_key = (
                round(float(candidate["synergy_score"]), 4),
                round(float(candidate.get("evidence_completeness", 0.0)), 4),
            )
            if tie_key != previous_key:
                rank = index + 1
                tie_group += 1
                previous_key = tie_key
            candidate["rank"] = rank
            candidate["tie_group"] = tie_group
            candidate["_tie_key"] = tie_key
        tie_counts: dict[tuple[float, float], int] = {}
        for candidate in scored_results:
            tie_counts[candidate["_tie_key"]] = (
                tie_counts.get(candidate["_tie_key"], 0) + 1
            )
        for candidate in scored_results:
            if (
                tie_counts[candidate["_tie_key"]] > 1
                and candidate.get("score_components", {}).get("pharmacology") is None
            ):
                candidate["tie_reason"] = (
                    "Insufficient drug-specific evidence; topology and proximity "
                    "were not enough to separate this candidate."
                )
            candidate.pop("_tie_key", None)
        return scored_results
