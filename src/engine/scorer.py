from __future__ import annotations

import math
from typing import Any

import networkx as nx


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _bounded_positive(value: float, scale: float = 4.0) -> float:
    """Map a non-negative value to a stable [0, 1) range."""
    return clamp01(1.0 - math.exp(-scale * max(value, 0.0)))


class PathwayScorer:
    @staticmethod
    def extract_relevant_component(
        graph: nx.Graph, required_nodes: set[str] | None = None
    ) -> nx.Graph:
        if graph.number_of_nodes() < 2:
            raise ValueError("NoPathwayFound: insufficient physical interactions.")

        required = {node.upper() for node in (required_nodes or set())}
        missing = required - set(graph.nodes)
        if missing:
            raise ValueError(
                "NoPathwayFound: requested targets absent from the physical network: "
                + ", ".join(sorted(missing))
            )

        components = list(nx.connected_components(graph))
        if required:
            matching = [component for component in components if required <= component]
            if not matching:
                raise ValueError(
                    "NoPathwayFound: primary and resistance targets are not connected."
                )
            nodes = max(
                matching, key=lambda component: (len(component), sorted(component))
            )
        else:
            nodes = max(
                components, key=lambda component: (len(component), sorted(component))
            )

        if len(nodes) < 2:
            raise ValueError("NoPathwayFound: insufficient physical interactions.")
        return graph.subgraph(nodes).copy()

    @staticmethod
    def calculate_bottleneck_centralities(graph: nx.Graph) -> dict[str, float]:
        """Calculate the documented hub-penalized composite centrality."""
        betweenness = nx.betweenness_centrality(graph, weight="weight")
        max_degree = max((graph.degree(node) for node in graph), default=1)
        result: dict[str, float] = {}
        for node in graph:
            degree = graph.degree(node)
            penalty = math.log2(degree + 2)
            composite = (
                betweenness.get(node, 0.0) + 0.5 * (degree / max(max_degree, 1))
            ) / penalty
            result[str(node)] = float(composite)
        return result

    @staticmethod
    def calculate_shortest_distance(
        graph: nx.Graph, source_node: str, target_node: str
    ) -> float | None:
        if source_node not in graph or target_node not in graph:
            return None
        if not nx.has_path(graph, source_node, target_node):
            return None
        return float(
            nx.dijkstra_path_length(graph, source_node, target_node, weight="weight")
        )

    @classmethod
    def rank_target_nodes(
        cls,
        graph: nx.Graph,
        primary_target: str,
        resistance_target: str,
        limit: int = 5,
    ) -> list[str]:
        centralities = cls.calculate_bottleneck_centralities(graph)
        ranked: list[tuple[float, str]] = []
        for node in graph:
            if node == primary_target:
                continue
            distance = cls.calculate_shortest_distance(graph, primary_target, str(node))
            if distance is None:
                continue
            topology = _bounded_positive(centralities.get(str(node), 0.0))
            proximity = math.exp(-distance)
            discovery_score = 0.65 * topology + 0.35 * proximity
            if node == resistance_target:
                discovery_score += 1.0
            ranked.append((discovery_score, str(node)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [node for _, node in ranked[:limit]]

    @classmethod
    def score_candidates(
        cls,
        graph: nx.Graph,
        primary_target: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        centralities = cls.calculate_bottleneck_centralities(graph)
        scored: list[dict[str, Any]] = []

        for candidate in candidates:
            target = str(candidate.get("secondary_target") or "").strip().upper()
            distance = cls.calculate_shortest_distance(graph, primary_target, target)
            if distance is None:
                continue

            topology = _bounded_positive(centralities.get(target, 0.0))
            proximity = clamp01(math.exp(-distance))
            phase = min(max(int(candidate.get("clinical_phase") or 0), 0), 4)
            indication_match = bool(candidate.get("indication_match"))
            combination_evidence = bool(candidate.get("combination_evidence"))
            status = str(candidate.get("clinical_status") or "unknown")
            status_signal = 1.0 if status == "active_or_completed" else 0.5
            if status == "stopped":
                status_signal = 0.0
            clinical = clamp01(
                0.45 * (phase / 4.0)
                + 0.25 * float(indication_match)
                + 0.20 * float(combination_evidence)
                + 0.10 * status_signal
            )

            median_pchembl = candidate.get("median_pchembl")
            pharmacology: float | None = None
            if median_pchembl is not None:
                try:
                    pharmacology = clamp01(
                        1.0 / (1.0 + math.exp(-(float(median_pchembl) - 7.0)))
                    )
                except (TypeError, ValueError, OverflowError):
                    pharmacology = None

            weighted = {
                "topology": (topology, 0.30),
                "proximity": (proximity, 0.25),
                "clinical_evidence": (clinical, 0.25),
            }
            if pharmacology is not None:
                weighted["pharmacology"] = (pharmacology, 0.20)
            weight_total = sum(weight for _, weight in weighted.values())
            priority = (
                sum(value * weight for value, weight in weighted.values())
                / weight_total
            )

            limitations: list[str] = []
            if not indication_match:
                limitations.append(
                    "No indication-specific clinical report matched the requested cancer type."
                )
            if not combination_evidence:
                limitations.append(
                    "No pair-level clinical report mentioning the primary drug was found."
                )
            if pharmacology is None:
                limitations.append(
                    "No quality-filtered human binding pChEMBL measurement was available."
                )
            limitations.append(
                "Priority score is heuristic and is not an experimental synergy measurement."
            )

            result = dict(candidate)
            result.update(
                {
                    "combination_priority_score": round(priority, 4),
                    "synergy_score": round(priority, 4),
                    "score_components": {
                        "topology": round(topology, 4),
                        "proximity": round(proximity, 4),
                        "pharmacology": (
                            round(pharmacology, 4) if pharmacology is not None else None
                        ),
                        "clinical_evidence": round(clinical, 4),
                    },
                    "hub_penalized_centrality": round(topology, 4),
                    "shortest_path_distance": round(distance, 4),
                    "limitations": limitations,
                }
            )
            scored.append(result)

        scored.sort(
            key=lambda item: (
                -item["combination_priority_score"],
                str(item.get("secondary_drug") or ""),
                str(item.get("secondary_target") or ""),
            )
        )
        return scored
