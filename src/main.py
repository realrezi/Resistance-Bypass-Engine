import asyncio
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.clients.base import cache
from src.clients.chembl import ChEMBLClient
from src.clients.open_targets import OpenTargetsClient
from src.clients.string_db import StringDBClient
from src.engine.graph_builder import build_signaling_graph
from src.engine.scorer import PathwayScorer
from src.schemas.models import (
    CombinationCandidate,
    ResistanceBypassReport,
    ResistanceRequest,
)
from src.services.id_mapper import IDMapper

app = FastAPI(
    title="Targeted Oncology Resistance Bypass Engine",
    description="Microservice modeling acquired drug resistance pathways in cancer",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sync_build_and_score(
    interactions: List[Dict[str, Any]],
    primary_target: str,
    raw_candidates: List[Dict[str, Any]],
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """CPU-bound worker function offloaded via asyncio.to_thread."""
    G = build_signaling_graph(interactions)
    if len(G.nodes) < 2:
        raise ValueError("NoPathwayFound: Insufficient biological interactions.")

    scored = PathwayScorer.score_candidates(G, primary_target, raw_candidates)
    pathway_nodes_count = len(G.nodes)

    if scored:
        shortest_path_distance = float(scored[0].get("shortest_path_distance", 2.0))
    else:
        shortest_path_distance = 2.0

    return pathway_nodes_count, shortest_path_distance, scored


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check diagnostic endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "cache_size_bytes": cache.volume(),
        "cache_limit_bytes": cache.size_limit,
    }


def _is_drug_withdrawn(drug_row: Dict[str, Any]) -> bool:
    """Check if an Open Targets drug row indicates a withdrawn status."""
    status = (drug_row.get("status") or "").strip().lower()
    return status == "withdrawn"


@app.post("/api/v1/analyze-resistance", response_model=ResistanceBypassReport)
async def analyze_resistance(req: ResistanceRequest) -> ResistanceBypassReport:
    """Analyze drug resistance pathways and rank dual-drug combination candidates."""
    id_mapper = IDMapper()

    try:
        mapped_primary = await id_mapper.map_identifier(req.primary_target)
        mapped_resistance = await id_mapper.map_identifier(req.resistance_marker)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"ID Resolution failed: {str(e)}")

    primary_target_canonical = mapped_primary.canonical_symbol
    resistance_marker_canonical = mapped_resistance.canonical_symbol

    chembl_client = ChEMBLClient()

    # Branching Evaluation
    if primary_target_canonical == resistance_marker_canonical:
        # On-Target Mutation Branching
        resistance_type = "On-Target Mutation"
        pathway_nodes_count = 1
        shortest_path_distance = 0.0

        molecules = await chembl_client.get_clinical_molecules(
            target_chembl_id=mapped_primary.chembl_target_id,
            max_phase_gte=2,
            withdrawn_flag=False,
        )

        ranked_combinations: List[CombinationCandidate] = []
        for mol in molecules[:10]:
            drug_name = mol.get("pref_name") or mol.get("molecule_chembl_id", "Unknown Drug")
            max_phase = mol.get("max_phase", 2) or 2

            ranked_combinations.append(
                CombinationCandidate(
                    secondary_drug=drug_name.upper(),
                    secondary_target=primary_target_canonical,
                    mechanism_of_action="Next-Generation Inhibitor",
                    clinical_phase=int(max_phase),
                    is_withdrawn=False,
                    synergy_score=1.0,
                    hub_penalized_centrality=1.0,
                    chembl_ic50_nm=None,
                    biological_rationale=f"On-target mutation in {primary_target_canonical}. Next-generation inhibitor overrides resistance.",
                )
            )

        if not ranked_combinations:
            # Fallback candidate if no specific clinical molecule returned
            ranked_combinations.append(
                CombinationCandidate(
                    secondary_drug=f"Next-Gen {primary_target_canonical} Inhibitor",
                    secondary_target=primary_target_canonical,
                    mechanism_of_action="Next-Generation Inhibitor",
                    clinical_phase=3,
                    is_withdrawn=False,
                    synergy_score=1.0,
                    hub_penalized_centrality=1.0,
                    chembl_ic50_nm=None,
                    biological_rationale=f"On-target mutation in {primary_target_canonical}. Next-generation inhibitor overrides resistance.",
                )
            )

        return ResistanceBypassReport(
            primary_target_canonical=primary_target_canonical,
            resistance_marker_canonical=resistance_marker_canonical,
            resistance_type=resistance_type,
            pathway_nodes_count=pathway_nodes_count,
            shortest_path_distance=shortest_path_distance,
            ranked_combinations=ranked_combinations,
        )

    else:
        # Off-Target Bypass Branching
        resistance_type = "Off-Target Bypass"

        string_client = StringDBClient()
        ot_client = OpenTargetsClient()

        # Fetch STRING-DB network
        interactions = await string_client.get_network(
            primary_target_canonical, resistance_marker_canonical
        )

        # Fetch Open Targets known drugs for resistance marker
        ot_drugs = await ot_client.get_known_drugs(mapped_resistance.ensembl_id)

        # Filter out withdrawn drugs — AGENTS.md: "ranks active, non-withdrawn clinical dual-drug combination therapies"
        ot_drugs = [d for d in ot_drugs if not _is_drug_withdrawn(d)]

        # Fetch ChEMBL activities for resistance target if available
        activity_map: Dict[str, float] = {}
        if mapped_resistance.chembl_target_id:
            activity_map = await chembl_client.get_target_activities(
                mapped_resistance.chembl_target_id
            )

        raw_candidates: List[Dict[str, Any]] = []

        if ot_drugs:
            for drug in ot_drugs:
                drug_name = drug.get("prefName") or drug.get("drugId") or "Unknown"
                moa = drug.get("mechanismOfAction") or "Bypass Pathway Inhibitor"
                phase = drug.get("phase", 2) or 2
                target_sym = drug.get("targetSymbol") or resistance_marker_canonical

                # Try matching activity by both prefName and drugId
                pchembl_val = activity_map.get(drug_name.upper())
                if pchembl_val is None:
                    drug_id = drug.get("drugId", "")
                    if drug_id:
                        pchembl_val = activity_map.get(drug_id.upper())

                raw_candidates.append(
                    {
                        "secondary_drug": drug_name.upper(),
                        "secondary_target": target_sym.upper(),
                        "mechanism_of_action": moa,
                        "clinical_phase": phase,
                        "is_withdrawn": False,
                        "pchembl_value": pchembl_val,
                        "biological_rationale": f"Inhibits bypass target {target_sym.upper()} to restore sensitivity to primary drug targeting {primary_target_canonical}.",
                    }
                )
        else:
            # Fallback candidate for resistance marker
            raw_candidates.append(
                {
                    "secondary_drug": f"{resistance_marker_canonical} Inhibitor",
                    "secondary_target": resistance_marker_canonical,
                    "mechanism_of_action": "Bypass Pathway Inhibitor",
                    "clinical_phase": 2,
                    "is_withdrawn": False,
                    "pchembl_value": None,
                    "biological_rationale": f"Inhibits bypass node {resistance_marker_canonical} to bypass resistance.",
                }
            )

        # THREAD SAFETY: Offload heavy NetworkX CPU-bound math via asyncio.to_thread
        try:
            pathway_nodes_count, shortest_path_distance, scored_raw = (
                await asyncio.to_thread(
                    _sync_build_and_score,
                    interactions,
                    primary_target_canonical,
                    raw_candidates,
                )
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        ranked_combinations: List[CombinationCandidate] = [
            CombinationCandidate(
                secondary_drug=c.get("secondary_drug", "Unknown"),
                secondary_target=c.get("secondary_target", resistance_marker_canonical),
                mechanism_of_action=c.get("mechanism_of_action", "Combination Therapy"),
                clinical_phase=int(c.get("clinical_phase", 2)),
                is_withdrawn=bool(c.get("is_withdrawn", False)),
                synergy_score=float(c.get("synergy_score", 0.0)),
                hub_penalized_centrality=float(c.get("hub_penalized_centrality", 0.0)),
                chembl_ic50_nm=c.get("chembl_ic50_nm"),
                biological_rationale=c.get(
                    "biological_rationale",
                    f"Targets bypass node {c.get('secondary_target')} to overcome {resistance_marker_canonical} resistance.",
                ),
            )
            for c in scored_raw
        ]

        return ResistanceBypassReport(
            primary_target_canonical=primary_target_canonical,
            resistance_marker_canonical=resistance_marker_canonical,
            resistance_type=resistance_type,
            pathway_nodes_count=pathway_nodes_count,
            shortest_path_distance=shortest_path_distance,
            ranked_combinations=ranked_combinations,
        )
