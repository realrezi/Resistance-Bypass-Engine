from typing import Any, Dict, List
from src.clients.base import BaseHTTPClient


class OpenTargetsClient(BaseHTTPClient):
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    KNOWN_DRUGS_QUERY = """
    query knownDrugsQuery($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        drugAndClinicalCandidates {
          count
          rows {
            id
            maxClinicalStage
            drug {
              id
              name
              drugType
              maximumClinicalStage
              mechanismsOfAction {
                rows {
                  mechanismOfAction
                  targetName
                }
              }
            }
          }
        }
      }
    }
    """

    async def get_known_drugs(self, ensembl_id: str, size: int = 100) -> List[Dict[str, Any]]:
        """Fetch known drugs for a target Ensembl ID via Open Targets GraphQL API."""
        clean_ensembl = ensembl_id.split(".")[0]
        json_payload = {
            "query": self.KNOWN_DRUGS_QUERY,
            "variables": {
                "ensemblId": clean_ensembl,
            },
        }
        data = await self.post_json(self.BASE_URL, json_data=json_payload)

        if not isinstance(data, dict):
            return []

        target_data = data.get("data", {}).get("target")
        if not target_data:
            return []

        candidates = target_data.get("drugAndClinicalCandidates", {})
        rows = candidates.get("rows", [])

        parsed_drugs: List[Dict[str, Any]] = []
        for row in rows[:size]:
            drug_obj = row.get("drug", {}) or {}
            drug_id = drug_obj.get("id", "")
            pref_name = drug_obj.get("name") or drug_id
            drug_type = drug_obj.get("drugType", "")
            moa_rows = drug_obj.get("mechanismsOfAction", {}).get("rows", [])
            moa = moa_rows[0].get("mechanismOfAction") if moa_rows else "Bypass Pathway Inhibitor"
            stage_str = row.get("maxClinicalStage") or drug_obj.get("maximumClinicalStage") or "PHASE_2"
            phase = 2
            if "PHASE_" in str(stage_str):
                try:
                    phase = int(str(stage_str).replace("PHASE_", ""))
                except ValueError:
                    phase = 2

            parsed_drugs.append(
                {
                    "drugId": drug_id,
                    "prefName": pref_name,
                    "drugType": drug_type,
                    "mechanismOfAction": moa,
                    "targetSymbol": target_data.get("approvedSymbol", ""),
                    "phase": phase,
                    "status": "active",
                }
            )

        return parsed_drugs

