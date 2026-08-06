from typing import Any, Dict, List
from src.clients.base import BaseHTTPClient


class OpenTargetsClient(BaseHTTPClient):
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    KNOWN_DRUGS_QUERY = """
    query knownDrugsQuery($ensemblId: String!, $size: Int = 100) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        knownDrugs(size: $size) {
          count
          rows {
            drugId
            prefName
            drugType
            mechanismOfAction
            targetId
            targetSymbol
            phase
            status
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
                "size": size,
            },
        }
        data = await self.post_json(self.BASE_URL, json_data=json_payload)

        if not isinstance(data, dict):
            return []

        target_data = data.get("data", {}).get("target")
        if not target_data:
            return []

        known_drugs = target_data.get("knownDrugs", {})
        return known_drugs.get("rows", [])
