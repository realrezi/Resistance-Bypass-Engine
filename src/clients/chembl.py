import statistics
from typing import Any, Dict, List, Optional
from src.clients.base import BaseHTTPClient


class ChEMBLClient(BaseHTTPClient):
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    async def get_clinical_molecules(
        self,
        target_chembl_id: Optional[str] = None,
        max_phase_gte: int = 2,
        withdrawn_flag: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch molecules with max_phase >= 2, withdrawn_flag = false, following page_meta.next pagination."""
        url = f"{self.BASE_URL}/molecule.json"
        params: Dict[str, Any] = {
            "molecule_dictionary__max_phase__gte": max_phase_gte,
            "withdrawn_flag": str(withdrawn_flag).lower(),
            "limit": limit,
        }
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id

        all_molecules: List[Dict[str, Any]] = []
        next_url: Optional[str] = url

        while next_url:
            if next_url == url:
                data = await self.get_json(next_url, params=params)
            else:
                full_next_url = (
                    next_url
                    if next_url.startswith("http")
                    else f"https://www.ebi.ac.uk{next_url}"
                )
                data = await self.get_json(full_next_url)

            if not isinstance(data, dict):
                break

            molecules = data.get("molecules", [])
            all_molecules.extend(molecules)

            page_meta = data.get("page_meta", {})
            next_path = page_meta.get("next")
            next_url = next_path if next_path else None

        return all_molecules

    async def get_target_activities(
        self,
        target_chembl_id: str,
        limit: int = 100,
    ) -> Dict[str, float]:
        """Fetch bioactivity values for target and group by pref_name using median pchembl_value."""
        url = f"{self.BASE_URL}/activity.json"
        params: Dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "pchembl_value__isnull": "false",
            "limit": limit,
        }

        all_activities: List[Dict[str, Any]] = []
        next_url: Optional[str] = url

        while next_url:
            if next_url == url:
                data = await self.get_json(next_url, params=params)
            else:
                full_next_url = (
                    next_url
                    if next_url.startswith("http")
                    else f"https://www.ebi.ac.uk{next_url}"
                )
                data = await self.get_json(full_next_url)

            if not isinstance(data, dict):
                break

            activities = data.get("activities", [])
            all_activities.extend(activities)

            page_meta = data.get("page_meta", {})
            next_path = page_meta.get("next")
            next_url = next_path if next_path else None

        drug_pchembl_map: Dict[str, List[float]] = {}
        for act in all_activities:
            pref_name = act.get("molecule_pref_name") or act.get("molecule_chembl_id")
            pchembl_val = act.get("pchembl_value")
            if pref_name and pchembl_val is not None:
                try:
                    val = float(pchembl_val)
                    drug_pchembl_map.setdefault(pref_name.upper(), []).append(val)
                except (ValueError, TypeError):
                    continue

        median_affinity_map: Dict[str, float] = {}
        for drug_name, values in drug_pchembl_map.items():
            if values:
                median_affinity_map[drug_name] = statistics.median(values)

        return median_affinity_map
