import os
import statistics
from typing import Any

from src.clients.base import BaseHTTPClient


class ChEMBLClient(BaseHTTPClient):
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    async def get_clinical_molecules(
        self,
        target_chembl_id: str | None = None,
        max_phase_gte: int = 2,
        withdrawn_flag: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch molecules with max_phase >= 2, withdrawn_flag = false, following page_meta.next pagination."""
        url = f"{self.BASE_URL}/molecule.json"
        params: dict[str, Any] = {
            "molecule_dictionary__max_phase__gte": max_phase_gte,
            "withdrawn_flag": str(withdrawn_flag).lower(),
            "limit": limit,
        }
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id

        all_molecules: list[dict[str, Any]] = []
        next_url: str | None = url
        page_count = 0
        # Follow the API cursor rather than silently stopping after five pages.
        # The high safety ceiling protects the service from a malformed cursor.
        max_pages = max(1, int(os.getenv("CHEMBL_MOLECULE_MAX_PAGES", "100")))

        while next_url and page_count < max_pages:
            page_count += 1
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
        limit: int = 500,
        max_pages: int | None = None,
    ) -> dict[str, float]:
        """Fetch bioactivity values for target and group by pref_name using median pchembl_value."""
        url = f"{self.BASE_URL}/activity.json"
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "pchembl_value__isnull": "false",
            "limit": limit,
        }

        all_activities: list[dict[str, Any]] = []
        if max_pages is None:
            max_pages = max(1, int(os.getenv("CHEMBL_ACTIVITY_MAX_PAGES", "2")))
        next_url: str | None = url
        page_count = 0

        while next_url and page_count < max_pages:
            page_count += 1
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

        drug_pchembl_map: dict[str, list[float]] = {}
        for act in all_activities:
            pref_name = act.get("molecule_pref_name") or act.get("molecule_chembl_id")
            pchembl_val = act.get("pchembl_value")
            if pref_name and pchembl_val is not None:
                try:
                    val = float(pchembl_val)
                    drug_pchembl_map.setdefault(pref_name.upper(), []).append(val)
                except (ValueError, TypeError):
                    continue

        median_affinity_map: dict[str, float] = {}
        for drug_name, values in drug_pchembl_map.items():
            if values:
                median_affinity_map[drug_name] = statistics.median(values)

        return median_affinity_map
