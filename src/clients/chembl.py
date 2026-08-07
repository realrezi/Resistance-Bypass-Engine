from __future__ import annotations

import statistics
from typing import Any

from src.clients.base import BaseHTTPClient


class ChEMBLClient(BaseHTTPClient):
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    async def get_target_activities(
        self,
        target_chembl_id: str,
        limit: int = 1000,
        max_records: int = 50_000,
    ) -> dict[str, dict[str, Any]]:
        """Return quality-filtered human binding measurements grouped by drug.

        IC50, Ki and Kd pChEMBL values are kept as a transparent aggregate. The
        count and measurement types are returned so callers can communicate the
        uncertainty rather than treating one median as definitive potency.
        """
        url = f"{self.BASE_URL}/activity.json"
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "target_organism": "Homo sapiens",
            "assay_type": "B",
            "standard_type__in": "IC50,Ki,Kd",
            "pchembl_value__isnull": "false",
            "data_validity_comment__isnull": "true",
            "potential_duplicate": "false",
            "limit": limit,
        }

        activities: list[dict[str, Any]] = []
        next_url: str | None = url
        seen_urls: set[str] = set()
        first_page = True

        while next_url and len(activities) < max_records:
            full_url = (
                next_url
                if next_url.startswith("http")
                else f"https://www.ebi.ac.uk{next_url}"
            )
            if full_url in seen_urls:
                break
            seen_urls.add(full_url)

            data = await self.get_json(full_url, params=params if first_page else None)
            first_page = False
            if not isinstance(data, dict):
                break
            page = data.get("activities") or []
            activities.extend(page[: max_records - len(activities)])
            next_url = (data.get("page_meta") or {}).get("next")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for activity in activities:
            name = activity.get("molecule_pref_name") or activity.get(
                "molecule_chembl_id"
            )
            value = activity.get("pchembl_value")
            if not name or value is None:
                continue
            try:
                pchembl = float(value)
            except (TypeError, ValueError):
                continue
            grouped.setdefault(str(name).upper(), []).append(
                {
                    "value": pchembl,
                    "type": activity.get("standard_type"),
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "assay_chembl_id": activity.get("assay_chembl_id"),
                    "document_chembl_id": activity.get("document_chembl_id"),
                }
            )

        result: dict[str, dict[str, Any]] = {}
        for name, measurements in grouped.items():
            values = [item["value"] for item in measurements]
            result[name] = {
                "median_pchembl": float(statistics.median(values)),
                "measurement_count": len(values),
                "measurement_types": sorted(
                    {str(item["type"]) for item in measurements if item.get("type")}
                ),
                "molecule_chembl_id": next(
                    (
                        item["molecule_chembl_id"]
                        for item in measurements
                        if item.get("molecule_chembl_id")
                    ),
                    None,
                ),
                "assay_ids": sorted(
                    {
                        str(item["assay_chembl_id"])
                        for item in measurements
                        if item.get("assay_chembl_id")
                    }
                )[:10],
                "document_ids": sorted(
                    {
                        str(item["document_chembl_id"])
                        for item in measurements
                        if item.get("document_chembl_id")
                    }
                )[:10],
            }
        return result
