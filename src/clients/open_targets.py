from __future__ import annotations

import re
from typing import Any

from src.clients.base import BaseHTTPClient


def _phase_number(value: Any) -> int:
    match = re.search(r"([0-4])", str(value or ""))
    return int(match.group(1)) if match else 0


def _context_tokens(value: str) -> set[str]:
    stop_words = {
        "and",
        "cancer",
        "carcinoma",
        "disease",
        "malignancy",
        "metastatic",
        "advanced",
        "tumor",
        "tumour",
        "solid",
        "cell",
        "cells",
        "the",
        "of",
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    tokens = {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in stop_words
    }
    if "nsclc" in normalized or {"non", "small", "lung"}.issubset(
        set(normalized.split())
    ):
        tokens.update({"nsclc", "lung"})
    if "cml" in normalized or {"chronic", "myeloid", "leukemia"}.issubset(
        set(normalized.split())
    ):
        tokens.update({"cml", "leukemia"})
    return tokens


class OpenTargetsClient(BaseHTTPClient):
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    CLINICAL_CANDIDATES_QUERY = """
    query clinicalCandidates($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        drugAndClinicalCandidates {
          count
          rows {
            id
            maxClinicalStage
            diseases {
              diseaseFromSource
              disease { id name }
            }
            clinicalReports {
              id source clinicalStage trialOverallStatus trialPhase url title
            }
            drug {
              id name drugType maximumClinicalStage
              drugWarnings { id warningType description country year }
              mechanismsOfAction {
                rows { mechanismOfAction targetName }
              }
            }
          }
        }
      }
    }
    """

    async def get_known_drugs(
        self,
        ensembl_id: str,
        cancer_type: str = "",
        primary_drug: str = "",
    ) -> list[dict[str, Any]]:
        """Return target-linked clinical agents with indication and trial provenance."""
        clean_ensembl = ensembl_id.split(".")[0]
        payload = {
            "query": self.CLINICAL_CANDIDATES_QUERY,
            "variables": {"ensemblId": clean_ensembl},
        }
        data = await self.post_json(self.BASE_URL, json_data=payload)
        if not isinstance(data, dict):
            return []
        if data.get("errors"):
            messages = "; ".join(
                str(item.get("message", "GraphQL error")) for item in data["errors"]
            )
            raise RuntimeError(f"Open Targets GraphQL error: {messages}")

        target_data = data.get("data", {}).get("target")
        if not target_data:
            return []

        rows = target_data.get("drugAndClinicalCandidates", {}).get("rows", [])
        requested_tokens = _context_tokens(cancer_type)
        primary_drug_lower = primary_drug.lower().strip()
        parsed: list[dict[str, Any]] = []

        for row in rows:
            drug = row.get("drug") or {}
            drug_id = drug.get("id")
            drug_name = drug.get("name") or drug_id
            if not drug_name:
                continue

            disease_entries = row.get("diseases") or []
            disease_labels: list[str] = []
            disease_ids: list[str] = []
            for entry in disease_entries:
                disease = entry.get("disease") or {}
                label = disease.get("name") or entry.get("diseaseFromSource")
                if label:
                    disease_labels.append(str(label))
                if disease.get("id"):
                    disease_ids.append(str(disease["id"]))

            disease_tokens = _context_tokens(" ".join(disease_labels))
            indication_match = bool(
                requested_tokens and requested_tokens & disease_tokens
            )

            reports = row.get("clinicalReports") or []
            statuses = {
                str(report.get("trialOverallStatus") or "").upper()
                for report in reports
                if report.get("trialOverallStatus")
            }
            active_statuses = {
                "RECRUITING",
                "ACTIVE_NOT_RECRUITING",
                "NOT_YET_RECRUITING",
                "ENROLLING_BY_INVITATION",
                "COMPLETED",
            }
            stopped_statuses = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
            if statuses & active_statuses:
                clinical_status = "active_or_completed"
            elif statuses and statuses <= stopped_statuses:
                clinical_status = "stopped"
            else:
                clinical_status = "reported"

            warnings = drug.get("drugWarnings") or []
            is_withdrawn = any(
                "withdraw" in str(warning.get("warningType") or "").lower()
                or "withdraw" in str(warning.get("description") or "").lower()
                for warning in warnings
            )

            combination_evidence = bool(
                primary_drug_lower
                and any(
                    primary_drug_lower in str(report.get("title") or "").lower()
                    for report in reports
                )
            )
            evidence = [
                {
                    "source": str(report.get("source") or "Open Targets"),
                    "record_id": report.get("id"),
                    "url": report.get("url"),
                    "title": report.get("title"),
                    "status": report.get("trialOverallStatus")
                    or report.get("clinicalStage"),
                }
                for report in reports[:8]
            ]
            for warning in warnings[:3]:
                evidence.append(
                    {
                        "source": "Open Targets drug warning",
                        "record_id": str(warning.get("id"))
                        if warning.get("id")
                        else None,
                        "title": warning.get("description")
                        or warning.get("warningType"),
                        "status": warning.get("warningType"),
                    }
                )

            moa_rows = (drug.get("mechanismsOfAction") or {}).get("rows", [])
            mechanism = (
                moa_rows[0].get("mechanismOfAction")
                if moa_rows
                else "Target-linked clinical agent"
            )
            phase = _phase_number(
                row.get("maxClinicalStage") or drug.get("maximumClinicalStage")
            )

            parsed.append(
                {
                    "drugId": drug_id or "",
                    "prefName": str(drug_name),
                    "drugType": drug.get("drugType") or "",
                    "mechanismOfAction": mechanism,
                    "targetSymbol": target_data.get("approvedSymbol", ""),
                    "phase": phase,
                    "clinicalStatus": clinical_status,
                    "isWithdrawn": is_withdrawn,
                    "indicationMatch": indication_match,
                    "combinationEvidence": combination_evidence,
                    "diseaseLabels": sorted(set(disease_labels)),
                    "diseaseIds": sorted(set(disease_ids)),
                    "evidence": evidence,
                }
            )

        parsed.sort(
            key=lambda item: (
                item["isWithdrawn"],
                not item["combinationEvidence"],
                not item["indicationMatch"],
                -item["phase"],
                item["prefName"].upper(),
            )
        )
        return parsed
