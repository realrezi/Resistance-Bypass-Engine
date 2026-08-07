import re
from datetime import UTC, datetime
from typing import Any

from src.clients.base import BaseHTTPClient
from src.schemas.evidence import EvidenceDirection, EvidenceLevel, EvidenceSource


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
            diseases {
              diseaseFromSource
              disease { id name }
            }
            clinicalReports {
              id
              source
              clinicalStage
              trialOverallStatus
              url
              title
            }
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

    async def get_known_drugs(
        self,
        ensembl_id: str,
        size: int = 100,
        cancer_type: str | None = None,
        primary_drug: str | None = None,
    ) -> list[dict[str, Any]]:
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

        parsed_drugs: list[dict[str, Any]] = []
        cancer_query = _normalize_text(cancer_type or "")
        primary_drug_query = _normalize_text(primary_drug or "")
        for row in rows[:size]:
            drug_obj = row.get("drug", {}) or {}
            drug_id = drug_obj.get("id", "")
            pref_name = drug_obj.get("name") or drug_id
            drug_type = drug_obj.get("drugType", "")
            moa_rows = drug_obj.get("mechanismsOfAction", {}).get("rows", [])
            moa = (
                moa_rows[0].get("mechanismOfAction")
                if moa_rows
                else "Bypass Pathway Inhibitor"
            )
            stage_str = row.get("maxClinicalStage") or drug_obj.get(
                "maximumClinicalStage"
            )
            phase: int | None = None
            if stage_str and "PHASE_" in str(stage_str):
                try:
                    phase = int(str(stage_str).replace("PHASE_", ""))
                except ValueError:
                    phase = None

            diseases = row.get("diseases", []) or []
            disease_names = [
                (item.get("disease", {}) or {}).get("name")
                or item.get("diseaseFromSource")
                for item in diseases
            ]
            disease_names = [name for name in disease_names if name]
            indication_match = _indication_matches(cancer_query, disease_names)

            reports = row.get("clinicalReports", []) or []
            report_evidence: list[EvidenceSource] = []
            statuses: list[str] = []
            combination_evidence = False
            for report in reports[:20]:
                title = report.get("title") or ""
                status = (report.get("trialOverallStatus") or "").upper()
                if status:
                    statuses.append(status)
                if primary_drug_query and primary_drug_query in _normalize_text(title):
                    combination_evidence = True
                evidence_direction = _status_direction(status)
                report_evidence.append(
                    EvidenceSource(
                        name=report.get("source") or "Open Targets clinical report",
                        stable_id=report.get("id"),
                        url=report.get("url"),
                        release="Open Targets Platform API v4",
                        retrieved_at=datetime.now(UTC).date(),
                        level=EvidenceLevel.PROSPECTIVE_CLINICAL
                        if "PHASE" in str(report.get("clinicalStage"))
                        else EvidenceLevel.CURATED_DATABASE,
                        direction=evidence_direction,
                        excerpt_or_field=(
                            f"clinicalStage={report.get('clinicalStage')}; "
                            f"trialOverallStatus={report.get('trialOverallStatus')}; "
                            f"title={title[:500]}"
                        ),
                        limitations=[
                            "A clinical report is not proof that the requested drug pair was evaluated or effective.",
                            "Status-derived direction is a triage signal, not an efficacy conclusion.",
                        ],
                    )
                )

            status = _aggregate_status(statuses)
            if not reports:
                status = "status_not_reported"

            if statuses and all(
                item in {"WITHDRAWN", "TERMINATED", "SUSPENDED"} for item in statuses
            ):
                status = "stopped_or_withdrawn"

            parsed_drugs.append(
                {
                    "drugId": drug_id,
                    "prefName": pref_name,
                    "drugType": drug_type,
                    "mechanismOfAction": moa,
                    "targetSymbol": target_data.get("approvedSymbol", ""),
                    "phase": phase,
                    "status": status,
                    "indicationMatch": indication_match,
                    "combinationEvidence": combination_evidence,
                    "clinicalStatus": status,
                    "diseaseNames": disease_names,
                    "evidence": report_evidence,
                }
            )

        return parsed_drugs


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _indication_matches(cancer_query: str, disease_names: list[str]) -> bool:
    """Use token-aware matching to avoid accidental substring matches."""
    if not cancer_query:
        return False
    query_tokens = set(re.findall(r"[a-z0-9]+", cancer_query))
    for disease in disease_names:
        disease_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(disease)))
        if query_tokens and (
            query_tokens <= disease_tokens or disease_tokens <= query_tokens
        ):
            return True
    return False


def _status_direction(status: str) -> EvidenceDirection:
    if status in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}:
        return EvidenceDirection.CONTRADICTS
    if status in {
        "RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "NOT_YET_RECRUITING",
        "ENROLLING_BY_INVITATION",
    }:
        return EvidenceDirection.NEUTRAL
    return EvidenceDirection.NEUTRAL


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "status_not_reported"
    if all(item in {"WITHDRAWN", "TERMINATED", "SUSPENDED"} for item in statuses):
        return "stopped_or_withdrawn"
    if any(
        item
        in {
            "RECRUITING",
            "NOT_YET_RECRUITING",
            "ENROLLING_BY_INVITATION",
            "ACTIVE_NOT_RECRUITING",
            "COMPLETED",
        }
        for item in statuses
    ):
        return "active_or_completed"
    return "status_reported"
