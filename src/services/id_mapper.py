import logging

from src.clients.base import BaseHTTPClient, _stable_cache_key, cache
from src.schemas.models import IDMappingResult

logger = logging.getLogger(__name__)


# Reviewed identifiers for the genes used by the built-in scenario library.
# Keeping this small reference set locally makes presets reproducible and prevents
# a temporary HGNC, UniProt, or ChEMBL outage from blocking the entire report.
# Other symbols continue through the live resolution workflow below.
CURATED_GENE_IDENTIFIERS: dict[str, dict[str, str | None]] = {
    "ABL1": {
        "ensembl_id": "ENSG00000097007",
        "uniprot_id": "P00519",
        "chembl_target_id": "CHEMBL1862",
    },
    "ALK": {
        "ensembl_id": "ENSG00000171094",
        "uniprot_id": "Q9UM73",
        "chembl_target_id": "CHEMBL4247",
    },
    "AR": {
        "ensembl_id": "ENSG00000169083",
        "uniprot_id": "P10275",
        "chembl_target_id": "CHEMBL1871",
    },
    "BRAF": {
        "ensembl_id": "ENSG00000157764",
        "uniprot_id": "P15056",
        "chembl_target_id": "CHEMBL5145",
    },
    "BRCA2": {
        "ensembl_id": "ENSG00000139618",
        "uniprot_id": "P51587",
        "chembl_target_id": None,
    },
    "CDK4": {
        "ensembl_id": "ENSG00000135446",
        "uniprot_id": "P11802",
        "chembl_target_id": "CHEMBL331",
    },
    "EGFR": {
        "ensembl_id": "ENSG00000146648",
        "uniprot_id": "P00533",
        "chembl_target_id": "CHEMBL203",
    },
    "ERBB2": {
        "ensembl_id": "ENSG00000141736",
        "uniprot_id": "P04626",
        "chembl_target_id": "CHEMBL1824",
    },
    "ESR1": {
        "ensembl_id": "ENSG00000091831",
        "uniprot_id": "P03372",
        "chembl_target_id": "CHEMBL206",
    },
    "FGFR2": {
        "ensembl_id": "ENSG00000066468",
        "uniprot_id": "P21802",
        "chembl_target_id": "CHEMBL4142",
    },
    "KIT": {
        "ensembl_id": "ENSG00000157404",
        "uniprot_id": "P10721",
        "chembl_target_id": "CHEMBL1936",
    },
    "KRAS": {
        "ensembl_id": "ENSG00000133703",
        "uniprot_id": "P01116",
        "chembl_target_id": "CHEMBL2189121",
    },
    "MAP2K1": {
        "ensembl_id": "ENSG00000169032",
        "uniprot_id": "Q02750",
        "chembl_target_id": "CHEMBL2964",
    },
    "MET": {
        "ensembl_id": "ENSG00000105976",
        "uniprot_id": "P08581",
        "chembl_target_id": "CHEMBL3717",
    },
    "PARP1": {
        "ensembl_id": "ENSG00000143799",
        "uniprot_id": "P09874",
        "chembl_target_id": "CHEMBL3105",
    },
    "PIK3CA": {
        "ensembl_id": "ENSG00000121879",
        "uniprot_id": "P42336",
        "chembl_target_id": "CHEMBL4005",
    },
    "RET": {
        "ensembl_id": "ENSG00000165731",
        "uniprot_id": "P07949",
        "chembl_target_id": "CHEMBL2041",
    },
    "ROS1": {
        "ensembl_id": "ENSG00000047936",
        "uniprot_id": "P08922",
        "chembl_target_id": "CHEMBL2431",
    },
}

CURATED_GENE_ALIASES = {"HER2": "ERBB2"}


class IDMapper(BaseHTTPClient):
    def __init__(self, timeout: float = 30.0):
        super().__init__(timeout=timeout)

    async def resolve_hgnc(self, symbol: str) -> tuple[str, str]:
        """Resolve canonical HGNC symbol and Ensembl ID (version stripped)."""
        url = f"https://rest.genenames.org/fetch/symbol/{symbol}"
        headers = {"Accept": "application/json"}
        data = await self.get_json(url, headers=headers)

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            # Fallback search across symbols, aliases, and previous symbols (e.g., HER2 -> ERBB2)
            search_url = f"https://rest.genenames.org/search/{symbol}"
            search_data = await self.get_json(search_url, headers=headers)
            search_docs = search_data.get("response", {}).get("docs", [])
            if search_docs:
                canonical = search_docs[0].get("symbol", "").upper()
                if canonical:
                    # Fetch official canonical document for the resolved symbol
                    fetch_url = f"https://rest.genenames.org/fetch/symbol/{canonical}"
                    fetch_data = await self.get_json(fetch_url, headers=headers)
                    docs = fetch_data.get("response", {}).get("docs", [])

        if not docs:
            raise ValueError(f"Gene symbol '{symbol}' not found in HGNC.")

        doc = docs[0]
        canonical_symbol = doc.get("symbol", symbol).upper()
        raw_ensembl = doc.get("ensembl_gene_id", "")
        # Strip Ensembl version suffixes (e.g. ENSG00000146648.15 -> ENSG00000146648)
        ensembl_id = raw_ensembl.split(".")[0] if raw_ensembl else ""

        return canonical_symbol, ensembl_id

    async def resolve_uniprot(self, canonical_symbol: str) -> str:
        """Resolve UniProt primary accession for human gene symbol."""
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene_exact:{canonical_symbol} AND organism_id:9606 AND reviewed:true",
            "format": "json",
            "size": 5,
        }
        data = await self.get_json(url, params=params)
        results = data.get("results", [])
        if not results:
            # Fallback search without gene_exact constraint
            params["query"] = (
                f"gene:{canonical_symbol} AND organism_id:9606 AND reviewed:true"
            )
            data = await self.get_json(url, params=params)
            results = data.get("results", [])

        if not results:
            raise ValueError(
                f"UniProt ID not found for gene symbol '{canonical_symbol}'."
            )

        return results[0]["primaryAccession"]

    async def resolve_chembl_target(self, uniprot_id: str) -> str | None:
        """Resolve ChEMBL target ID strictly filtered for SINGLE PROTEIN targets."""
        url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
        params = {"target_components__accession": uniprot_id}
        data = await self.get_json(url, params=params)
        targets = data.get("targets", [])

        # Strict filter for target_type == 'SINGLE PROTEIN' to prevent target multiplicity bugs
        single_protein_targets = [
            t for t in targets if t.get("target_type") == "SINGLE PROTEIN"
        ]

        if single_protein_targets:
            return single_protein_targets[0].get("target_chembl_id")

        return None

    async def map_identifier(self, symbol: str) -> IDMappingResult:
        """Complete ID resolution flow returning an IDMappingResult instance."""
        clean_symbol = symbol.strip().upper()
        curated_symbol = CURATED_GENE_ALIASES.get(clean_symbol, clean_symbol)
        curated = CURATED_GENE_IDENTIFIERS.get(curated_symbol)
        if curated:
            return IDMappingResult(
                original_input=symbol,
                canonical_symbol=curated_symbol,
                ensembl_id=str(curated["ensembl_id"]),
                uniprot_id=str(curated["uniprot_id"]),
                chembl_target_id=curated["chembl_target_id"],
            )

        cache_key = _stable_cache_key("ID-MAP", "hgnc-uniprot-chembl", clean_symbol)
        try:
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                return IDMappingResult.model_validate(cached)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("ID mapping cache read failed: %s", exc)
        canonical_symbol, ensembl_id = await self.resolve_hgnc(clean_symbol)
        uniprot_id = await self.resolve_uniprot(canonical_symbol)
        chembl_target_id = await self.resolve_chembl_target(uniprot_id)

        result = IDMappingResult(
            original_input=symbol,
            canonical_symbol=canonical_symbol,
            ensembl_id=ensembl_id,
            uniprot_id=uniprot_id,
            chembl_target_id=chembl_target_id,
        )
        try:
            cache.set(cache_key, result.model_dump(mode="json"), expire=604800)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("ID mapping cache write failed: %s", exc)
        return result
