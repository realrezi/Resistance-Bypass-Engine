from typing import Optional
from src.clients.base import BaseHTTPClient
from src.schemas.models import IDMappingResult


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
            "query": f"gene_exact:{canonical_symbol} AND organism_id:9606",
            "format": "json",
        }
        data = await self.get_json(url, params=params)
        results = data.get("results", [])
        if not results:
            # Fallback search without gene_exact constraint
            params["query"] = f"gene:{canonical_symbol} AND organism_id:9606"
            data = await self.get_json(url, params=params)
            results = data.get("results", [])

        if not results:
            raise ValueError(f"UniProt ID not found for gene symbol '{canonical_symbol}'.")

        return results[0]["primaryAccession"]

    async def resolve_chembl_target(self, uniprot_id: str) -> Optional[str]:
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
        canonical_symbol, ensembl_id = await self.resolve_hgnc(clean_symbol)
        uniprot_id = await self.resolve_uniprot(canonical_symbol)
        chembl_target_id = await self.resolve_chembl_target(uniprot_id)

        return IDMappingResult(
            original_input=symbol,
            canonical_symbol=canonical_symbol,
            ensembl_id=ensembl_id,
            uniprot_id=uniprot_id,
            chembl_target_id=chembl_target_id,
        )
