from typing import Any

from src.clients.base import CONTACT_EMAIL, BaseHTTPClient


class StringDBClient(BaseHTTPClient):
    BASE_URL = "https://string-db.org/api/json/network"

    async def get_network(
        self,
        t_primary: str,
        t_resistance: str,
        add_nodes: int = 25,
        required_score: int = 400,
        species: int = 9606,
        network_type: str = "physical",
    ) -> list[dict[str, Any]]:
        """Fetch protein interaction network from STRING-DB."""
        params = {
            "identifiers": f"{t_primary}\r{t_resistance}",
            "species": species,
            "required_score": required_score,
            "add_nodes": add_nodes,
            "network_type": network_type,
            "caller_identity": f"mailto:{CONTACT_EMAIL}",
        }
        data = await self.get_json(self.BASE_URL, params=params)
        if isinstance(data, list):
            return data
        return []
