# Placeholder-free minimal client would require an RDF endpoint.
# For MVP we keep distance computation local (routing.distance).
# This module is implemented but unused.
class KGClient:
    def query(self, sparql: str) -> list[dict]:
        raise NotImplementedError("Attach an RDF store to use KGClient.")
