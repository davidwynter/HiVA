from hiva.knowledge.matching import kg_distance

class DummyClient:
    def query(self, sparql: str):
        # Return count=1 to simulate a match
        return [{"c":"1"}]

def test_kg_distance_mock():
    client = DummyClient()
    d = kg_distance(client,
                    {"capabilities_uri":["http://example.org/Calc"]},
                    {"capabilities_uri":["http://example.org/Math"]})
    assert d == 0.0
