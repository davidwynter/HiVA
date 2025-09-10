import pytest
from hiva.routing.distance import KnowledgeDistanceAblation

def test_tag_distance():
    dist = KnowledgeDistanceAblation(mode="tag")
    d = dist({"capabilities":["math","calc"]}, {"capabilities":["math"]})
    assert 0.0 <= d <= 1.0
    assert d == 0.0  # perfect overlap

def test_kg_distance_mock(monkeypatch):
    # mock KG client query to pretend there is always a match (count=1)
    class DummyClient:
        def query(self, sparql: str):
            return [{"c":"1"}]
    # Inject dummy into ablation object
    dist = KnowledgeDistanceAblation.__new__(KnowledgeDistanceAblation)
    from hiva.routing.distance import KnowledgeDistanceKG
    kg = KnowledgeDistanceKG.__new__(KnowledgeDistanceKG)
    kg.client = DummyClient()
    dist.impl = kg
    d = dist({"capabilities_uri":["http://example.org/Calc"]},
             {"capabilities_uri":["http://example.org/Math"]})
    assert d == 0.0  # overlap=1.0 -> distance 0.0

def test_kg_requires_endpoint():
    with pytest.raises(ValueError):
        KnowledgeDistanceAblation(mode="kg")
