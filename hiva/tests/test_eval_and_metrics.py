from hiva.eval.datasets import python_expr_dataset
from hiva.eval.metrics import CostAndAccuracy

def test_dataset_and_metrics():
    ds = python_expr_dataset()
    assert len(ds) >= 3
    m = CostAndAccuracy()
    m.add(acc=1.0, cost=0.0)
    s = m.summary()
    assert s["avg_acc"] > 0.0 and s["n"] == 1
