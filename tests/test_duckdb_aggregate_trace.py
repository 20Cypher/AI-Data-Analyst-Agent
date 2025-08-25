import tempfile
from pathlib import Path
import pandas as pd

from ..tools.duckdb_sql import DuckDBTool

class _Ctx:
    def __init__(self, run_dir, dataset_id):
        self.run_dir = run_dir
        self.dataset_id = dataset_id

def test_aggregate_returns_meta_and_safe_order_by():
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "data"
        arts_dir = Path(d) / "artifacts"
        data_dir.mkdir()
        arts_dir.mkdir()

        # Create small dataset
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "B"],
            "value": [1, 2, 3, 4, 5],
        })
        csv = data_dir / "t1.csv"
        df.to_csv(csv, index=False)

        tool = DuckDBTool(str(data_dir), artifacts_dir=str(arts_dir))
        ctx = _Ctx(run_dir=str(arts_dir / "run1"), dataset_id="t1")
        Path(ctx.run_dir).mkdir(parents=True, exist_ok=True)

        res = tool.aggregate(group_by=["category"], metrics=["sum(value)"], context=ctx)
        assert res["rows"] >= 2
        assert res.get("group_by") == ["category"]
        assert res.get("metrics") == ["sum(value)"]
        assert "query" in res
        assert " ORDER BY category" in res["query"]
