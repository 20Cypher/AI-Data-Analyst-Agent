import tempfile
from pathlib import Path
import pandas as pd
from ..agent.controller import AgentController

def test_controller_reports_progress_and_completes_without_llm():
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "data"
        arts_dir = Path(d) / "artifacts"
        data_dir.mkdir()
        arts_dir.mkdir()

        # Prepare a tiny dataset with a numeric column (viz fallback-friendly)
        df = pd.DataFrame({"x": [1,2,3], "y": [10, 20, 30]})
        (data_dir / "mini.csv").write_text(df.to_csv(index=False), encoding="utf-8")

        ac = AgentController(str(arts_dir), str(data_dir), max_tool_calls=50, max_runtime=120)

        events = []
        def progress_cb(ev):
            events.append(ev)

        res = ac.analyze("basic question", "mini", progress_cb=progress_cb)
        assert res["status"] in ("success",), f"run failed: {res}"
        # progress events captured
        assert events, "no progress events captured"
        # last event should be ~100
        assert int(events[-1].get("progress", 0)) >= 99
