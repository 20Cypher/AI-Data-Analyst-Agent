import tempfile
from pathlib import Path

from ..tools.python_repl import PythonREPLTool

def test_repl_blocks_os_import():
    with tempfile.TemporaryDirectory() as d:
        tool = PythonREPLTool(artifacts_dir=d)
        out = tool.exec("import os\nx=1", context={})
        assert not out["success"]
        assert "Import blocked" in (out.get("error") or "") or "blocked" in (out.get("error") or "")

def test_repl_allows_simple_math():
    with tempfile.TemporaryDirectory() as d:
        tool = PythonREPLTool(artifacts_dir=d)
        out = tool.exec("a = 1 + 2\nprint(a)", context={})
        assert out["success"]
        assert "variables" in out and out["variables"].get("a") == 3
        assert "output" in out and "3" in out["output"]
