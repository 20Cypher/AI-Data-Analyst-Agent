"""Tool router for dispatching operations to appropriate tools.

This router is a simple dependency container. The AgentController is responsible
for calling the right method on each tool (e.g., aggregate, visualize, etc.).
Use:
    router.get_tool("duckdb_sql").aggregate(...)
    router.get_tool("viz").create_chart(...)

We intentionally do NOT expose a generic `dispatch(step, ...)` here to avoid
duplicating execution logic and to keep typing/contexts explicit.
"""
from typing import Dict, Any, Optional
from .duckdb_sql import DuckDBTool
from .python_repl import PythonREPLTool
from .viz import VizTool
from .file_io import FileIOTool
from .notebook_builder import NotebookBuilderTool
from .validation import ValidationTool

class ToolRouter:
    """Routes tool requests to appropriate implementations (DI container)."""

    def __init__(self, artifacts_dir: str, data_dir: str, llm: Optional[Any] = None):
        # Note: pass artifacts_dir to DuckDBTool so it can persist frames under the run folder
        self.tools = {
            "duckdb_sql": DuckDBTool(data_dir, artifacts_dir),
            "python_repl": PythonREPLTool(artifacts_dir),
            "viz": VizTool(artifacts_dir),
            "file_io": FileIOTool(artifacts_dir, data_dir, llm=llm),
            "notebook_builder": NotebookBuilderTool(artifacts_dir),
            "validation": ValidationTool(),
        }

    def get_tool(self, name: str):
        """Get tool instance by name."""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name]