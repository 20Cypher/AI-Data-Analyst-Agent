"""Memory and context management for agent runs"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Context:
    run_id: str
    question: str
    dataset_id: str
    run_dir: str
    start_time: float
    plan: Optional[Any] = None
    dataframes: Dict[str, str] = field(default_factory=dict)  # name -> pickle path
    artifacts: Dict[str, str] = field(default_factory=dict)   # type -> path
    computed_values: Dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0

class Memory:
    """Manages context and state during agent execution"""
    
    def update_context(self, context: Context, step: Any, result: Dict[str, Any]) -> None:
        """Update context based on step execution result"""
        
        # Update dataframe registry
        if "df_pickle_path" in result:
            df_name = result.get("df_name", f"df_{step.id}")
            context.dataframes[df_name] = result["df_pickle_path"]
        
        # Update artifacts registry
        if "outfile" in result:
            artifact_type = self._get_artifact_type(result["outfile"])
            context.artifacts[artifact_type] = result["outfile"]
        
        # Update computed values for grounding
        if "computed_values" in result:
            context.computed_values.update(result["computed_values"])
        
        # Store aggregation results
        if step.action == "aggregate" and "preview" in result:
            context.computed_values[f"aggregation_{step.id}"] = result["preview"]
    
    def _get_artifact_type(self, filepath: str) -> str:
        """Determine artifact type from file path"""
        path = Path(filepath)
        if path.suffix == ".png":
            return "chart"
        elif path.suffix == ".md":
            return "summary"
        elif path.suffix == ".ipynb":
            return "notebook"
        else:
            return "other"
    
    def get_context_for_summary(self, context: Context) -> Dict[str, Any]:
        """Prepare context data for summary generation"""
        return {
            "question": context.question,
            "dataset_id": context.dataset_id,
            "computed_values": context.computed_values,
            "artifacts": context.artifacts,
            "dataframes": list(context.dataframes.keys())
        }