"""File I/O operations and bundle creation"""
import json
import hashlib
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from agent.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE
from agent.llm import build_llm

logger = logging.getLogger(__name__)

class FileIOTool:
    def __init__(self, artifacts_dir: str, data_dir: str, llm: ChatOpenAI = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        self.llm = llm or build_llm()
    
    def save_markdown(self, text: str, filename: str = "summary.md") -> str:
        """Save markdown text to file"""
        filepath = self.artifacts_dir / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return str(filepath)
    
    def hash_dataset(self, dataset_path: str) -> str:
        """Generate SHA256 hash of dataset"""
        hasher = hashlib.sha256()
        
        with open(dataset_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def make_bundle(self, run_dir: str) -> Dict[str, Any]:
        """Create zip bundle of all artifacts"""
        run_path = Path(run_dir)
        bundle_path = run_path / "bundle.zip"
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all files in run directory
            for file_path in run_path.rglob('*'):
                if file_path.is_file() and file_path.name != "bundle.zip":
                    arcname = file_path.relative_to(run_path)
                    zf.write(file_path, arcname)
        
        return {
            "bundle_path": str(bundle_path),
            "success": True
        }
    
    def generate_summary(self, context: Any) -> Dict[str, Any]:
        """Generate analysis summary using LLM"""
        try:
            # Prepare context for summary
            context_text = self._format_context_for_summary(context)
            
            # Create summary prompt
            user_prompt = SUMMARY_USER_TEMPLATE.format(
                question=context.question,
                context=context_text
            )
            
            # Generate summary
            messages = [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            summary_text = response.content
            
            # Save summary
            summary_path = self.save_markdown(summary_text, f"{context.run_id}/summary.md")
            
            return {
                "summary_path": summary_path,
                "summary_text": summary_text,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Create fallback summary
            fallback_summary = f"""# Analysis Summary

**Question:** {context.question}

**Dataset:** {context.dataset_id}

**Status:** Analysis completed with basic processing.

**Note:** Detailed summary generation encountered an issue: {str(e)}
"""
            summary_path = self.save_markdown(fallback_summary, f"{context.run_id}/summary.md")
            
            return {
                "summary_path": summary_path,
                "summary_text": fallback_summary,
                "success": False,
                "error": str(e)
            }
    
    def _format_context_for_summary(self, context: Any) -> str:
        lines = [f"Dataset: {context.dataset_id}"]

        # Computed values (flat)
        if hasattr(context, 'computed_values') and context.computed_values:
            lines.append("Computed Values:")
            for key, value in context.computed_values.items():
                # For nested dicts like topk/extrema show compact JSON-ish
                if key in ("topk", "bottomk", "extrema"):
                    import json
                    try:
                        snippet = json.dumps(value, default=str)[:4000]
                    except Exception:
                        snippet = str(value)[:4000]
                    lines.append(f"- {key}: {snippet}")
                else:
                    lines.append(f"- {key}: {value}")

        # Artifacts
        if getattr(context, 'artifacts', None):
            lines.append("Generated Artifacts:")
            for artifact_type, path in context.artifacts.items():
                lines.append(f"- {artifact_type}: {path}")

        # DataFrames
        if getattr(context, 'dataframes', None):
            lines.append("DataFrames:")
            for name, path in context.dataframes.items():
                lines.append(f"- {name}: {path}")

        return "\n".join(lines)
