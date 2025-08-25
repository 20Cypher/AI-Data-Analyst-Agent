"""Fixed Jupyter notebook builder for creating analysis notebooks"""
import nbformat as nbf
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import textwrap as tw

logger = logging.getLogger(__name__)


class NotebookBuilderTool:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)

    def compose(self, context: Any) -> Dict[str, Any]:
        """Build Jupyter notebook from analysis context"""
        try:
            # Create new notebook
            nb = nbf.v4.new_notebook()

            # Add title cell
            title_cell = nbf.v4.new_markdown_cell(f"""# Analysis: {context.question}
            **Dataset:** {context.dataset_id}  
            **Run ID:** {context.run_id}  
            **Generated:** {self._get_timestamp()}
            """)
            nb.cells.append(title_cell)

            # Add setup cell
            setup_cell = nbf.v4.new_code_cell(tw.dedent("""
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                from pathlib import Path

                try:
                    import seaborn as sns
                    _HAS_SEABORN = True
                except Exception:
                    _HAS_SEABORN = False

                plt.style.use('default')
                if _HAS_SEABORN:
                    sns.set_palette("husl")
                %matplotlib inline
            """))
            nb.cells.append(setup_cell)

            # Add data loading cell
            data_loading_cell = self._create_data_loading_cell(context)
            nb.cells.append(data_loading_cell)

            # environment info cell
            env_info_cell = nbf.v4.new_code_cell(tw.dedent("""# Environment info (for reproducibility)
            import sys, platform
            import pandas as pd
            try:
                import duckdb
                duckdb_version = duckdb.__version__
            except Exception:
                duckdb_version = "not installed"
            import matplotlib

            print("Python:", sys.version.replace("\\n"," "))
            print("Platform:", platform.platform())
            print("pandas:", pd.__version__)
            print("duckdb:", duckdb_version)
            print("matplotlib:", matplotlib.__version__)"""))
            nb.cells.append(env_info_cell)


            # Add analysis cells based on plan steps
            if hasattr(context, 'plan') and context.plan:
                for step in context.plan.steps:
                    cell = self._create_cell_for_step(step, context)
                    if cell:
                        nb.cells.append(cell)

            # Add summary cell
            summary_cell = self._create_summary_cell(context)
            nb.cells.append(summary_cell)

            # Save notebook
            notebook_path = Path(context.run_dir) / "analysis.ipynb"
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)

            return {
                "notebook_path": str(notebook_path),
                "success": True,
                "cell_count": len(nb.cells)
            }

        except Exception as e:
            logger.error(f"Notebook creation failed: {e}")
            raise

    def _create_data_loading_cell(self, context):
        data_dir = (Path(context.run_dir).parents[1] / "data").as_posix()
        code = f"""
        from pathlib import Path
        import pandas as pd

        DATA_DIR = Path("{data_dir}")
        dataset_path = DATA_DIR / "{context.dataset_id}.csv"
        if not dataset_path.exists():
            dataset_path = DATA_DIR / "{context.dataset_id}.parquet"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {{dataset_path}}")

        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_parquet(dataset_path)

        print("Dataset path:", dataset_path)
        print("Dataset shape:", df.shape)
        print("Columns:", list(df.columns))
        df.head()
        """
        return nbf.v4.new_code_cell(tw.dedent(code))

    def _create_cell_for_step(self, step: Any, context: Any) -> nbf.NotebookNode:
        """Create notebook cell for a plan step"""
        if step.action == "inspect_schema":
            return nbf.v4.new_code_cell(tw.dedent("""# Dataset Schema and Info
print("Dataset Info:")
print(df.info())
print("\\nBasic Statistics:")
print(df.describe())
print("\\nMissing Values:")
print(df.isnull().sum())"""))

        elif step.action == "prepare_data":
            operations_text = "\\n".join(getattr(step, 'operations', []))
            return nbf.v4.new_markdown_cell(f"""## Data Preparation

Applied operations:
{operations_text}""")

        elif step.action == "aggregate":
            group_by = getattr(step, 'group_by', [])
            metrics = getattr(step, 'metrics', [])

            code = f"""# Aggregation Analysis
"""
            if group_by and metrics:
                group_clause = ", ".join([f'"{col}"' for col in group_by])
                code += f"""
# Group by {', '.join(group_by)}
result = df.groupby([{group_clause}]).agg({{
"""
                for metric in metrics:
                    if "(" in metric:  # e.g., "sum(sales)"
                        func, col = metric.split("(")
                        col = col.rstrip(")")
                        code += f'    "{col}": "{func}",\\n'

                code += """}).reset_index()
print("Aggregation Results:")
print(result)
result.head(10)"""

            return nbf.v4.new_code_cell(tw.dedent(code))

        elif step.action == "visualize":
            chart_type = getattr(step, 'type', 'line')
            x_col = getattr(step, 'x', 'index')
            y_col = getattr(step, 'y', 'value')

            code = f"""# {chart_type.title()} Chart
plt.figure(figsize=(10, 6))
"""
            if chart_type == "line":
                code += f'plt.plot(df["{x_col}"], df["{y_col}"], marker="o")'
            elif chart_type == "bar":
                code += f'plt.bar(df["{x_col}"], df["{y_col}"])'
            elif chart_type == "boxplot":
                code += f'plt.boxplot(df["{y_col}"].dropna())'
            elif chart_type == "scatter":
                code += f'plt.scatter(df["{x_col}"], df["{y_col}"], alpha=0.6)'

            code += f"""
plt.xlabel("{x_col}")
plt.ylabel("{y_col}")
plt.title("{chart_type.title()} Chart")
plt.tight_layout()
plt.show()"""

            return nbf.v4.new_code_cell(tw.dedent(code))

        elif step.action == "interpret":
            return nbf.v4.new_markdown_cell(f"""## Interpretation

Analysis of the results shows patterns in the data that address the original question: "{context.question}"
""")

        return None

    def _create_summary_cell(self, context: Any) -> nbf.NotebookNode:
        """Create summary cell"""
        return nbf.v4.new_markdown_cell("""## Summary

This analysis was generated automatically by the AI Data Analyst Agent. 
The notebook contains reproducible code that can be re-executed to generate the same results.

For the complete analysis bundle including charts and detailed summary, refer to the accompanying files.
""")

    def _get_timestamp(self) -> str:
        """Get current timestamp for notebook"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
