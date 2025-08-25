"""Visualization tool for creating charts and plots"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

# Use non-interactive backend
matplotlib.use('Agg')
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class VizTool:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        
    def create_chart(self, chart_type: str, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Create a chart and save it as PNG"""
        try:
            # Ensure charts directory exists
            charts_dir = Path(context.run_dir) / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # Load data
            df = self._load_data_for_viz(context)
            if df is None or df.empty:
                raise ValueError("No data available for visualization")
            
            # Create chart based on type
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "line":
                self._create_line_chart(df, params, ax)
            elif chart_type == "bar":
                self._create_bar_chart(df, params, ax)
            elif chart_type == "boxplot":
                self._create_boxplot(df, params, ax)
            elif chart_type == "scatter":
                self._create_scatter_chart(df, params, ax)
            elif chart_type == "heatmap":
                self._create_heatmap(df, params, ax)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Save chart
            outfile = params.get("outfile", f"charts/{chart_type}_chart.png")
            if not outfile.startswith("charts/"):
                outfile = f"charts/{outfile}"
            
            full_path = Path(context.run_dir) / outfile
            full_path.parent.mkdir(exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return {
                "outfile": str(full_path),
                "chart_type": chart_type,
                "success": True,
                "computed_values": {f"{chart_type}_chart_created": True}
            }
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            plt.close('all')  # Clean up any open figures
            raise
    
    def _load_data_for_viz(self, context: Any) -> pd.DataFrame:
        """Load data for visualization from context"""
        # Try to load the most recent DataFrame from context
        if hasattr(context, 'dataframes') and context.dataframes:
            # Get the most recent DataFrame
            df_name, pickle_path = list(context.dataframes.items())[-1]
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load DataFrame from {pickle_path}: {e}")
        
        # Fallback: load original dataset
        data_dir = Path(context.run_dir).parent.parent / "data"
        dataset_path = data_dir / f"{context.dataset_id}.csv"
        
        if not dataset_path.exists():
            dataset_path = data_dir / f"{context.dataset_id}.parquet"
        
        if dataset_path.exists():
            if dataset_path.suffix == '.csv':
                return pd.read_csv(dataset_path)
            else:
                return pd.read_parquet(dataset_path)
        
        return None
    
    def _create_line_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        """Create line chart"""
        x_col = params.get("x")
        y_col = params.get("y")
        
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            ax.plot(df[x_col], df[y_col], marker='o')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            # Fallback: plot first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                ax.plot(df[numeric_cols[0]])
                ax.set_ylabel(numeric_cols[0])
        
        ax.set_title("Line Chart")
    
    def _create_bar_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        """Create bar chart"""
        x_col = params.get("x")
        y_col = params.get("y")
        
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            ax.bar(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            # Fallback: bar chart of first categorical and numeric columns
            cat_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=['number']).columns
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                grouped = df.groupby(cat_cols[0])[num_cols[0]].sum()
                ax.bar(grouped.index, grouped.values)
                ax.set_xlabel(cat_cols[0])
                ax.set_ylabel(num_cols[0])
        
        ax.set_title("Bar Chart")
    
    def _create_boxplot(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        """Create boxplot"""
        y_col = params.get("y")
        
        if y_col and y_col in df.columns:
            ax.boxplot(df[y_col].dropna())
            ax.set_ylabel(y_col)
        else:
            # Fallback: boxplot of all numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                ax.boxplot([df[col].dropna() for col in numeric_cols[:5]])
                ax.set_xticklabels(numeric_cols[:5], rotation=45)
        
        ax.set_title("Boxplot")
    
    def _create_scatter_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        """Create scatter plot"""
        x_col = params.get("x")
        y_col = params.get("y")
        
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            # Fallback: scatter of first two numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
        
        ax.set_title("Scatter Plot")
    
    def _create_heatmap(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        """Create heatmap"""
        # Use correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm', center=0)
        else:
            # Fallback: simple heatmap of data
            sns.heatmap(df.head(10).select_dtypes(include=['number']), ax=ax)
        
        ax.set_title("Heatmap")