"""Visualization tool for creating charts and plots (readability-optimized)"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import logging
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

# --------- helpers for nicer charts ---------

def _shorten_label(s: str, width: int = 18) -> str:
    """Wrap long labels to multiple lines for readability."""
    if not isinstance(s, str):
        s = str(s)
    # 2 lines max by default (to avoid tall x-axis)
    wrapped = textwrap.fill(s, width=width, max_lines=2, placeholder="…")
    return wrapped

def _ensure_readable_axes(ax, rotate: int = 0, x_fontsize: int = 10, y_fontsize: int = 9):
    """Apply common readability tweaks."""
    ax.grid(axis='y', linestyle='--', alpha=0.25)

    # For bar charts with few categories (<=6), don't rotate labels
    num_x_ticks = len(ax.get_xticklabels())
    if num_x_ticks <= 6:
        rotate = 0
    elif num_x_ticks <= 10:
        rotate = 30
    else:
        rotate = 45

    ax.tick_params(axis='x', labelrotation=rotate, labelsize=x_fontsize)
    ax.tick_params(axis='y', labelsize=y_fontsize)

    # Use thousands separator on y where appropriate
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, p: f"{int(x):,}" if float(x).is_integer() else f"{x:,.2f}"
    ))

    # Improve layout to prevent label cutoff
    plt.tight_layout()

def _maybe_coerce_datetime(s: pd.Series) -> pd.Series:
    """Try to parse to datetime if it's object-like and looks like dates."""
    if s is None:
        return s
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_object_dtype(s):
        # attempt parse with errors='coerce' and accept if at least half parse
        parsed = pd.to_datetime(s, errors='coerce')
        if parsed.notna().mean() >= 0.5:
            return parsed
    return s

def _first_numeric(df: pd.DataFrame) -> Optional[str]:
    cols = df.select_dtypes(include=['number']).columns.tolist()
    return cols[0] if cols else None

def _first_categorical(df: pd.DataFrame) -> Optional[str]:
    # 'object' or low-cardinality category-like column
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        # prefer the first with reasonable cardinality
        for c in obj_cols:
            if df[c].nunique(dropna=True) <= 50:
                return c
        return obj_cols[0]
    # if none, try any column with small cardinality
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 50:
            return c
    return None

def _limit_top_n(df: pd.DataFrame, x: str, y: str, top_n: int = 30) -> pd.DataFrame:
    """Limit to top N by y (descending) when top_n is specified."""
    if x not in df.columns or y not in df.columns:
        return df

    # Group by x and sum y values (in case there are duplicates)
    dff = df.groupby(x, as_index=False, observed=True)[y].sum()

    # Sort by y descending
    dff = dff.sort_values(by=y, ascending=False)

    # Always apply top_n limit when it's less than the default (indicates user wants limiting)
    if top_n < 30:  # User specified a limit
        dff = dff.head(top_n)
    elif len(dff) > top_n:  # Fallback for readability when many items
        dff = dff.head(top_n)

    return dff

def _auto_title(base: str, x: Optional[str], y: Optional[str], params: Dict[str, Any] = None) -> str:
    """Auto-generate chart title with column names and top_n awareness."""
    params = params or {}

    # Check for top_n parameter to customize title
    top_n = params.get("top_n", 30)
    if top_n and top_n < 30:  # Only add "Top N" if it's actually limiting results
        base = f"Top {top_n} {base}"

    parts = [base]
    if x: parts.append(f"by {x.replace('_', ' ').title()}")
    if y: parts.append(f"({y.replace('_', ' ').title()})")
    return " ".join(parts)

# ------------------------------------------------------------

class VizTool:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)

    def _extract_top_n_from_question(self, question: str) -> Optional[int]:
        """Extract top N from question as fallback if planner missed it."""
        if not question:
            return None

        import re
        patterns = [
            r'\btop\s+(\d+)\b',
            r'\bbest\s+(\d+)\b',
            r'\bhighest\s+(\d+)\b',
            r'\bmost\s+(\d+)\b',
            r'\blargest\s+(\d+)\b',
            r'\bfirst\s+(\d+)\b',
            r'\bbiggest\s+(\d+)\b',
            r'\btop-(\d+)\b',
        ]

        question_lower = question.lower()
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                n = int(match.group(1))
                if 1 <= n <= 100:  # reasonable bounds
                    return n
        return None
        
    def create_chart(self, chart_type: str, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Create a chart and save it as PNG (with smart legibility defaults)."""
        try:
            charts_dir = Path(context.run_dir) / "charts"
            charts_dir.mkdir(exist_ok=True)

            df = self._load_data_for_viz(context)
            if df is None or df.empty:
                raise ValueError("No data available for visualization")

            # FALLBACK: Extract top_n from question if not provided by planner
            if "top_n" not in params or params.get("top_n", 30) >= 30:
                question = getattr(context, 'question', '')
                extracted_n = self._extract_top_n_from_question(question)
                if extracted_n:
                    params["top_n"] = extracted_n
                    logger.info(f"Fallback: Extracted top_n={extracted_n} from question: {question}")

            # Create chart
            figsize = params.get("figsize", (12, 6))  # a little wider by default
            fig, ax = plt.subplots(figsize=figsize)

            chart_type = (chart_type or "").lower().strip()
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
            if not str(outfile).startswith("charts/"):
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
            plt.close('all')
            raise
    
    def _load_data_for_viz(self, context: Any) -> pd.DataFrame:
        """Load data for visualization from context"""
        # Try the most recent DataFrame produced in this run
        if hasattr(context, 'dataframes') and context.dataframes:
            df_name, pickle_path = list(context.dataframes.items())[-1]
            try:
                with open(pickle_path, 'rb') as f:
                    df = pickle.load(f)
                return df
            except Exception as e:
                logger.warning(f"Failed to load DataFrame from {pickle_path}: {e}")

        # Fallback: original dataset
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

    # ---------------- specific chart creators ----------------

    def _create_line_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        x_col = params.get("x")
        y_col = params.get("y")

        if x_col and x_col in df.columns:
            x = _maybe_coerce_datetime(df[x_col])
        else:
            x = None

        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            dff = df[[x_col, y_col]].copy()

            # Special handling for period strings (like "2023-01")
            if dff[x_col].dtype == 'object' and pd.api.types.is_string_dtype(dff[x_col]):
                # Check if it looks like period format (YYYY-MM, YYYY-Q1, etc.)
                sample = str(dff[x_col].iloc[0]) if len(dff) > 0 else ""
                if len(sample.split('-')) >= 2:
                    try:
                        # Try to convert to periods first, then to datetime for plotting
                        periods = pd.to_datetime(dff[x_col], errors='coerce')
                        if periods.notna().mean() > 0.8:  # Most values parsed successfully
                            dff[x_col] = periods
                            x = periods
                    except:
                        pass

            if x is not None:
                dff[x_col] = x

            # Sort by x column for proper line ordering
            dff = dff.sort_values(by=x_col)

            # Group by x and sum y in case of duplicates
            if dff.groupby(x_col).size().max() > 1:
                dff = dff.groupby(x_col, as_index=False)[y_col].sum()

            ax.plot(dff[x_col], dff[y_col], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_title(_auto_title("Line Chart", x_col, y_col))

            if pd.api.types.is_datetime64_any_dtype(dff[x_col]):
                locator = AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
                ax.tick_params(axis='x', rotation=45, labelsize=9)
            else:
                # For non-datetime x (like period strings), rotate labels if many
                if len(dff) > 6:
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
        else:
            # Fallback: first numeric vs index
            y_first = _first_numeric(df)
            if y_first:
                ax.plot(df[y_first].values, marker='o', linewidth=2, markersize=6)
                ax.set_ylabel(y_first)
                ax.set_title(_auto_title("Line Chart", "index", y_first))
                ax.set_xlabel("index")
            else:
                # nothing sensible to plot
                ax.text(0.5, 0.5, "No numeric columns to plot", ha="center", va="center")
        _ensure_readable_axes(ax)

    def _create_bar_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        x_col = params.get("x")
        y_col = params.get("y")

        # If user specified x/y and they're valid, use them
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            dff = df[[x_col, y_col]].copy()
            # If multiple rows per x, aggregate by sum for clarity
            if dff.groupby(x_col).size().max() > 1:
                dff = dff.groupby(x_col, as_index=False, observed=True)[y_col].sum()
            # limit to top N for readability
            dff = _limit_top_n(dff, x_col, y_col, top_n=params.get("top_n", 30))
            dff = dff.sort_values(by=y_col, ascending=False)
            # shorter labels
            labels = dff[x_col].astype(str).map(lambda s: _shorten_label(s, width=18))
            # Create colors that work for any number of bars
            n_bars = len(labels)
            if n_bars <= 10:
                # Use distinct colors for small sets
                colors = plt.cm.tab10(range(n_bars))
            else:
                # Use a continuous colormap for larger sets
                colors = plt.cm.viridis(np.linspace(0, 1, n_bars))

            bars = ax.bar(labels, dff[y_col].values, color=colors)

            # Add value labels on top of bars for clarity
            for bar, value in zip(bars, dff[y_col].values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:,.0f}', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel(x_col.title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_title(_auto_title("Products", x_col, y_col, params))
        else:
            # Fallback: group first categorical by sum of first numeric
            cat = _first_categorical(df)
            num = _first_numeric(df)
            if cat and num:
                dff = df.groupby(cat, as_index=False, observed=True)[num].sum()
                dff = dff.sort_values(by=num, ascending=False)
                dff = dff.head(params.get("top_n", 30))
                labels = dff[cat].astype(str).map(lambda s: _shorten_label(s, width=18))
                # Create colors that work for any number of bars
                n_bars = len(labels)
                if n_bars <= 10:
                    # Use distinct colors for small sets
                    colors = plt.cm.tab10(range(n_bars))
                else:
                    # Use a continuous colormap for larger sets
                    colors = plt.cm.viridis(np.linspace(0, 1, n_bars))

                bars = ax.bar(labels, dff[num].values, color=colors)

                # Add value labels on top of bars for clarity
                for bar, value in zip(bars, dff[num].values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:,.0f}', ha='center', va='bottom', fontsize=9)

                ax.set_xlabel(cat.replace('_', ' ').title())
                ax.set_ylabel(num.replace('_', ' ').title())
                ax.set_title(_auto_title("Items", cat, num, params))
            else:
                ax.text(0.5, 0.5, "Need a categorical and a numeric column for a bar chart",
                        ha="center", va="center")
        _ensure_readable_axes(ax)

    def _create_boxplot(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        y_col = params.get("y")
        if y_col and y_col in df.columns:
            ax.boxplot(df[y_col].dropna(), vert=True)
            ax.set_ylabel(y_col)
            ax.set_title(_auto_title("Boxplot", None, y_col))
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                cols = numeric_cols[: min(8, len(numeric_cols))]
                ax.boxplot([df[c].dropna() for c in cols], vert=True)
                ax.set_xticklabels([_shorten_label(c, width=12) for c in cols], rotation=45, ha='right')
                ax.set_title("Boxplot (first numeric columns)")
            else:
                ax.text(0.5, 0.5, "No numeric columns for boxplot", ha="center", va="center")
        _ensure_readable_axes(ax, rotate=45)

    def _create_scatter_chart(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        x_col = params.get("x")
        y_col = params.get("y")
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            xx = _maybe_coerce_datetime(df[x_col])
            if pd.api.types.is_datetime64_any_dtype(xx):
                # If dates on x, plot against ordinal
                ax.scatter(xx, df[y_col], alpha=0.6, s=25)
                locator = AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
                ax.tick_params(axis='x', rotation=0, labelsize=8)
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=25)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(_auto_title("Scatter Plot", x_col, y_col))
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, s=25)
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_title(_auto_title("Scatter Plot", numeric_cols[0], numeric_cols[1]))
            else:
                ax.text(0.5, 0.5, "Need at least two numeric columns for scatter", ha="center", va="center")
        _ensure_readable_axes(ax, rotate=0)

    def _create_heatmap(self, df: pd.DataFrame, params: Dict[str, Any], ax):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', center=0, fmt=".2f",
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Heatmap (Correlation)")
        else:
            sns.heatmap(df.head(10).select_dtypes(include=['number']), ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title("Heatmap (Sample numeric values)")
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=9)