import pandas as pd
from typing import Dict, Any, List

# ----------------- helpers -----------------

def _is_single_numeric_cell(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if df.shape == (1, 1):
        return pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    return False

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _looks_like_percent(colname: str) -> bool:
    name = (colname or "").lower()
    return ("percent" in name) or ("percentage" in name) or name.endswith("_pct") or name.endswith("_share") or name.endswith("_ratio")

def _looks_like_revenue_share(question: str, colname: str) -> bool:
    """Detect if this looks like a revenue share analysis based on question and column name"""
    question_lower = (question or "").lower()
    col_lower = (colname or "").lower()

    revenue_keywords = ["revenue", "sales", "income", "earnings"]
    share_keywords = ["percentage", "percent", "share", "breakdown", "proportion"]

    question_has_revenue = any(kw in question_lower for kw in revenue_keywords)
    question_has_share = any(kw in question_lower for kw in share_keywords)
    col_has_revenue = any(kw in col_lower for kw in revenue_keywords)

    return question_has_revenue and question_has_share and col_has_revenue

def _fmt_val(val: Any, colname: str | None = None, question: str | None = None) -> str:
    """Human-friendly formatting with percentage awareness."""
    try:
        v = float(val)
    except Exception:
        return str(val)

    # Enhanced percentage detection
    if _looks_like_percent(colname or ""):
        if 0.0 <= v <= 1.0:
            return f"{v * 100:.1f}%"
        elif v > 1.0 and v <= 100.0:  # Already in percentage form
            return f"{v:.1f}%"

    # Special case for revenue percentage questions
    if question and _looks_like_revenue_share(question, colname or ""):
        if 0.0 <= v <= 1.0:
            return f"{v * 100:.1f}%"
        elif v > 1.0 and v <= 100.0:
            return f"{v:.1f}%"

    # Use comma separators for large numbers
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:.2f}"

def _pick_metric_columns(df: pd.DataFrame, group_by: List[str]) -> List[str]:
    """Prefer metric-ish columns; otherwise any numeric not in group_by."""
    num = [c for c in _numeric_cols(df) if c not in group_by]
    if not num:
        return []
    # lightweight priority: prefer common aggregate prefixes
    def score(c: str) -> int:
        lc = c.lower()
        if lc.startswith(("sum_", "avg_", "mean_", "count_", "min_", "max_", "median_", "std_", "var_", "metric_")):
            return 0
        return 1
    return sorted(num, key=score)

# ----------------- main API -----------------

def summarize_result(df: pd.DataFrame, meta: Dict[str, Any] | None = None, question: str | None = None) -> str:
    """
    Produce a safe, non-misleading textual summary of a result frame.

    Rules of thumb:
    - Never report min/max/std on a single scalar.
    - If aggregated with groups, summarize top/bottom groups per metric (up to 2 metrics).
    - If a single-row result with multiple metrics, list the key-value pairs.
    - Otherwise, fall back to row/column counts and compact numeric hints.
    """
    meta = meta or {}
    is_agg = bool(meta.get("is_aggregate", False))
    group_by = list(meta.get("group_by") or [])
    # metrics from meta is optional; we rely more on actual columns present
    # but keep it for future use
    _ = meta.get("metrics") or []

    if df is None or df.empty:
        return "No rows returned."

    # Case A: single numeric cell (typical global aggregate)
    if _is_single_numeric_cell(df):
        col = df.columns[0]
        val = _fmt_val(df.iloc[0, 0], col, question)
        if is_agg:
            return f"{col} is {val} (aggregated result)."
        return f"{col} is {val}."

    # Case B: aggregated with groups -> report top/bottom for up to 2 metric columns
    if is_agg and group_by:
        metric_cols = _pick_metric_columns(df, group_by)
        if not metric_cols:
            # No numeric metrics; just report group count
            return f"Grouped by {', '.join(group_by)}. Returned {len(df)} groups."

        # Special handling for percentage/share questions with categories
        if (question and any(kw in question.lower() for kw in ["percentage", "percent", "share"]) and
            len(metric_cols) >= 1 and len(df) <= 10):  # Small number of categories

            primary_metric = metric_cols[0]

            # Check if we have percentage columns already
            percentage_cols = [col for col in df.columns if "percentage" in col.lower() or "percent" in col.lower()]

            if percentage_cols:
                # Use existing percentage column
                percentage_col = percentage_cols[0]
                sdf = df.sort_values(by=percentage_col, ascending=False)

                def g_label(row):
                    return ", ".join(f"{row[g]}" for g in group_by if g in df.columns)

                category_results = []
                for _, row in sdf.iterrows():
                    category = g_label(row)
                    value = _fmt_val(row[percentage_col], percentage_col, question)
                    category_results.append(f"{category}: {value}")

                return f"Revenue breakdown by category: " + ", ".join(category_results) + f". ({len(df)} categories total)"

            # Check if we can compute percentages from revenue columns
            elif "total_revenue" in df.columns and len(group_by) > 0:
                # Compute percentages on-the-fly
                total = df["total_revenue"].sum()
                sdf = df.sort_values(by="total_revenue", ascending=False)

                def g_label(row):
                    return ", ".join(f"{row[g]}" for g in group_by if g in df.columns)

                category_results = []
                for _, row in sdf.iterrows():
                    category = g_label(row)
                    percentage = (row["total_revenue"] / total) * 100
                    value = f"{percentage:.1f}%"
                    category_results.append(f"{category}: {value}")

                return f"Revenue breakdown by category: " + ", ".join(category_results) + f". ({len(df)} categories total)"

            else:
                # Fallback to showing raw values
                sdf = df.sort_values(by=primary_metric, ascending=False)

                def g_label(row):
                    return ", ".join(f"{row[g]}" for g in group_by if g in df.columns)

                category_results = []
                for _, row in sdf.iterrows():
                    category = g_label(row)
                    value = _fmt_val(row[primary_metric], primary_metric, question)
                    category_results.append(f"{category}: {value}")

                return f"Revenue breakdown by category: " + ", ".join(category_results) + f". ({len(df)} categories total)"

        # Default groupby reporting for non-percentage questions
        snippets: List[str] = []
        for metric in metric_cols[:2]:  # cap to two metrics to avoid verbosity
            sdf = df.sort_values(by=metric, ascending=False)
            top = sdf.iloc[0]
            bottom = sdf.iloc[-1]

            def g_label(row):
                return ", ".join(f"{g}={row[g]}" for g in group_by if g in df.columns)

            snippets.append(
                f"Using {metric}: top={g_label(top)} ({metric}={_fmt_val(top[metric], metric, question)}), "
                f"bottom={g_label(bottom)} ({metric}={_fmt_val(bottom[metric], metric, question)})."
            )
        return f"Grouped by {', '.join(group_by)}. " + " ".join(snippets) + f" Returned {len(df)} groups."

    # Case C: single-row multi-column result (often post-aggregation without group_by)
    if len(df) == 1:
        rec = df.to_dict("records")[0]
        # Show up to 6 key/value pairs compactly
        pairs = []
        for k, v in rec.items():
            pairs.append(f"{k}={_fmt_val(v, k, question)}")
            if len(pairs) >= 6:
                break
        return " | ".join(pairs)

    # Case D: general multi-row result, show compact numeric hints (non-descriptive)
    num_cols = _numeric_cols(df)
    if num_cols:
        pieces = []
        for c in num_cols[:3]:  # cap to avoid verbosity
            s = df[c].describe()
            mean_txt = _fmt_val(s.get("mean", float("nan")), c, question) if "mean" in s else "-"
            min_txt = _fmt_val(s.get("min", float("nan")), c, question) if "min" in s else "-"
            max_txt = _fmt_val(s.get("max", float("nan")), c, question) if "max" in s else "-"
            pieces.append(f"{c}: mean={mean_txt}, min={min_txt}, max={max_txt}")
        return f"Returned {len(df)} rows. " + " | ".join(pieces)

    # Fallback
    return f"Returned {len(df)} rows and {len(df.columns)} columns."