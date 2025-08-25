"""DuckDB tool for SQL operations on datasets"""
import duckdb
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List
import hashlib

logger = logging.getLogger(__name__)

class DuckDBTool:
    def __init__(self, data_dir: str, artifacts_dir: str | None = None):
        self.data_dir = Path(data_dir)
        # when context is not provided to `run`, fall back to this artifacts_dir (or data_dir as last resort)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else self.data_dir

    def run(self, query: str, sources: Dict[str, str], context: Any = None) -> Dict[str, Any]:
        try:
            conn = duckdb.connect()
            try:
                for name, path in sources.items():
                    if path.endswith('.csv'):
                        conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
                    elif path.endswith('.parquet'):
                        conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
                    else:
                        raise ValueError(f"Unsupported source: {path}")

                cur = conn.execute(query)
                df = cur.fetchdf()  # <- robust: get a pandas DataFrame directly

                # output dir scoped to the run if available
                base_dir = Path(getattr(context, "run_dir", self.artifacts_dir))
                out_dir = base_dir / "frames"
                out_dir.mkdir(parents=True, exist_ok=True)

                qhash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
                pickle_path = out_dir / f"result_{qhash}.pkl"
                df.to_pickle(pickle_path)

                return {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "preview": df.head().to_dict('records') if not df.empty else [],
                    "df_pickle_path": str(pickle_path),
                    "computed_values": self._extract_computed_values(df),
                }
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")
            raise

    def inspect_schema(self, dataset_id: str, sources: Dict[str, str]) -> Dict[str, Any]:
        try:
            source_path = sources.get("main")
            if not source_path:
                raise ValueError("No main data source provided")

            conn = duckdb.connect()
            try:
                if source_path.endswith('.csv'):
                    conn.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM read_csv_auto('{source_path}')")
                elif source_path.endswith('.parquet'):
                    conn.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet('{source_path}')")
                else:
                    raise ValueError(f"Unsupported file format: {source_path}")

                schema_rows = conn.execute("DESCRIBE data").fetchall()
                schema = {row[0]: row[1] for row in schema_rows}

                row_count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
                sample_df = conn.execute("SELECT * FROM data LIMIT 5").fetchdf()

                return {
                    "rows": row_count,
                    "columns": schema,
                    "sample": sample_df.to_dict('records'),
                    "computed_values": {"total_rows": row_count, "column_count": len(schema)},
                }
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Schema inspection failed: {e}")
            raise

    def aggregate(self, group_by: List[str], metrics: List[str], context: Any) -> Dict[str, Any]:
        """Perform aggregation operations with robust aliasing and safe ordering."""
        try:
            if not metrics:
                raise ValueError("At least one metric is required for aggregation")

            # Build SELECT clause
            select_parts: List[str] = []
            group_clause = ", ".join(group_by) if group_by else ""

            if group_by:
                # (Optional) could validate columns exist via schema
                select_parts.extend(group_by)

            metric_aliases: List[str] = []
            for m in metrics:
                m_stripped = m.strip()
                # Handle patterns like "sum(col)", "avg(col)", "count(*)"
                if "(" in m_stripped and m_stripped.endswith(")"):
                    func, inner = m_stripped.split("(", 1)
                    func = func.strip()
                    inner = inner[:-1].strip()  # drop trailing ')'
                    if inner == "*" or inner == "1":
                        alias = f"{func.lower()}_all"
                        select_parts.append(f"{func}({inner}) AS {alias}")
                    else:
                        # remove potential quotes around column
                        col = inner.strip().strip('"').strip("'")
                        alias = f"{func.lower()}_{col}"
                        select_parts.append(f"{func}({col}) AS {alias}")
                    metric_aliases.append(alias)
                else:
                    # If metric already aliased or a raw expression, keep as-is
                    # Try to parse "expr AS alias" (basic)
                    lower = m_stripped.lower()
                    if " as " in lower:
                        select_parts.append(m_stripped)
                        alias = m_stripped.split(" as ", 1)[1].strip()
                        metric_aliases.append(alias)
                    else:
                        # Last resort: wrap as-is, but create a stable alias
                        alias = "metric_" + str(len(metric_aliases) + 1)
                        select_parts.append(f"{m_stripped} AS {alias}")
                        metric_aliases.append(alias)

            select_clause = ", ".join(select_parts)
            query = f"SELECT {select_clause} FROM data"

            if group_by:
                query += f" GROUP BY {group_clause}"
                # Only order when grouping; ordering a single-row aggregate is not needed
                query += f" ORDER BY {group_by[0]}"

            # Execute query
            dataset_path = self.data_dir / f"{context.dataset_id}.csv"
            if not dataset_path.exists():
                dataset_path = self.data_dir / f"{context.dataset_id}.parquet"

            sources = {"data": str(dataset_path)}
            res = self.run(query, sources, context=context)

            # Add context for evaluation harness & trace readers
            res["group_by"] = group_by or []
            res["metrics"] = metrics or []
            res["query"] = query
            return res

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

    def _extract_computed_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key computed values for grounding"""
        values = {}

        # Basic stats
        if len(df) > 0:
            values["result_count"] = len(df)

            # For numeric columns, add basic stats
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                values[f"{col}_mean"] = df[col].mean()
                values[f"{col}_sum"] = df[col].sum()
                values[f"{col}_min"] = df[col].min()
                values[f"{col}_max"] = df[col].max()

        return values