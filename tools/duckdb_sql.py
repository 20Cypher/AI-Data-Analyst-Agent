"""DuckDB tool for SQL operations on datasets"""
import duckdb
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import hashlib
import re

logger = logging.getLogger(__name__)

def _q(name: str) -> str:
    """Quote an identifier for DuckDB (safe for spaces/case/reserved words)."""
    # If already quoted, return as-is
    if name.startswith('"') and name.endswith('"'):
        return name
    # Escape internal quotes by doubling them
    escaped = name.replace('"', '""')
    return '"' + escaped + '"'

def _parse_metric(expr: str) -> Tuple[str, str]:
    """
    Parse a metric string and return (select_sql_fragment, alias).
    Handles:
      - sum(col), avg(col), min(col), max(col), count(*)
      - already-aliased: "<expr> AS alias"
      - raw expression fallback: "expr" -> "expr AS metric_n"
    """
    s = expr.strip()
    lower = s.lower()

    # Pre-aliased form
    if " as " in lower:
        # Preserve original case to avoid altering user intent
        left, alias = re.split(r"\s+as\s+", s, flags=re.IGNORECASE, maxsplit=1)
        alias = alias.strip()
        return f"{left} AS {_q(alias)}", alias

    # Function-like metric: func(inner)
    m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*?)\s*\)\s*$", s)
    if m:
        func = m.group(1)
        inner = m.group(2)
        func_l = func.lower()

        # Special cases: count(*), count(1)
        if inner in ("*", "1"):
            alias = f"{func_l}_all"
            return f"{func}({inner}) AS {_q(alias)}", alias

        # Strip quotes around column and quote it safely
        col = inner.strip().strip('"').strip("'")
        alias = f"{func_l}_{col}"
        return f"{func}({_q(col)}) AS {_q(alias)}", alias

    # Raw expression fallback: will alias later (caller provides unique name)
    return s, ""


class DuckDBTool:
    def __init__(self, data_dir: str, artifacts_dir: str | None = None):
        self.data_dir = Path(data_dir)
        # when context is not provided to `run`, fall back to this artifacts_dir (or data_dir as last resort)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else self.data_dir

    # ---- low-level helpers -------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        # Practical defaults; harmless if DuckDB ignores or changes later
        try:
            conn.execute("PRAGMA threads=4")
        except Exception:
            pass
        return conn

    def _register_sources(self, conn: duckdb.DuckDBPyConnection, sources: Dict[str, str]) -> None:
        for name, path in sources.items():
            if path.endswith(".csv"):
                conn.execute(f"CREATE OR REPLACE VIEW {_q(name)} AS SELECT * FROM read_csv_auto('{path}')")
            elif path.endswith(".parquet"):
                conn.execute(f"CREATE OR REPLACE VIEW {_q(name)} AS SELECT * FROM read_parquet('{path}')")
            else:
                raise ValueError(f"Unsupported source: {path}")

    def _dataset_path(self, dataset_id: str) -> Path:
        p = self.data_dir / f"{dataset_id}.csv"
        if not p.exists():
            p = self.data_dir / f"{dataset_id}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found in {self.data_dir}")
        return p

    def _frame_outdir(self, context: Any) -> Path:
        base_dir = Path(getattr(context, "run_dir", self.artifacts_dir))
        out_dir = base_dir / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ---- public API --------------------------------------------------------

    def run(self, query: str, sources: Dict[str, str], context: Any = None) -> Dict[str, Any]:
        """Execute SQL and persist result as a pickle (deterministic name)."""
        try:
            conn = self._connect()
            try:
                self._register_sources(conn, sources)
                df: pd.DataFrame = conn.execute(query).fetchdf()

                out_dir = self._frame_outdir(context)
                qhash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
                pickle_path = out_dir / f"result_{qhash}.pkl"
                df.to_pickle(pickle_path)

                return {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "preview": df.head().to_dict("records") if not df.empty else [],
                    "df_pickle_path": str(pickle_path),
                    "computed_values": self._extract_computed_values(df),
                    "query": query,
                }
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")
            raise

    def inspect_schema(self, dataset_id: str, sources: Dict[str, str]) -> Dict[str, Any]:
        """Inspect dataset schema and basic statistics."""
        try:
            conn = self._connect()
            try:
                source_path = sources.get("main")
                if not source_path:
                    raise ValueError("No main data source provided")

                self._register_sources(conn, {"data": source_path})
                schema_rows = conn.execute("DESCRIBE data").fetchall()
                schema = {row[0]: row[1] for row in schema_rows}
                row_count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
                sample_df = conn.execute("SELECT * FROM data LIMIT 5").fetchdf()

                return {
                    "rows": row_count,
                    "columns": schema,
                    "sample": sample_df.to_dict("records"),
                    "computed_values": {"total_rows": row_count, "column_count": len(schema)},
                }
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Schema inspection failed: {e}")
            raise

    def aggregate(self, group_by: List[str], metrics: List[str], context: Any) -> Dict[str, Any]:
        """Perform aggregation with robust aliasing, identifier quoting, and schema checks."""
        try:
            if not metrics:
                raise ValueError("At least one metric is required for aggregation")

            # Resolve dataset path & build the minimal source map
            dataset_path = self._dataset_path(context.dataset_id)
            sources = {"data": str(dataset_path)}

            # Discover schema columns so we can validate group_by & metrics
            conn = self._connect()
            try:
                self._register_sources(conn, sources)
                existing_cols = {
                    row[0] for row in conn.execute("DESCRIBE data").fetchall()
                }
            finally:
                conn.close()

            # Build SELECT list
            select_parts: List[str] = []
            quoted_group = []
            if group_by:
                for g in group_by:
                    g = g.strip()
                    if g not in existing_cols:
                        raise ValueError(f"Group-by column not found in data: {g}")
                    qg = _q(g)
                    select_parts.append(qg)
                    quoted_group.append(qg)

            metric_aliases: List[str] = []
            metric_sqls: List[str] = []
            metric_counter = 0

            for m in metrics:
                frag, alias = _parse_metric(m)
                if alias == "":
                    # raw expression: quote alias deterministically
                    metric_counter += 1
                    alias = f"metric_{metric_counter}"
                    frag = f"{frag} AS {_q(alias)}"

                # If expression referenced a column, sanity check it exists (best-effort quick parse)
                # We only check a simple pattern "func(col)" to avoid false positives on complex SQL.
                cm = re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s+AS\s+", frag)
                if cm:
                    col = cm.group(1)
                    if col != "*" and col not in existing_cols:
                        raise ValueError(f"Metric references missing column: {col}")

                metric_aliases.append(alias)
                metric_sqls.append(frag)

            select_parts.extend(metric_sqls)
            select_clause = ", ".join(select_parts)

            query = f"SELECT {select_clause} FROM data"
            if quoted_group:
                # GROUP BY with quoted identifiers
                query += " GROUP BY " + ", ".join(quoted_group)
                query += f" ORDER BY {quoted_group[0]}"

            # Execute
            res = self.run(query, sources, context=context)
            res["group_by"] = group_by or []
            res["metrics"] = metrics or []
            return res

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

    # ---- utilities ---------------------------------------------------------

    def _extract_computed_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract a few simple grounding stats."""
        values: Dict[str, Any] = {}
        if len(df) > 0:
            values["result_count"] = int(len(df))
            for col in df.select_dtypes(include=["number"]).columns:
                try:
                    values[f"{col}_mean"] = float(df[col].mean())
                    values[f"{col}_sum"] = float(df[col].sum())
                    values[f"{col}_min"] = float(df[col].min())
                    values[f"{col}_max"] = float(df[col].max())
                except Exception:
                    # Be resilient to weird numeric types
                    pass
        return values