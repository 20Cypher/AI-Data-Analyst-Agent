"""Python REPL tool for data processing and analysis"""
import pandas as pd
import numpy as np
import logging
import pickle
import ast
import sys
from io import StringIO
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import os
import signal
import re  # <-- NEW

logger = logging.getLogger(__name__)


class PythonREPLTool:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        # Modules we may allow to be imported explicitly (still discouraged; env already provides pd/np)
        self.allowed_modules = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'datetime', 'math'}
        # Soft timeout for user code (seconds). Uses SIGALRM on Unix; on Windows, timeout is ignored.
        self.timeout_seconds = int(os.getenv("REPL_TIMEOUT_SECONDS", "10"))
    
    @staticmethod
    def _raise_timeout(signum, frame):
        raise TimeoutError("Execution timed out")

    # ---------- NEW: helpers for robust date handling ----------
    def _is_dateish(self, name: str) -> bool:
        """Heuristic: does the column name look like a date/time?"""
        n = name.lower()
        return any(k in n for k in ["date", "time", "timestamp", "created", "updated"])

    def _auto_parse_and_enrich_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date-like columns and add useful date parts.
        - For each detected date column 'col': col_year, col_month, col_day, col_week, col_quarter, col_ym
        - Also create generic 'year','month','day','ym' from the first detected date column
        """
        df = df.copy()
        date_cols: List[str] = []

        for col in df.columns:
            s = df[col]
            looks_like_date = self._is_dateish(col) or pd.api.types.is_datetime64_any_dtype(s)

            if looks_like_date or s.dtype == "object":
                try:
                    # Try common date formats first to avoid warnings
                    if s.dtype == "object" and len(s) > 0:
                        sample = str(s.iloc[0]) if not pd.isna(s.iloc[0]) else ""
                        if "-" in sample and len(sample) == 10:  # YYYY-MM-DD format
                            parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
                        else:
                            # Suppress dateutil warnings for unclear formats
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                parsed = pd.to_datetime(s, errors="coerce")
                    else:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            parsed = pd.to_datetime(s, errors="coerce")
                except Exception:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(s, errors="coerce")
                # treat as date if name suggests it OR parsing succeeds for enough rows
                if self._is_dateish(col) or parsed.notna().mean() > 0.6:
                    df[col] = parsed
                    date_cols.append(col)

        for col in date_cols:
            df[f"{col}_year"]    = df[col].dt.year
            df[f"{col}_month"]   = df[col].dt.month
            df[f"{col}_day"]     = df[col].dt.day
            # isocalendar().week returns UInt; cast to nullable int to avoid dtype issues
            df[f"{col}_week"]    = df[col].dt.isocalendar().week.astype("Int64")
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_ym"]      = df[col].dt.to_period("M").astype(str)

        # Provide generic names for the first detected date column
        if date_cols:
            primary = date_cols[0]
            if "year" not in df.columns:
                df["year"] = df[f"{primary}_year"]
            if "month" not in df.columns:
                df["month"] = df[f"{primary}_month"]
            if "day" not in df.columns:
                df["day"] = df[f"{primary}_day"]
            if "ym" not in df.columns:
                df["ym"] = df[f"{primary}_ym"]

        return df
    # -----------------------------------------------------------

    def exec(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code with restricted environment + timeout + output truncation."""
        try:
            # Validate code safety via AST
            self._validate_code(code)

            # Set up execution environment with strict builtins
            env = self._create_environment(context)

            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # Soft timeout (Unix only)
            use_alarm = hasattr(signal, "SIGALRM") and os.name != "nt"
            if use_alarm:
                signal.signal(signal.SIGALRM, self._raise_timeout)
                signal.alarm(self.timeout_seconds)

            try:
                exec(code, env)
                # Capture any new variables (safe types only)
                result_vars = {}
                for name, value in env.items():
                    if not name.startswith('__') and name not in ['pd', 'np', 'plt', 'sns']:
                        if isinstance(value, pd.DataFrame):
                            # Save DataFrame
                            pickle_path = self.artifacts_dir / f"df_{name}.pkl"
                            value.to_pickle(pickle_path)
                            result_vars[name] = str(pickle_path)
                        elif isinstance(value, (int, float, str, list, dict, bool, type(None))):
                            result_vars[name] = value

                output = captured_output.getvalue()
                # Truncate very large outputs
                MAX_CHARS = 10_000
                if len(output) > MAX_CHARS:
                    output = output[:MAX_CHARS] + "\n... [truncated]\n"

                return {
                    "output": output,
                    "variables": result_vars,
                    "success": True
                }

            finally:
                # Always restore stdout and cancel alarm
                sys.stdout = old_stdout
                if use_alarm:
                    signal.alarm(0)

        except TimeoutError:
            return {
                "output": "",
                "variables": {},
                "success": False,
                "error": f"Execution timed out after {self.timeout_seconds}s"
            }
        except Exception as e:
            logger.error(f"Python execution failed: {e}")
            return {
                "output": "",
                "variables": {},
                "success": False,
                "error": str(e)
            }

    def prepare_data(self, operations: List[str], context: Any) -> Dict[str, Any]:
        """Prepare data based on specified operations"""
        try:
            # Load main dataset
            dataset_path = self._get_dataset_path(context)
            df = self._load_dataset(dataset_path)

            # NEW: Auto-parse dates and add date parts up-front
            df = self._auto_parse_and_enrich_dates(df)

            # Apply operations (existing DSL still supported)
            for op in operations:
                df = self._apply_operation(df, op)

            # Save prepared data
            pickle_path = self.artifacts_dir / context.run_id / f"prepared_data.pkl"
            pickle_path.parent.mkdir(exist_ok=True)
            df.to_pickle(pickle_path)

            return {
                "df_pickle_path": str(pickle_path),
                "df_name": "prepared_data",
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head().to_dict('records'),
                "computed_values": {
                    "prepared_rows": len(df),
                    "prepared_columns": list(df.columns)
                }
            }

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def interpret(self, prompts: List[str], context: Any) -> Dict[str, Any]:
        """Generate interpretation based on computed results"""
        # This is a placeholder for interpretation logic
        # In a real implementation, this might call an LLM or generate insights
        
        interpretation_parts = []
        
        for prompt in prompts:
            if "summarize" in prompt.lower():
                interpretation_parts.append("Data analysis completed with computed results.")
            elif "trend" in prompt.lower():
                interpretation_parts.append("Trends have been analyzed in the dataset.")
            elif "pattern" in prompt.lower():
                interpretation_parts.append("Patterns in the data have been identified.")
            else:
                interpretation_parts.append(f"Analysis completed for: {prompt}")
        
        interpretation = " ".join(interpretation_parts)

        return {
            "interpretation": interpretation,
            "computed_values": {
                "interpretation_generated": True,
                "prompt_count": len(prompts)
            }
        }

    def _get_dataset_path(self, context: Any) -> Path:
        """Get the path to the dataset"""
        data_dir = Path(context.run_dir).parent.parent / "data"
        
        # Try CSV first
        csv_path = data_dir / f"{context.dataset_id}.csv"
        if csv_path.exists():
            return csv_path
            
        # Try Parquet
        parquet_path = data_dir / f"{context.dataset_id}.parquet"
        if parquet_path.exists():
            return parquet_path
            
        raise FileNotFoundError(f"Dataset {context.dataset_id} not found")

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Load dataset from file"""
        if dataset_path.suffix == '.csv':
            return pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.parquet':
            return pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    def _apply_operation(self, df: pd.DataFrame, operation: str) -> pd.DataFrame:
        """Apply a single operation to the DataFrame"""
        if operation.startswith("filter:"):
            condition = operation.split(":", 1)[1].strip()
            return self._apply_filter(df, condition)
            
        elif operation.startswith("parse_dates:"):
            column = operation.split(":", 1)[1].strip()
            if column in df.columns:
                df = df.copy()
                df[column] = pd.to_datetime(df[column], errors='coerce')
                # also add date parts for that column
                df[f"{column}_year"]    = df[column].dt.year
                df[f"{column}_month"]   = df[column].dt.month
                df[f"{column}_day"]     = df[column].dt.day
                df[f"{column}_week"]    = df[column].dt.isocalendar().week.astype("Int64")
                df[f"{column}_quarter"] = df[column].dt.quarter
                df[f"{column}_ym"]      = df[column].dt.to_period("M").astype(str)
                
        elif operation.startswith("drop_na:"):
            column = operation.split(":", 1)[1].strip()
            if column in df.columns:
                df = df.dropna(subset=[column])
            else:
                df = df.dropna()
                
        elif operation.startswith("remove_duplicates"):
            df = df.drop_duplicates()
            
        elif operation.startswith("sort:"):
            column = operation.split(":", 1)[1].strip()
            if column in df.columns:
                df = df.sort_values(column)
                
        elif operation.startswith("rename:"):
            # Format: rename:old_name->new_name
            rename_spec = operation.split(":", 1)[1].strip()
            if "->" in rename_spec:
                old_name, new_name = rename_spec.split("->", 1)
                old_name = old_name.strip()
                new_name = new_name.strip()
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})

        elif operation.startswith("calculate:"):
            # Format: calculate: column_name = expression
            calc_spec = operation.split(":", 1)[1].strip()
            if "=" in calc_spec:
                try:
                    # This is a simple implementation - we could expand this to handle more complex calculations
                    # For now, handle percentage calculations specifically
                    if "percentage" in calc_spec and "*" in calc_spec and "100" in calc_spec:
                        # Handle percentage = (total_revenue / overall_revenue) * 100
                        left, right = calc_spec.split("=", 1)
                        col_name = left.strip()
                        expr = right.strip()

                        # Parse simple percentage expressions
                        if "(" in expr and ")" in expr and "/" in expr and "*" in expr:
                            # Extract variables from expression like (total_revenue / overall_revenue) * 100
                            match = re.search(r'\(\s*(\w+)\s*/\s*(\w+)\s*\)\s*\*\s*(\d+)', expr)
                            if match:
                                numerator = match.group(1)
                                denominator = match.group(2)
                                multiplier = float(match.group(3))

                                if numerator in df.columns and denominator in df.columns:
                                    df[col_name] = (df[numerator] / df[denominator]) * multiplier
                        logger.info(f"Applied calculation: {calc_spec}")
                    else:
                        logger.warning(f"Complex calculation not supported: {calc_spec}")
                except Exception as e:
                    logger.error(f"Calculation failed: {calc_spec}, error: {e}")

        elif operation == "extract_month" or operation.startswith("extract_month"):
            # Extract month (YYYY-MM format) from date columns
            date_cols = []
            for col in df.columns:
                if self._is_dateish(col) or pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)

            if date_cols:
                date_col = date_cols[0]  # Use first date column
                df['month'] = df[date_col].dt.to_period('M').astype(str)
                logger.info(f"Extracted month from {date_col}")
            else:
                logger.warning("No date column found for month extraction")

        elif operation == "extract_year" or operation.startswith("extract_year"):
            # Extract year from date columns
            date_cols = []
            for col in df.columns:
                if self._is_dateish(col) or pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)

            if date_cols:
                date_col = date_cols[0]  # Use first date column
                df['year'] = df[date_col].dt.year.astype(str)
                logger.info(f"Extracted year from {date_col}")
            else:
                logger.warning("No date column found for year extraction")

        elif operation == "extract_quarter" or operation.startswith("extract_quarter"):
            # Extract quarter from date columns
            date_cols = []
            for col in df.columns:
                if self._is_dateish(col) or pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)

            if date_cols:
                date_col = date_cols[0]  # Use first date column
                df['quarter'] = df[date_col].dt.to_period('Q').astype(str)
                logger.info(f"Extracted quarter from {date_col}")
            else:
                logger.warning("No date column found for quarter extraction")

        else:
            logger.warning(f"Unknown operation: {operation}")

        return df

    def _validate_code(self, code: str) -> None:
        """AST-based validation: block dangerous imports/objects and risky calls."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        banned_names = {
            "os", "sys", "subprocess", "shutil", "socket", "pathlib",
            "builtins", "__import__", "eval", "exec", "compile", "open",
            "input", "help", "breakpoint"
        }
        banned_attrs = {
            # process/file/system-ish
            "system", "popen", "walk", "remove", "rmtree", "unlink", "rename",
            "replace", "spawn", "kill", "fork", "execv", "execve", "chdir",
            "chmod", "chown", "symlink", "mkfifo", "mknod", "rmdir",
            # networking-ish
            "connect", "bind", "listen", "accept"
        }

        for node in ast.walk(tree):
            # Block imports except allowlist
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    root = (alias.name or "").split(".", 1)[0]
                    if root not in self.allowed_modules:
                        raise ValueError(f"Import blocked: {alias.name}")

            # Block access to dangerous names
            if isinstance(node, ast.Name):
                if node.id in banned_names:
                    raise ValueError(f"Use of '{node.id}' is not allowed")

            # Block attribute access on dangerous modules or dangerous attributes
            if isinstance(node, ast.Attribute):
                # e.g., os.system, sys.exit, subprocess.Popen
                if isinstance(node.value, ast.Name) and node.value.id in banned_names:
                    raise ValueError(f"Access to '{node.value.id}.{node.attr}' is not allowed")
                if node.attr in banned_attrs:
                    raise ValueError(f"Use of attribute '{node.attr}' is not allowed")

            # Block direct calls to banned functions
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_names:
                    raise ValueError(f"Call to '{func.id}' is not allowed")
                if isinstance(func, ast.Attribute):
                    # Something like module.attr(...)
                    if isinstance(func.value, ast.Name) and func.value.id in banned_names:
                        raise ValueError(f"Call to '{func.value.id}.{func.attr}' is not allowed")
                    if func.attr in banned_attrs:
                        raise ValueError(f"Call to attribute '{func.attr}' is not allowed")


    def _create_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create safe execution environment (minimal builtins)."""
        SAFE_BUILTINS = {
            'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'min': min, 'max': max, 'sum': sum, 'abs': abs,
            'round': round, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'print': print, 'sorted': sorted,
            'any': any, 'all': all, 'bool': bool
        }

        env = {
            '__builtins__': SAFE_BUILTINS,
            'pd': pd,
            'np': np
        }

        # Load any existing DataFrames from context
        for name, pickle_path in context.get('dataframes', {}).items():
            try:
                with open(pickle_path, 'rb') as f:
                    env[name] = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load DataFrame {name}: {e}")

        return env


    def _apply_filter(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Apply a simple SQL‑like filter safely.
        Supports:
        - col == 'x', !=, >, >=, <, <=
        - col CONTAINS 'x' (case‑insensitive)
        - col ISNULL / NOTNULL
        - col BETWEEN a AND b    (a/b can be numbers or dates in 'YYYY‑MM‑DD')
        - Chaining with AND / OR  (left‑to‑right, AND > OR precedence)
        """
        try:
            # Normalize whitespace but keep case for column names
            c = " ".join(condition.strip().split())

            # 1) Split on OR (lowest precedence)
            parts_or = self._split_outside_quotes(c, " OR ")
            if len(parts_or) > 1:
                mask = pd.Series(False, index=df.index)
                for part in parts_or:
                    mask = mask | self._eval_and_chain(df, part)
                return df[mask]

            # 2) Otherwise it's a pure AND chain (or single predicate)
            return df[self._eval_and_chain(df, c)]

        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return df

    def _eval_and_chain(self, df: pd.DataFrame, clause: str) -> pd.Series:
        """Evaluate a chain of AND‑ed predicates."""
        parts_and = self._split_outside_quotes(clause, " AND ")
        mask = pd.Series(True, index=df.index)
        for p in parts_and:
            mask = mask & self._eval_simple_predicate(df, p.strip())
        return mask

    def _split_outside_quotes(self, text: str, sep: str) -> List[str]:
        """Split by sep ignoring separators inside single/double quotes."""
        out, buf, q = [], [], None
        i, n, m = 0, len(text), len(sep)
        while i < n:
            ch = text[i]
            if q:
                buf.append(ch)
                if ch == q:
                    q = None
                i += 1
            else:
                if ch in ("'", '"'):
                    q = ch
                    buf.append(ch)
                    i += 1
                elif text[i:i+m] == sep:
                    out.append("".join(buf).strip())
                    buf = []
                    i += m
                else:
                    buf.append(ch)
                    i += 1
        out.append("".join(buf).strip())
        return out

    def _parse_literal(self, s: str):
        """Parse numeric or date literal; fallback to raw string."""
        t = s.strip().strip("'\"")
        # Try number
        try:
            return float(t) if "." in t else int(t)
        except Exception:
            pass
        # Try date
        try:
            return pd.to_datetime(t)
        except Exception:
            pass
        return t

    def _eval_simple_predicate(self, df: pd.DataFrame, pred: str) -> pd.Series:
        """Evaluate one predicate like "col >= 5", "date BETWEEN '2023-01-01' AND '2023-12-31'", "col IN ('a', 'b')"."""
        p = pred.strip()

        # IN operator (e.g., "product_category IN ('Electronics', 'Books', 'Home')" or "product_category in ['Electronics', 'Books', 'Home']")
        if " IN " in p.upper():
            # Handle both parentheses and square brackets
            match = re.search(r'(.+?)\s+IN\s*[\(\[](.+?)[\)\]]', p, re.IGNORECASE)
            if match:
                col = match.group(1).strip()
                values_str = match.group(2).strip()
                if col not in df.columns:
                    return pd.Series(True, index=df.index)

                # Parse the values inside parentheses/brackets
                values = []
                for val in values_str.split(','):
                    val = val.strip().strip("'\"")
                    values.append(val)

                return df[col].isin(values)

        # BETWEEN
        if " BETWEEN " in p.upper():
            # col BETWEEN a AND b
            left, rest = self._split_outside_quotes(p, " BETWEEN ")
            if " AND " not in rest.upper():
                return pd.Series(True, index=df.index)
            a, b = self._split_outside_quotes(rest, " AND ")
            col = left.strip()
            if col not in df.columns:
                return pd.Series(True, index=df.index)
            lo = self._parse_literal(a)
            hi = self._parse_literal(b)
            series = df[col]
            # If both parse as datetimes, coerce series as datetime too
            if isinstance(lo, pd.Timestamp) or isinstance(hi, pd.Timestamp):
                series = pd.to_datetime(series, errors="coerce")
            return (series >= lo) & (series <= hi)

        # CONTAINS (case‑insensitive)
        low = p.lower()
        idx = low.find(" contains ")
        if idx != -1:
            col = p[:idx].strip()
            val = p[idx+len(" contains "):].strip().strip("'\"")
            if col not in df.columns:
                return pd.Series(True, index=df.index)
            return df[col].astype(str).str.contains(val, case=False, na=False)

        # ISNULL / NOTNULL
        if low.endswith(" isnull"):
            col = p[:-len(" isnull")].strip()
            return df[col].isnull() if col in df.columns else pd.Series(True, index=df.index)
        if low.endswith(" notnull"):
            col = p[:-len(" notnull")].strip()
            return df[col].notnull() if col in df.columns else pd.Series(True, index=df.index)

        # Binary comparisons (order of ops matters)
        for op in ["<=", ">=", "==", "!=", "<", ">"]:
            if op in p:
                col, lit = p.split(op, 1)
                col = col.strip()
                if col not in df.columns:
                    return pd.Series(True, index=df.index)
                series = df[col]
                rhs = self._parse_literal(lit)
                # If rhs is a date, coerce series to datetime
                if isinstance(rhs, pd.Timestamp):
                    series = pd.to_datetime(series, errors="coerce")
                try:
                    if op == "==": return series == rhs
                    if op == "!=": return series != rhs
                    if op == ">":  return series >  rhs
                    if op == ">=": return series >= rhs
                    if op == "<":  return series <  rhs
                    if op == "<=": return series <= rhs
                except Exception:
                    # Type mismatch → try numeric coercion if possible
                    try:
                        return pd.to_numeric(series, errors="coerce").map(lambda x: x).__getattribute__({">": "__gt__", ">=": "__ge__", "<": "__lt__", "<=": "__le__"}[op])(rhs)
                    except Exception:
                        return pd.Series(True, index=df.index)
        # Unknown → pass-through
        logger.warning(f"Could not parse predicate: {pred}")
        return pd.Series(True, index=df.index)
