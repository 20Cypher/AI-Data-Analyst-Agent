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

            # Apply operations
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
        """Apply filter condition to DataFrame"""
        try:
            df_copy = df.copy()
            c = condition.strip()
            # contains: "<col> contains '<value>'"
            idx = c.lower().find(" contains ")
            if idx != -1:
                col = c[:idx].strip()
                val = c[idx+len(" contains "):].strip().strip("'\"")
                if col in df.columns:
                    return df[df[col].astype(str).str.contains(val, case=False, na=False)]

            # isnull/notnull as suffix operators: "<col> isnull"
            if c.lower().endswith(" isnull"):
                col = c[: -len(" isnull")].strip()
                if col in df.columns:
                    return df[df[col].isnull()]
            if c.lower().endswith(" notnull"):
                col = c[: -len(" notnull")].strip()
                if col in df.columns:
                    return df[df[col].notnull()]
                
            # Handle different filter types
            if "==" in condition:
                column, value = condition.split("==", 1)
                column = column.strip()
                value = value.strip().strip("'\"")
                
                if column in df_copy.columns:
                    # Try to convert value to appropriate type
                    if df_copy[column].dtype in ['int64', 'float64']:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    
                    return df_copy[df_copy[column] == value]
                    
            elif "!=" in condition:
                column, value = condition.split("!=", 1)
                column = column.strip()
                value = value.strip().strip("'\"")
                
                if column in df_copy.columns:
                    # Try to convert value to appropriate type
                    if df_copy[column].dtype in ['int64', 'float64']:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    
                    return df_copy[df_copy[column] != value]
                    
            elif ">" in condition and not ">=" in condition:
                column, value = condition.split(">", 1)
                column = column.strip()
                value = float(value.strip())
                
                if column in df_copy.columns:
                    return df_copy[df_copy[column] > value]
                    
            elif ">=" in condition:
                column, value = condition.split(">=", 1)
                column = column.strip()
                value = float(value.strip())
                
                if column in df_copy.columns:
                    return df_copy[df_copy[column] >= value]
                    
            elif "<" in condition and not "<=" in condition:
                column, value = condition.split("<", 1)
                column = column.strip()
                value = float(value.strip())
                
                if column in df_copy.columns:
                    return df_copy[df_copy[column] < value]
                    
            elif "<=" in condition:
                column, value = condition.split("<=", 1)
                column = column.strip()
                value = float(value.strip())
                
                if column in df_copy.columns:
                    return df_copy[df_copy[column] <= value]
                    
            elif "contains" in condition.lower():
                # Format: column contains 'value'
                parts = condition.lower().split("contains", 1)
                if len(parts) == 2:
                    column = parts[0].strip()
                    value = parts[1].strip().strip("'\"")
                    
                    if column in df_copy.columns:
                        return df_copy[df_copy[column].astype(str).str.contains(value, na=False)]
                        
            elif "isnull" in condition.lower():
                column = condition.lower().replace("isnull", "").strip()
                if column in df_copy.columns:
                    return df_copy[df_copy[column].isnull()]
                    
            elif "notnull" in condition.lower():
                column = condition.lower().replace("notnull", "").strip()
                if column in df_copy.columns:
                    return df_copy[df_copy[column].notnull()]
            
            # If no condition matched, log and return original DataFrame
            logger.warning(f"Could not parse filter condition: {condition}")
            return df_copy
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return df  # Return original DataFrame if filter fails