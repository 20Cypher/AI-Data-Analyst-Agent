"""Validation tools for data quality and result checking"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ValidationReport(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class ValidationTool:
    def __init__(self):
        self.max_missing_ratio = 0.8
        self.min_rows = 1

    def _count_iqr_outliers(self, s: pd.Series) -> int:
        """Return count of outliers using Tukey's IQR method (1.5*IQR fences)."""
        s = s.dropna()
        if s.empty:
            return 0
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0  # all values identical or very tight — no IQR-based outliers
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((s < lower) | (s > upper)).sum())

        
    def validate_schema(self, df: pd.DataFrame, expectations: Dict[str, Any]) -> ValidationReport:
        """Validate DataFrame against expected schema"""
        errors = []
        warnings = []
        
        # Check required columns
        required_cols = expectations.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        expected_types = expectations.get('column_types', {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    warnings.append(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check row count
        if len(df) < self.min_rows:
            errors.append(f"Dataset has {len(df)} rows, minimum required: {self.min_rows}")
        
        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={
                "row_count": len(df),
                "column_count": len(df.columns),
                "missing_columns": len(missing_cols)
            }
        )
    
    def preflight(self, df: pd.DataFrame) -> ValidationReport:
        """Pre-analysis validation checks"""
        errors = []
        warnings = []
        
        # Check for empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty")
            return ValidationReport(valid=False, errors=errors, warnings=warnings, metrics={})
        
        # Check for excessive missing values
        missing_ratios = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratios[missing_ratios > self.max_missing_ratio].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with high missing values (>{self.max_missing_ratio*100}%): {high_missing_cols}")
        
        # Check for date parsing issues
        potential_date_cols = df.select_dtypes(include=['object']).columns
        for col in potential_date_cols:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                try:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise')
                except:
                    warnings.append(f"Column {col} appears to be a date but parsing may fail")
        
        # Check for numeric sanity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].min() == df[col].max():
                warnings.append(f"Column {col} has constant values")
            
            # Check for outliers using IQR (Tukey fences)
            if len(df) > 10:  # Only for datasets with sufficient rows
                try:
                    outliers = self._count_iqr_outliers(df[col])
                    if outliers > 0:
                        warnings.append(
                            f"Column {col} has {outliers} potential extreme outliers (IQR method)"
                        )
                except Exception as _:
                    # Be resilient — don't let validation fail due to a numeric quirk
                    pass
        
        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "high_missing_columns": len(high_missing_cols)
            }
        )
    
    def postflight(self, run_dir: str) -> ValidationReport:
        """Post-analysis validation of artifacts"""
        run_path = Path(run_dir)
        errors = []
        warnings = []
        
        # Check for required artifacts
        required_files = ["analysis.ipynb", "summary.md", "trace.json"]
        for filename in required_files:
            if not (run_path / filename).exists():
                errors.append(f"Missing required file: {filename}")
        
        # Check for charts
        charts_dir = run_path / "charts"
        if not charts_dir.exists():
            errors.append("No charts directory found")
        else:
            chart_files = list(charts_dir.glob("*.png"))
            if len(chart_files) == 0:
                errors.append("No chart files found in charts directory")
        
        # Validate notebook can be opened
        notebook_path = run_path / "analysis.ipynb"
        if notebook_path.exists():
            try:
                import nbformat
                with open(notebook_path, 'r') as f:
                    nb = nbformat.read(f, as_version=4)
                if len(nb.cells) == 0:
                    warnings.append("Notebook has no cells")
            except Exception as e:
                errors.append(f"Notebook validation failed: {e}")
        
        # Check summary content
        summary_path = run_path / "summary.md"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    content = f.read()
                if len(content.strip()) < 50:
                    warnings.append("Summary appears to be very short")
            except Exception as e:
                warnings.append(f"Could not validate summary: {e}")
        
        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={
                "artifacts_found": len([f for f in required_files if (run_path / f).exists()]),
                "charts_count": len(list(charts_dir.glob("*.png"))) if charts_dir.exists() else 0
            }
        )
    
    def validate_analysis_result(self, context: Any, result: Dict[str, Any]) -> ValidationReport:
        """Validate analysis results meet success criteria"""
        errors = []
        warnings = []
        
        # Check success criteria from plan
        if hasattr(context, 'plan') and context.plan:
            for criterion in context.plan.success_criteria:
                if not self._check_criterion(criterion, context, result):
                    errors.append(f"Success criterion not met: {criterion}")
        
        # Check for computed values (grounding)
        if not hasattr(context, 'computed_values') or not context.computed_values:
            warnings.append("No computed values found for grounding summary")
        
        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={
                "criteria_checked": len(context.plan.success_criteria) if hasattr(context, 'plan') and context.plan else 0,
                "computed_values_count": len(context.computed_values) if hasattr(context, 'computed_values') else 0
            }
        )
    
    def _check_criterion(self, criterion: str, context: Any, result: Dict[str, Any]) -> bool:
        """Check if a specific success criterion is met"""
        criterion_lower = criterion.lower()
        
        if "chart saved" in criterion_lower:
            charts_dir = Path(context.run_dir) / "charts"
            return charts_dir.exists() and len(list(charts_dir.glob("*.png"))) > 0
        
        elif "summary" in criterion_lower:
            summary_path = Path(context.run_dir) / "summary.md"
            return summary_path.exists() and summary_path.stat().st_size > 0
        
        elif "notebook" in criterion_lower:
            notebook_path = Path(context.run_dir) / "analysis.ipynb"
            return notebook_path.exists() and notebook_path.stat().st_size > 0
        
        elif "computed values" in criterion_lower:
            return hasattr(context, 'computed_values') and len(context.computed_values) > 0
        
        return True  # Unknown criteria pass by default