"""Tests for validation module"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path

from ..tools.validation import ValidationTool, ValidationReport

class TestValidationTool:
    def setup_method(self):
        self.validator = ValidationTool()
    
    def test_validate_schema_success(self):
        """Test successful schema validation"""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30],
            'salary': [50000, 60000]
        })
        
        expectations = {
            'required_columns': ['name', 'age'],
            'column_types': {'age': 'int', 'salary': 'int'}
        }
        
        report = self.validator.validate_schema(df, expectations)
        
        assert report.valid
        assert len(report.errors) == 0
        assert report.metrics['row_count'] == 2
        assert report.metrics['column_count'] == 3
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns"""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })
        
        expectations = {
            'required_columns': ['name', 'age', 'salary']
        }
        
        report = self.validator.validate_schema(df, expectations)
        
        assert not report.valid
        assert len(report.errors) == 1
        assert 'salary' in report.errors[0]
    
    def test_preflight_empty_dataset(self):
        """Test preflight validation with empty dataset"""
        df = pd.DataFrame()
        
        report = self.validator.preflight(df)
        
        assert not report.valid
        assert 'empty' in report.errors[0].lower()
    
    def test_preflight_high_missing_values(self):
        """Test preflight validation with high missing values"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [None, None, None, None, 1]  # 80% missing
        })
        
        report = self.validator.preflight(df)
        
        assert report.valid  # Should still be valid, just warnings
        assert len(report.warnings) > 0
        assert 'col2' in str(report.warnings)
    
    def test_postflight_missing_artifacts(self):
        """Test postflight validation with missing artifacts"""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self.validator.postflight(temp_dir)
            
            assert not report.valid
            assert len(report.errors) >= 3  # Missing required files
    
    def test_postflight_complete_artifacts(self):
        """Test postflight validation with complete artifacts"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create required files
            (temp_path / "analysis.ipynb").write_text('{"cells": []}')
            (temp_path / "summary.md").write_text("# Summary\nTest summary")
            (temp_path / "trace.json").write_text('{"status": "completed"}')
            
            # Create charts directory with a chart
            charts_dir = temp_path / "charts"
            charts_dir.mkdir()
            (charts_dir / "test_chart.png").write_bytes(b"fake png data")
            
            report = self.validator.postflight(temp_dir)
            
            assert report.valid
            assert len(report.errors) == 0