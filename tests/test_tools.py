"""Tests for tools module"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path

from ..tools.duckdb_sql import DuckDBTool
from ..tools.viz import VizTool
from ..tools.file_io import FileIOTool

class TestDuckDBTool:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tool = DuckDBTool(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_query(self):
        """Test basic SQL query execution"""
        # Create test CSV
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        csv_path = Path(self.temp_dir) / "test.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Run query
        result = self.tool.run(
            "SELECT AVG(age) as avg_age FROM main",
            {"main": str(csv_path)}
        )
        
        assert result["rows"] == 1
        assert "avg_age" in result["columns"]
        assert abs(result["preview"][0]["avg_age"] - 30.0) < 0.01

class TestVizTool:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tool = VizTool(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_mock_context(self):
        """Create mock context for testing"""
        class MockContext:
            def __init__(self):
                self.run_id = "test_run"
                self.run_dir = self.temp_dir
                self.dataset_id = "test_data"
                self.dataframes = {}
        
        return MockContext()
    
    def test_line_chart_creation(self):
        """Test line chart creation"""
        # Create test data
        test_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        # Save as pickle for context
        pickle_path = Path(self.temp_dir) / "test_data.pkl"
        test_data.to_pickle(pickle_path)
        
        # Create context
        context = self.create_mock_context()
        context.dataframes = {"test": str(pickle_path)}
        
        # Create chart
        result = self.tool.create_chart(
            "line", 
            {"x": "x", "y": "y", "outfile": "charts/test_line.png"},
            context
        )
        
        assert result["success"]
        assert "test_line.png" in result["outfile"]
        assert Path(result["outfile"]).exists()

class TestFileIOTool:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tool = FileIOTool(self.temp_dir, self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_markdown(self):
        """Test markdown file saving"""
        content = "# Test Summary\n\nThis is a test summary."
        
        result = self.tool.save_markdown(content, "test_summary.md")
        
        assert Path(result).exists()
        with open(result, 'r') as f:
            assert f.read() == content
    
    def test_hash_dataset(self):
        """Test dataset hashing"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        hash1 = self.tool.hash_dataset(str(test_file))
        hash2 = self.tool.hash_dataset(str(test_file))
        
        assert hash1 == hash2  # Same file should have same hash
        assert len(hash1) == 64  # SHA256 hash length
    
    def test_make_bundle(self):
        """Test bundle creation"""
        # Create test files
        run_dir = Path(self.temp_dir) / "test_run"
        run_dir.mkdir()
        
        (run_dir / "analysis.ipynb").write_text('{"cells": []}')
        (run_dir / "summary.md").write_text("# Summary")
        
        charts_dir = run_dir / "charts"
        charts_dir.mkdir()
        (charts_dir / "chart.png").write_bytes(b"fake png")
        
        result = self.tool.make_bundle(str(run_dir))
        
        assert result["success"]
        assert Path(result["bundle_path"]).exists()
        assert result["bundle_path"].endswith(".zip")