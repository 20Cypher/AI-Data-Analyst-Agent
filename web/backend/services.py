"""Fixed business logic services for the web backend"""
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from fastapi import UploadFile
import shutil
import time
import asyncio

from .models import DatasetInfo

class DatasetService:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize uploaded filename"""
        # Remove path components
        filename = os.path.basename(filename)
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        # Ensure reasonable length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:90] + ext
        return filename
    
    async def save_uploaded_file(self, file: UploadFile, filename: str) -> str:
        """Save uploaded file and return dataset ID"""
        # Generate dataset ID (filename without extension)
        dataset_id = os.path.splitext(filename)[0]
        filepath = self.data_dir / filename
        
        # Save file
        with open(filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return dataset_id
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if dataset exists"""
        csv_path = self.data_dir / f"{dataset_id}.csv"
        parquet_path = self.data_dir / f"{dataset_id}.parquet"
        return csv_path.exists() or parquet_path.exists()
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all available datasets"""
        datasets = []
        
        for filepath in self.data_dir.glob("*"):
            if filepath.suffix in ['.csv', '.parquet']:
                try:
                    dataset_id = filepath.stem
                    info = self._get_dataset_info(filepath, dataset_id)
                    datasets.append(info)
                except Exception as e:
                    # Skip problematic files
                    continue
        
        return datasets
    
    def _get_dataset_info(self, filepath: Path, dataset_id: str) -> DatasetInfo:
        """Get dataset information"""
        try:
            # Get file stats
            stat = filepath.stat()
            
            # Try to read dataset for schema info
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath, nrows=0)  # Just read headers
            else:
                df = pd.read_parquet(filepath)
                df = df.head(0)  # Just headers
            
            return DatasetInfo(
                dataset_id=dataset_id,
                filename=filepath.name,
                rows=None,  # Don't count rows for performance
                columns=list(df.columns),
                size_bytes=stat.st_size,
                upload_time=stat.st_mtime
            )
        
        except Exception:
            # Fallback for problematic files
            stat = filepath.stat()
            return DatasetInfo(
                dataset_id=dataset_id,
                filename=filepath.name,
                size_bytes=stat.st_size,
                upload_time=stat.st_mtime
            )

class AnalysisService:
    def __init__(self, agent_controller):
        self.agent_controller = agent_controller
    
    def analyze(self, question: str, dataset_id: str, progress_cb: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
        """Run analysis using agent controller"""
        return self.agent_controller.analyze(question, dataset_id, progress_cb=progress_cb)