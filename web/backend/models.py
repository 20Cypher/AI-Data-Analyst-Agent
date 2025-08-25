"""Pydantic models for API requests and responses"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    dataset_id: str
    question: str

class AnalysisResponse(BaseModel):
    run_id: str
    status: str
    bundle_path: Optional[str] = None
    error: Optional[str] = None

class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    rows: Optional[int] = None
    columns: Optional[List[str]] = None
    size_bytes: Optional[int] = None
    upload_time: Optional[float] = None

class RunStatus(BaseModel):
    run_id: str
    status: str
    progress: int
    messages: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    bundle_path: Optional[str] = None