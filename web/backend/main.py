"""Fixed FastAPI backend for AI Data Analyst Agent"""
import os
import sys
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directories to path for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from web.backend.models import AnalysisRequest, AnalysisResponse, DatasetInfo, RunStatus
from web.backend.services import AnalysisService, DatasetService
from agent.controller import AgentController

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Data Analyst Agent", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
DATA_DIR = os.getenv("DATA_DIR", "data")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "20"))
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME_SECONDS", "300"))

# Ensure directories exist
Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
Path(DATA_DIR).mkdir(exist_ok=True)

# Initialize services
agent_controller = AgentController(ARTIFACTS_DIR, DATA_DIR, MAX_TOOL_CALLS, MAX_RUNTIME)
analysis_service = AnalysisService(agent_controller)
dataset_service = DatasetService(DATA_DIR)

# Store running analyses
running_analyses: Dict[str, Dict[str, Any]] = {}

# Static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def root():
    """Root endpoint - serve frontend or API info"""
    frontend_index = frontend_dir / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    else:
        return {
            "message": "AI Data Analyst Agent API",
            "version": "1.0.0",
            "docs_url": "/docs"
        }

@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a dataset file"""
    try:
        # Sanitize filename
        safe_filename = dataset_service.sanitize_filename(file.filename)
        
        # Check file extension
        if not safe_filename.endswith(('.csv', '.parquet')):
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
        
        # Save file
        dataset_id = await dataset_service.save_uploaded_file(file, safe_filename)
        
        logger.info(f"Dataset uploaded: {dataset_id}")
        return {
            "dataset_id": dataset_id,
            "filename": safe_filename,
            "status": "uploaded"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def list_datasets() -> List[DatasetInfo]:
    """List available datasets"""
    try:
        datasets = dataset_service.list_datasets()
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start analysis of a dataset"""
    try:
        # Validate dataset exists
        if not dataset_service.dataset_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Generate run ID
        run_id = str(uuid4())
        
        # Initialize run status
        running_analyses[run_id] = {
            "status": "starting",
            "progress": 0,
            "messages": [],
            "start_time": None,
            "end_time": None,
            "result": None
        }
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis_background,
            run_id,
            request.question,
            request.dataset_id
        )
        
        logger.info(f"Analysis started: {run_id}")
        return {"run_id": run_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}")
async def get_run_status(run_id: str) -> RunStatus:
    """Get status of a running analysis"""
    if run_id not in running_analyses:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_data = running_analyses[run_id]
    
    return RunStatus(
        run_id=run_id,
        status=run_data["status"],
        progress=run_data["progress"],
        messages=run_data["messages"],
        start_time=run_data["start_time"],
        end_time=run_data["end_time"],
        bundle_path=run_data.get("result", {}).get("bundle_path") if run_data["result"] else None
    )

@app.get("/runs/{run_id}/artifacts")
async def list_run_artifacts(run_id: str) -> Dict[str, Any]:
    """List summary and charts for a finished run."""
    if run_id not in running_analyses:
        raise HTTPException(status_code=404, detail="Run not found")
    run_data = running_analyses[run_id]
    if run_data.get("status") != "completed" or not run_data.get("result"):
        raise HTTPException(status_code=400, detail="Analysis not completed")

    bundle_path = run_data["result"].get("bundle_path")
    if not bundle_path or not Path(bundle_path).exists():
        raise HTTPException(status_code=404, detail="Bundle not found")

    run_dir = Path(bundle_path).parent
    charts_dir = run_dir / "charts"
    charts = []
    if charts_dir.exists():
        for p in sorted(charts_dir.glob("*.png")):
            charts.append(f"/runs/{run_id}/files/{p.relative_to(run_dir).as_posix()}")

    summary_rel = "summary.md"
    summary_url = f"/runs/{run_id}/files/{summary_rel}" if (run_dir / summary_rel).exists() else None

    return {
        "summary": summary_url,
        "charts": charts
    }

@app.get("/runs/{run_id}/files/{path:path}")
async def get_run_file(run_id: str, path: str):
    """Serve a specific artifact file within the run directory (safe join)."""
    if run_id not in running_analyses:
        raise HTTPException(status_code=404, detail="Run not found")
    run_data = running_analyses[run_id]
    if not run_data.get("result") or not run_data["result"].get("bundle_path"):
        raise HTTPException(status_code=400, detail="Artifacts unavailable")

    bundle_path = run_data["result"]["bundle_path"]
    run_dir = Path(bundle_path).parent

    # Safe path resolution (prevent traversal)
    requested = (run_dir / path).resolve()
    base = run_dir.resolve()
    if str(requested).startswith(str(base)) and requested.exists() and requested.is_file():
        return FileResponse(str(requested))
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/runs/{run_id}/bundle")
async def download_bundle(run_id: str):
    """Download analysis bundle"""
    if run_id not in running_analyses:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_data = running_analyses[run_id]
    
    if run_data["status"] != "completed" or not run_data.get("result"):
        raise HTTPException(status_code=400, detail="Analysis not completed or failed")
    
    bundle_path = run_data["result"].get("bundle_path")
    if not bundle_path or not Path(bundle_path).exists():
        raise HTTPException(status_code=404, detail="Bundle not found")
    
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename=f"analysis_{run_id}.zip"
    )

@app.get("/runs/{run_id}/stream")
async def stream_run_logs(run_id: str):
    """Stream live logs for a running analysis"""
    if run_id not in running_analyses:
        raise HTTPException(status_code=404, detail="Run not found")
    
    async def generate_logs():
        last_message_count = 0
        
        while True:
            run_data = running_analyses.get(run_id)
            if not run_data:
                break
            
            # Send new messages
            messages = run_data["messages"]
            new_messages = messages[last_message_count:]
            
            for message in new_messages:
                yield f"data: {message}\n\n"
            
            last_message_count = len(messages)
            
            # Check if analysis is complete
            if run_data["status"] in ["completed", "failed", "error"]:
                yield f"data: Analysis {run_data['status']}\n\n"
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",   # was text/plain
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

async def run_analysis_background(run_id: str, question: str, dataset_id: str):
    """Run analysis in background task"""
    import time
    
    def progress_cb(update: Dict[str, Any]):
        # update contains {"progress": int, "message": str}
        try:
            p = int(update.get("progress", 0))
        except Exception:
            p = 0
        msg = update.get("message")
        run_data = running_analyses.get(run_id)
        if not run_data:
            return
        # increase progress monotonically
        if p and p > run_data.get("progress", 0):
            run_data["progress"] = p
        if msg:
            run_data["messages"].append(msg)

    try:
        # Update status
        running_analyses[run_id].update({
            "status": "running",
            "start_time": time.time(),
            "messages": ["Starting analysis..."]
        })
        
        # Run analysis
        result = analysis_service.analyze(question, dataset_id, progress_cb=progress_cb)
        
        # Update status
        running_analyses[run_id].update({
            "status": "completed" if result["status"] == "success" else "failed",
            "end_time": time.time(),
            "progress": 100,
            "result": result,
            "messages": running_analyses[run_id]["messages"] + [f"Analysis {result['status']}"]
        })
        
        logger.info(f"Analysis completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {run_id} - {e}")
        running_analyses[run_id].update({
            "status": "error",
            "end_time": time.time(),
            "messages": running_analyses[run_id]["messages"] + [f"Error: {str(e)}"]
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")