"""Main agent controller for orchestrating analysis execution"""
import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from .planner import Planner
from .memory import Memory, Context
from .policies import PolicyManager
from ..tools.router import ToolRouter
from ..tools.validation import ValidationTool
from .llm import build_llm

logger = logging.getLogger(__name__)

class AgentController:
    def __init__(self, artifacts_dir: str, data_dir: str, max_tool_calls: int = 20, max_runtime: int = 300):
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)

        self.artifacts_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Build one LLM and share
        shared_llm = build_llm()

        # Initialize components
        self.planner = Planner(llm=shared_llm)  # <-- inject
        self.memory = Memory()
        self.policy_manager = PolicyManager(max_tool_calls, max_runtime)
        self.tool_router = ToolRouter(str(artifacts_dir), str(data_dir), llm=shared_llm)  # <-- inject
        self.validator = ValidationTool()

        logger.info("AgentController initialized")
    
    def analyze(self, question: str, dataset_id: str, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Main analysis entry point"""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Create run directory
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "charts").mkdir(exist_ok=True)
        
        self._report(progress_cb, 5, f"Run {run_id}: initialized artifacts directory")

        # Initialize context
        context = Context(
            run_id=run_id,
            question=question,
            dataset_id=dataset_id,
            run_dir=str(run_dir),
            start_time=start_time
        )
        
        try:
            # Step 1: Load and validate dataset
            dataset_schema = self._load_dataset_schema(dataset_id)
            df = self._load_dataset(dataset_id)
            
            # Validate dataset
            preflight_report = self.validator.preflight(df)
            if not preflight_report.valid:
                raise ValueError(f"Dataset validation failed: {preflight_report.errors}")
            
            self._report(progress_cb, 10, "Dataset loaded; preflight validation passed")
            
            # Step 2: Create execution plan
            plan = self.planner.create_plan(question, dataset_schema)
            context.plan = plan
            
            logger.info(f"Created plan with {len(plan.steps)} steps for run {run_id}")

            self._report(progress_cb, 20, f"Plan created with {len(plan.steps)} steps")
            
            # Step 3: Execute plan
            trace = self._execute_plan(plan, context, progress_cb=progress_cb, base_progress=20, end_progress=85)
            
            # Step 4: Generate summary
            summary_result = self.tool_router.get_tool("file_io").generate_summary(context)

            self._report(progress_cb, 90, "Summary generated")
            
            # Step 5: Create notebook
            notebook_result = self.tool_router.get_tool("notebook_builder").compose(context)

            self._report(progress_cb, 93, "Notebook composed")
            
            # Step 6: Validate results
            postflight_report = self.validator.postflight(str(run_dir))

            self._report(progress_cb, 96, "Postflight validation passed")
            
            # Step 7: Create bundle
            bundle_result = self.tool_router.get_tool("file_io").make_bundle(str(run_dir))

            self._report(progress_cb, 99, "Bundle created")
            
            # Save trace
            trace_path = run_dir / "trace.json"
            with open(trace_path, 'w') as f:
                json.dump(trace, f, indent=2, default=str)
            
            # Save dataset hash for reproducibility
            dataset_path = self.data_dir / f"{dataset_id}.csv"
            if not dataset_path.exists():
                dataset_path = self.data_dir / f"{dataset_id}.parquet"
            
            if dataset_path.exists():
                dataset_hash = self.tool_router.get_tool("file_io").hash_dataset(str(dataset_path))
                (run_dir / "dataset_hash.txt").write_text(dataset_hash)
            
            end_time = time.time()
            
            self._report(progress_cb, 100, "Analysis completed")

            return {
                "status": "success",
                "run_id": run_id,
                "duration": end_time - start_time,
                "bundle_path": bundle_result["bundle_path"],
                "trace": trace,
                "validation": {
                    "preflight": preflight_report.dict(),
                    "postflight": postflight_report.dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for run {run_id}: {e}")
            self._report(progress_cb, 100, f"Analysis failed: {e}")
            
            # Save error trace
            error_trace = {
                "status": "error",
                "error": str(e),
                "run_id": run_id,
                "question": question,
                "dataset_id": dataset_id,
                "duration": time.time() - start_time,
                "steps_completed": getattr(context, 'tool_calls', 0)
            }
            
            try:
                trace_path = run_dir / "trace.json"
                with open(trace_path, 'w') as f:
                    json.dump(error_trace, f, indent=2, default=str)
            except:
                pass  # Don't fail on trace saving
            
            return {
                "status": "error",
                "run_id": run_id,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _load_dataset_schema(self, dataset_id: str) -> Dict[str, Any]:
        """Load dataset schema information"""
        import pandas as pd
        
        dataset_path = self.data_dir / f"{dataset_id}.csv"
        if not dataset_path.exists():
            dataset_path = self.data_dir / f"{dataset_id}.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path, nrows=1000)  # Sample for schema
        else:
            df = pd.read_parquet(dataset_path)
            df = df.head(1000)
        
        # Build schema info
        schema = {
            "rows": len(df),
            "columns": {col: str(df[col].dtype) for col in df.columns}
        }
        
        return schema
    
    def _load_dataset(self, dataset_id: str):
        """Load full dataset for validation"""
        import pandas as pd
        
        dataset_path = self.data_dir / f"{dataset_id}.csv"
        if not dataset_path.exists():
            dataset_path = self.data_dir / f"{dataset_id}.parquet"
        
        if dataset_path.suffix == '.csv':
            return pd.read_csv(dataset_path)
        else:
            return pd.read_parquet(dataset_path)
    
    def _execute_plan(self, plan, context: Context, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None, base_progress: int = 20, end_progress: int = 85) -> Dict[str, Any]:
        """Execute the analysis plan step by step"""
        trace = {
            "run_id": context.run_id,
            "question": context.question,
            "dataset_id": context.dataset_id,
            "plan": plan.dict(),
            "steps": [],
            "start_time": context.start_time,
            "status": "running"
        }

        total_steps = max(len(plan.steps), 1)
        progress_span = max(end_progress - base_progress, 1)
        
        for i, step in enumerate(plan.steps):
            # Check budget constraints
            if not self.policy_manager.check_budget(context):
                logger.warning(f"Budget exceeded at step {i}")
                break

            step_start_msg = f"Executing step {step.id}: {step.action}"
            logger.info(step_start_msg)
            self._report(progress_cb, base_progress + int((i / total_steps) * progress_span), step_start_msg)
                        
            try:
                # Execute step based on action type
                result = self._execute_step(step, context)
                
                # Update context with results
                self.memory.update_context(context, step, result)
                context.tool_calls += 1
                
                # Record in trace
                step_trace = {
                    "step_id": step.id,
                    "action": step.action,
                    "params": step.dict(),
                    "result": result,
                    "timestamp": time.time()
                }
                trace["steps"].append(step_trace)
                
                logger.info(f"Step {step.id} completed successfully")
                self._report(progress_cb, base_progress + int(((i + 1) / total_steps) * progress_span), f"Step {step.id} {step.action} completed")
                
            except Exception as e:
                logger.error(f"Step {step.id} failed: {e}")
                
                step_trace = {
                    "step_id": step.id,
                    "action": step.action,
                    "params": step.dict(),
                    "error": str(e),
                    "timestamp": time.time()
                }
                trace["steps"].append(step_trace)
                err_msg = f"Step {step.id} failed: {e}"
                logger.error(err_msg)
                self._report(progress_cb, base_progress + int(((i + 1) / total_steps) * progress_span), err_msg)
                # Continue with next step (resilient execution)
                continue
        
        trace["end_time"] = time.time()
        trace["status"] = "completed"
        
        return trace
    
    def _execute_step(self, step, context: Context) -> Dict[str, Any]:
        """Execute a single step based on its action type"""
        
        if step.action == "inspect_schema":
            dataset_path = self.data_dir / f"{context.dataset_id}.csv"
            if not dataset_path.exists():
                dataset_path = self.data_dir / f"{context.dataset_id}.parquet"
            
            sources = {"main": str(dataset_path)}
            return self.tool_router.get_tool("duckdb_sql").inspect_schema(context.dataset_id, sources)
        
        elif step.action == "prepare_data":
            operations = getattr(step, 'operations', [])
            return self.tool_router.get_tool("python_repl").prepare_data(operations, context)
        
        elif step.action == "aggregate":
            group_by = getattr(step, 'group_by', [])
            metrics = getattr(step, 'metrics', [])
            return self.tool_router.get_tool("duckdb_sql").aggregate(group_by, metrics, context)
        
        elif step.action == "visualize":
            chart_params = {
                "type": getattr(step, 'type', 'line'),
                "x": getattr(step, 'x', None),
                "y": getattr(step, 'y', None),
                "outfile": getattr(step, 'outfile', f"charts/chart_{step.id}.png")
            }
            return self.tool_router.get_tool("viz").create_chart(
                chart_params["type"], chart_params, context
            )
        
        elif step.action == "interpret":
            prompts = getattr(step, 'prompts', [])
            return self.tool_router.get_tool("python_repl").interpret(prompts, context)
        
        elif step.action == "compose_notebook":
            return self.tool_router.get_tool("notebook_builder").compose(context)
        
        elif step.action == "bundle_outputs":
            return self.tool_router.get_tool("file_io").make_bundle(context.run_dir)
        
        else:
            raise ValueError(f"Unknown action: {step.action}")
        
    def _report(self, progress_cb: Optional[Callable[[Dict[str, Any]], None]], progress: int, message: str) -> None:
        """Send a progress update if a callback is provided."""
        if not progress_cb:
            return
        try:
            progress_cb({"progress": int(progress), "message": message})
        except Exception:
            # Never let UI reporting crash the run
            pass
