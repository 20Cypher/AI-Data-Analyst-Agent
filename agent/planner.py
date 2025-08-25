"""Planning module for converting natural language questions to executable plans"""
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .prompts import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE
from .llm import build_llm

logger = logging.getLogger(__name__)

class PlanStep(BaseModel):
    id: int
    action: str
    notes: Optional[str] = None
    operations: Optional[List[str]] = None
    group_by: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    type: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    outfile: Optional[str] = None
    prompts: Optional[List[str]] = None

class Plan(BaseModel):
    steps: List[PlanStep]
    success_criteria: List[str]

class Planner:
    def __init__(self, llm: Optional[ChatOpenAI] = None, model_name: str = "gpt-4o"):
        self.llm = llm or build_llm()
        
    def _strip_code_fences(self, text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("```", 1)[1]
            if t.lstrip().startswith("json"):
                t = t.split("\n", 1)[1]
            t = t.rsplit("```", 1)[0]
        return t.strip()
    
    def create_plan(self, question: str, dataset_schema: Dict[str, Any]) -> Plan:
        """Convert natural language question and dataset schema into executable plan"""
        try:
            # Format schema for prompt
            schema_text = self._format_schema(dataset_schema)
            
            # Create prompt
            user_prompt = PLAN_USER_TEMPLATE.format(
                question=question,
                schema=schema_text
            )
            
            # Call LLM
            messages = [
                SystemMessage(content=PLAN_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            raw = self._strip_code_fences(response.content)
            plan_dict = json.loads(raw)
            plan = Plan(**plan_dict)
            logger.info(f"Created plan with {len(plan.steps)} steps")
            return Plan(**plan_dict)
            
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            # Fallback to basic plan
            return self._create_fallback_plan(question)
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format dataset schema for prompt"""
        lines = ["Dataset Schema:"]
        lines.append(f"- Rows: {schema.get('rows', 'unknown')}")
        lines.append("- Columns:")
        
        for col, dtype in schema.get('columns', {}).items():
            lines.append(f"  - {col}: {dtype}")
            
        return "\n".join(lines)
    
    def _create_fallback_plan(self, question: str) -> Plan:
        """Create a basic fallback plan when LLM planning fails"""
        return Plan(
            steps=[
                PlanStep(id=1, action="inspect_schema", notes="Inspect dataset structure"),
                # keep very safe prep that always works
                PlanStep(id=2, action="prepare_data", notes="Basic data preparation", operations=[
                    "remove_duplicates"
                ]),
                # a generic aggregation that always succeeds
                # (no group_by, simple count(*) is safe for any table)
                PlanStep(id=3, action="aggregate", notes="Basic aggregation", group_by=[], metrics=["count(*)"]),
                # lightweight viz that can render without explicit x/y (falls back to first numeric)
                PlanStep(id=4, action="visualize", type="line", outfile="charts/basic_chart.png"),
                PlanStep(id=5, action="interpret", prompts=["Summarize findings"]),
                PlanStep(id=6, action="compose_notebook"),
                PlanStep(id=7, action="bundle_outputs")
            ],
            success_criteria=[
                "At least one chart saved",
                "Summary generated",
                "Notebook created"
            ]
        )
