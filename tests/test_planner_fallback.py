import pytest
from ..agent.planner import Planner

SUPPORTED = {
    "inspect_schema",
    "prepare_data",
    "aggregate",
    "visualize",
    "interpret",
    "compose_notebook",
    "bundle_outputs",
}

def test_fallback_plan_supported_actions_only():
    p = Planner(llm=None)  # llm None => we’ll call the fallback directly
    plan = p._create_fallback_plan("any question")
    assert plan.steps, "fallback plan should have steps"
    assert all(step.action in SUPPORTED for step in plan.steps), \
        f"Unsupported actions found: {[s.action for s in plan.steps if s.action not in SUPPORTED]}"
