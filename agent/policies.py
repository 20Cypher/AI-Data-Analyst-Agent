"""Policy management for budget limits and execution controls"""
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

class PolicyManager:
    """Manages execution policies like budgets, retries, and timeouts"""
    
    def __init__(self, max_tool_calls: int = 20, max_runtime: int = 300):
        self.max_tool_calls = max_tool_calls
        self.max_runtime = max_runtime
    
    def check_budget(self, context: Any) -> bool:
        """Check if execution is within budget limits"""
        
        # Check tool calls limit
        if context.tool_calls >= self.max_tool_calls:
            logger.warning(f"Tool calls limit exceeded: {context.tool_calls}/{self.max_tool_calls}")
            return False
        
        # Check runtime limit
        elapsed = time.time() - context.start_time
        if elapsed >= self.max_runtime:
            logger.warning(f"Runtime limit exceeded: {elapsed}s/{self.max_runtime}s")
            return False
        
        return True
    
    def should_retry(self, attempt: int, max_retries: int = 3) -> bool:
        """Determine if operation should be retried"""
        return attempt < max_retries
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return min(2 ** attempt, 30)  # Cap at 30 seconds