#!/usr/bin/env python3
"""Evaluation harness for AI Data Analyst Agent"""
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from agent import AgentController

class EvaluationHarness:
    def __init__(self, artifacts_dir: str = "artifacts", data_dir: str = "data"):
        self.agent = AgentController(artifacts_dir, data_dir)
        self.results = []
        
    def load_tasks(self, tasks_file: str = "evals/tasks.jsonl") -> List[Dict]:
        """Load evaluation tasks from JSONL file"""
        tasks = []
        with open(tasks_file, 'r') as f:
            for line in f:
                tasks.append(json.loads(line.strip()))
        return tasks
    
    def run_single_task(self, task: Dict) -> Dict[str, Any]:
        """Run a single evaluation task"""
        start_time = time.time()
        
        try:
            # Run analysis
            result = self.agent.analyze(task["question"], task["dataset_id"])
            
            # Check assertions
            assertions_passed = self._check_assertions(
                task.get("assertions", []), 
                result
            )
            
            success = result["status"] == "success" and assertions_passed["all_passed"]
            
            return {
                "task_id": task["id"],
                "question": task["question"],
                "dataset_id": task["dataset_id"],
                "success": success,
                "status": result["status"],
                "duration": time.time() - start_time,
                "assertions": assertions_passed,
                "bundle_path": result.get("bundle_path"),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            return {
                "task_id": task["id"],
                "question": task["question"],
                "dataset_id": task["dataset_id"],
                "success": False,
                "status": "error",
                "duration": time.time() - start_time,
                "error": str(e),
                "assertions": {"all_passed": False, "details": {}}
            }
    
    def _check_assertions(self, assertions: List[str], result: Dict) -> Dict[str, Any]:
        """Check if assertions are met"""
        details = {}
        
        for assertion in assertions:
            if assertion == "chart_created":
                details[assertion] = self._check_chart_created(result)
            elif assertion == "monthly_aggregation":
                details[assertion] = self._check_monthly_aggregation(result)
            elif assertion == "trend_analysis":
                details[assertion] = self._check_trend_analysis(result)
            elif assertion == "regional_analysis":
                details[assertion] = self._check_regional_analysis(result)
            elif assertion == "average_calculation":
                details[assertion] = self._check_average_calculation(result)
            # ---- New assertions implemented below ----
            elif assertion == "aggregation_result":
                details[assertion] = self._check_aggregation_result(result)
            elif assertion == "top_5_ranking":
                details[assertion] = self._check_top_5_ranking(result)
            elif assertion == "category_analysis":
                details[assertion] = self._check_category_analysis(result)
            elif assertion == "distance_analysis":
                details[assertion] = self._check_distance_analysis(result)
            elif assertion == "fare_analysis":
                details[assertion] = self._check_fare_analysis(result)
            elif assertion == "distribution_analysis":
                details[assertion] = self._check_distribution_analysis(result)
            elif assertion == "passenger_analysis":
                details[assertion] = self._check_passenger_analysis(result)
            elif assertion == "outlier_detection":
                details[assertion] = self._check_outlier_detection(result)
            elif assertion == "boxplot_visualization":
                details[assertion] = self._check_boxplot_visualization(result)
            elif assertion == "payment_analysis":
                details[assertion] = self._check_payment_analysis(result)
            elif assertion == "tip_analysis":
                details[assertion] = self._check_tip_analysis(result)
            elif assertion == "comparison_made":
                details[assertion] = self._check_comparison_made(result)
            elif assertion == "correlation_analysis":
                details[assertion] = self._check_correlation_analysis(result)
            elif assertion == "relationship_analysis":
                details[assertion] = self._check_relationship_analysis(result)
            elif assertion == "scatter_plot":
                details[assertion] = self._check_scatter_plot(result)
            else:
                details[assertion] = True  # Default pass for unknown assertions
        
        all_passed = all(details.values())
        
        return {
            "all_passed": all_passed,
            "details": details,
            "passed_count": sum(1 for v in details.values() if v),
            "total_count": len(assertions)
        }

    def _check_chart_created(self, result: Dict) -> bool:
        """Check if at least one chart was created"""
        if result["status"] != "success":
            return False
            
        bundle_path = result.get("bundle_path")
        if not bundle_path:
            return False
            
        # Check if charts directory exists and has PNG files
        run_dir = Path(bundle_path).parent
        charts_dir = run_dir / "charts"
        
        return charts_dir.exists() and len(list(charts_dir.glob("*.png"))) > 0
    
    def _check_monthly_aggregation(self, result: Dict) -> bool:
        """Check if monthly aggregation was performed"""
        # This is a simplified check - in practice, you'd examine the trace
        trace = result.get("trace", {})
        steps = trace.get("steps", [])
        
        for step in steps:
            if step.get("action") == "aggregate":
                group_by = step.get("result", {}).get("group_by", [])
                if any("month" in str(gb).lower() for gb in group_by):
                    return True
        
        return False
    
    def _check_trend_analysis(self, result: Dict) -> bool:
        """Check if trend analysis was mentioned in summary"""
        if result["status"] != "success":
            return False
            
        bundle_path = result.get("bundle_path")
        if not bundle_path:
            return False
            
        try:
            run_dir = Path(bundle_path).parent
            summary_path = run_dir / "summary.md"
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    content = f.read().lower()
                return "trend" in content
        except:
            pass
        
        return False
    
    def _check_regional_analysis(self, result: Dict) -> bool:
        """Check if regional analysis was performed"""
        trace = result.get("trace", {})
        steps = trace.get("steps", [])
        
        for step in steps:
            if step.get("action") == "aggregate":
                group_by = step.get("result", {}).get("group_by", [])
                if any("region" in str(gb).lower() for gb in group_by):
                    return True
        
        return False
    
    def _check_average_calculation(self, result: Dict) -> bool:
        """Check if average calculation was performed"""
        trace = result.get("trace", {})
        steps = trace.get("steps", [])
        
        for step in steps:
            if step.get("action") == "aggregate":
                metrics = step.get("result", {}).get("metrics", [])
                if any("avg" in str(m).lower() or "mean" in str(m).lower() for m in metrics):
                    return True
        
        return False
    
    def _get_trace_steps(self, result: Dict) -> List[Dict[str, Any]]:
        trace = result.get("trace", {}) or {}
        return trace.get("steps", []) or []

    def _read_summary_text(self, result: Dict) -> str:
        """Read summary.md (lowercased) if available; return empty string on failure."""
        try:
            bundle_path = result.get("bundle_path")
            if not bundle_path:
                return ""
            run_dir = Path(bundle_path).parent
            summary_path = run_dir / "summary.md"
            if summary_path.exists():
                return summary_path.read_text(encoding="utf-8").lower()
        except Exception:
            pass
        return ""

    def _columns_or_query_mentions(self, res: Dict[str, Any], keywords: List[str]) -> bool:
        """Return True if any keyword (lowercased) appears in result['columns'] or result['query']."""
        kws = [k.lower() for k in keywords]
        # columns
        for col in res.get("columns", []) or []:
            c = str(col).lower()
            if any(k in c for k in kws):
                return True
        # query
        q = str(res.get("query", "")).lower()
        if any(k in q for k in kws):
            return True
        return False

    def _any_visualization(self, result: Dict, chart_types: List[str]) -> bool:
        """True if any visualize step produced one of the given chart types."""
        ctset = {c.lower() for c in chart_types}
        for step in self._get_trace_steps(result):
            if step.get("action") == "visualize":
                r = step.get("result", {}) or {}
                ctype = str(r.get("chart_type", "")).lower()
                if ctype in ctset:
                    return True
        return False

    def _any_aggregate(self, result: Dict) -> bool:
        """True if at least one aggregate step exists."""
        return any(step.get("action") == "aggregate" for step in self._get_trace_steps(result))

    # ---------- New concrete checks ----------

    def _check_aggregation_result(self, result: Dict) -> bool:
        """At least one aggregation produced rows."""
        for step in self._get_trace_steps(result):
            if step.get("action") == "aggregate":
                res = step.get("result", {}) or {}
                if res.get("rows", 0) > 0:
                    return True
                if res.get("preview"):
                    return True
                if res.get("computed_values", {}).get("result_count", 0) > 0:
                    return True
        return False

    def _check_top_5_ranking(self, result: Dict) -> bool:
        """Detect top-5 via LIMIT 5 or result count == 5."""
        for step in self._get_trace_steps(result):
            if step.get("action") == "aggregate":
                res = step.get("result", {}) or {}
                if res.get("computed_values", {}).get("result_count") == 5:
                    return True
                if len(res.get("preview", [])) == 5:
                    return True
                q = str(res.get("query", "")).lower()
                if "limit" in q and " 5" in q:
                    return True
        return False

    def _check_category_analysis(self, result: Dict) -> bool:
        """Group-by on a 'category'-like column or presence in selected columns."""
        for step in self._get_trace_steps(result):
            if step.get("action") == "aggregate":
                res = step.get("result", {}) or {}
                gbs = [str(g).lower() for g in res.get("group_by", []) or []]
                if any("category" in g for g in gbs):
                    return True
                if self._columns_or_query_mentions(res, ["category"]):
                    return True
        return False

    def _check_distance_analysis(self, result: Dict) -> bool:
        """Evidence of distance analysis (columns/query mention 'distance' or summary mentions it)."""
        for step in self._get_trace_steps(result):
            res = step.get("result", {}) or {}
            if self._columns_or_query_mentions(res, ["distance"]):
                return True
        # fallback: check summary text
        return "distance" in self._read_summary_text(result)

    def _check_fare_analysis(self, result: Dict) -> bool:
        """Evidence of fare analysis (columns/query mention 'fare')."""
        for step in self._get_trace_steps(result):
            res = step.get("result", {}) or {}
            if self._columns_or_query_mentions(res, ["fare"]):
                return True
        return "fare" in self._read_summary_text(result)

    def _check_distribution_analysis(self, result: Dict) -> bool:
        """Bar chart is an acceptable signal for 'distribution' in this agent."""
        return self._any_visualization(result, ["bar"])

    def _check_passenger_analysis(self, result: Dict) -> bool:
        """Mentions of passenger-related fields."""
        for step in self._get_trace_steps(result):
            res = step.get("result", {}) or {}
            if self._columns_or_query_mentions(res, ["passenger"]):
                return True
        return "passenger" in self._read_summary_text(result)

    def _check_outlier_detection(self, result: Dict) -> bool:
        """Either preflight warned about outliers or a boxplot was produced."""
        # Check preflight warnings
        pre = (result.get("validation") or {}).get("preflight") or {}
        warnings = pre.get("warnings", []) or []
        if any("outlier" in str(w).lower() for w in warnings):
            return True
        # Or a boxplot visualization
        return self._any_visualization(result, ["boxplot"])

    def _check_boxplot_visualization(self, result: Dict) -> bool:
        return self._any_visualization(result, ["boxplot"])

    def _check_payment_analysis(self, result: Dict) -> bool:
        for step in self._get_trace_steps(result):
            res = step.get("result", {}) or {}
            if self._columns_or_query_mentions(res, ["payment"]):
                return True
        return "payment" in self._read_summary_text(result)

    def _check_tip_analysis(self, result: Dict) -> bool:
        for step in self._get_trace_steps(result):
            res = step.get("result", {}) or {}
            if self._columns_or_query_mentions(res, ["tip"]):
                return True
        return "tip" in self._read_summary_text(result)

    def _check_comparison_made(self, result: Dict) -> bool:
        """Comparison typically implies grouping or a comparative chart."""
        if any((step.get("action") == "aggregate" and (step.get("result", {}) or {}).get("group_by")) 
            for step in self._get_trace_steps(result)):
            return True
        # Or a chart type suggestive of comparison
        return self._any_visualization(result, ["bar", "scatter"])

    def _check_correlation_analysis(self, result: Dict) -> bool:
        """Correlation often signaled by scatter or heatmap (corr matrix)."""
        if self._any_visualization(result, ["scatter", "heatmap"]):
            return True
        return "correlation" in self._read_summary_text(result)

    def _check_relationship_analysis(self, result: Dict) -> bool:
        """Relationship → often a scatter or heatmap as well."""
        if self._any_visualization(result, ["scatter", "heatmap"]):
            return True
        return "relationship" in self._read_summary_text(result)

    def _check_scatter_plot(self, result: Dict) -> bool:
        return self._any_visualization(result, ["scatter"])

    
    def run_all_tasks(self, tasks_file: str = "evals/tasks.jsonl") -> Dict[str, Any]:
        """Run all evaluation tasks"""
        tasks = self.load_tasks(tasks_file)
        
        print(f"Running {len(tasks)} evaluation tasks...")
        
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] Running task {task['id']}: {task['question'][:60]}...")
            
            result = self.run_single_task(task)
            self.results.append(result)
            
            status_symbol = "✅" if result["success"] else "❌"
            print(f"  {status_symbol} {result['status']} ({result['duration']:.1f}s)")
            
            if not result["success"] and "error" in result:
                print(f"    Error: {result['error']}")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        self._print_metrics_table(metrics)
        
        return {
            "metrics": metrics,
            "results": self.results,
            "tasks_run": len(tasks),
            "timestamp": time.time()
        }
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        if not self.results:
            return {}
        
        success_count = sum(1 for r in self.results if r["success"])
        total_count = len(self.results)
        success_rate = success_count / total_count
        
        durations = [r["duration"] for r in self.results if "duration" in r]
        median_latency = sorted(durations)[len(durations)//2] if durations else 0
        avg_latency = sum(durations) / len(durations) if durations else 0
        
        # Count failures by reason
        failure_reasons = {}
        for result in self.results:
            if not result["success"]:
                reason = result.get("status", "unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            "success_rate": success_rate,
            "success_count": success_count,
            "total_count": total_count,
            "median_latency": median_latency,
            "avg_latency": avg_latency,
            "failure_reasons": failure_reasons
        }
    
    def _print_metrics_table(self, metrics: Dict[str, Any]) -> None:
        """Print metrics in a formatted table"""
        print(f"Success Rate:      {metrics['success_rate']:.1%} ({metrics['success_count']}/{metrics['total_count']})")
        print(f"Median Latency:    {metrics['median_latency']:.1f}s")
        print(f"Average Latency:   {metrics['avg_latency']:.1f}s")
        
        if metrics['failure_reasons']:
            print("\nFailure Breakdown:")
            for reason, count in metrics['failure_reasons'].items():
                print(f"  {reason}: {count}")
        
        print()

def main():
    """Main entry point for evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AI Data Analyst Agent evaluations")
    parser.add_argument("--tasks", default="evals/tasks.jsonl", help="Path to tasks file")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output", help="Output file for results JSON")
    
    args = parser.parse_args()
    
    # Initialize harness
    harness = EvaluationHarness(args.artifacts_dir, args.data_dir)
    
    # Run evaluations
    results = harness.run_all_tasks(args.tasks)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()