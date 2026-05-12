"""
src/evaluation/run_evaluation.py

Main entry point for running end-to-end evaluation.
Supports:
  - Baseline evaluation (BM25 + Flan-T5)
  - Full system evaluation (BGE + Reranker + Tools + Llama 3)
  - Side-by-side comparison

Usage:
    python -m src.evaluation.run_evaluation --system baseline
    python -m src.evaluation.run_evaluation --system full
    python -m src.evaluation.run_evaluation --system both
"""

import json
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluator import evaluate, print_report, compare_reports

def load_test_cases(path: str = "data/processed/test_cases.jsonl"):
    if not Path(path).exists():
        print(f"Error: {path} not found.")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def run_baseline(test_cases):
    from src.baseline.baseline_system import BaselineSystem
    print("\n" + "="*60)
    print("  RUNNING BASELINE EVALUATION")
    print("="*60)
    system = BaselineSystem()
    results = system.batch_run(test_cases)
    
    out_path = Path("data/processed/baseline_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  Baseline results saved to {out_path}")
    return results

def run_full(test_cases):
    from src.pipeline.inference_pipeline import TelecomCopilot
    print("\n" + "="*60)
    print("  RUNNING FULL SYSTEM EVALUATION")
    print("="*60)
    system = TelecomCopilot()
    results = system.batch_run(test_cases)
    
    out_path = Path("data/processed/full_system_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  Full system results saved to {out_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, choices=["baseline", "full", "both"], default="both")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--test-path", type=str, default="data/processed/test_cases.jsonl",
                        help="Path to test cases JSONL")
    args = parser.parse_args()
    
    test_cases = load_test_cases(args.test_path)
    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"  [Info] Limited to {args.limit} test cases.")
        
    if not test_cases:
        sys.exit(1)
        
    baseline_results = None
    full_results = None
    
    if args.system in ["baseline", "both"]:
        baseline_results = run_baseline(test_cases)
        
    if args.system in ["full", "both"]:
        full_results = run_full(test_cases)
        
    # Final Reporting
    if args.system == "baseline":
        metrics = evaluate(baseline_results, label="baseline")
        print_report(metrics)
    elif args.system == "full":
        # Need baseline results for win rate if available
        b_path = Path("data/processed/baseline_results.jsonl")
        if b_path.exists():
            with open(b_path) as f:
                baseline_results = [json.loads(l) for l in f]
        
        metrics = evaluate(full_results, label="full_system", baseline_results=baseline_results)
        print_report(metrics)
    elif args.system == "both":
        b_metrics = evaluate(baseline_results, label="baseline")
        f_metrics = evaluate(full_results, label="full_system", baseline_results=baseline_results)
        
        print_report(b_metrics)
        print_report(f_metrics)
        compare_reports(b_metrics, f_metrics)
