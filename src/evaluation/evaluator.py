"""
src/evaluation/evaluator.py

Shared evaluation harness for the Telecom Copilot RAG system.
Computes a comprehensive suite of metrics across Retrieval, Generation, DPO, and Serving.

Metrics:
  - Retrieval: Recall@1, Recall@5, MRR@10
  - Generation: Citation Recall@1, BERTScore F1, ROUGE-L, Groundedness Score
  - RAG: Hallucination Rate (1 - Groundedness)
  - DPO: Win Rate, Reward Margin
  - Novel: GEA (Grounded Escalation Accuracy), OARR (Outage-Aware Response Rate)
  - Serving: Avg Latency, P95 Latency
"""

import json
import argparse
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

# ─── Metric Helpers ───────────────────────────────────────────────────────────

def avg(lst):
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 4) if lst else None

# ─── Retrieval Metrics ────────────────────────────────────────────────────────

def retrieval_recall_at_k(result: Dict, k: int = 1) -> Optional[float]:
    """1.0 if gold_section_id in top-k retrieved results."""
    gold_id = result.get("gold_section_id")
    if not gold_id:
        return None
    
    retrieved = result.get("retrieved", [])
    top_k_ids = [p.get("section_id") for p in retrieved[:k]]
    return 1.0 if gold_id in top_k_ids else 0.0

def mrr_at_10(result: Dict) -> Optional[float]:
    """Mean Reciprocal Rank of the gold_section_id within top-10."""
    gold_id = result.get("gold_section_id")
    if not gold_id:
        return None
    
    retrieved = result.get("retrieved", [])
    for i, p in enumerate(retrieved[:10]):
        if p.get("section_id") == gold_id:
            return 1.0 / (i + 1)
    return 0.0

# ─── Generation Metrics ───────────────────────────────────────────────────────

def citation_recall_at_1(result: Dict) -> Optional[float]:
    """Check if gold_doc_id is correctly cited."""
    gold_doc_id = result.get("gold_doc_id")
    if not gold_doc_id:
        return None

    # Check structured citations
    for c in result.get("citations", []):
        if c.get("doc_id", "") == gold_doc_id:
            return 1.0

    # Weak fallback: check text mention
    if gold_doc_id in result.get("answer", ""):
        return 0.5

    return 0.0

def rouge_l_score(result: Dict) -> float:
    """ROUGE-L recall vs gold answer."""
    from rouge_score import rouge_scorer
    gold = result.get("gold_answer", "")
    pred = result.get("answer", "")
    if not gold or not pred:
        return 0.0
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gold, pred)
    return round(scores['rougeL'].recall, 4)

def bertscore_f1_batch(results: List[Dict]) -> List[float]:
    """BERTScore F1 for a batch of results."""
    from bert_score import score
    
    preds = [r.get("answer", "") for r in results]
    golds = [r.get("gold_answer", "") for r in results]
    
    # Filter out empty pairs
    valid_indices = [i for i, (p, g) in enumerate(zip(preds, golds)) if p and g]
    if not valid_indices:
        return [0.0] * len(results)
    
    valid_preds = [preds[i] for i in valid_indices]
    valid_golds = [golds[i] for i in valid_indices]
    
    P, R, F1 = score(valid_preds, valid_golds, lang="en", verbose=False, model_type="distilbert-base-uncased")
    
    scores = [0.0] * len(results)
    for i, idx in enumerate(valid_indices):
        scores[idx] = round(float(F1[i]), 4)
    return scores

def groundedness_score(result: Dict, nli_model=None) -> float:
    """NLI-based groundedness: is answer entailed by context?"""
    if not result.get("retrieved") or not result.get("answer"):
        return 0.0
    
    context = " ".join([p.get("text", "") for p in result["retrieved"][:3]])
    answer = result["answer"]
    
    if nli_model is None:
        from sentence_transformers import CrossEncoder
        nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    
    # Label mapping for nli-deberta-v3-small: 0: contradiction, 1: neutral, 2: entailment
    scores = nli_model.predict([(context, answer)])
    entailment_prob = np.exp(scores[0]) / np.sum(np.exp(scores[0]))
    return round(float(entailment_prob[2]), 4)

# ─── Novel Telecom Metrics ───────────────────────────────────────────────────

def grounded_escalation_accuracy(result: Dict) -> Optional[float]:
    gold = result.get("should_escalate")
    pred = result.get("escalated", False)
    if gold is None:
        return None
    return 1.0 if (gold == pred) else 0.0

def outage_aware_response_rate(result: Dict) -> Optional[float]:
    if not result.get("requires_outage_check", False):
        return None
    tool_trace = result.get("tool_trace", [])
    tools = [t.get("tool", t) if isinstance(t, dict) else t for t in tool_trace]
    return 1.0 if "CheckNetworkStatus" in tools else 0.0

# ─── DPO Metrics ──────────────────────────────────────────────────────────────

def reward_margin(result: Dict) -> Optional[float]:
    """log π(chosen)/π(rejected) - usually provided by the DPO trainer/eval script."""
    # Look for 'reward_margin' or compute from logps if available
    return result.get("reward_margin")

def win_rate_llm_judge(results: List[Dict], baseline_results: List[Dict]) -> float:
    """Compare two systems using LLM-as-judge via HF API."""
    from huggingface_hub import InferenceClient
    import os
    
    client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=os.environ.get("HF_TOKEN"))
    
    wins = 0
    total = 0
    
    # Compare top 50 samples for speed/cost
    samples = min(50, len(results))
    for i in range(samples):
        r_full = results[i]
        r_base = next((b for b in baseline_results if b.get("test_id") == r_full.get("test_id")), None)
        
        if not r_base:
            continue
            
        prompt = (
            "You are an impartial judge evaluating two AI assistant answers.\n"
            f"Question: {r_full['query']}\n"
            f"Ground Truth: {r_full.get('gold_answer', 'N/A')}\n\n"
            f"Answer A: {r_base['answer']}\n\n"
            f"Answer B: {r_full['answer']}\n\n"
            "Which answer is better? Respond with only 'A' or 'B'."
        )
        
        try:
            resp = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=2)
            judge_choice = resp.choices[0].message.content.strip().upper()
            if 'B' in judge_choice:
                wins += 1
            total += 1
        except:
            continue
            
    return round(wins / total, 4) if total > 0 else 0.0

# ─── Aggregate Evaluation ─────────────────────────────────────────────────────

def evaluate(results: List[Dict], label: str = "system", baseline_results: List[Dict] = None) -> Dict:
    print(f"  [Evaluator] Computing metrics for '{label}'...")
    
    metrics = {
        "label": label,
        "n": len(results),
        # Retrieval
        "recall_at_1": avg([retrieval_recall_at_k(r, 1) for r in results]),
        "recall_at_5": avg([retrieval_recall_at_k(r, 5) for r in results]),
        "mrr_at_10": avg([mrr_at_10(r) for r in results]),
        # Generation
        "citation_recall_at_1": avg([citation_recall_at_1(r) for r in results]),
        "rouge_l": avg([rouge_l_score(r) for r in results]),
        # DPO
        "reward_margin": avg([reward_margin(r) for r in results]),
        # Novel
        "gea": avg([grounded_escalation_accuracy(r) for r in results]),
        "oarr": avg([outage_aware_response_rate(r) for r in results]),
        # Serving
        "avg_latency_ms": avg([r.get("latency_ms", 0) for r in results]),
    }
    
    # Batch metrics
    print("  [Evaluator] Computing BERTScore...")
    metrics["bertscore_f1"] = avg(bertscore_f1_batch(results))
    
    print("  [Evaluator] Computing Groundedness...")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    g_scores = [groundedness_score(r, nli_model) for r in results]
    metrics["groundedness"] = avg(g_scores)
    metrics["hallucination_rate"] = round(1.0 - metrics["groundedness"], 4)
    
    if label != "baseline" and baseline_results:
        print("  [Evaluator] Computing Win Rate (LLM-as-judge)...")
        metrics["win_rate"] = win_rate_llm_judge(results, baseline_results)
    
    # Latency P95
    lats = sorted([r.get("latency_ms", 0) for r in results])
    metrics["p95_latency_ms"] = lats[int(len(lats)*0.95)] if lats else 0
    
    return metrics

def evaluate_retriever(retriever, test_cases: List[Dict]) -> Dict:
    """Evaluates a retriever object in isolation."""
    print(f"  [Evaluator] evaluating retriever isolation...")
    results = []
    for case in test_cases:
        query = case["query"]
        t0 = time.time()
        hits = retriever.search(query, top_k=10)
        latency = (time.time() - t0) * 1000
        results.append({
            "gold_section_id": case.get("gold_section_id"),
            "retrieved": hits,
            "latency_ms": latency
        })
    
    return {
        "recall_at_1": avg([retrieval_recall_at_k(r, 1) for r in results]),
        "recall_at_5": avg([retrieval_recall_at_k(r, 5) for r in results]),
        "mrr_at_10": avg([mrr_at_10(r) for r in results]),
        "avg_latency_ms": avg([r["latency_ms"] for r in results])
    }

# ─── Printing & Comparison ────────────────────────────────────────────────────

def print_report(m: Dict):
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT: {m['label'].upper()}")
    print(f"  n = {m['n']} test cases")
    print(f"{'='*70}")
    
    groups = [
        ("Retrieval", ["recall_at_1", "recall_at_5", "mrr_at_10"]),
        ("Generation", ["citation_recall_at_1", "bertscore_f1", "rouge_l", "groundedness"]),
        ("RAG", ["hallucination_rate"]),
        ("DPO", ["win_rate", "reward_margin"]),
        ("Novel", ["gea", "oarr"]),
        ("Serving", ["avg_latency_ms", "p95_latency_ms"])
    ]
    
    for group_name, keys in groups:
        print(f"\n  [{group_name}]")
        for k in keys:
            val = m.get(k)
            if val is None: continue
            print(f"    {k:<25} {val:>10.4f}" if isinstance(val, float) else f"    {k:<25} {val:>10}")
    print(f"\n{'='*70}\n")

def compare_reports(b: Dict, f: Dict):
    print(f"\n{'='*85}")
    print(f"  COMPARISON: BASELINE vs FULL SYSTEM")
    print(f"{'='*85}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Full Sys':>12} {'Delta':>12} {'%':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    all_keys = [
        "recall_at_1", "recall_at_5", "mrr_at_10", 
        "citation_recall_at_1", "bertscore_f1", "rouge_l", "groundedness", 
        "hallucination_rate", "reward_margin", "gea", "oarr", 
        "avg_latency_ms", "p95_latency_ms"
    ]
                
    for k in all_keys:
        bv = b.get(k)
        fv = f.get(k)
        
        # Handle None values
        bv = bv if bv is not None else 0.0
        fv = fv if fv is not None else 0.0
        
        delta = fv - bv
        pct = (delta / bv * 100) if bv != 0 else 0.0
        arrow = "+" if delta > 0 else "-" if delta < 0 else " "
        print(f"  {k:<25} {bv:>12.4f} {fv:>12.4f} {arrow}{abs(delta):>11.4f} {arrow}{abs(pct):>8.1f}%")
    
    print(f"{'='*85}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str)
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    
    if args.compare and args.results and args.baseline:
        with open(args.results) as f:
            res = [json.loads(l) for l in f]
        with open(args.baseline) as f:
            base = [json.loads(l) for l in f]
            
        b_metrics = evaluate(base, "baseline")
        f_metrics = evaluate(res, "full_system", baseline_results=base)
        
        print_report(b_metrics)
        print_report(f_metrics)
        compare_reports(b_metrics, f_metrics)
    elif args.results:
        with open(args.results) as f:
            res = [json.loads(l) for l in f]
        metrics = evaluate(res, "system")
        print_report(metrics)
