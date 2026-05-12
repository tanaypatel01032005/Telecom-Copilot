import json
import argparse
import re
import time
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

# ─── Heavy Imports (with error handling) ──────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn
    from sentence_transformers import CrossEncoder
    from huggingface_hub import InferenceClient
except ImportError as e:
    print(f"\n[!] WARNING: Missing dependencies for evaluation: {e}")
    print("    Metrics like ROUGE, BERTScore, and Groundedness will be skipped.")
    print("    To fix, run: .\\.venv\\Scripts\\pip install -r requirements.txt\n")

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
    try:
        gold = result.get("gold_answer", "")
        pred = result.get("answer", "")
        if not gold or not pred:
            return 0.0
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(gold, pred)
        return round(scores['rougeL'].recall, 4)
    except:
        return 0.0

def bertscore_f1_batch(results: List[Dict]) -> List[float]:
    """BERTScore F1 for a batch of results."""
    try:
        preds = [r.get("answer", "") for r in results]
        golds = [r.get("gold_answer", "") for r in results]
        
        valid_indices = [i for i, (p, g) in enumerate(zip(preds, golds)) if p and g]
        if not valid_indices:
            return [0.0] * len(results)
        
        valid_preds = [preds[i] for i in valid_indices]
        valid_golds = [golds[i] for i in valid_indices]
        
        P, R, F1 = bert_score_fn(valid_preds, valid_golds, lang="en", verbose=False, model_type="distilbert-base-uncased")
        
        scores = [0.0] * len(results)
        for i, idx in enumerate(valid_indices):
            scores[idx] = round(float(F1[i]), 4)
        return scores
    except:
        return [0.0] * len(results)

def groundedness_score_batch(results: List[Dict], nli_model: CrossEncoder) -> List[float]:
    """NLI-based groundedness: is answer entailed by context? (Batched)"""
    pairs = []
    valid_indices = []
    
    for i, r in enumerate(results):
        if r.get("retrieved") and r.get("answer"):
            context = " ".join([p.get("text", "") for p in r["retrieved"][:3]])
            answer = r["answer"]
            pairs.append((context, answer))
            valid_indices.append(i)
    
    scores_out = [0.0] * len(results)
    if not pairs:
        return scores_out
        
    # Batch predict
    logits = nli_model.predict(pairs, batch_size=16)
    
    # Softmax to get entailment prob (index 2 for nli-deberta-v3-small)
    for i, idx in enumerate(valid_indices):
        l = logits[i]
        probs = np.exp(l) / np.sum(np.exp(l))
        scores_out[idx] = round(float(probs[2]), 4)
        
    return scores_out

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
    return result.get("reward_margin")

def win_rate_llm_judge(results: List[Dict], baseline_results: List[Dict]) -> float:
    """Compare two systems using LLM-as-judge via HF API."""
    try:
        client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=os.environ.get("HF_TOKEN"))
        wins, total = 0, 0
        samples = min(50, len(results))
        
        for i in range(samples):
            r_full = results[i]
            r_base = next((b for b in baseline_results if b.get("test_id") == r_full.get("test_id")), None)
            if not r_base: continue
                
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
                if 'B' in judge_choice: wins += 1
                total += 1
            except: continue
        return round(wins / total, 4) if total > 0 else 0.0
    except:
        return 0.0

# ─── Aggregate Evaluation ─────────────────────────────────────────────────────

def evaluate(results: List[Dict], label: str = "system", baseline_results: List[Dict] = None) -> Dict:
    print(f"  [Evaluator] Computing metrics for '{label}' (n={len(results)})...")
    
    metrics = {
        "label": label,
        "n": len(results),
        "recall_at_1": avg([retrieval_recall_at_k(r, 1) for r in results]),
        "recall_at_5": avg([retrieval_recall_at_k(r, 5) for r in results]),
        "mrr_at_10": avg([mrr_at_10(r) for r in results]),
        "citation_recall_at_1": avg([citation_recall_at_1(r) for r in results]),
        "rouge_l": avg([rouge_l_score(r) for r in results]),
        "reward_margin": avg([reward_margin(r) for r in results]),
        "gea": avg([grounded_escalation_accuracy(r) for r in results]),
        "oarr": avg([outage_aware_response_rate(r) for r in results]),
        "avg_latency_ms": avg([r.get("latency_ms", 0) for r in results]),
    }
    
    # BERTScore (Batched)
    if 'bert_score_fn' in globals():
        print("  [Evaluator] Computing BERTScore (batched)...")
        metrics["bertscore_f1"] = avg(bertscore_f1_batch(results))
    
    # Groundedness (Batched)
    if 'CrossEncoder' in globals():
        print("  [Evaluator] Computing Groundedness (batched NLI)...")
        nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        g_scores = groundedness_score_batch(results, nli_model)
        metrics["groundedness"] = avg(g_scores)
        metrics["hallucination_rate"] = round(1.0 - metrics["groundedness"], 4) if metrics["groundedness"] else 0.0
    
    # Win Rate
    if label != "baseline" and baseline_results and 'InferenceClient' in globals():
        print("  [Evaluator] Computing Win Rate (LLM-as-judge)...")
        metrics["win_rate"] = win_rate_llm_judge(results, baseline_results)
    
    # Latency P95
    lats = sorted([r.get("latency_ms", 0) for r in results])
    metrics["p95_latency_ms"] = lats[int(len(lats)*0.95)] if lats else 0
    
    return metrics

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
        bv = b.get(k) if b.get(k) is not None else 0.0
        fv = f.get(k) if f.get(k) is not None else 0.0
        delta = fv - bv
        pct = (delta / bv * 100) if bv != 0 else 0.0
        arrow = "+" if delta > 0 else "-" if delta < 0 else " "
        print(f"  {k:<25} {bv:>12.4f} {fv:>12.4f} {arrow}{abs(delta):>11.4f} {arrow}{abs(pct):>8.1f}%")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    # Auto-detect project root (src/evaluation/evaluator.py -> parents[2] is root)
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "evaluation" else SCRIPT_DIR.parent
    
    DEFAULT_RESULTS = BASE_DIR / "data" / "processed" / "full_system_results.jsonl"
    DEFAULT_BASE = BASE_DIR / "data" / "processed" / "baseline_results.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--baseline", type=str, default=str(DEFAULT_BASE))
    parser.add_argument("--compare", action="store_true", default=True)
    args = parser.parse_args()
    
    path_res = Path(args.results)
    path_base = Path(args.baseline)

    if args.compare and path_res.exists() and path_base.exists():
        print(f"\n[Evaluator] Auto-comparing:\n    {path_res}\n    vs\n    {path_base}\n")
        with open(path_res) as f: res = [json.loads(l) for l in f]
        with open(path_base) as f: base = [json.loads(l) for l in f]
        b_metrics = evaluate(base, "baseline")
        f_metrics = evaluate(res, "full_system", baseline_results=base)
        print_report(b_metrics)
        print_report(f_metrics)
        compare_reports(b_metrics, f_metrics)
    elif path_res.exists():
        print(f"\n[Evaluator] Evaluating {path_res} (no baseline found for comparison)...")
        with open(path_res) as f: res = [json.loads(l) for l in f]
        metrics = evaluate(res, "system")
        print_report(metrics)
    else:
        print(f"\n[!] Error: Results file not found at {path_res}")
        print("    Ensure you have run the baseline and full system pipelines first.")
        print(f"    Looked in: {path_res.parent}")
