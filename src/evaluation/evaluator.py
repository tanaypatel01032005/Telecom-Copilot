"""
src/evaluation/evaluator.py

Shared evaluation harness used for BOTH:
  - Week 1: baseline_results.jsonl
  - Week 5: full_system_results.jsonl

Running the exact same metrics on the exact same test_cases.jsonl
(built from MultiDoc2Dial validation split) guarantees a fair comparison.

Metrics computed:
  1. Citation Recall@1        (grounding — REQUIRED by spec)
  2. Answer Coverage Score    (token-level ROUGE-1 recall vs gold answer)
  3. Grounded Escalation Accuracy / GEA  (novel telecom metric)
  4. Outage-Aware Response Rate / OARR   (novel telecom metric)
  5. Avg / P95 Latency (ms)

Run:
    python -m src.evaluation.evaluator --results data/processed/baseline_results.jsonl
    python -m src.evaluation.evaluator --results data/processed/full_system_results.jsonl
    python -m src.evaluation.evaluator --compare  (side-by-side table)
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional


# ─── Metric 1: Citation Recall@1 (grounding) ─────────────────────────────────

def citation_recall_at_1(result: Dict) -> Optional[float]:
    """
    1.0 if the gold doc_id is referenced in either:
      (a) result["citations"] list (full system always produces these)
      (b) the answer text itself (weak proxy for baseline)
    Returns None for out-of-scope queries (no gold_doc_id).
    """
    gold_doc_id = result.get("gold_doc_id")
    if not gold_doc_id:
        return None   # out-of-scope → skip

    # (a) Structured citations (full system)
    for c in result.get("citations", []):
        if c.get("doc_id", "") == gold_doc_id:
            return 1.0

    # (b) Raw text mention (very weak — baseline almost never scores)
    if gold_doc_id in result.get("answer", ""):
        return 0.5   # partial credit for accidental text match

    return 0.0


# ─── Metric 2: Answer Coverage Score ─────────────────────────────────────────

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","can",
    "to","of","in","on","at","by","for","with","from","that","this",
    "it","or","and","if","you","your","i","we","our","they","their",
    "hi","hello","yes","no","please","thank","thanks","okay","ok",
}

def answer_coverage_score(result: Dict) -> float:
    """
    Token-level recall: fraction of content words in the gold answer
    that appear in the system's answer. A simple ROUGE-1 recall proxy.
    """
    gold   = result.get("gold_answer", "")
    answer = result.get("answer", "")
    if not gold or not answer:
        return 0.0

    def content_tokens(text):
        return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS

    gold_toks   = content_tokens(gold)
    answer_toks = content_tokens(answer)
    if not gold_toks:
        return 1.0
    return round(len(gold_toks & answer_toks) / len(gold_toks), 4)


# ─── Metric 3: Grounded Escalation Accuracy (GEA) ────────────────────────────

def grounded_escalation_accuracy(result: Dict) -> Optional[float]:
    """
    Novel telecom metric.
    Only evaluated on cases where should_escalate is defined.
    1.0 if escalation decision matches gold. 0.0 otherwise.

    Baseline always scores 0 on cases that should_escalate=True
    (it never escalates), and 1.0 on cases where should_escalate=False
    (it correctly does not escalate — but for the wrong reason).
    """
    gold = result.get("should_escalate")
    pred = result.get("escalated", False)
    if gold is None:
        return None
    return 1.0 if (gold == pred) else 0.0


# ─── Metric 4: Outage-Aware Response Rate (OARR) ─────────────────────────────

def outage_aware_response_rate(result: Dict) -> Optional[float]:
    """
    Novel telecom metric.
    Only evaluated on cases where requires_outage_check=True.
    1.0 if CheckNetworkStatus appears in tool_trace. 0.0 otherwise.

    Baseline always scores 0.0 (no tools).
    """
    if not result.get("requires_outage_check", False):
        return None   # not applicable

    tool_trace = result.get("tool_trace", [])
    tools = [
        t.get("tool", t) if isinstance(t, dict) else t
        for t in tool_trace
    ]
    return 1.0 if "CheckNetworkStatus" in tools else 0.0


# ─── Aggregate ────────────────────────────────────────────────────────────────

def avg(lst):
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 4) if lst else None


def evaluate(results: List[Dict], label: str = "system") -> Dict:
    cr_scores    = []
    cov_scores   = []
    gea_scores   = []
    oarr_scores  = []
    latencies    = []

    for r in results:
        cr_scores.append(citation_recall_at_1(r))
        cov_scores.append(answer_coverage_score(r))
        gea_scores.append(grounded_escalation_accuracy(r))
        oarr_scores.append(outage_aware_response_rate(r))
        latencies.append(r.get("latency_ms", 0))

    lat_sorted = sorted(l for l in latencies if l)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else None

    return {
        "label":                         label,
        "n":                             len(results),
        "citation_recall_at_1":          avg(cr_scores),
        "answer_coverage_score":         avg(cov_scores),
        "grounded_escalation_accuracy":  avg(gea_scores),
        "outage_aware_response_rate":    avg(oarr_scores),
        "avg_latency_ms":                avg(latencies),
        "p95_latency_ms":                round(p95, 1) if p95 else None,
        # Per-domain breakdown
        "by_domain": _by_domain(results),
        "by_source": _by_source(results),
    }


def _by_domain(results: List[Dict]) -> Dict:
    from collections import defaultdict
    domains = defaultdict(list)
    for r in results:
        domains[r.get("domain", "unknown")].append(r)
    return {
        dom: {
            "n":                    len(rs),
            "citation_recall":      avg([citation_recall_at_1(r) for r in rs]),
            "answer_coverage":      avg([answer_coverage_score(r) for r in rs]),
        }
        for dom, rs in domains.items()
    }


def _by_source(results: List[Dict]) -> Dict:
    from collections import defaultdict
    sources = defaultdict(list)
    for r in results:
        sources[r.get("source", "unknown")].append(r)
    return {
        src: {"n": len(rs)}
        for src, rs in sources.items()
    }


def print_report(metrics: Dict):
    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT — {metrics['label'].upper()}")
    print(f"  n = {metrics['n']} test cases")
    print(f"{'='*65}")
    rows = [
        ("Citation Recall@1",                  "grounding ★", metrics["citation_recall_at_1"]),
        ("Answer Coverage Score (ROUGE-1 R)",   "",            metrics["answer_coverage_score"]),
        ("Grounded Escalation Accuracy (GEA)",  "novel",       metrics["grounded_escalation_accuracy"]),
        ("Outage-Aware Response Rate (OARR)",   "novel",       metrics["outage_aware_response_rate"]),
        ("Avg Latency (ms)",                    "",            metrics["avg_latency_ms"]),
        ("P95 Latency (ms)",                    "",            metrics["p95_latency_ms"]),
    ]
    for name, tag, val in rows:
        tag_str = f"[{tag}]" if tag else ""
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {name:<42} {tag_str:<14} {val_str:>8}")
    print(f"{'='*65}")

    if metrics.get("by_domain"):
        print(f"\n  Per-domain Citation Recall:")
        for dom, stats in sorted(metrics["by_domain"].items()):
            cr = stats.get("citation_recall")
            cr_str = f"{cr:.4f}" if cr is not None else "  N/A "
            print(f"    {dom:<20} n={stats['n']:<5}  citation_recall={cr_str}")


def compare_reports(baseline: Dict, full: Dict):
    print(f"\n{'='*75}")
    print(f"  COMPARISON: BASELINE vs FULL SYSTEM")
    print(f"{'='*75}")
    print(f"  {'Metric':<42} {'Baseline':>10} {'Full Sys':>10} {'Delta':>8} {'%':>6}")
    print(f"  {'-'*42} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")

    keys = [
        ("Citation Recall@1",                 "citation_recall_at_1"),
        ("Answer Coverage Score",             "answer_coverage_score"),
        ("Grounded Escalation Accuracy",      "grounded_escalation_accuracy"),
        ("Outage-Aware Response Rate",        "outage_aware_response_rate"),
    ]
    for name, key in keys:
        b = baseline.get(key)
        f = full.get(key)
        if b is None or f is None:
            print(f"  {name:<42} {'N/A':>10} {'N/A':>10}")
            continue
        delta = f - b
        pct   = (delta / b * 100) if b > 0 else float("inf")
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {name:<42} {b:>10.4f} {f:>10.4f} "
              f"{arrow}{abs(delta):>6.4f} {arrow}{abs(pct):>4.0f}%")
    print(f"{'='*75}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        help="Path to results JSONL file to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs full system results")
    parser.add_argument("--label",   type=str, default="system")
    args = parser.parse_args()

    if args.compare:
        b_path = Path("data/processed/baseline_results.jsonl")
        f_path = Path("data/processed/full_system_results.jsonl")
        if not b_path.exists() or not f_path.exists():
            print("Need both baseline_results.jsonl and full_system_results.jsonl")
        else:
            with open(b_path) as f:
                b_results = [json.loads(l) for l in f]
            with open(f_path) as f:
                f_results = [json.loads(l) for l in f]
            b_metrics = evaluate(b_results, "baseline")
            f_metrics = evaluate(f_results, "full_system")
            print_report(b_metrics)
            print_report(f_metrics)
            compare_reports(b_metrics, f_metrics)

    elif args.results:
        with open(args.results) as f:
            results = [json.loads(l) for l in f]
        metrics = evaluate(results, args.label)
        print_report(metrics)

        out = Path(args.results).with_suffix(".eval.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n  Saved metrics → {out}")

    else:
        parser.print_help()
