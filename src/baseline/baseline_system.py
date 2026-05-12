"""
src/baseline/baseline_system.py

Baseline system — evaluated on the SAME MultiDoc2Dial validation data
that the full trained system will be evaluated on in Week 5.

What baseline does:
  - BM25 retrieval (un-finetuned, no dense embeddings)
  - Direct generation (un-tuned Flan-T5-base via HuggingFace — no LoRA, no DPO)
  - No tool calling
  - No citation enforcement
  - No escalation logic

This establishes the floor numbers for all metrics.
The full system (Weeks 2-4) will be evaluated on the same test_cases.jsonl
to show quantitative improvement.

Run:
    python -m src.baseline.baseline_system --demo
    python -m src.baseline.baseline_system --eval
"""

import json
import os

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/transformers"
import time
import math
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


# ─── BM25 (no external library) ───────────────────────────────────────────────

class BM25:
    """Okapi BM25 over a passage list. k1=1.5, b=0.75."""

    def __init__(self, passages: List[Dict], k1=1.5, b=0.75):
        self.passages  = passages
        self.k1, self.b = k1, b
        self.N         = len(passages)
        self.tokenised = [self._tok(p["full_text"]) for p in passages]
        self.avg_dl    = sum(len(t) for t in self.tokenised) / max(self.N, 1)
        df: Counter    = Counter()
        for tokens in self.tokenised:
            df.update(set(tokens))
        self.idf = {
            w: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for w, f in df.items()
        }

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k=5) -> List[Dict]:
        q_terms = self._tok(query)
        scores  = []
        for idx, tokens in enumerate(self.tokenised):
            tf_map = Counter(tokens)
            dl     = len(tokens)
            score  = sum(
                self.idf.get(t, 0) *
                (tf_map.get(t, 0) * (self.k1 + 1)) /
                (tf_map.get(t, 0) + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                for t in q_terms
            )
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            p = dict(self.passages[idx])
            p["retrieval_score"] = round(score, 4)
            results.append(p)
        return results


# ─── Generator — un-tuned Flan-T5-base ────────────────────────────────────────────────────────────────────

def load_generator():
    """
    Skipping local model loading because we are using the Hugging Face API.
    """
    return None, None


def generate_with_hf_api(prompt: str) -> str:
    """Generate using Hugging Face Serverless Inference API for Llama-3-8B-Instruct."""
    try:
        from huggingface_hub import InferenceClient
        import os
        # Use environment variable for HF_TOKEN
        token = os.environ.get("HF_TOKEN")
        client = InferenceClient("google/flan-t5-base", token=os.environ.get("HF_TOKEN"))
        response = client.text_generation(prompt, max_new_tokens=150)
        return response.strip()
    except Exception as e:
        return f"[GENERATION ERROR: {e}]"


def generate_with_api(prompt: str) -> str:
    """Fallback: Anthropic API (un-tuned, raw generation — no system prompt rubric)."""
    try:
        import anthropic
        client   = anthropic.Anthropic()
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 200,
            messages   = [{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"[GENERATION ERROR: {e}]"


def build_baseline_prompt(query: str, passages: List[Dict], history: List[Dict]) -> str:
    """
    Simple RAG prompt — NO citation format enforcement, NO rubric,
    NO tool descriptions. This is deliberately bare.
    """
    ctx = "\n\n".join(
        f"[Doc {i+1}] {p['heading']}\n{p['text']}"
        for i, p in enumerate(passages[:3])
    )
    hist = ""
    if history:
        hist = "\nConversation:\n"
        for turn in history[-2:]:
            role = turn.get("role", turn.get("speaker", ""))
            utt  = turn.get("utterance", turn.get("text", ""))
            hist += f"  {role}: {utt}\n"

    return (
        f"Answer the customer question using the context below.\n"
        f"{hist}\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )


# ─── Baseline System ──────────────────────────────────────────────────────────

class BaselineSystem:
    """
    Retrieve-then-read baseline over the real MultiDoc2Dial KB.
    No tools. No citation enforcement. No preference alignment.
    """

    def __init__(self, kb_path: str = "data/processed/kb_passages.jsonl"):
        if not Path(kb_path).exists():
            raise FileNotFoundError(
                f"KB not found: {kb_path}\n"
                "Run: python -m src.ingestion.kb_builder first."
            )
        with open(kb_path) as f:
            self.passages = [json.loads(line) for line in f]

        self.retriever = BM25(self.passages)
        self.model, self.tokenizer = load_generator()

        total = len(self.passages)
        md2d  = sum(1 for p in self.passages if p["source"] == "multidoc2dial")
        overlay = total - md2d
        print(f"  [Baseline] KB: {total:,} passages "
              f"({md2d:,} MD2D + {overlay} telecom overlay). BM25 ready.")

    def run(self, query: str, history: Optional[List[Dict]] = None) -> Dict:
        start_ms = time.time() * 1000
        history  = history or []

        # Step 1: BM25 retrieval (un-tuned)
        retrieved = self.retriever.search(query, top_k=5)

        # Step 2: Generate with Hugging Face API
        prompt = build_baseline_prompt(query, retrieved, history)
        answer = generate_with_hf_api(prompt)

        return {
            "system":     "baseline",
            "query":      query,
            "answer":     answer,
            "citations":  [],          # baseline produces NO citations
            "tool_trace": [],          # baseline calls NO tools
            "escalated":  False,       # baseline NEVER escalates
            "ticket_id":  None,
            "confidence": None,
            "retrieved":  retrieved[:3],
            "latency_ms": round(time.time() * 1000 - start_ms, 1),
        }

    def batch_run(self, test_cases: List[Dict], max_workers: int = 10) -> List[Dict]:
        """Runs the baseline system on all test cases using parallel threads."""
        from concurrent.futures import ThreadPoolExecutor
        results = [None] * len(test_cases)
        
        def _process(idx):
            case = test_cases[idx]
            print(f"  [Baseline] [{idx+1:03d}/{len(test_cases)}] {case['query'][:50]}...")
            res = self.run(case["query"], case.get("history", []))
            res.update({
                "test_id":          case.get("test_id"),
                "gold_doc_id":      case.get("gold_doc_id"),
                "gold_section_id":  case.get("gold_section_id"),
                "gold_answer":      case.get("gold_answer"),
                "should_escalate":  case.get("should_escalate", False),
                "requires_outage_check": case.get("requires_outage_check", False),
                "domain":           case.get("domain", "unknown"),
                "source":           case.get("source", "unknown"),
            })
            results[idx] = res

        print(f"\n  [Baseline] Starting parallel batch run with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_process, range(len(test_cases))))
            
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick 5-query demo")
    parser.add_argument("--eval", action="store_true",
                        help="Run full evaluation on test_cases.jsonl")
    args = parser.parse_args()

    system = BaselineSystem()

    if args.demo or not args.eval:
        print("\n=== BASELINE DEMO ===")
        demo_queries = [
            "Can I renew my driver's license online?",
            "What documents do I need to apply for student aid?",
            "How do I report a change of address?",
            "My 4G is not working in Ahmedabad, is there an outage?",
            "I was charged roaming fees but never left India.",
        ]
        for q in demo_queries:
            r = system.run(q)
            print(f"\nQ: {q}")
            print(f"A: {r['answer'][:200]}")
            print(f"   Citations: {r['citations']} | Tools: {r['tool_trace']} | {r['latency_ms']}ms")

    if args.eval:
        test_path = Path("data/processed/test_cases.jsonl")
        if not test_path.exists():
            print("No test_cases.jsonl found. Run training_data_builder first.")
        else:
            with open(test_path) as f:
                cases = [json.loads(line) for line in f]
            print(f"\nRunning baseline on {len(cases)} test cases...")
            results = system.batch_run(cases)

            out = Path("data/processed/baseline_results.jsonl")
            with open(out, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"Saved {len(results)} results -> {out}")
