"""
src/pipeline/inference_pipeline.py

Week 3 — Full Inference Pipeline (ReAct-style).

Connects all trained components:
  Retriever (fine-tuned, Week 2)
  Reranker  (fine-tuned, Week 2)
  Tools     (SearchKB, GetPolicy, CreateTicket, CheckNetworkStatus)
  Generator (DoRA fine-tuned, Week 3)

Control flow (ReAct-inspired, not copied — our own schema + routing):
  1. Tool Policy: classify query → which tool(s) to call
  2. Tool execution loop: call tools, collect evidence
  3. Escalation check: decide if KB is sufficient
  4. Generator: produce cited answer from evidence
  5. Post-process: parse citations, format response

Difference from baseline:
  Baseline: BM25 → un-tuned T5 → no citations, no tools, no escalation
  Full sys:  Dense+Reranker → DoRA T5 → structured citations + tool calls + escalation

Run:
    python -m src.pipeline.inference_pipeline --demo
    python -m src.pipeline.inference_pipeline --eval
"""
import os

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import time
import re
import argparse
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
from typing import Dict, List, Optional, Tuple

# ─── Tool policy (rule-based, upgraded to classifier in Week 4) ──────────────

def tool_policy(
    query:   str,
    history: List[Dict],
) -> List[Tuple[str, Dict]]:
    """
    Decides which tools to call and with what params.

    Week 3: rule-based (keyword + pattern matching).
    Week 4: replaced by a fine-tuned BERT classifier.

    Returns list of (tool_name, params) tuples in call order.
    Multiple tools can be called sequentially.

    Design decisions:
      - SearchKB is ALWAYS first (grounding before anything else)
      - CheckNetworkStatus is added for network-flavour queries
      - GetPolicy is added when a specific section is referenced
      - CreateTicket is added when escalation triggers fire
    """
    q = query.lower()

    calls = []

    # ── Always: SearchKB ──────────────────────────────────────────
    # Infer category from keywords for better retrieval precision
    category = "any"
    if any(w in q for w in ["bill","charge","invoice","payment","dispute","refund","autopay"]):
        category = "billing"
    elif any(w in q for w in ["signal","4g","5g","network","outage","slow","speed","data"]):
        category = "network"
    elif any(w in q for w in ["roam","international","abroad","travel","country"]):
        category = "roaming"
    elif any(w in q for w in ["plan","recharge","prepaid","postpaid","upgrade","downgrade"]):
        category = "plans"
    elif any(w in q for w in ["sim","esim","device","phone","handset","return","imei"]):
        category = "device"
    elif any(w in q for w in ["account","kyc","port","aadhaar","password","otp","hack"]):
        category = "account"
    elif any(w in q for w in ["escalat","trai","nodal","complaint","unresolved","supervisor"]):
        category = "escalation"

    calls.append(("SearchKB", {"query": query, "category_filter": category, "top_k": 5}))

    # ── Network queries: also check live outage status ─────────────
    network_kw = ["signal","4g","5g","network","outage","down","not working",
                  "slow internet","no service","coverage"]
    if any(kw in q for kw in network_kw):
        # Try to extract region from query
        region = _extract_region(query)
        svc    = "5G" if "5g" in q else "4G" if "4g" in q else "all"
        calls.append(("CheckNetworkStatus", {"region": region, "service_type": svc}))

    # ── Policy queries: add GetPolicy for specific section lookup ──
    policy_kw = ["policy","rule","regulation","what happens","how long","deadline",
                 "eligible","entitle","trai","compensation","days"]
    if any(kw in q for kw in policy_kw):
        # GetPolicy will be called AFTER SearchKB returns a section_id
        calls.append(("GetPolicy", {"_deferred": True}))

    return calls


def _extract_region(query: str) -> str:
    """
    Attempts to extract a city/region from the query.
    Falls back to 'Unknown' if nothing found.
    """
    cities = [
        "mumbai","delhi","bangalore","bengaluru","hyderabad","chennai",
        "kolkata","pune","ahmedabad","surat","jaipur","lucknow","noida",
        "gurgaon","gurugram","chandigarh","indore","bhopal","patna",
    ]
    q_lower = query.lower()
    for city in cities:
        if city in q_lower:
            return city.capitalize()
    # Look for pincode pattern
    match = re.search(r"\b[1-9][0-9]{5}\b", query)
    if match:
        return match.group(0)
    return "Unknown"


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class TelecomCopilot:
    """
    Full inference pipeline. Instantiated once, handles many queries.

    Component loading is lazy — if a checkpoint doesn't exist (e.g. running
    before Week 2/3 training), falls back gracefully to simpler components.
    """

    def __init__(
        self,
        BASE_DIR = Path(__file__).resolve().parents[2],
        retriever_path:  str = str(BASE_DIR / "checkpoints/retriever"),
        reranker_path:   str = str(BASE_DIR / "checkpoints/reranker"),
        generator_path:  str = str(BASE_DIR / "checkpoints/generator"),
        index_dir:       str = str(BASE_DIR / "data/index"),
        kb_path:         str = str(BASE_DIR / "data/processed/kb_passages.jsonl"),
        span_index_path: str = str(BASE_DIR / "data/processed/span_index.json")):
        import sys
        sys.path.insert(0, ".")

        print("\n" + "="*55)
        print("  TELECOM COPILOT — LOADING COMPONENTS")
        print("="*55)

        # Tools (always available)
        from src.tools.tool_executor import ToolExecutor, seed_network_status_feed
        seed_network_status_feed()
        self.tools = ToolExecutor(
            retriever_path  = retriever_path,
            reranker_path   = reranker_path,
            index_dir       = index_dir,
            kb_path         = kb_path,
            span_index_path = span_index_path,
        )

        # Generator (DoRA fine-tuned)
        self.generator = None
        try:
            from src.generation.train_generator import Generator
            self.generator = Generator(generator_path)
            print("  [Pipeline] DoRA generator loaded.")
        except Exception as e:
            print(f"  [Pipeline] Generator unavailable ({e}). Using API fallback.")

        print("="*55 + "\n")

    def _api_generate(self, prompt: str) -> str:
        """Anthropic API fallback when local generator not available."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp   = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 250,
                system     = (
                    "You are a concise telecom customer support agent. "
                    "Always cite sources as [SOURCE: doc_id, section_id]. "
                    "Escalate when KB is insufficient. Be empathetic and brief."
                ),
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            return f"[Generation error: {e}]"

    def run(
        self,
        query:   str,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Full pipeline inference.

        Args:
            query:   Customer's query string
            history: List of previous turns [{"role": "user"/"agent", "utterance": str}]

        Returns:
            {
              system, query, answer, citations,
              tool_trace, escalated, ticket_id,
              confidence, retrieved, latency_ms
            }
        """
        from src.tools.tool_executor import should_escalate
        from src.generation.train_generator import build_input_prompt

        t0      = time.time()
        history = history or []

        tool_trace    = []
        all_retrieved = []
        outage_info   = None
        ticket_id     = None
        escalated     = False

        # ── Step 1: Tool policy — decide what to call ──────────────
        planned_calls = tool_policy(query, history)

        # ── Step 2: Execute tool loop ──────────────────────────────
        for tool_name, params in planned_calls:

            # Deferred GetPolicy: fill section_id from SearchKB results
            if params.get("_deferred") and tool_name == "GetPolicy":
                if all_retrieved:
                    params = {"section_id": all_retrieved[0].get("section_id", "")}
                else:
                    continue   # skip if nothing retrieved yet

            result = self.tools.execute(tool_name, params)
            tool_trace.append({
                "tool":   tool_name,
                "params": {k: v for k, v in params.items() if not k.startswith("_")},
                "output_summary": _summarise_tool_output(tool_name, result),
            })

            # Accumulate evidence
            if tool_name == "SearchKB":
                all_retrieved.extend(result.get("passages", []))
            elif tool_name == "CheckNetworkStatus":
                outage_info = result
            elif tool_name == "GetPolicy" and "policy_text" in result:
                # Add GetPolicy result as a pseudo-passage for the generator
                all_retrieved.insert(0, {
                    "doc_id":     result["doc_id"],
                    "section_id": result["section_id"],
                    "heading":    result["heading"],
                    "text":       result["policy_text"],
                    "dense_score": 1.0,   # authoritative — treat as top result
                })

        # Deduplicate passages
        seen   = set()
        unique = []
        for p in all_retrieved:
            key = p.get("section_id", p.get("doc_id", ""))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        all_retrieved = unique[:5]

        # ── Step 3: Escalation check ───────────────────────────────
        top_score   = all_retrieved[0].get("dense_score",
                       all_retrieved[0].get("rerank_score", 0.0)) if all_retrieved else 0.0
        esc, reason = should_escalate(query, all_retrieved, top_score, history)

        if esc:
            escalated = True
            # Determine severity
            severity = "critical" if reason == "security_concern" else \
                       "high"     if "high_value" in reason       else \
                       "medium"
            category = _infer_ticket_category(query)
            ticket_result = self.tools.execute("CreateTicket", {
                "summary":  f"[{reason}] {query[:150]}",
                "category": category,
                "severity": severity,
            })
            ticket_id = ticket_result.get("ticket_id")
            tool_trace.append({
                "tool":   "CreateTicket",
                "params": {"summary": query[:80], "category": category, "severity": severity},
                "output_summary": f"Ticket {ticket_id} created ({severity})",
            })

        # ── Step 4: Build generation context ──────────────────────
        # Inject outage info as top context if relevant
        gen_context = list(all_retrieved)
        if outage_info and outage_info.get("active_incident"):
            outage_passage = {
                "doc_id":     "network_status_live",
                "section_id": f"outage_{outage_info.get('incident_id','')}",
                "heading":    f"Live Network Status — {outage_info['region']}",
                "text":       (
                    f"ACTIVE {outage_info['status'].upper()} in {outage_info['region']}: "
                    f"{outage_info.get('incident_summary','')} "
                    f"Estimated resolution: {outage_info.get('estimated_resolution','unknown')}. "
                    f"Compensation eligible: {outage_info.get('compensation_eligible', False)}."
                ),
                "dense_score": 1.0,
            }
            gen_context.insert(0, outage_passage)

        # ── Step 5: Generate answer ────────────────────────────────
        if self.generator:
            gen_result = self.generator.generate(query, gen_context, history)
            answer     = gen_result["answer"]
            citations  = gen_result["citations"]
            raw_output = gen_result["raw_output"]
        else:
            # API fallback: inject context into prompt
            from src.generation.train_generator import build_input_prompt
            prompt    = build_input_prompt(query, gen_context, history)
            raw_output = self._api_generate(prompt)
            # Parse citations from API output
            citations = []
            for m in re.finditer(r"\[SOURCE:\s*([^,\]]+),\s*([^\]]+)\]", raw_output):
                citations.append({"doc_id": m.group(1).strip(),
                                   "section_id": m.group(2).strip()})
            answer = re.sub(r"\[SOURCE:[^\]]+\]", "", raw_output).strip()

        # ── Step 6: Augment answer with ticket / outage info ───────
        if escalated and ticket_id:
            ticket_note = (
                f" Your issue has been escalated — ticket {ticket_id} created. "
                f"Our team will contact you within the specified time."
            )
            if ticket_note.strip() not in answer:
                answer = answer + ticket_note

        if escalated and ticket_id:
            ticket_note = (
                f" Your issue has been escalated — ticket {ticket_id} created. "
                f"Our team will contact you within the specified time."
            )
            if ticket_note.strip() not in answer:
                answer = answer + ticket_note


        # =========================
        # ADD THIS NEW BLOCK HERE
        # =========================
        if not answer or len(answer.strip()) < 5:
            answer = (
                "Please provide more details about your issue. "
                "Include your city, service type (4G/5G/Broadband), "
                "and describe the problem."
            )

        # REMOVE FAKE DOC_ID CITATIONS
        clean_citations = []
        for c in citations:
            if isinstance(c, dict):
                if c.get("doc_id") != "doc_id":
                    clean_citations.append(c)

        citations = clean_citations


        latency_ms = round((time.time() - t0) * 1000, 1)

        return {
            "system":     "full_pipeline",
            "query":      query,
            "answer":     answer,
            "citations":  citations,
            "tool_trace": tool_trace,
            "escalated":  escalated,
            "ticket_id":  ticket_id,
            "confidence": round(float(top_score), 4),
            "retrieved":  all_retrieved[:3],
            "latency_ms": latency_ms,
        }

    def batch_run(self, test_cases: List[Dict]) -> List[Dict]:
        """Runs the full pipeline on all test cases for evaluation."""
        results = []
        for i, case in enumerate(test_cases):
            print(f"  [{i+1:03d}/{len(test_cases)}] {case['query'][:60]}...")
            result = self.run(case["query"], case.get("history", []))
            result.update({
                "test_id":          case.get("test_id"),
                "gold_doc_id":      case.get("gold_doc_id"),
                "gold_section_id":  case.get("gold_section_id"),
                "gold_answer":      case.get("gold_answer"),
                "should_escalate":  case.get("should_escalate", False),
                "requires_outage_check": case.get("requires_outage_check", False),
                "domain":           case.get("domain", "unknown"),
                "source":           case.get("source", "unknown"),
            })
            results.append(result)
        return results


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _summarise_tool_output(tool_name: str, result: Dict) -> str:
    if tool_name == "SearchKB":
        n = len(result.get("passages", []))
        top = result.get("passages", [{}])[0].get("doc_id", "?") if n else "none"
        return f"{n} passages, top={top}"
    elif tool_name == "GetPolicy":
        return result.get("heading", result.get("error", "?"))
    elif tool_name == "CreateTicket":
        return f"ticket={result.get('ticket_id','?')} eta={result.get('eta_hours','?')}h"
    elif tool_name == "CheckNetworkStatus":
        return f"status={result.get('status','?')} incident={result.get('active_incident',False)}"
    return str(result)[:60]


def _infer_ticket_category(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["bill","charge","invoice","payment","refund"]):
        return "billing"
    if any(w in q for w in ["roam","international","abroad"]):
        return "roaming"
    if any(w in q for w in ["network","signal","4g","5g","speed","outage"]):
        return "network"
    if any(w in q for w in ["sim","esim","device","phone","return"]):
        return "device"
    if any(w in q for w in ["hack","fraud","unauthorized","security"]):
        return "fraud"
    if any(w in q for w in ["account","port","kyc","otp"]):
        return "account"
    return "general"


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Interactive demo on 5 test queries")
    parser.add_argument("--eval", action="store_true",
                        help="Run full evaluation on test_cases.jsonl")
    parser.add_argument("--query", type=str,
                        help="Single query to run")
    args = parser.parse_args()

    pipeline = TelecomCopilot()

    DEMO_QUERIES = [
        "How do I dispute a wrong charge on my bill?",
        "My 4G is not working in Mumbai since this morning.",
        "I was charged Rs. 12000 for roaming on a 2-day trip.",
        "Can I downgrade my postpaid plan this month?",
        "This billing issue has been going on for 10 days and nobody resolved it.",
    ]

    if args.query:
        result = pipeline.run(args.query)
        print(f"\nQuery   : {result['query']}")
        print(f"Answer  : {result['answer']}")
        print(f"Cites   : {result['citations']}")
        print(f"Tools   : {[t['tool'] for t in result['tool_trace']]}")
        print(f"Escalate: {result['escalated']} (ticket: {result['ticket_id']})")
        print(f"Latency : {result['latency_ms']} ms")

    elif args.demo or not args.eval:
        print("\n" + "="*60)
        print("  FULL PIPELINE — DEMO")
        print("="*60)
        for q in DEMO_QUERIES:
            result = pipeline.run(q)
            print(f"\nQ : {q}")
            print(f"A : {result['answer'][:200]}")
            print(f"  Citations : {result['citations']}")
            print(f"  Tools     : {[t['tool'] for t in result['tool_trace']]}")
            print(f"  Escalated : {result['escalated']} | Ticket: {result['ticket_id']}")
            print(f"  Latency   : {result['latency_ms']} ms")
            print("-"*40)

    if args.eval:
        test_path = Path("data/processed/test_cases.jsonl")
        if not test_path.exists():
            print("No test_cases.jsonl. Run Week 1 training_data_builder first.")
        else:
            with open(test_path) as f:
                cases = [json.loads(l) for l in f]
            print(f"\nRunning full pipeline on {len(cases)} test cases...")
            results = pipeline.batch_run(cases)

            out = Path("data/processed/full_system_results.jsonl")
            with open(out, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"Saved {len(results)} results → {out}")

            # Auto-compute metrics
            sys.path.insert(0, ".")
            from src.evaluation.evaluator import evaluate, print_report
            metrics = evaluate(results, label="full_system")
            print_report(metrics)

            mpath = Path("data/processed/full_system_results.eval.json")
            with open(mpath, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"Metrics saved → {mpath}")
